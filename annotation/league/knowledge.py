"""Patch-pinned static knowledge for a VLM; never a source of match state.

Run ``python -m annotation.league.knowledge --help`` for the JSON handoff CLI.
Only the standard library is required. See README.md for the input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from ingestion.common.replay_state import atomic_write_json, utc_now


DD = "https://ddragon.leagueoflegends.com"
CD = "https://raw.communitydragon.org"
# Capture unsupported expressions too: they must remain visible knowledge gaps.
PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
STATS_BLOCK = re.compile(r"<stats\b[^>]*>(.*?)</stats\s*>", re.I | re.S)
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
STAT_LINE = re.compile(rf"^\s*({NUMBER})\s*(%)?\s+([A-Za-z][A-Za-z '\-/]+?)\s*$")

# All normalized percentages use percentage points (0.3 -> 30), not fractions.
STRUCTURED_STATS = {
    "FlatHPPoolMod": ("health", "flat", 1),
    "FlatMPPoolMod": ("mana", "flat", 1),
    "FlatPhysicalDamageMod": ("attack_damage", "flat", 1),
    "FlatMagicDamageMod": ("ability_power", "flat", 1),
    "FlatArmorMod": ("armor", "flat", 1),
    "FlatSpellBlockMod": ("magic_resist", "flat", 1),
    "FlatMovementSpeedMod": ("move_speed", "flat", 1),
    "PercentMovementSpeedMod": ("move_speed", "percent", 100),
    "PercentAttackSpeedMod": ("attack_speed", "percent", 100),
    "FlatCritChanceMod": ("critical_strike_chance", "percent", 100),
    "PercentLifeStealMod": ("life_steal", "percent", 100),
}
STAT_LABELS = {
    "health": "health", "mana": "mana", "attack damage": "attack_damage",
    "ability power": "ability_power", "armor": "armor",
    "magic resist": "magic_resist", "magic resistance": "magic_resist",
    "ability haste": "ability_haste", "move speed": "move_speed",
    "movement speed": "move_speed", "attack speed": "attack_speed",
    "critical strike chance": "critical_strike_chance", "life steal": "life_steal",
    "omnivamp": "omnivamp", "lethality": "lethality", "tenacity": "tenacity",
    "armor penetration": "armor_penetration", "magic penetration": "magic_penetration",
    "base health regen": "base_health_regen", "base mana regen": "base_mana_regen",
    "heal and shield power": "heal_and_shield_power",
}


def plain_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>|</li\s*>", "\n", value, flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", value)).replace("\xa0", " ")


def identity_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    return "".join(c for c in value if c.isalnum())


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_item_stats(item: dict, source: str = "ddragon") -> dict:
    """Merge structured stats and header text without inventing units or formulas."""
    facts: dict[str, dict] = {}
    conflicts, unparsed = [], []

    def add(name: str, value: float, unit: str, path: str) -> None:
        key = f"{name}:{unit}"
        candidate = {"value": value, "unit": unit, "source_fields": [path]}
        if key not in facts:
            facts[key] = candidate
        elif math.isclose(facts[key]["value"], value, abs_tol=1e-8):
            facts[key]["source_fields"].append(path)
        else:
            conflicts.append({"stat": key, "candidates": [facts[key], candidate]})

    for field, value in item.get("stats", {}).items():
        if field in STRUCTURED_STATS and numeric(value):
            name, unit, multiplier = STRUCTURED_STATS[field]
            add(name, value * multiplier, unit, f"{source}.stats.{field}")

    for index, block in enumerate(STATS_BLOCK.findall(item.get("description", ""))):
        for line in plain_text(block).splitlines():
            line = " ".join(line.split())
            if not line:
                continue
            match = STAT_LINE.fullmatch(line)
            label = match.group(3).casefold() if match else ""
            if not match or label not in STAT_LABELS:
                unparsed.append(line)
                continue
            value = float(match.group(1).replace(",", ""))
            add(STAT_LABELS[label], value, "percent" if match.group(2) else "flat",
                f"{source}.description.stats[{index}]: {line}")

    for conflict in conflicts:
        facts.pop(conflict["stat"], None)
    return {"values": facts, "conflicts": conflicts, "unparsed": unparsed,
            "raw_structured_stats": item.get("stats", {})}


def resolve_tooltip(spell: dict, fallback: dict[str, dict] | None = None) -> dict:
    """Resolve only explicit bindings; preserve rank arrays and scaling semantics."""
    variables = {v["key"]: v for v in spell.get("vars", [])
                 if isinstance(v, dict) and isinstance(v.get("key"), str)}
    bindings: dict[str, dict] = {}

    def substitute(match: re.Match) -> str:
        key = match.group(1).strip()
        binding = None
        if re.fullmatch(r"e\d+", key):
            index = int(key[1:])
            effects = spell.get("effectBurn", [])
            if isinstance(effects, list) and 0 < index < len(effects) and effects[index] not in (None, ""):
                binding = {"value": effects[index], "kind": "rank_values",
                           "source_field": f"ddragon.effectBurn[{index}]"}
        elif re.fullmatch(r"[af]\d+", key):
            variable = variables.get(key, {})
            coeff = variable.get("coeff")
            if numeric(coeff) or (isinstance(coeff, list) and coeff and all(numeric(v) for v in coeff)):
                binding = {"value": coeff, "kind": "coefficient", "link": variable.get("link"),
                           "source_field": f"ddragon.vars[{key}].coeff"}
        elif key in {"cost", "cooldown"} and spell.get(key + "Burn") not in (None, ""):
            binding = {"value": spell[key + "Burn"], "kind": "rank_values",
                       "source_field": f"ddragon.{key}Burn"}
        if binding is None:
            binding = (fallback or {}).get(key.casefold())
        if binding is None:
            return match.group(0)
        bindings[key] = binding
        value = binding["value"]
        rendered = "/".join(str(v) for v in value) if isinstance(value, list) else str(value)
        if binding.get("kind") == "coefficient":
            # A coefficient is not a final damage/healing amount.
            rendered += f" × {binding.get('link') or 'unspecified scaling stat'}"
        return rendered

    template = spell.get("tooltip", "")
    rendered = PLACEHOLDER.sub(substitute, template)
    unresolved = sorted({key.strip() for key in PLACEHOLDER.findall(rendered)})
    return {"template": template, "rendered_html": rendered, "text": plain_text(rendered),
            "bindings": bindings, "unresolved": unresolved,
            "status": "unresolved" if unresolved else "resolved_template",
            "note": "Rank values and coefficients are static; no current match values were computed."}


class ReferenceError(RuntimeError):
    """Missing, ambiguous, mismatched, or invalid static reference evidence."""


@dataclass(frozen=True)
class Patch:
    gameplay_patch: str
    ddragon_version: str
    cdragon_version: str
    locale: str = "en_US"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d+\.\d+", self.gameplay_patch):
            raise ValueError("gameplay_patch must be explicit, for example 26.13")
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.ddragon_version):
            raise ValueError("ddragon_version must be a pinned build, not latest")
        gp_major, gp_minor = map(int, self.gameplay_patch.split("."))
        dd_major, dd_minor, _ = map(int, self.ddragon_version.split("."))
        if gp_minor != dd_minor or gp_major not in {dd_major, dd_major + 10}:
            raise ValueError("Gameplay and Data Dragon patch mapping is inconsistent")
        if self.cdragon_version != self.ddragon_version.rsplit(".", 1)[0]:
            raise ValueError("CommunityDragon and Data Dragon must share a patch family")
        if self.locale != "en_US":
            raise ValueError("The current item-stat parser supports en_US only")


class JsonCache:
    """Pinned URL cache with content hashes; supports reproducible offline runs."""

    def __init__(self, root: Path, *, offline: bool = False):
        self.root, self.offline = root, offline

    def get(self, url: str) -> tuple[Any, dict]:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc not in {"ddragon.leagueoflegends.com", "raw.communitydragon.org"}:
            raise ReferenceError("Only the fixed static-reference hosts are supported")
        path = self.root / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists():
            try:
                envelope = json.loads(path.read_text())
                if not isinstance(envelope, dict) or not all(isinstance(envelope.get(k), str)
                        for k in ("url", "sha256", "retrieved_at", "raw_json")):
                    raise ValueError("Invalid cache envelope")
                raw = envelope["raw_json"]
                json.loads(raw)
            except (ValueError, OSError) as exc:
                raise ReferenceError(f"Invalid reference cache: {path}") from exc
            if envelope["url"] != url or hashlib.sha256(raw.encode()).hexdigest() != envelope["sha256"]:
                raise ReferenceError(f"Reference cache integrity check failed: {path}")
        else:
            if self.offline:
                raise ReferenceError(f"Reference not cached for offline use: {url}")
            try:
                request = Request(url, headers={"User-Agent": "gameplay-reference-resolver/0.1"})
                with urlopen(request, timeout=30) as response:
                    # No silent redirects to mutable/latest or other-version data.
                    if response.url != url:
                        raise ReferenceError(f"Unexpected reference redirect: {response.url}")
                    payload = response.read(20_000_001)
                    if len(payload) > 20_000_000:
                        raise ReferenceError("Reference exceeds the 20 MB limit")
                    raw = payload.decode("utf-8")
                json.loads(raw)  # Never cache an error page as reference data.
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReferenceError(f"Static reference unavailable: {url}: {exc}") from exc
            envelope = {"url": url, "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                        "retrieved_at": utc_now(), "raw_json": raw}
            atomic_write_json(path, envelope)
        provenance = {k: envelope[k] for k in ("url", "sha256", "retrieved_at")}
        return json.loads(raw), provenance


def named_bin_bindings(document: dict, spell_id: str) -> tuple[dict, list]:
    """Extract exact named BIN values within an exact spell record, never by proximity.

    Calculation graphs are returned as unresolved evidence rather than evaluated.
    e1/a1 bindings are not guessed from arbitrary raw data-value names.
    """
    candidates: dict[str, list] = {}
    calculations = []
    for path, record in document.items():
        if not isinstance(record, dict):
            continue
        if record.get("mScriptName", "").casefold() != spell_id.casefold() and path.rsplit("/", 1)[-1].casefold() != spell_id.casefold():
            continue
        spell = record.get("mSpell", record)
        for value in spell.get("mDataValues", spell.get("DataValues", [])):
            name = value.get("mName", value.get("name"))
            values = value.get("mValues", value.get("values"))
            if isinstance(name, str) and isinstance(values, list) and values and all(numeric(v) for v in values):
                candidates.setdefault(name.casefold(), []).append({
                    "value": values, "kind": "raw_rank_values",
                    "source_field": f"cdragon.bin:{path}.mDataValues[{name}]",
                    "rank_indexing": "raw; do not assume element zero is learned rank one"})
        for name, calculation in spell.get("mSpellCalculations", {}).items():
            calculations.append({"name": name, "record": path, "raw": calculation,
                                 "status": "requires_calculation_interpreter"})
    bindings = {key: values[0] for key, values in candidates.items()
                if len({json.dumps(v["value"]) for v in values}) == 1}
    return bindings, calculations


class Resolver:
    def __init__(self, patch: Patch, cache: JsonCache):
        self.patch, self.cache = patch, cache

    def _dd(self, path: str) -> tuple[dict, dict]:
        data, provenance = self.cache.get(
            f"{DD}/cdn/{self.patch.ddragon_version}/data/{self.patch.locale}/{path}")
        if not isinstance(data, dict) or data.get("version") != self.patch.ddragon_version:
            raise ReferenceError("Data Dragon response version does not match the requested build")
        return data, provenance

    def _cd(self, path: str) -> tuple[Any, dict]:
        return self.cache.get(f"{CD}/{self.patch.cdragon_version}/{path}")

    def roster(self) -> dict:
        data, provenance = self._dd("champion.json")
        entries = [{"internal_id": c["id"], "numeric_id": str(c["key"]), "name": c["name"],
                    "aliases": [c["id"], str(c["key"]), c["name"]],
                    "icon_url": f"{DD}/cdn/{self.patch.ddragon_version}/img/champion/{c['id']}.png"}
                   for c in data["data"].values()]
        return {"patch": asdict(self.patch), "champions": sorted(entries, key=lambda c: c["internal_id"]),
                "provenance": provenance}

    def _identify(self, query: str, entity_type: str) -> dict:
        if any(c in query for c in "/\\:\n\r"):
            raise ReferenceError("Entity text must be a roster name or ID, not a path or URL")
        if entity_type == "champion":
            entries = self.roster()["champions"]
        elif entity_type == "item":
            data, _ = self._dd("item.json")
            entries = [{"internal_id": key, "numeric_id": key, "name": item["name"]}
                       for key, item in data["data"].items() if item.get("maps", {}).get("11")]
        else:
            raise ValueError("entity_type must be champion, item, or buff")
        key = identity_key(query)
        matches = [e for e in entries if key in {identity_key(e[field])
                   for field in ("internal_id", "numeric_id", "name")}]
        # Numeric IDs disambiguate names; never fuzzy-match OCR strings.
        if not matches:
            raise ReferenceError(f"No exact patch-roster match for {entity_type}: {query!r}")
        if len(matches) != 1:
            raise ReferenceError(f"Ambiguous {entity_type} name {query!r}; use an explicit ID")
        return matches[0]

    def champion(self, query: str) -> dict:
        identity = self._identify(query, "champion")
        data, provenance = self._dd(f"champion/{quote(identity['internal_id'], safe='')}.json")
        champion = data["data"].get(identity["internal_id"])
        if not champion or str(champion.get("key")) != identity["numeric_id"]:
            raise ReferenceError("Champion detail identity does not match the patch roster")
        passive_description = champion.get("passive", {}).get("description", "")
        passive_gaps = sorted(set(PLACEHOLDER.findall(passive_description)
                                  + re.findall(r"@([^@]+)@", passive_description)))
        needs_fallback = bool(passive_gaps) or any(resolve_tooltip(s)["unresolved"] for s in champion.get("spells", []))
        sources, failures = [provenance], []
        bin_data, cd_champion = {}, {}
        if needs_fallback:
            internal = identity["internal_id"].lower()
            for kind, path in [
                ("bin", f"game/data/characters/{internal}/{internal}.bin.json"),
                ("lcu", f"plugins/rcp-be-lol-game-data/global/default/v1/champions/{identity['numeric_id']}.json"),
            ]:
                try:
                    fallback, source = self._cd(path)
                    if not isinstance(fallback, dict):
                        raise ReferenceError("Expected a CommunityDragon JSON object")
                    if kind == "lcu" and str(fallback.get("id")) != identity["numeric_id"]:
                        raise ReferenceError("CommunityDragon champion identity mismatch")
                    if kind == "bin":
                        bin_data = fallback
                    else:
                        cd_champion = fallback
                    sources.append(source)
                except ReferenceError as exc:
                    failures.append(str(exc))
        cd_spells = {s.get("spellKey", "").upper(): s for s in cd_champion.get("spells", [])}
        spells = []
        for slot, spell in zip("QWER", champion.get("spells", [])):
            bindings, calculations = named_bin_bindings(bin_data, spell["id"])
            resolved = resolve_tooltip(spell, bindings)
            cd_spell = cd_spells.get(slot, {})
            spells.append({
                "slot": slot, "id": spell["id"], "name": spell["name"],
                "icon_url": f"{DD}/cdn/{self.patch.ddragon_version}/img/spell/{spell['image']['full']}",
                "description": plain_text(spell.get("description", "")),
                "base_cooldown_by_rank": spell.get("cooldown"),
                "tooltip": resolved,
                "cdragon": {"description": plain_text(cd_spell.get("description", "")),
                            "dynamic_template": cd_spell.get("dynamicDescription"),
                            "raw_named_values": bindings, "calculations": calculations,
                            "unresolved_template_tokens": re.findall(r"@([^@]+)@", cd_spell.get("dynamicDescription", "")),
                            "status": "raw_static_evidence_not_executed"},
            })
        gaps = {s["slot"]: s["tooltip"]["unresolved"] for s in spells if s["tooltip"]["unresolved"]}
        if passive_gaps:
            gaps["passive"] = passive_gaps
        return {"entity_type": "champion", "identity": identity,
                "icon_url": f"{DD}/cdn/{self.patch.ddragon_version}/img/champion/{identity['internal_id']}.png",
                "passive": {"name": champion.get("passive", {}).get("name"),
                            "description": plain_text(passive_description), "unresolved": passive_gaps},
                "spells": spells, "provenance": sources, "fallback_errors": failures,
                "knowledge_gaps": gaps}

    def item(self, query: str, required_stats: list[str] | None = None) -> dict:
        identity = self._identify(query, "item")
        data, source = self._dd("item.json")
        item = data["data"][identity["internal_id"]]
        stats = normalize_item_stats(item)
        sources, failures, fallback = [source], [], None
        required_stats = required_stats or []
        present = {key.split(":")[0] for key in stats["values"]}
        # Description parsing is stage two; CDragon is stage three, when parsing
        # left actual unknown/conflicting fields or an unresolved template.
        if (not stats["values"] or stats["conflicts"] or stats["unparsed"]
                or set(required_stats) - present or "{{" in item.get("description", "")):
            try:
                entries, cd_source = self._cd("plugins/rcp-be-lol-game-data/global/default/v1/items.json")
                if not isinstance(entries, list):
                    raise ReferenceError("Expected a CommunityDragon item array")
                matches = [e for e in entries if str(e.get("id")) == identity["numeric_id"]]
                if len(matches) != 1:
                    raise ReferenceError("CommunityDragon item identity unavailable or ambiguous")
                cd_item = matches[0]
                fallback = {"description": plain_text(cd_item.get("description", "")),
                            "stats": normalize_item_stats(cd_item, "cdragon")}
                sources.append(cd_source)
                blocked = {c["stat"] for c in stats["conflicts"]}
                for key, fact in fallback["stats"]["values"].items():
                    if key not in stats["values"] and key not in blocked:
                        stats["values"][key] = fact
                    elif key in stats["values"] and not math.isclose(stats["values"][key]["value"], fact["value"], abs_tol=1e-8):
                        stats["conflicts"].append({"stat": key, "candidates": [stats["values"].pop(key), fact]})
            except ReferenceError as exc:
                failures.append(str(exc))
        present = {key.split(":")[0] for key in stats["values"]}
        return {"entity_type": "item", "identity": identity, "stats": stats,
                "missing_requested_stats": sorted(set(required_stats) - present),
                "description": plain_text(item.get("description", "")),
                "components": item.get("from", []), "builds_into": item.get("into", []),
                "gold": item.get("gold"),
                "icon_url": f"{DD}/cdn/{self.patch.ddragon_version}/img/item/{item['image']['full']}",
                "cdragon": fallback, "provenance": sources, "fallback_errors": failures}

    def buff(self, query: str, champion_query: str) -> dict:
        """Buff names are champion-scoped raw names, not arbitrary global guesses."""
        identity = self._identify(champion_query, "champion")
        internal = identity["internal_id"].lower()
        data, source = self._cd(f"game/data/characters/{internal}/{internal}.bin.json")
        matches = []
        for path, record in data.items():
            if not isinstance(record, dict) or "mBuff" not in record:
                continue
            names = [path.rsplit("/", 1)[-1], record.get("mScriptName", ""), record.get("ObjectName", "")]
            if identity_key(query) in {identity_key(n) for n in names if n}:
                matches.append({"record_path": path, "script_name": record.get("mScriptName"),
                                "raw": record["mBuff"]})
        if len(matches) != 1:
            raise ReferenceError("Buff requires an exact, unique name in the anchored champion BIN")
        return {"entity_type": "buff", "champion": identity, "buff": matches[0],
                "status": "raw_static_evidence_not_localized", "provenance": [source]}

    def handle_query(self, request: dict) -> dict:
        """Intercept a model JSON request. Only static, patch-pinned fields leave here."""
        allowed = {"query_knowledge", "entity_type", "champion", "ocr_evidence", "required_stats"}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("Unsupported knowledge-request fields")
        query = request.get("query_knowledge")
        if not isinstance(query, str) or not query.strip() or len(query) > 160:
            raise ValueError("query_knowledge must be a nonempty name or ID (max 160 characters)")
        entity_type = request.get("entity_type", "champion")
        required_stats = request.get("required_stats", [])
        if (not isinstance(required_stats, list) or len(required_stats) > 30
                or any(not isinstance(s, str) or s not in set(STAT_LABELS.values()) for s in required_stats)):
            raise ValueError("required_stats must contain recognized canonical stat names")
        if required_stats and entity_type != "item":
            raise ValueError("required_stats applies only to items")
        envelope = {"patch": asdict(self.patch), "query": request,
                    "scope": "static_reference_only", "match_observation_verified": False}
        try:
            if entity_type == "champion":
                result = self.champion(query)
            elif entity_type == "item":
                result = self.item(query, required_stats)
            elif entity_type == "buff":
                anchor = request.get("champion")
                if not isinstance(anchor, str) or not anchor.strip():
                    raise ValueError("Buff queries require an anchored champion")
                result = self.buff(query, anchor)
            else:
                raise ValueError("entity_type must be champion, item, or buff")
        except ReferenceError as exc:
            return {**envelope, "status": "unresolved", "error": str(exc)}
        partial = (result.get("knowledge_gaps") or result.get("fallback_errors")
                   or result.get("missing_requested_stats") or result.get("stats", {}).get("conflicts")
                   or result.get("stats", {}).get("unparsed"))
        return {**envelope, "status": "partial" if partial else "ok", "reference": result}

    def system_context(self) -> str:
        """Complete patch roster plus an OCR/lookup contract, without a match roster."""
        roster = self.roster()
        lines = [f"{c['internal_id']}|{c['numeric_id']}|{c['name']}" for c in roster["champions"]]
        return (
            f"Static reference patch: {json.dumps(asdict(self.patch), sort_keys=True)}\n"
            "Valid champions (internal ID|numeric ID|display name):\n" + "\n".join(lines) + "\n\n"
            "Ground identities in visible HUD text or icon evidence. HUD text can be a player name, "
            "not a champion name. Match OCR only against this roster; unresolved or overlapping "
            "identities stay unknown. Never invent a match participant from this list.\n"
            'To request static knowledge, output JSON only, for example {"query_knowledge":"Smolder"}, '
            'or {"entity_type":"item","query_knowledge":"6610"}. Buff requests require '
            'entity_type=buff and champion=<anchored champion>. Include ocr_evidence when available.\n'
            "References describe possible mechanics, not evidence of a cast, hit, purchase, current "
            "cooldown or buff. Do not use remembered kits to fill missing reference fields. "
            "Keep unresolved variables, raw rank indexing and calculation graphs explicit.\n"
        )


def run_knowledge_loop(
    generate: Callable[[list[dict]], str | dict],
    messages: list[dict],
    resolver: Resolver,
    *,
    max_queries: int = 3,
) -> dict:
    """Model-independent adapter: intercept JSON, retrieve, and re-prompt.

    ``generate`` is the caller's Qwen inference function. Video/image content in
    ``messages`` is passed through. No annotation logs or match state are added.
    Each request, including invalid/repeated ones, consumes the lookup budget.
    """
    if not isinstance(max_queries, int) or isinstance(max_queries, bool) or not 0 <= max_queries <= 20:
        raise ValueError("max_queries must be an integer between zero and twenty")
    conversation = [{"role": "system", "content": resolver.system_context()}, *messages]
    queries = []
    while True:
        output = generate(conversation)
        if isinstance(output, str):
            try:
                request = json.loads(output)
            except json.JSONDecodeError:
                request = None
        elif isinstance(output, dict):
            request = output
        else:
            raise ValueError("generate must return text or a structured JSON object")
        if not isinstance(request, dict) or "query_knowledge" not in request:
            return {"status": "answered", "answer": output, "knowledge_queries": queries,
                    "messages": conversation}
        if len(queries) >= max_queries:
            return {"status": "lookup_limit", "answer": None, "knowledge_queries": queries,
                    "messages": conversation}
        try:
            response = resolver.handle_query(request)
        except ValueError as exc:
            response = {"status": "invalid_request", "error": str(exc)}
        queries.append({"request": request, "response": response})
        conversation.append({"role": "assistant", "content": json.dumps(request, ensure_ascii=False)})
        conversation.append({"role": "user", "content":
            "Static reference response; this does not verify any match observation. "
            "Use the supplied pixels for match facts. Answer or request another bounded lookup.\n"
            + json.dumps(response, ensure_ascii=False, allow_nan=False)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Explicit gameplay/DD/CD patch mapping")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/league_knowledge"))
    parser.add_argument("--offline", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("roster", help="Complete valid champion roster as JSON")
    sub.add_parser("context", help="System-prompt roster and lookup instructions")
    lookup = sub.add_parser("lookup", help="Resolve one champion, item, or anchored buff")
    lookup.add_argument("name")
    lookup.add_argument("--entity-type", choices=["champion", "item", "buff"], default="champion")
    lookup.add_argument("--champion")
    sub.add_parser("respond", help="Read one JSON knowledge request from stdin")
    args = parser.parse_args(argv)
    try:
        patch = Patch(**json.loads(args.manifest.read_text()))
        resolver = Resolver(patch, JsonCache(args.cache_dir, offline=args.offline))
        if args.command == "context":
            print(resolver.system_context())
            return 0
        if args.command == "roster":
            result = resolver.roster()
        elif args.command == "lookup":
            request = {"query_knowledge": args.name, "entity_type": args.entity_type}
            if args.champion:
                request["champion"] = args.champion
            result = resolver.handle_query(request)
        else:
            result = resolver.handle_query(json.load(sys.stdin))
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 2 if result.get("status") == "unresolved" else 0
    except (OSError, ValueError, ReferenceError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
