"""Official versioned ability descriptions, independent of footage mechanics."""
import hashlib
import io
from pathlib import Path
import re

from .core import atomic_json, read_json
from .knowledge import DD, JsonCache, plain_text, resolve_tooltip
from .visual_assets import download


def cache_abilities(cache, reference_version="16.13.1", champions=("all",), *, offline=False):
    from PIL import Image
    if not re.fullmatch(r"\d+\.\d+\.\d+", reference_version):
        raise ValueError("Ability reference version must be an explicit Data Dragon build")
    root = Path(cache)/"abilities"/reference_version
    raw_cache = JsonCache(root/"raw", offline=offline)
    prefix = f"{DD}/cdn/{reference_version}"
    roster, roster_source = raw_cache.get(prefix+"/data/en_US/champion.json")
    if roster.get("version") != reference_version:
        raise ValueError("Ability roster version mismatch")
    names = sorted(roster["data"]) if list(champions) == ["all"] else sorted(set(champions))
    if not names or set(names)-roster["data"].keys():
        raise ValueError("Ability champions must use official roster identifiers")
    target = root/"catalog.json"
    previous = read_json(target) if target.exists() else {"champions": {}}
    previous_icons = {a["icon"]["path"]: a["icon"]["sha256"] for c in previous["champions"].values() for a in c["abilities"]}
    catalog = {"schema_version": 1, "reference_version": reference_version, "locale": "en_US",
               "source_patch": None, "mechanics_enabled": False, "roster_provenance": roster_source,
               "champions": dict(previous["champions"]), "failures": dict(previous.get("failures", {})),
               "status": "downloading", "scope": "official_static_descriptions_only"}
    def icon(kind, filename):
        url = prefix+"/img/"+kind+"/"+filename
        if Path(filename).name != filename:
            raise ValueError("Invalid official ability icon filename")
        path = root/"icons"/kind/filename
        if not path.exists():
            if offline:
                raise ValueError("Official ability icon not cached for offline use")
            raw = download(url)
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(raw); temporary.replace(path)
        raw = path.read_bytes()
        expected = previous_icons.get(str(path.relative_to(root)))
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Cached official icon hash mismatch")
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        return {"url": url, "path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}
    for name in names:
        try:
            detail, source = raw_cache.get(prefix+"/data/en_US/champion/"+name+".json")
            if detail.get("version") != reference_version or set(detail["data"]) != {name}:
                raise ValueError("Official champion reference identity/version mismatch")
            champion = detail["data"][name]
            if len(champion.get("spells", [])) != 4:
                raise ValueError("Official reference does not contain four ordered ability slots")
            passive = champion["passive"]
            abilities = [{"slot": "passive", "id": name+":passive", "name": passive["name"],
                          "description_html": passive.get("description", ""),
                          "description": plain_text(passive.get("description", "")),
                          "icon": icon("passive", passive["image"]["full"])}]
            for slot, spell in zip("QWER", champion["spells"]):
                abilities.append({"slot": slot, "id": name+":"+slot, "official_spell_id": spell["id"],
                    "name": spell["name"], "description_html": spell.get("description", ""),
                    "description": plain_text(spell.get("description", "")), "tooltip": resolve_tooltip(spell),
                    "icon": icon("spell", spell["image"]["full"])})
            catalog["champions"][name] = {"name": champion["name"], "abilities": abilities, "provenance": source}
            catalog["failures"].pop(name, None)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            catalog["failures"][name] = str(exc)
        atomic_json(target, catalog)
    catalog["roster_count"] = len(roster["data"])
    catalog["requested_count"] = len(names)
    catalog["status"] = "partial" if catalog["failures"] else "complete_all" if len(catalog["champions"]) == len(roster["data"]) else "complete_requested"
    atomic_json(target, catalog)
    return {"status": catalog["status"], "catalog": str(target), "reference_version": reference_version,
            "cached_champions": len(catalog["champions"]), "requested_champions": len(names),
            "failures": catalog["failures"], "source_patch": None, "mechanics_enabled": False}
