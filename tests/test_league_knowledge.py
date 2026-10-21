from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation.league.knowledge import (
    CD, DD, JsonCache, Patch, ReferenceError, Resolver, main,
    named_bin_bindings, normalize_item_stats, resolve_tooltip,
    run_knowledge_loop,
)


PIN = Patch("26.13", "16.13.1", "16.13")
DD_BASE = f"{DD}/cdn/16.13.1/data/en_US/"
CD_BASE = f"{CD}/16.13/"


def spell(**changes):
    return {"id": "AatroxQ", "name": "The Darkin Blade", "image": {"full": "AatroxQ.png"},
            "description": "A visible attack.", "tooltip": "Damage {{ e1 }} (+{{ a1 }}) {{ missing }}",
            "effectBurn": [None, "10/20/30"], "cooldown": [14, 12, 10, 8, 6],
            "vars": [{"key": "a1", "coeff": [0.7], "link": "spelldamage"}], **changes}


def references():
    aatrox = {"id": "Aatrox", "key": "266", "name": "Aatrox"}
    monkey = {"id": "MonkeyKing", "key": "62", "name": "Wukong"}
    bin_data = {
        "Characters/Aatrox/Spells/AatroxQ": {
            "mScriptName": "AatroxQ", "mSpell": {
                "DataValues": [{"name": "Missing", "values": [1, 2, 3]}],
                "mSpellCalculations": {"QDamage": {"__type": "GameCalculation", "mFormulaParts": []}},
            }, "mBuff": {"mDescription": "Buff_AatroxQ_Tooltip"}}}
    return {
        DD_BASE + "champion.json": {"version": "16.13.1", "data": {"Aatrox": aatrox, "MonkeyKing": monkey}},
        DD_BASE + "champion/Aatrox.json": {"version": "16.13.1", "data": {"Aatrox": {
            **aatrox, "passive": {"name": "Deathbringer Stance"}, "spells": [spell()]}}},
        DD_BASE + "champion/MonkeyKing.json": {"version": "16.13.1", "data": {"MonkeyKing": {
            **monkey, "spells": []}}},
        DD_BASE + "item.json": {"version": "16.13.1", "data": {"6610": {
            "name": "Sundered Sky", "maps": {"11": True}, "image": {"full": "6610.png"},
            "description": "<stats><attention>400</attention> Health<br>10 Ability Haste</stats>",
            "stats": {"FlatHPPoolMod": 400}, "from": ["1028"]}}},
        CD_BASE + "game/data/characters/aatrox/aatrox.bin.json": bin_data,
        CD_BASE + "plugins/rcp-be-lol-game-data/global/default/v1/champions/266.json": {
            "id": 266, "spells": [{"spellKey": "q", "description": "Static reference",
                                    "dynamicDescription": "Damage @QDamage@"}]},
        CD_BASE + "plugins/rcp-be-lol-game-data/global/default/v1/items.json": [
            {"id": 6610, "description": "<stats>400 Health<br>10 Ability Haste<br>5 Tenacity</stats>"}],
    }


class FixtureCache:
    def __init__(self):
        self.data = references()
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        if url not in self.data:
            raise ReferenceError(f"fixture unavailable: {url}")
        return copy.deepcopy(self.data[url]), {"url": url, "sha256": "fixture", "retrieved_at": "fixture"}


class ItemParsingTests(unittest.TestCase):
    def test_description_fills_missing_haste_without_parsing_passive_damage(self):
        result = normalize_item_stats({"stats": {"FlatHPPoolMod": 400}, "description":
            "<stats><attention>400</attention> Health<br>10 Ability Haste</stats>"
            "<passive>Deal 999 damage</passive>"})
        self.assertEqual(result["values"]["ability_haste:flat"]["value"], 10)
        self.assertEqual(len(result["values"]["health:flat"]["source_fields"]), 2)
        self.assertEqual(len(result["values"]), 2)

    def test_percentages_decimals_and_thousands_keep_units(self):
        result = normalize_item_stats({"stats": {"PercentAttackSpeedMod": 0.125},
            "description": "<stats>12.5% Attack Speed<br>1,200 Health<br>7.5 Ability Haste</stats>"})
        self.assertEqual(result["values"]["attack_speed:percent"]["value"], 12.5)
        self.assertEqual(result["values"]["health:flat"]["value"], 1200)
        self.assertEqual(result["values"]["ability_haste:flat"]["value"], 7.5)
        self.assertFalse(result["conflicts"])

    def test_conflicts_remain_unknown_and_unknown_labels_are_not_guessed(self):
        result = normalize_item_stats({"stats": {"FlatHPPoolMod": 400},
            "description": "<stats>450 Health<br>10 Unknown Stat<br>5-10 Armor</stats>"})
        self.assertNotIn("health:flat", result["values"])
        self.assertEqual(result["conflicts"][0]["stat"], "health:flat")
        self.assertEqual(result["unparsed"], ["10 Unknown Stat", "5-10 Armor"])

    def test_flat_and_percent_move_speed_are_separate(self):
        result = normalize_item_stats({"stats": {"FlatMovementSpeedMod": 25},
            "description": "<stats>25 Move Speed<br>5% Move Speed</stats>"})
        self.assertEqual(set(result["values"]), {"move_speed:flat", "move_speed:percent"})


class TooltipTests(unittest.TestCase):
    def test_effectburn_is_one_based_and_coeff_retains_scaling(self):
        result = resolve_tooltip(spell())
        self.assertIn("10/20/30", result["text"])
        self.assertIn("0.7 × spelldamage", result["text"])
        self.assertEqual(result["bindings"]["a1"]["value"], [0.7])
        self.assertEqual(result["unresolved"], ["missing"])

    def test_zero_is_a_value_but_missing_effects_are_not_zero(self):
        result = resolve_tooltip(spell(tooltip="{{e1}} {{e2}} {{e9}} {{f1}}",
            effectBurn=[None, "0", None], vars=[{"key": "f1", "coeff": 0, "link": "attackdamage"}]))
        self.assertEqual(result["bindings"]["e1"]["value"], "0")
        self.assertEqual(result["bindings"]["f1"]["value"], 0)
        self.assertEqual(result["unresolved"], ["e2", "e9"])

    def test_missing_coefficients_do_not_use_their_array_position(self):
        result = resolve_tooltip(spell(vars=[{"key": "f1", "coeff": [1.2]}]))
        self.assertIn("a1", result["unresolved"])

    def test_unsupported_placeholder_expressions_are_not_reported_resolved(self):
        result = resolve_tooltip(spell(tooltip="{{ speed*100 }}% {{ named.value }} {{ e1 }}"))
        self.assertEqual(result["unresolved"], ["named.value", "speed*100"])
        self.assertEqual(result["status"], "unresolved")
        self.assertIn("{{ speed*100 }}", result["text"])

    def test_exact_bin_named_values_fill_gaps_and_graphs_remain_raw(self):
        doc = references()[CD_BASE + "game/data/characters/aatrox/aatrox.bin.json"]
        bindings, calculations = named_bin_bindings(doc, "AatroxQ")
        result = resolve_tooltip(spell(), bindings)
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["bindings"]["missing"]["value"], [1, 2, 3])
        self.assertEqual(calculations[0]["status"], "requires_calculation_interpreter")

    def test_other_spells_and_conflicting_records_cannot_supply_bindings(self):
        doc = references()[CD_BASE + "game/data/characters/aatrox/aatrox.bin.json"]
        self.assertEqual(named_bin_bindings(doc, "DifferentSpell"), ({}, []))
        doc["duplicate"] = {"mScriptName": "AatroxQ", "mSpell": {
            "DataValues": [{"name": "Missing", "values": [99]}]}}
        self.assertNotIn("missing", named_bin_bindings(doc, "AatroxQ")[0])


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.cache = FixtureCache()
        self.resolver = Resolver(PIN, self.cache)

    def test_aliases_and_numeric_ids_share_the_same_roster_identity(self):
        for query in ["MonkeyKing", "Wukong", "62", "wukong"]:
            with self.subTest(query=query):
                response = self.resolver.handle_query({"query_knowledge": query})
                self.assertEqual(response["reference"]["identity"]["internal_id"], "MonkeyKing")
                self.assertFalse(response["match_observation_verified"])

    def test_unknown_ocr_is_not_fuzzy_matched_or_used_as_a_path(self):
        for query in ["Aatr0x", "../Aatrox", "some-summoner-name", "https://example.com"]:
            self.assertEqual(self.resolver.handle_query({"query_knowledge": query})["status"], "unresolved")
        self.assertTrue(all(url.endswith("champion.json") for url in self.cache.calls))

    def test_champion_fallback_is_pinned_and_has_source_provenance(self):
        response = self.resolver.handle_query({"query_knowledge": "Aatrox"})
        spell_result = response["reference"]["spells"][0]
        self.assertEqual(spell_result["tooltip"]["unresolved"], [])
        self.assertEqual(spell_result["cdragon"]["unresolved_template_tokens"], ["QDamage"])
        self.assertEqual(len(response["reference"]["provenance"]), 3)
        self.assertTrue(all("latest" not in url and "127.0.0.1" not in url for url in self.cache.calls))

    def test_unavailable_fallback_preserves_gaps(self):
        self.cache.data = {k: v for k, v in self.cache.data.items() if not k.startswith(CD)}
        result = self.resolver.handle_query({"query_knowledge": "Aatrox"})
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["reference"]["knowledge_gaps"], {"Q": ["missing"]})
        self.assertEqual(len(result["reference"]["fallback_errors"]), 2)

    def test_passive_variables_remain_explicit_knowledge_gaps(self):
        self.cache.data[DD_BASE + "champion/Aatrox.json"]["data"]["Aatrox"]["passive"]["description"] = "Heals {{ passiveamount }}."
        result = self.resolver.handle_query({"query_knowledge": "Aatrox"})
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["reference"]["knowledge_gaps"]["passive"], ["passiveamount"])

    def test_description_recovery_avoids_unnecessary_cdragon_call(self):
        result = self.resolver.handle_query({"entity_type": "item", "query_knowledge": "6610"})
        self.assertEqual(result["reference"]["stats"]["values"]["ability_haste:flat"]["value"], 10)
        self.assertFalse(any(url.startswith(CD) for url in self.cache.calls))

    def test_requested_missing_stat_triggers_cdragon_and_preserves_source(self):
        result = self.resolver.handle_query({"entity_type": "item", "query_knowledge": "6610",
                                             "required_stats": ["tenacity"]})
        fact = result["reference"]["stats"]["values"]["tenacity:flat"]
        self.assertTrue(fact["source_fields"][0].startswith("cdragon"))
        self.assertEqual(result["reference"]["missing_requested_stats"], [])

    def test_item_conflicts_across_sources_are_not_silently_overwritten(self):
        self.cache.data[CD_BASE + "plugins/rcp-be-lol-game-data/global/default/v1/items.json"][0]["description"] = "<stats>900 Health</stats>"
        result = self.resolver.handle_query({"entity_type": "item", "query_knowledge": "6610",
                                             "required_stats": ["tenacity"]})
        self.assertEqual(result["status"], "partial")
        self.assertNotIn("health:flat", result["reference"]["stats"]["values"])
        self.assertEqual(result["reference"]["missing_requested_stats"], ["tenacity"])

    def test_item_name_collisions_require_an_id(self):
        self.cache.data[DD_BASE + "item.json"]["data"]["9999"] = copy.deepcopy(
            self.cache.data[DD_BASE + "item.json"]["data"]["6610"])
        self.assertEqual(self.resolver.handle_query({"entity_type": "item", "query_knowledge": "Sundered Sky"})["status"], "unresolved")

    def test_buffs_require_a_champion_and_exact_bin_name(self):
        with self.assertRaises(ValueError):
            self.resolver.handle_query({"entity_type": "buff", "query_knowledge": "AatroxQ"})
        result = self.resolver.handle_query({"entity_type": "buff", "champion": "Aatrox", "query_knowledge": "AatroxQ"})
        self.assertEqual(result["reference"]["buff"]["script_name"], "AatroxQ")

    def test_wrong_version_or_detail_identity_is_rejected(self):
        self.cache.data[DD_BASE + "champion.json"]["version"] = "16.20.1"
        self.assertEqual(self.resolver.handle_query({"query_knowledge": "Aatrox"})["status"], "unresolved")
        self.cache.data = references()
        self.cache.data[DD_BASE + "champion/Aatrox.json"]["data"]["Aatrox"]["key"] = "9999"
        self.assertEqual(self.resolver.handle_query({"query_knowledge": "Aatrox"})["status"], "unresolved")

    def test_system_context_is_complete_roster_not_asserted_match_roster(self):
        context = self.resolver.system_context()
        self.assertIn("MonkeyKing|62|Wukong", context)
        self.assertIn("Never invent a match participant", context)

    def test_malformed_or_arbitrary_requests_are_rejected(self):
        for request in [[], {"query_knowledge": 12}, {"query_knowledge": "Aatrox", "url": "https://example.com"},
                        {"query_knowledge": "6610", "entity_type": "item", "required_stats": ["current_hp"]}]:
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.resolver.handle_query(request)

    def test_model_loop_injects_roster_then_reference_and_preserves_video(self):
        video_message = {"role": "user", "content": [{"type": "video", "video": "example.mp4"}]}
        calls = []

        def generate(messages):
            calls.append(copy.deepcopy(messages))
            return '{"query_knowledge":"MonkeyKing"}' if len(calls) == 1 else "Wukong, if confirmed in the frame."

        result = run_knowledge_loop(generate, [video_message], self.resolver)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(calls[0][1], video_message)
        self.assertIn("MonkeyKing|62|Wukong", calls[0][0]["content"])
        self.assertIn('"name": "Wukong"', calls[1][-1]["content"])
        self.assertEqual(len(result["knowledge_queries"]), 1)

    def test_model_loop_is_bounded_even_on_repeated_invalid_requests(self):
        def generate(messages):
            return {"query_knowledge": "Aatrox", "url": "https://example.com"}

        result = run_knowledge_loop(generate, [], self.resolver, max_queries=2)
        self.assertEqual(result["status"], "lookup_limit")
        self.assertIsNone(result["answer"])
        self.assertEqual(len(result["knowledge_queries"]), 2)
        self.assertEqual(result["knowledge_queries"][0]["response"]["status"], "invalid_request")

    def test_query_budget_can_disable_retrieval(self):
        result = run_knowledge_loop(lambda _: {"query_knowledge": "Aatrox"}, [], self.resolver, max_queries=0)
        self.assertEqual(result["status"], "lookup_limit")
        self.assertEqual(result["knowledge_queries"], [])


class CacheAndCliTests(unittest.TestCase):
    def test_patch_mismatch_and_mutable_versions_are_rejected(self):
        for args in [("26.13", "latest", "16.13"), ("26.13", "16.13.1", "16.20"),
                     ("26.13", "16.20.1", "16.20"), ("26.13", "16.13.1", "16.13", "ko_KR")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Patch(*args)

    def test_cache_hash_is_verified_and_offline_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as root:
            cache = JsonCache(Path(root), offline=True)
            url = DD_BASE + "champion.json"
            with patch("annotation.league.knowledge.urlopen") as network:
                with self.assertRaises(ReferenceError):
                    cache.get(url)
                raw = json.dumps({"version": "16.13.1"})
                entry = {"url": url, "raw_json": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest(), "retrieved_at": "fixture"}
                file = Path(root) / (hashlib.sha256(url.encode()).hexdigest() + ".json")
                file.write_text(json.dumps(entry))
                self.assertEqual(cache.get(url)[0]["version"], "16.13.1")
                entry["raw_json"] = "{}"
                file.write_text(json.dumps(entry))
                with self.assertRaises(ReferenceError):
                    cache.get(url)
                network.assert_not_called()

    def test_cli_intercepts_json_request(self):
        import io
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "patch.json"
            manifest.write_text(json.dumps(PIN.__dict__))
            output = io.StringIO()
            with patch("annotation.league.knowledge.JsonCache", return_value=FixtureCache()), \
                    patch("sys.stdin", io.StringIO('{"query_knowledge":"MonkeyKing"}')), patch("sys.stdout", output):
                status = main(["--manifest", str(manifest), "respond"])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["reference"]["identity"]["name"], "Wukong")


if __name__ == "__main__":
    unittest.main()
