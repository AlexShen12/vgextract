"""Populate portable patch references before offline GPU allocations."""
from concurrent.futures import ThreadPoolExecutor
from .core import atomic_json


def cache_references(store, resolver, champions=None, base_only=False):
    roster = resolver.roster()
    resolver._dd("item.json")
    if base_only:
        result = {"patch": roster["patch"], "champions": len(roster["champions"]),
                  "cached_details": 0, "scope": "roster_and_items_only"}
    else:
        names = [c["internal_id"] for c in roster["champions"]] if not champions or champions == ["all"] else champions
        def cache(name):
            response = resolver.handle_query({"entity_type": "champion", "query_knowledge": name})
            reference = response.get("reference", {})
            return {"entity": name, "status": response["status"],
                    "knowledge_gaps": reference.get("knowledge_gaps", {}),
                    "fallback_errors": reference.get("fallback_errors", []), "error": response.get("error")}
        with ThreadPoolExecutor(max_workers=4) as executor:
            details = list(executor.map(cache, names))
        try:
            resolver._cd("plugins/rcp-be-lol-game-data/global/default/v1/items.json")
            item_fallback = "cached"
        except RuntimeError as exc:
            item_fallback = str(exc)
        result = {"patch": roster["patch"], "champions": len(roster["champions"]),
                  "cached_details": sum(d["status"] != "unresolved" for d in details),
                  "details": details, "item_fallback": item_fallback,
                  "scope": "static_patch_only; unresolved formulas remain explicit"}
    atomic_json(store.path("reference_summary.json"), result)
    return result
