"""Rebuild derived facts from effective v2 evidence after append-only corrections."""

from __future__ import annotations

from collections import defaultdict
from .core import consistent_review, read_json


@consistent_review
def normalize(store, config):
    key = store.stage_key("normalize", config, ["observations"])
    stage = store.path("stages/normalize.json")
    if (
        stage.exists()
        and read_json(stage).get("key") == key
        and store.path("normalized_observations.jsonl").exists()
    ):
        return store.rows("normalized_observations")
    rows = effective_intake(
        store.effective(store.rows("observations"), "observations"),
        {c["record_id"] for c in store.corrections() if c["target"] == "observations"},
    )
    store.write("normalized_observations", rows)
    store.commit_stage("normalize", key, observations=len(rows))
    return rows


def accepted(row):
    return (
        row.get("review_status") != "rejected"
        and row.get("visibility") == "visible"
        and (
            row.get("acceptance") == "accepted"
            or row.get("review_status") == "human-verified"
        )
    )


def effective_intake(rows, explicitly_corrected=()):
    rows = [dict(row) for row in rows]
    indexed = {row["id"]: row for row in rows}
    corrected = set(explicitly_corrected)
    for row in rows:
        if not row.get("derived_from") or row["id"] in corrected:
            continue
        sources = [indexed[key] for key in row["derived_from"] if key in indexed]
        value = None
        if len(sources) == len(row["derived_from"]):
            if row["field"] in {"kills", "deaths", "assists"} and len(sources) == 1:
                source = sources[0]
                if accepted(source) and isinstance(source["value"], dict):
                    value = source["value"].get(row["field"])
                    row["raw_text"] = source.get("raw_text")
            elif row["field"] == "inventory" and all(
                accepted(source)
                or (
                    source.get("acceptance") == "absent"
                    and source.get("review_status") != "rejected"
                )
                for source in sources
            ):
                value = [
                    source["value"] if accepted(source) else None for source in sources
                ]
        row.update(
            value=value,
            parsed_value=value,
            acceptance="accepted" if value is not None else "unknown",
            visibility="visible" if value is not None else "unknown",
        )
        row["review_status"] = (
            "human-verified"
            if value is not None
            and all(
                source.get("review_status") == "human-verified" for source in sources
            )
            else "unreviewed"
        )
    groups = defaultdict(list)
    for row in rows:
        if row.get("schema_version", 1) >= 2:
            groups[(row["time"], row["slot_id"])].append(row)
    for group in groups.values():
        portraits = [row for row in group if row["field"] == "identity"]
        if not portraits:
            continue
        verified = [
            row
            for row in portraits
            if accepted(row) and row.get("review_status") == "human-verified"
        ]
        identities = {
            row["value"]
            for row in (verified or portraits)
            if accepted(row) and isinstance(row["value"], str)
        }
        identity = next(iter(identities)) if len(identities) == 1 else "unknown"
        for row in group:
            if row["id"] not in corrected:
                row.update(
                    entity=identity,
                    resolved_identity=None if identity == "unknown" else identity,
                )
    return rows
