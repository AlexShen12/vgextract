"""Reference-assisted HUD identity from agreement between independent ability slots.

Reference asset versions identify pictures only, never the gameplay patch/mechanics.
"""
from pathlib import Path

from .core import digest, read_json, uid
from .media import file_hash
from .readers import IconMatcher

ICON_POLICY = {"version": 1, "score_min": .45, "margin_min": .06,
               "minimum_agreeing_slots": 3, "recheck_seconds": .25}


def guard_kit_changes(readings, identities):
    """A newly recognized kit is a continuity boundary, never an activation."""
    previous, boundaries = {}, set()
    for row in sorted(identities, key=lambda r: r["time"]):
        key = (row["segment_id"], row["owner_epoch"], row["owner_id"])
        name = row.get("hud_champion")
        if name:
            if previous.get(key) and previous[key] != name:
                boundaries.add((row["frame_id"], row["owner_id"]))
            previous[key] = name
    return [{**r, "state": "unknown", "identity_changed": True} if (r["frame_id"], r["owner_id"]) in boundaries
            else r for r in readings]


class AbilityIdentity:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        manifest = read_json(self.path)
        self.version = manifest.get("patch", {}).get("ddragon_version") or manifest.get("reference_version")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Ability identity references need an explicit asset version")
        self.matchers = {}
        entries = [r for r in manifest["assets"] if r["kind"] == "spell" and r.get("owner")
                   and r["id"].rsplit(":", 1)[-1] in ("Q", "W", "E", "R")]
        for entry in entries:
            if entry["id"].rsplit(":", 1)[0] != entry["owner"]:
                raise ValueError("Ability reference owner differs from its identifier")
            path = (self.path.parent / entry["path"]).resolve()
            if not path.is_relative_to(self.path.parent) or file_hash(path) != entry["sha256"]:
                raise ValueError("Ability reference path/hash mismatch")
        self.signature = digest({"assets": entries, "version": self.version, "policy": ICON_POLICY})
        for slot in ("Q", "W", "E", "R"):
            subset = [e for e in entries if e["id"].endswith(":"+slot)]
            if len(subset) < 2:
                raise ValueError("Ability references need multiple champions for every Q/W/E/R slot")
            self.matchers[slot] = IconMatcher(self.path.parent, {"assets": subset},
                {"icon": ICON_POLICY["score_min"], "icon_margin": ICON_POLICY["margin_min"]})

    @staticmethod
    def consensus(readings):
        votes = {}
        for slot, reading in readings.items():
            if reading["acceptance"] == "accepted":
                champion = reading["value"].rsplit(":", 1)[0]
                votes.setdefault(champion, []).append(slot)
        winner = [name for name, slots in votes.items() if len(slots) >= ICON_POLICY["minimum_agreeing_slots"]]
        # Any strong contradictory slot prevents binding, even if three others agree.
        return winner[0] if len(winner) == 1 and len(votes) == 1 else None

    def identify(self, image, owner):
        crops = {s["id"]: image.crop(s["box"]).convert("RGB") for s in owner["slots"]
                 if s["kind"] == "ability" and s["id"] in self.matchers}
        readings = {slot: self.matchers[slot].read(crop, {"kind": "spell"}) for slot, crop in crops.items()}
        return {"champion": self.consensus(readings), "slots": readings,
                "identity_scope": "visible_ability_kit_candidate",
                "image_signature": digest({slot: [crop.size, crop.tobytes().hex()] for slot, crop in crops.items()}),
                "reference_version": self.version, "reference_signature": self.signature,
                "policy": ICON_POLICY, "gameplay_patch": None, "mechanics_enabled": False,
                "scores_are_calibrated_confidence": False}

    def read_frames(self, frames, profile, mode):
        from PIL import Image
        samples, rows = {}, []
        for frame in frames:
            if not frame["layout_valid"]:
                continue
            for owner in profile["owners"]:
                key = (frame["segment_id"], frame["owner_epochs"][owner["id"]], owner["id"])
                previous = samples.get(key)
                if previous is None or frame["time"]-previous["time"] >= ICON_POLICY["recheck_seconds"]-.001:
                    with Image.open(frame["native_path"]) as image:
                        result = self.identify(image, owner)
                    previous = {"time": frame["time"], "frame_id": frame["id"], "result": result,
                                "id": uid("ability-identity", frame["id"], self.signature, owner["id"], result.get("image_signature"))}
                    samples[key] = previous
                rows.append({"id": uid("icon-identity-row", previous["id"], frame["id"], mode),
                    "frame_id": frame["id"], "frame_ids": [previous["frame_id"]], "time": frame["time"],
                    "mode": mode, "owner_id": owner["id"], "segment_id": frame["segment_id"],
                    "owner_epoch": frame["owner_epochs"][owner["id"]], "hud_champion": previous["result"]["champion"],
                    "scene_champion": None, "scene_box": None, "actor_track": None,
                    "actor_description": "HUD owner identified from ability icons", "binding_visible": False,
                    "identity_source": "ability-icons", "reference_assisted": True,
                    "request_id": previous["id"], "evidence": previous["result"],
                    "derived_from_stable_selection": frame["id"] != previous["frame_id"]})
        return rows
