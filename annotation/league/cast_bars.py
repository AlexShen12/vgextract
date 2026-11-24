"""CPU selected-player cast bars, reference identities, and correction-aware channels."""
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
from pathlib import Path
import re
import unicodedata

from .core import atomic_json, digest, finite, implementation_hash, read_json, read_rows, uid, write_rows


def text_key(text):
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", text).casefold())


def text_preprocess(image):
    import numpy as np
    from PIL import Image, ImageOps
    image=image.convert("RGB").resize((image.width*4,image.height*4),Image.Resampling.BICUBIC)
    pixels=np.asarray(image,dtype=float)
    glyphs=(pixels.min(axis=2)>=130)&(np.ptp(pixels,axis=2)<=90)
    return ImageOps.expand(Image.fromarray(np.where(glyphs,0,255).astype("uint8")),border=12,fill=255)


class AbilityCatalog:
    def __init__(self, path, aliases=()):
        self.path = Path(path)
        self.data = read_json(self.path)
        self.version = self.data["reference_version"]
        if self.data.get("status") != "complete_all":
            raise ValueError("A complete ability catalog is required")
        self.abilities = {a["id"]: {**a, "champion": name} for name, c in self.data["champions"].items()
                          for a in c["abilities"] if a["slot"] in ("Q", "W", "E", "R")}
        supplements = None
        if len(self.path.parents) >= 3:
            cd_version = self.version.rsplit(".", 1)[0]
            supplement_path = self.path.parents[2]/"skin-references"/(self.version+"_"+cd_version)/"manifest.json"
            if supplement_path.exists():
                supplements = read_json(supplement_path)
                if supplements.get("reference_version") != self.version or supplements.get("communitydragon_version") != cd_version:
                    raise ValueError("Ability supplements use a different patch family")
                for ability in self.abilities.values():
                    supplement = supplements.get("ability_supplements", {}).get(ability["champion"], {}).get(ability["slot"])
                    if supplement:
                        ability["communitydragon"] = {**supplement, "reference_version": cd_version}
        self.names = defaultdict(set)
        for identifier, ability in self.abilities.items():
            self.names[text_key(ability["name"])].add(identifier)
        for alias in aliases:
            if not alias.get("reviewer") or alias.get("verified") is not True or alias["ability_id"] not in self.abilities:
                raise ValueError("Ability aliases require reviewed catalog IDs")
            self.names[text_key(alias["text"])].add(alias["ability_id"])
        self.signature = digest({"catalog": self.data, "aliases": list(aliases), "supplements": supplements})

    def resolve(self, text, champion=None, fuzzy=None):
        key = text_key(text)
        empty = {"ability_id": None, "channel_kind": "unknown", "resolution": "unresolved"}
        if key == "recall":
            return {"ability_id": None, "channel_kind": "recall", "resolution": "exact"}
        if not key:
            return empty
        identifiers = self.names.get(key, set())
        method = "exact"
        if not identifiers and fuzzy:
            if not fuzzy.get("development_freeze"):
                raise ValueError("Fuzzy bar matching requires a development freeze")
            ranked = sorted(((SequenceMatcher(None, key, name).ratio(), name) for name in self.names), reverse=True)
            if (ranked and ranked[0][0] >= fuzzy["score"] and
                    ranked[0][0]-(ranked[1][0] if len(ranked) > 1 else 0) >= fuzzy["margin"]):
                identifiers = self.names[ranked[0][1]]; method = "ocr_tolerant"
        if champion:
            compatible = {i for i in identifiers if self.abilities[i]["champion"] == champion}
            if identifiers and not compatible:
                return {**empty, "resolution": "champion_conflict"}
            identifiers = compatible
        if len(identifiers) != 1:
            return empty
        return {"ability_id": next(iter(identifiers)), "channel_kind": "ability", "resolution": method}

    def publish(self, root):
        path = root/"cast-catalogs"/(self.signature+".json")
        if not path.exists():
            atomic_json(path, {"reference_version": self.version, "catalog_id": self.signature,
                              "gameplay_patch": None, "abilities": self.abilities})


class CastBarReader:
    def __init__(self, root, catalog, geometry, masks=()):
        self.root, self.catalog, self.geometry, self.masks = Path(root), catalog, geometry, list(masks)
        self.signature = digest({"catalog": catalog.signature, "geometry": geometry, "masks": self.masks,
                                 "implementation": implementation_hash(["cast_bars.py", "readers.py", "numeric_reader.py"])})
        for name in ("text_box", "progress_box"):
            box = geometry[name]
            if len(box) != 4 or not all(type(x) is int for x in box) or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]:
                raise ValueError("Invalid native cast-bar geometry")
        catalog.publish(self.root)

    def locate(self, image):
        """Cast and recall bars can occupy different rows within the same HUD layout."""
        import numpy as np
        cfg = dict(self.geometry)
        box = cfg["progress_box"]
        search = cfg.get("search_box", [max(0,box[0]-16), max(0,box[1]-24),
                                        min(image.width,box[2]+140), min(image.height,box[3]+24)])
        search = [search[0], max(search[1],box[1]-24), search[2], min(search[3],box[3]+24)]
        x0,y0,x1,y1 = search
        pixels = np.asarray(image.crop(search).convert("RGB"),dtype=float)
        cyan=(pixels[...,1]>90)&(pixels[...,2]>90)&(pixels[...,1]>pixels[...,0]*1.3)
        rows=cyan.sum(axis=1)
        if not rows.size or rows.max()<max(8,(x1-x0)*.025):return cfg
        chosen=None
        for peak in np.argsort(rows)[::-1]:
            if rows[peak]<max(8,(x1-x0)*.025):break
            start=int(peak)
            while start>0 and rows[start-1]>=rows[peak]*.3:start-=1
            end=int(peak)+1
            while end<len(rows) and rows[end]>=rows[peak]*.3:end+=1
            if not 2<=end-start<=18:continue
            a,b=y0+start,y0+end
            upper=np.asarray(image.crop([box[0],max(0,a-3),box[2],a]).convert("RGB"))
            lower=np.asarray(image.crop([box[0],b,box[2],min(image.height,b+3)]).convert("RGB"))
            borders=[float((p.max(axis=2)<100).mean()) for p in (upper,lower) if p.size]
            if len(borders)==2 and min(borders)>=.35:
                chosen=a;break
        if chosen is None:return cfg
        delta=chosen-box[1]
        for key in ("text_box","progress_box"):
            a,b,c,d=cfg[key]
            cfg[key]=[a,max(0,b+delta),c,min(image.height,d+delta)]
        return cfg

    def read(self, image, frame, identity):
        import numpy as np
        from .numeric_reader import numeric_preprocess
        from .readers import tesseract
        cfg = self.locate(image)
        boxes = [cfg[name] for name in ("text_box", "progress_box")]
        occluded = (not frame.get("layout_valid", True) or any(b[2] > image.width or b[3] > image.height for b in boxes)
                    or any(a < box[2] and c > box[0] and b < box[3] and d > box[1]
                           for a, b, c, d in self.masks for box in boxes))
        crops = [image.crop(box).convert("RGB") for box in boxes]
        crop_hash = digest([[crop.size, hashlib.sha256(crop.tobytes()).hexdigest()] for crop in crops])
        key = digest([self.signature, crop_hash, boxes, occluded])
        directory = self.root/"cast-bar-cache"/key
        cached = directory/"reading.json"
        if cached.exists():
            pixels = read_json(cached)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            for name, crop in zip(("text", "progress"), crops):
                crop.save(directory/(name+".png"))
            p = np.asarray(crops[1], dtype=float)
            cyan = (p[..., 1] > 90) & (p[..., 2] > 90) & (p[..., 1] > p[..., 0]*1.3)
            fraction = float((cyan.mean(axis=0) >= .3).mean())
            raw = {"raw_text": "", "score": 0}
            if not occluded and fraction > cfg.get("minimum_fill", .02):
                text_preprocess(crops[0]).save(directory/"ocr.png")
                raw = tesseract(directory/"ocr.png")
                if (raw.get("score",0)<cfg.get("ocr_confidence",.85) or
                        self.catalog.resolve(raw.get("raw_text",""))["channel_kind"]=="unknown"):
                    numeric_preprocess(crops[0]).save(directory/"ocr-alternate.png")
                    alternate=tesseract(directory/"ocr-alternate.png")
                    if alternate.get("score",0)>raw.get("score",0):raw=alternate
            visibility = ("unknown" if occluded else "absent" if fraction <= cfg.get("minimum_fill", .02)
                          else "readable" if raw.get("score", 0) >= cfg.get("ocr_confidence", .85)
                          and text_key(raw.get("raw_text", "")) else "unreadable")
            pixels = {"raw_text": raw.get("raw_text", ""), "ocr_score": raw.get("score", 0),
                      "visibility": visibility, "progress_fraction": fraction,
                      "text_path": str((directory/"text.png").relative_to(self.root)),
                      "progress_path": str((directory/"progress.png").relative_to(self.root))}
            atomic_json(cached, pixels)
        champion = identity.get("hud_champion")
        resolved = self.catalog.resolve(pixels["raw_text"], champion, cfg.get("fuzzy")) if pixels["visibility"] == "readable" else {
            "ability_id": None, "channel_kind": "unknown", "resolution": "unresolved"}
        return {**pixels, **resolved, "schema_version": 1, "id": uid("cast-bar", frame["id"], identity["owner_id"], self.signature),
                "frame_id": frame["id"], "time": frame["time"], "owner_id": identity["owner_id"],
                "owner_epoch": frame["owner_epochs"][identity["owner_id"]], "segment_id": frame["segment_id"],
                "champion": champion, "catalog_id": self.catalog.signature, "reader_signature": self.signature,
                "reference_version": self.catalog.version, "box": cfg["text_box"], "review_status": "unreviewed"}


def channel_occurrences(rows):
    active, result = {}, []
    for row in sorted(rows, key=lambda r: (r["time"], r["id"])):
        owner = row["owner_id"]
        key = (row["segment_id"], row["owner_epoch"], row["catalog_id"])
        previous = active.get(owner)
        known = row["visibility"] == "readable" and row["channel_kind"] != "unknown" and row.get("review_status") != "rejected"
        identity = (row["channel_kind"], row["ability_id"])
        if previous and (previous["continuity"] != key or row["time"]-previous["last_observed"] > .25
                         or row["visibility"] == "absent" or row.get("review_status") == "rejected"
                         or (known and previous["identity"] != identity)
                         or (known and row["progress_fraction"]+.15 < previous["last_progress"])):
            active.pop(owner); previous = None
        if not known:
            continue
        if previous is None:
            event = {"id": uid("channel", row["id"]), "start": row["time"], "end": row["time"],
                     "owner_id": owner, "owner_epoch": row["owner_epoch"], "segment_id": row["segment_id"],
                     "channel_kind": row["channel_kind"], "ability_id": row["ability_id"],
                     "catalog_id": row["catalog_id"], "observed_text": row["raw_text"],
                     "evidence_ids": [], "frame_ids": [], "terminal_status": "ending_unknown",
                     "sampling_fps": 16, "target": None, "outcome": None}
            result.append(event)
            previous = {"continuity": key, "identity": identity, "event": event}
            active[owner] = previous
        event = previous["event"]
        event["end"] = row["time"]
        event["evidence_ids"].append(row["id"]); event["frame_ids"].append(row["frame_id"])
        previous.update(last_observed=row["time"], last_progress=row["progress_fraction"])
    return result


def rebuild(root):
    root = Path(root)
    rows = {r["id"]: r for r in read_rows(root/"cast_bar_readings.jsonl")}
    for path in sorted((root/"cast-bar-reviews").glob("*.json")):
        correction = read_json(path)
        if correction["record_id"] in rows:
            rows[correction["record_id"]].update(correction["changes"])
    effective = list(rows.values())
    events = channel_occurrences(effective)
    write_rows(root/"effective_cast_bar_readings.jsonl", effective)
    write_rows(root/"channel_occurrences.jsonl", events)
    write_rows(root/"channel_timeline.jsonl", [{**e, "kind": "observed_channel"} for e in events])
    return effective, events


def correct_reading(root, record_id, changes, reviewer):
    import fcntl
    from .core import now
    root = Path(root)
    allowed = {"raw_text", "ability_id", "channel_kind", "visibility", "review_status"}
    if not isinstance(changes, dict) or set(changes)-allowed or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Invalid cast-bar correction")
    directory = root/"cast-bar-reviews"; directory.mkdir(exist_ok=True)
    with (directory/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, _ = rebuild(root)
        original = next((r for r in rows if r["id"] == record_id), None)
        if original is None:
            raise ValueError("Unknown cast-bar reading")
        row = {**original, **changes}
        if row["visibility"] not in {"readable", "unreadable", "unknown", "absent"} or row["channel_kind"] not in {"recall", "ability", "unknown"}:
            raise ValueError("Invalid bar visibility or channel kind")
        if row["review_status"] not in {"unreviewed", "approved", "corrected", "rejected"} or not isinstance(row["raw_text"], str) or len(row["raw_text"]) > 500:
            raise ValueError("Invalid bar review status/text")
        catalog = read_json(root/"cast-catalogs"/(row["catalog_id"]+".json"))
        ability = catalog["abilities"].get(row["ability_id"])
        if (row["channel_kind"] == "ability" and (not ability or (row.get("champion") and ability["champion"] != row["champion"]))) or (row["channel_kind"] != "ability" and row["ability_id"] is not None):
            raise ValueError("Corrected ability conflicts with the source owner/catalog")
        revision = len(list(directory.glob("*.json")))+1
        record = {"record_id": record_id, "changes": changes, "reviewer": reviewer,
                  "created_at": now(), "revision": revision}
        atomic_json(directory/f"{revision:08d}.json", record)
        rebuild(root)
    return record
