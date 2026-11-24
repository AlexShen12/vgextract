"""Experimental appearance retrieval. Tracks, champion names and skins are separate."""
from collections import Counter, defaultdict, deque
from pathlib import Path
import time

import numpy as np

from .core import atomic_json, digest, read_json
from .media import file_hash

ARMS = tuple((gallery, context) for gallery in ("gameplay", "rendered", "combined") for context in (False, True))
BUDGET_SECONDS = 8*60*60
UNKNOWN_NAMES={"unknown","null","none","unresolved","?",""}


def canonical_champion(value):
    return None if value is None or (isinstance(value,str) and value.strip().casefold() in UNKNOWN_NAMES) else value


def reference_approved(entry):
    human=entry.get("reviewer_kind")=="human" and entry.get("verified") is True
    model=(entry.get("reviewer_kind")=="model" and entry.get("verified") is not True
           and entry.get("source_reviewed") is True and entry.get("identity_approved") is True)
    return bool(entry.get("reviewer") and entry.get("identity_evidence") and (human or model))


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-9:
        raise ValueError("Invalid appearance embedding")
    return vector/np.linalg.norm(vector)


class BodyGallery:
    def __init__(self, entries, vectors, *, encoder_signature, kind="combined", excluded_groups=()):
        if kind not in {"gameplay", "rendered", "combined"} or len(entries) != len(vectors):
            raise ValueError("Invalid body gallery")
        selected = []
        for entry, vector in zip(entries, vectors):
            if entry["reference_kind"] not in {"gameplay","rendered"}:
                raise ValueError("Unsupported body reference kind")
            if kind != "combined" and entry["reference_kind"] != kind:
                continue
            if entry.get("match_group") in excluded_groups:
                raise ValueError("Evaluation match leaked into the body gallery")
            if not reference_approved(entry):
                raise ValueError("Body references need independently reviewed identities/render audits")
            if not canonical_champion(entry.get("champion")) or not entry.get("sha256"):
                raise ValueError("Body reference needs identity and source hash")
            selected.append((entry, unit(vector)))
        if not selected:
            raise ValueError("Empty body gallery")
        self.entries, self.vectors = zip(*selected)
        self.matrix = np.asarray(self.vectors)
        self.signature = digest({"entries": list(self.entries), "vectors": self.matrix.tolist(),
                                 "encoder": encoder_signature, "kind": kind})

    def match(self, vector, policy, roster=None):
        if not policy.get("development_freeze"):
            raise ValueError("Body matching thresholds must be frozen on development references")
        scores = self.matrix @ unit(vector)
        grouped = defaultdict(list)
        for score, entry in zip(scores, self.entries):
            if roster is None or entry["champion"] in roster:
                grouped[entry["champion"]].append((float(score), entry))
        ranked = sorted(((max(values, key=lambda p:p[0])[0], champion, values)
                         for champion, values in grouped.items()), reverse=True)
        if not ranked:
            return {"champion": None, "skin_id": None, "reason": "no compatible references", "score": None}
        score, champion, values = ranked[0]
        margin = score-(ranked[1][0] if len(ranked) > 1 else -1)
        best = max(values, key=lambda p:p[0])[1]
        accepted = score >= policy["score"] and margin >= policy["margin"]
        return {"champion": champion if accepted else None, "skin_id": None,
                "best_reference_skin": best.get("skin_id"), "score": score, "margin": margin,
                "reason": "accepted appearance" if accepted else "ambiguous appearance",
                "reference_id": best["id"], "reference_sha256": best["sha256"], "gallery_id": self.signature,
                "scores_are_probabilities": False,
                "candidates": [{"champion": name, "score": s} for s,name,_ in ranked[:3]]}


class TrackIdentity:
    def __init__(self):
        self.windows, self.times, self.contexts, self.evidence = {}, {}, {}, {}

    def update(self, row, prediction):
        track = row["track_id"]
        context = (row["segment_id"], row["owner_epoch"], row.get("track_epoch", 0), row.get("name_key"))
        if self.contexts.get(track) != context or row["time"]-self.times.get(track, row["time"]) > .5:
            self.windows[track] = deque(maxlen=5)
            self.evidence[track] = deque(maxlen=5)
        self.contexts[track] = context
        if track in self.times and row["time"] <= self.times[track]:
            raise ValueError("Repeated/out-of-order body samples cannot vote")
        if track in self.times and row["time"]-self.times[track] < .249:
            raise ValueError("Identity votes require 4 FPS sampling")
        self.times[track] = row["time"]
        window = self.windows[track]
        name = canonical_champion(prediction.get("champion")) if row.get("visible", False) else None
        if name and any(previous and previous != name for previous in window):
            window.clear()
            self.evidence[track].clear()
        window.append(name)
        self.evidence[track].append(row.get("evidence_id"))
        votes = Counter(n for n in window if n)
        winner = next((n for n, count in votes.items() if count >= 3), None)
        result = {**prediction, "champion": winner if name else None, "skin_id": None,
                "identity_votes": votes.get(winner, 0), "track_id": track,
                "team": row.get("team", "unknown"), "segment_id": row["segment_id"],
                "owner_epoch": row["owner_epoch"], "time": row["time"]}
        if "evidence_id" in row:
            result["identity_vote_ids"] = [identifier for identity, identifier in zip(window, self.evidence[track])
                                           if winner and identity == winner and identifier]
        return result


class FrozenEncoder:
    """Load only a locally pinned DINOv2 checkout and state dict; never torch.hub online."""
    def __init__(self, lock, device="cpu"):
        import torch
        self.torch = torch
        lock = read_json(Path(lock))
        code = Path(lock["code_root"]).resolve(); weights = Path(lock["weights"]).resolve()
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code, check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=code, check=True, capture_output=True, text=True).stdout
        if commit != lock["code_revision"] or dirty or file_hash(weights) != lock["weights_sha256"] or lock["architecture"] != "dinov2_vits14":
            raise ValueError("Encoder checkout/weights differ from the frozen pin")
        from .core import implementation_hash
        import PIL
        self.signature = digest({"pin":{k: lock[k] for k in ("code_revision", "weights_sha256", "architecture")},
            "implementation":implementation_hash(["body_identity.py"]),"torch":torch.__version__,
            "cuda":torch.version.cuda,"numpy":np.__version__,"Pillow":PIL.__version__,"dtype":"float32",
            "device":torch.cuda.get_device_name(0) if str(device).startswith("cuda") else str(device)})
        self.model = torch.hub.load(str(code), "dinov2_vits14", source="local", pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model.eval().requires_grad_(False).to(device); self.device = device

    def encode(self, image, mask):
        # Appearance inputs exclude background, bars, names and HUD pixels.
        rgb = np.asarray(image.convert("RGB")); mask = np.asarray(mask.convert("L")) > 127
        if mask.shape != rgb.shape[:2] or not mask.any():
            raise ValueError("Body embedding requires a nonempty aligned foreground mask")
        from PIL import Image
        rgb = np.where(mask[...,None], rgb, 127).astype("uint8")
        return self.describe(Image.fromarray(rgb).resize((224, 224)))[0]

    def describe(self, image, patch_layer=None):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        value = (rgb-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
        tensor = self.torch.from_numpy(value.transpose(2,0,1).copy()).float().unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            if patch_layer is None:
                result = self.model(tensor).float().cpu().numpy()[0]
                return unit(result), None
            output = self.model.forward_features(tensor)
            if patch_layer == -1:
                patches = output["x_norm_patchtokens"]
            else:
                layer = len(self.model.blocks) + patch_layer
                patches = self.model.get_intermediate_layers(tensor, n=[layer])[0]
            patches = self.torch.nn.functional.normalize(patches.float(), dim=-1).cpu().numpy()[0]
            return unit(output["x_norm_clstoken"].float().cpu().numpy()[0]), patches


def healthbar_proposals(image, profile, *, require_coherent_fill=False):
    """Anonymous proposals from each calibrated palette, without selected HUD-health filtering."""
    from .action_body import candidates
    result = []
    for team, color in (("friendly", "green"), ("friendly", "blue"), ("enemy", "red")):
        cfg = dict(profile["body_binder"])
        cfg["hud_health_box"] = [0, 0, 1, 1]
        cfg["health_tolerance"] = 1.0
        view = image
        if color == "red":
            r,g,b = image.convert("RGB").split()
            from PIL import Image
            view = Image.merge("RGB", (g,r,b))
        elif color == "blue":
            r,g,b = image.convert("RGB").split()
            from PIL import Image
            view = Image.merge("RGB", (r,b,g))
        found, _ = candidates(view, {**profile, "body_binder": cfg}, require_coherent_fill=require_coherent_fill)
        for candidate in found:
            # Remove the overhead bar/level/name area from proposed appearance pixels.
            box = candidate["actor_region"][:]
            box[1] = max(box[1], candidate["bar_box"][3]+2)
            if box[1] < box[3]:
                result.append({**candidate, "actor_region": box, "team": team,
                               "champion": None, "proposal_only": True})
    return result


def run_arms(root, prepared, gallery_entries, gallery_vectors, query_vectors, policy, only_arm=None):
    """Six paired arms share immutable proposals/tracks and identical query embeddings."""
    from .core import write_rows
    root = Path(root)
    if len(prepared["samples"]) != len(query_vectors):
        raise ValueError("Each body sample needs one embedding")
    runtime = {"started": time.time(), "encoder_signature": prepared["encoder_signature"]}
    reports = []
    for gallery_kind, context in ARMS:
        gallery = BodyGallery(gallery_entries, gallery_vectors, encoder_signature=prepared["encoder_signature"],
                              kind=gallery_kind, excluded_groups=prepared["evaluation_match_groups"])
        rows = []
        arm = gallery_kind+("-context" if context else "-appearance")
        if only_arm is not None and arm != only_arm:
            continue
        for output in predict_samples(prepared["samples"], query_vectors, gallery, policy, context):
            rows.append({**output, "arm": arm, "experimental": True})
            # Checkpoint completed samples, including early failures/time-limit recovery.
            if len(rows)%32 == 0:
                write_rows(root/arm/"predictions.jsonl", rows)
        write_rows(root/arm/"predictions.jsonl", rows)
        reports.append({"arm": arm, "samples": len(rows), "named": sum(r["champion"] is not None for r in rows),
                        "gallery_id": gallery.signature})
    if not reports:
        raise ValueError("Unknown body comparison arm")
    result = {"experimental": True, "release_passed": False, "arms": reports, "runtime": runtime,
              "prepared_hash": digest(prepared), "policy": policy, "gpu_ceiling_seconds": BUDGET_SECONDS}
    atomic_json(root/((only_arm+"-report.json") if only_arm else "report.json"), result)
    return result


def predict_samples(samples, vectors, gallery, policy, context=False):
    tracker=TrackIdentity()
    for sample,vector in sorted(zip(samples,vectors),key=lambda p:(p[0]["time"],p[0]["id"])):
        prediction=gallery.match(vector,policy,sample.get("accepted_roster") if context else None)
        hud=sample.get("accepted_hud_champion") if context and sample.get("hud_body_binding_accepted") else None
        if hud and prediction.get("champion")!=hud:
            prediction.update(champion=None,reason="appearance conflicts with accepted HUD/body binding")
        yield {**tracker.update(sample,prediction), "id": sample["id"],
               **{key: sample[key] for key in ("source_sha256", "frame_id", "body_box", "split") if key in sample}}
