"""Conservative GPU allocation reservations, including failed attempts."""
import fcntl
import os
from pathlib import Path

from .core import atomic_json, digest, now, read_json


def reconcile(root, submissions, job_receipts=()):
    """Import submitted allocations once per job, retaining the requested wall time."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "gpu-budget.json"
        data = read_json(path) if path.exists() else {"ceiling_seconds": 28800, "recovery_reserve_seconds": 3600, "attempts": []}
        by_job = {row["slurm_job_id"]: row for row in data["attempts"] if row.get("slurm_job_id")}
        for receipt in submissions:
            for job in receipt["jobs"]:
                seconds = job.get("gpu_seconds", receipt.get("gpu_limit_seconds", 0) if job["stage"] == "compare" else 0)
                if type(seconds) is not int or seconds < 0:
                    raise ValueError("Invalid submitted GPU allocation")
                job_id = str(job["job_id"])
                existing = by_job.get(job_id)
                if existing and existing["reserved_seconds"] != seconds:
                    raise ValueError(f"Conflicting allocation for job {job_id}")
                if existing or not seconds:
                    continue
                entry = {"stage": job["stage"], "reserved_seconds": seconds, "created_at": receipt["submitted_at"],
                         "slurm_job_id": job_id, "status": "submitted", "recovery": bool(job.get("recovery")),
                         "attempt": len(data["attempts"]) + 1, "submission_hash": digest(receipt)}
                data["attempts"].append(entry)
                by_job[job_id] = entry
        for receipt in job_receipts:
            entry = by_job.get(str(receipt["job_id"]))
            if entry:
                entry.update(cluster_state=receipt["state"], job_receipt_hash=digest(receipt))
                state = receipt["state"].split()[0].rstrip("+").upper()
                if state == "COMPLETED":
                    entry["status"] = "completed"
                elif state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}:
                    entry["status"] = "failed"
                if "elapsed_seconds" in receipt:
                    entry["elapsed_seconds"] = receipt["elapsed_seconds"]
        reserved = sum(row["reserved_seconds"] for row in data["attempts"])
        regular = sum(row["reserved_seconds"] for row in data["attempts"] if not row["recovery"])
        if reserved > data["ceiling_seconds"] or regular > data["ceiling_seconds"] - data["recovery_reserve_seconds"]:
            raise ValueError("Submitted jobs exceed the campaign allocation ceiling")
        atomic_json(path, data)
    return {"ledger": str(path), "reserved_seconds": reserved, "attempts": len(data["attempts"])}


def reserve(root,stage,seconds,*,recovery=False):
    if type(seconds) is not int or seconds<0:raise ValueError("Invalid GPU allocation")
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/".budget.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/"gpu-budget.json"
        data=read_json(path) if path.exists() else {"ceiling_seconds":28800,"recovery_reserve_seconds":3600,"attempts":[]}
        job_id = os.environ.get("SLURM_JOB_ID")
        existing = next((row for row in data["attempts"] if job_id and row.get("slurm_job_id") == job_id and seconds), None)
        if existing:
            if existing["status"] != "submitted" or existing["reserved_seconds"] != seconds or existing["recovery"] != recovery:
                raise ValueError("Job already claimed or submitted with a different GPU allocation")
            existing["status"] = "reserved"
            atomic_json(path, data)
            return existing
        reserved = sum(row["reserved_seconds"] for row in data["attempts"])
        regular = sum(row["reserved_seconds"] for row in data["attempts"] if not row["recovery"])
        if (reserved + seconds > data["ceiling_seconds"]
                or (not recovery and regular + seconds > data["ceiling_seconds"] - data["recovery_reserve_seconds"])):
            raise ValueError("Body campaign GPU ceiling/recovery reserve exceeded")
        entry={"stage":stage,"reserved_seconds":seconds,"created_at":now(),"slurm_job_id":os.environ.get("SLURM_JOB_ID"),
               "status":"reserved","recovery":recovery,"attempt":len(data["attempts"])+1}
        data["attempts"].append(entry);atomic_json(path,data)
    return entry


def finish(root,attempt,elapsed,status):
    root=Path(root)
    with (root/".budget.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);data=read_json(root/"gpu-budget.json")
        row=next(a for a in data["attempts"] if a["attempt"]==attempt)
        row.update(elapsed_seconds=elapsed,status=status)
        # Reservations are deliberately not refunded on early failure.
        atomic_json(root/"gpu-budget.json",data)


def qualify(root):
    root=Path(root)
    atomic_json(root/"qualification.json",{"passed":False,"status":"checking","created_at":now()})
    try:return _qualify(root)
    except BaseException as exc:
        atomic_json(root/"qualification.json",{"passed":False,"status":"failed","error":str(exc),"created_at":now()})
        raise


def _qualify(root):
    from .core import digest
    root=Path(root);report=read_json(root/"preflight/report.json")
    if report.get("status")!="completed" or report.get("peak_gpu_bytes",float("inf"))>36*1024**3 or report.get("encoder_smoke",{}).get("passed") is not True:
        raise ValueError("Body SAM preflight failed or exceeded the GPU memory gate")
    prepared=read_json(root/"prepared.json")
    if report["prepared_hash"]!=digest(prepared):raise ValueError("Preflight used different inputs")
    if report.get("peak_host_memory_bytes",float("inf"))>80*1024**3:raise ValueError("Body preflight exceeds host-memory gate")
    frames=sum(job.get("end_frame_index", len(job["frames"])) - job.get("seed_frame_index", 0)
               for job in prepared["jobs"])
    projection=report.get("model_load_seconds",float("inf"))+report.get("inference_wall_seconds",float("inf"))*frames/max(1,report.get("sampled_frames",0))
    if projection>1800*.8:raise ValueError("Body tracking projection exceeds 80% of its allocation; prepare a smaller campaign chunk")
    # Independent sparse source masks are required; SAM output cannot self-qualify.
    import numpy as np
    from PIL import Image
    from .media import file_hash
    source=next(c for c in prepared["clips"] if c["id"]==prepared["jobs"][0]["source_clip"]);reference=source.get("seed_reference")
    if not reference or reference.get("source_only") is not True or not reference.get("reviewer"):
        raise ValueError("Independent preflight seed-mask reference is missing")
    if file_hash(Path(reference["path"]))!=reference["sha256"]:raise ValueError("Seed reference changed")
    result=read_json(root/"preflight"/(prepared["jobs"][0]["id"]+"-complete.json"))
    ious=[];seed_inputs=[{"path":reference["path"],"sha256":reference["sha256"]}]
    with Image.open(reference["path"]) as image:ref=np.asarray(image)>127
    for frame in result["frames"]:
        if frame["frame_index"] + report.get("source_frame_offset", 0) != reference["frame_index"]:continue
        for mask in frame["masks"]:
            if file_hash(Path(mask["mask_path"]))!=mask["sha256"]:raise ValueError("Predicted seed mask changed")
            seed_inputs.append({"path":mask["mask_path"],"sha256":mask["sha256"]})
            with Image.open(mask["mask_path"]) as image:pred=np.asarray(image)>127
            if pred.shape!=ref.shape:raise ValueError("Seed mask geometry mismatch")
            ious.append(float((ref&pred).sum()/max(1,(ref|pred).sum())))
    if not ious or max(ious)<.2:raise ValueError("Body preflight fails independent seed overlap")
    result={"passed":True,"prepared_hash":digest(prepared),"preflight_hash":digest(report),"seed_iou":max(ious),
            "projected_tracking_seconds":projection,"seed_inputs":seed_inputs,"created_at":now()}
    atomic_json(root/"qualification.json",result);return result
