"""One budget-accounted stage for an explicitly submitted isolated body campaign."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .core import atomic_json, read_json
from .recognition_budget import reserve, finish


def execute_identity(stage, root, cache, *, intake_run, policy_path, gallery_kind, device, budget_root, gpu_seconds,
                     slot_skim=None, recovery=False):
    if device == "cuda" and (budget_root is None or gpu_seconds <= 0 or not (budget_root / "gpu-budget.json").exists()):
        raise ValueError("GPU identity work needs a reconciled campaign ledger and allocation seconds")
    reservation = reserve(budget_root, stage, gpu_seconds, recovery=recovery) if device == "cuda" else None
    started = time.monotonic()
    status = "failed"
    try:
        if stage == "identify":
            from .body_operational import identify
            result = identify(root, intake_run, cache, policy_path=policy_path, gallery_kind=gallery_kind,
                              device=device, slot_skim=slot_skim)
        else:
            from .body_appearance import fit
            result = fit(root, cache, gallery_kind=gallery_kind, device=device)
        status = "completed"
        return result
    finally:
        if reservation:
            finish(budget_root, reservation["attempt"], time.monotonic() - started, status)


def execute(stage,root,cache,*,arm=None,recovery=False,encoder_lock=None,device="cuda",budget_root=None):
    args=SimpleNamespace(stage=stage,root=Path(root),cache=Path(cache),arm=arm,recovery=recovery)
    seconds={"preflight":1800,"track":1800,"embed":1800,"run":0,"qualify":0}[args.stage]
    if stage=="embed" and device=="cpu":seconds=0
    ledger = Path(budget_root) if budget_root is not None else args.root
    if budget_root is not None and not (ledger / "gpu-budget.json").is_file():
        raise ValueError("Shared body campaign ledger is missing. Reconcile allocations first.")
    attempt=reserve(ledger,args.stage+("-"+args.arm if args.arm else ""),seconds,recovery=args.recovery)
    started=time.monotonic();status="failed"
    try:
        from .body_campaign import track,embed
        if args.stage in {"preflight","track"}: result=track(args.root,args.cache,args.stage=="preflight")
        elif args.stage=="qualify":
            from .recognition_budget import qualify
            result=qualify(args.root)
        elif args.stage=="embed": result=embed(args.root,encoder_lock or args.root/"encoder-lock.json",device)
        else:
            from .body_identity import run_arms
            from .body_policy import validate_freeze
            policy=read_json(args.root/"policy.json");data=validate_freeze(args.root,policy)
            result=run_arms(args.root/"comparisons",data,data["gallery"],data["gallery_vectors"],data["query_vectors"],
                            policy,only_arm=args.arm)
        if args.stage=="preflight":
            import gc,torch
            from PIL import Image
            from .body_identity import FrozenEncoder
            gc.collect();torch.cuda.empty_cache()
            encoder=FrozenEncoder(args.root/"encoder-lock.json","cuda")
            reference=read_json(args.root/"prepared.json")["gallery"][0]
            with Image.open(reference["path"]) as image,Image.open(reference["mask_path"]) as mask:
                vector=encoder.encode(image,mask)
            result["encoder_smoke"]={"passed":len(vector)==384,"signature":encoder.signature,"dimensions":len(vector)}
            if not result["encoder_smoke"]["passed"]:raise ValueError("Unexpected body embedding dimensions")
            result["peak_gpu_bytes"]=torch.cuda.max_memory_allocated()
            atomic_json(args.root/"preflight/report.json",result)
        status="completed"
        atomic_json(args.root/"stages"/f"{attempt['attempt']:08d}.json",{"stage":stage,"status":status,"result":result})
        return result
    except BaseException as exc:
        atomic_json(args.root/"stages"/f"{attempt['attempt']:08d}.json",{"stage":stage,"status":"failed","error":str(exc)})
        raise
    finally:
        finish(ledger,attempt["attempt"],time.monotonic()-started,status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("preflight","qualify","track","embed","run"));p.add_argument("--arm")
    p.add_argument("--root",type=Path,required=True);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--recovery",action="store_true");p.add_argument("--budget-root",type=Path);args=p.parse_args()
    print(json.dumps(execute(args.stage,args.root,args.cache,arm=args.arm,recovery=args.recovery,budget_root=args.budget_root),indent=2))


if __name__=="__main__":main()
