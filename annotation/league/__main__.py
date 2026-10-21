"""Run with python -m annotation.league --help."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import sys
import time
from pathlib import Path

from .core import Config, StopRequested, Store, atomic_json, now, read_json, consistent_review
from .knowledge import JsonCache, Patch, Resolver

ROOT = Path(__file__).resolve().parents[2]


def fresh(store, name, key):
    path = store.path(f"stages/{name}.json")
    return path.exists() and read_json(path).get("key") == key


@consistent_review
def ensure_fused(store, config):
    from .timeline import fuse
    if not store.path("observations.jsonl").exists() or not store.path("event_candidates.jsonl").exists():
        raise ValueError("Run annotate before fusion/question construction")
    if not fresh(store, "annotate", store.fingerprint(config)):
        raise ValueError("Annotations are stale for this source/configuration/model/code; rerun annotate")
    key = store.stage_key("fuse", config, ["observations", "event_candidates"])
    if not fresh(store, "fuse", key):
        fuse(store, config)


@consistent_review
def ensure_questions(store, config, resolver):
    from .questions import generate_qa
    ensure_fused(store, config)
    key = store.stage_key("generate-qa", config, ["states", "events", "issues"])
    if not fresh(store, "generate-qa", key):
        generate_qa(store, config, resolver)


@consistent_review
def perform_evaluation(store, config, runner, resolver):
    from .evaluation import evaluate
    ensure_questions(store, config, resolver)
    return evaluate(store, config, runner, resolver)


@consistent_review
def perform_report(store, config, gpu_hour_rate=None):
    from .evaluation import report
    key = store.stage_key("evaluate", config, ["questions"])
    if store.rows("predictions") and not fresh(store, "evaluate", key):
        raise ValueError("Predictions are stale after corrections/config changes; rerun evaluate")
    if store.rows("questions") and not fresh(store, "generate-qa", store.stage_key(
            "generate-qa", config, ["states", "events", "issues"])):
        raise ValueError("Questions are stale; rerun generate-qa/evaluate before reporting")
    return report(store, config, gpu_hour_rate)


def parser():
    p = argparse.ArgumentParser(description="Fixed-route LoL annotation and independent memory evaluation")
    p.add_argument("--run-dir", type=Path, default=ROOT / "data/league_runs/aatrox-26.13")
    p.add_argument("--config", type=Path)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/league_knowledge")
    p.add_argument("--offline", action="store_true", help="Use cached patch references only")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        s = sub.add_parser(command)
        s.add_argument("--video", type=Path, default=ROOT / "data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4")
        s.add_argument("--start", type=float, default=0.0)
        s.add_argument("--end", type=float)
        if command == "run":
            s.add_argument("--retry-failed", action="store_true")
    s = sub.add_parser("annotate")
    s.add_argument("--retry-failed", action="store_true")
    for command in ("fuse", "generate-qa", "evaluate", "preflight", "cache-models", "processor-preview"):
        sub.add_parser(command)
    s = sub.add_parser("cache-references")
    s.add_argument("--champions", nargs="+", default=["all"], help="Static details to pre-cache; defaults to the complete patch roster")
    s.add_argument("--base-only", action="store_true", help="Roster/items only; champion mechanics lookups will need network later")
    s = sub.add_parser("review")
    s.add_argument("--port", type=int, default=8765)
    s = sub.add_parser("report")
    s.add_argument("--gpu-hour-rate", type=float)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = Store(args.run_dir)
    config_path = args.config or (store.path("config.json") if store.path("config.json").exists()
                                  else Path(__file__).with_name("aatrox.json"))
    runner = None
    try:
        config = Config.load(config_path)
        store.root.mkdir(parents=True, exist_ok=True)
        store.install_signals()
        resolver = Resolver(Patch(**config.patch), JsonCache(args.cache_dir, offline=args.offline))
        if args.command == "review":
            from .review import create_app
            import uvicorn
            uvicorn.run(create_app(store), host="127.0.0.1", port=args.port)
            return 0
        # Do not allow two batch writers to reuse the same run directory concurrently.
        with store.path(".writer.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another pipeline writer owns this run directory") from exc
            started = time.monotonic()
            command = args.command
            result = {}
            if command in {"prepare", "run"}:
                from .media import prepare
                result = prepare(store, config, args.video, args.start, args.end)
            if command == "cache-references":
                from .references import cache_references
                cached = cache_references(store, resolver, args.champions, args.base_only)
                uncached = [d for d in cached.get("details", []) if d["status"] == "unresolved"]
                result = {k: v for k, v in cached.items() if k != "details"}
                result.update(uncached_champions=uncached, summary_path=str(store.path("reference_summary.json")))
            if command == "processor-preview":
                from .inference import processor_preview
                result = processor_preview(store, config)
            if command in {"cache-models", "preflight"}:
                from .inference import pin_models, preflight
                result = pin_models(store, config, download=True) if command == "cache-models" else preflight(store, config)
            if command in {"annotate", "evaluate", "run"}:
                from .inference import QwenRunner
                from .media import file_hash
                manifest = read_json(store.path("manifest.json"))
                if file_hash(Path(manifest["source"]["path"])) != manifest["source"]["sha256"]:
                    raise ValueError("Source video changed; prepare a new run")
                runner = QwenRunner(store, config)
            if command in {"annotate", "run"}:
                from .observers import annotate
                from .inference import processor_preview
                layout = read_json(store.path("layout_validation.json"))
                if layout.get("preprocessing_key") != store.fingerprint(config):
                    processor_preview(store, config)
                result = annotate(store, config, runner, resolver, retry_failed=args.retry_failed)
            if command in {"fuse", "generate-qa", "evaluate", "run"}:
                ensure_fused(store, config)
            if command in {"generate-qa", "evaluate", "run"}:
                ensure_questions(store, config, resolver)
                result = read_json(store.path("question_coverage.json"))
            if command in {"evaluate", "run"}:
                result = perform_evaluation(store, config, runner, resolver)
            if command in {"report", "run"}:
                if command == "run":
                    atomic_json(store.path("timings/run.json"), {"wall_seconds": time.monotonic()-started,
                                "completed_at": now(), "command": command})
                result = perform_report(store, config, getattr(args, "gpu_hour_rate", None))
            atomic_json(store.path(f"timings/{command}.json"), {"wall_seconds": time.monotonic()-started,
                       "completed_at": now(), "command": command})
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 2 if command == "cache-references" and result.get("uncached_champions") else 0
    except StopRequested as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
