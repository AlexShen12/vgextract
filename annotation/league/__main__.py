"""Run with python -m annotation.league --help."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import time
from pathlib import Path

from .core import Config, DEFAULT_RUN, StopRequested, Store, atomic_json, now, read_json, consistent_review
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
        generate_qa(store, config, resolver if config.mechanics else None)


@consistent_review
def ensure_captions(store, config):
    from .captions import DEPENDENCIES, generate_captions
    ensure_fused(store, config)
    key = store.stage_key("generate-captions", config, DEPENDENCIES)
    if (not fresh(store, "generate-captions", key) or not store.path("captions.jsonl").exists()
            or not store.path("caption_coverage.json").exists()
            or not store.path("caption_omissions.jsonl").exists()):
        generate_captions(store, config)


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
    p = argparse.ArgumentParser(description="Calibrated League intake and independent memory evaluation")
    p.add_argument("--run-dir", type=Path, default=Path(os.environ.get("LEAGUE_RUN_DIR",str(ROOT / DEFAULT_RUN))))
    p.add_argument("--config", type=Path)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/league_knowledge")
    p.add_argument("--offline", action="store_true", help="Use cached patch references only")
    p.add_argument("--debug", action="store_true", help="Include a traceback when a stage fails")
    p.add_argument("--mechanics", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--evaluation", action=argparse.BooleanOptionalAction, default=None)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        s = sub.add_parser(command)
        s.add_argument("--video", type=Path, help="Source path; defaults to the configured Midbeast recording")
        s.add_argument("--start", type=float, default=0.0)
        s.add_argument("--end", type=float)
        if command == "run":
            s.add_argument("--retry-failed", action="store_true")
    s = sub.add_parser("annotate")
    s.add_argument("--retry-failed", action="store_true")
    s.add_argument("--sample-times", type=Path, help="Benchmark labels JSON or a JSON list of source times")
    s.add_argument("--pilot", action="store_true", help="Extract five representative one-minute intervals")
    for command in ("fuse", "generate-captions", "generate-qa", "evaluate", "preflight", "processor-preview", "cache-assets"):
        sub.add_parser(command)
    s = sub.add_parser("cache-models")
    s.add_argument("--suite", choices=("intake", "events", "audio", "all", "qwen35"), default="intake")
    s.add_argument("--solver", action="store_true")
    s.add_argument("--tracking", action="store_true", help="Also cache the gated SAM 3.1 tracking checkpoint")
    s = sub.add_parser("experiment")
    s.add_argument("--suite", choices=("crops", "events", "audio", "actions", "captions", "all", "hp-vlm", "scene-identity", "ability-alignment", "cooldown-identity", "qwen35"), required=True)
    s.add_argument("--labels", type=Path, help="Directory containing the independent experiment label manifests")
    s.add_argument("--initialize", action="store_true", help="Save source evidence and unverified label templates without loading GPU models")
    s.add_argument("--draft", action="store_true", help="Diagnostic predictions only; never release acceptance")
    s.add_argument("--freeze", action="store_true", help="For actions/captions: freeze development choices before held-out scoring")
    s.add_argument("--infer", action="store_true", help="Actions/captions: add pinned local scene inference to CPU detection")
    s.add_argument("--prepare-only", action="store_true", help="Qwen3.5: prepare source-only evidence for both splits")
    s.add_argument("--preflight-only", action="store_true", help="Qwen3.5: qualify image/JSON inference for each model")
    s.add_argument("--action-profile", type=Path)
    s.add_argument("--icon-references", type=Path, help="Versioned Q/W/E/R reference manifest (defaults to asset cache)")
    s.add_argument("--assistance", type=Path, help="Obsolete for actions v3 (rejected); use --icon-references")
    s = sub.add_parser("caption-pilot", help="Selected-player caption and clip pilot, isolated from production")
    s.add_argument("--start", type=float, required=True, help="Source-video seconds")
    s.add_argument("--end", type=float, required=True)
    s.add_argument("--infer", action="store_true", help="Run pinned GPU scene inference; default is CPU plus compatible cached scene evidence")
    s.add_argument("--action-profile", type=Path)
    s.add_argument("--icon-references", type=Path)
    s = sub.add_parser("action-window", help="Prepare or inspect experimental champion actions in a source-video interval")
    s.add_argument("--start", type=float, required=True, help="Source-video seconds, not the in-game clock")
    s.add_argument("--end", type=float, required=True)
    s.add_argument("--fps", type=float, default=4, help="Scene sampling rate (2, 4 or 8 FPS; HUD uses 16)")
    s.add_argument("--hud-box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="Compatibility argument; must match the action profile's HUD box")
    s.add_argument("--infer", action="store_true", help="Run the pinned local GPU model; default only saves inspectable inputs")
    s.add_argument("--detect", action="store_true", help="Run CPU player binding, farming and recall detection without loading Qwen")
    s.add_argument("--action-profile", type=Path)
    s.add_argument("--icon-references", type=Path, help="Versioned Q/W/E/R reference manifest (defaults to asset cache)")
    s.add_argument("--assistance", type=Path, help="Obsolete for actions v3 (rejected); use --icon-references")
    s = sub.add_parser("intake-benchmark")
    s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--freeze", action="store_true", help="Freeze thresholds using development labels only")
    s.add_argument("--draft", action="store_true", help="Diagnostic scores for unverified labels; cannot authorize a release")
    s = sub.add_parser("cache-references")
    s.add_argument("--champions", nargs="+", default=["all"], help="Static details to pre-cache; defaults to the complete patch roster")
    s.add_argument("--base-only", action="store_true", help="Roster/items only; champion mechanics lookups will need network later")
    s = sub.add_parser("cache-abilities", help="Cache official descriptions independently of the footage patch")
    s.add_argument("--reference-version", default="16.13.1")
    s.add_argument("--champions", nargs="+", default=["all"])
    s = sub.add_parser("cache-qwen35-references", help="Cache versioned portraits and ability icons without enabling mechanics")
    s.add_argument("--reference-version", default="16.13.1")
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
                                  else Path(__file__).with_name("midbeast.json"))
    runner = None
    try:
        config = Config.load(config_path)
        config.asset_cache = os.environ.get("LEAGUE_ASSET_CACHE", config.asset_cache)
        if args.mechanics is not None:
            config.mechanics = args.mechanics
        if args.evaluation is not None:
            config.evaluation_enabled = args.evaluation
        store.root.mkdir(parents=True, exist_ok=True)
        store.install_signals()
        resolver = None
        if (config.mechanics and args.command != "cache-abilities") or args.command == "cache-references":
            if not config.patch:
                raise ValueError("Static mechanics requires an explicitly established patch")
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
            if command in {"annotate", "run", "prepare", "processor-preview", "preflight"} and store.path("manifest.json").exists():
                from .intake import require_v2
                require_v2(store)
            if command not in {"prepare", "run"} and (args.mechanics is not None or args.evaluation is not None):
                atomic_json(store.path("config.json"), config.to_dict())
            if command in {"prepare", "run"}:
                from .media import prepare
                result = prepare(store, config, args.video or ROOT / config.source_video, args.start, args.end)
            if command == "cache-references":
                from .references import cache_references
                cached = cache_references(store, resolver, args.champions, args.base_only)
                uncached = [d for d in cached.get("details", []) if d["status"] == "unresolved"]
                result = {k: v for k, v in cached.items() if k != "details"}
                result.update(uncached_champions=uncached, summary_path=str(store.path("reference_summary.json")))
            if command == "cache-abilities":
                from .ability_catalog import cache_abilities
                result = cache_abilities(config.asset_cache, args.reference_version, args.champions, offline=args.offline)
            if command == "cache-assets":
                if not config.patch:
                    raise ValueError("Deferred visual assets require an explicitly established patch")
                from .visual_assets import cache_assets
                result = cache_assets(Path(config.asset_cache), config.patch)
            if command == "intake-benchmark":
                from .intake_benchmark import benchmark
                result = benchmark(store, config, args.labels, freeze=args.freeze, draft=args.draft)
            if command == "processor-preview":
                from .inference import processor_preview
                result = processor_preview(store, config)
            if command == "cache-models":
                from .local_models import cache_models
                roles = {"intake": ["intake"], "events": ["intake", "event_alternative"],
                         "audio": ["asr", "aligner"], "all": ["intake", "event_alternative", "asr", "aligner"],
                         "qwen35": ["qwen35_2b", "qwen35_4b", "qwen35_9b", "qwen35_27b"]}[args.suite]
                if args.tracking:
                    roles += ["tracking"]
                result = cache_models(config.asset_cache, roles)
                if args.solver:
                    from .inference import pin_models
                    result["solver"] = pin_models(store, config, download=True)
            if command == "cache-qwen35-references":
                from .qwen35_references import cache_references
                result = cache_references(config.asset_cache, args.reference_version, offline=args.offline)
            if command == "experiment":
                if args.suite in {"hp-vlm", "scene-identity", "ability-alignment", "cooldown-identity", "qwen35"}:
                    from .qwen35_study import experiment_qwen35
                    if args.assistance:
                        raise ValueError("Qwen3.5 studies do not accept identity assistance")
                    result = experiment_qwen35(store, config, args.suite, labels=args.labels,
                        initialize_labels=args.initialize, draft=args.draft, freeze=args.freeze, infer=args.infer,
                        preflight=args.preflight_only, prepare_only=args.prepare_only, action_profile=args.action_profile, icon_references=args.icon_references)
                elif args.suite == "captions":
                    from .caption_benchmark import experiment_captions
                    if args.assistance:
                        raise ValueError("Caption pilots do not accept identity assistance")
                    result = experiment_captions(store, config, labels=args.labels, initialize=args.initialize,
                        draft=args.draft, freeze=args.freeze, infer=args.infer,
                        action_profile=args.action_profile, icon_references=args.icon_references)
                elif args.suite == "actions":
                    from .action_benchmark_v3 import experiment_actions
                    result = experiment_actions(store, config, labels=args.labels, initialize=args.initialize,
                                                draft=args.draft, freeze=args.freeze,
                                                action_profile=args.action_profile, assistance=args.assistance, icon_references=args.icon_references,
                                                infer=args.infer)
                else:
                    if args.infer:
                        raise ValueError("experiment --infer is available only with --suite actions or captions")
                    from .experiments import experiment
                    result = experiment(store, config, args.suite, labels=args.labels,
                                        initialize=args.initialize, draft=args.draft)
            if command == "action-window":
                from .action_v3 import action_window
                result = action_window(store, config, args.start, args.end, fps=args.fps,
                                       hud_box=args.hud_box, infer=args.infer, detect=args.detect,
                                       action_profile=args.action_profile, assistance=args.assistance, icon_references=args.icon_references)
            if command == "caption-pilot":
                from .caption_pilot import caption_pilot
                result = caption_pilot(store, config, args.start, args.end, infer=args.infer,
                                       action_profile=args.action_profile, icon_references=args.icon_references)
            if command == "preflight":
                from .intake import preflight
                result = preflight(store, config)
            if command in {"annotate", "evaluate", "run"}:
                from .media import file_hash
                manifest = read_json(store.path("manifest.json"))
                if file_hash(Path(manifest["source"]["path"])) != manifest["source"]["sha256"]:
                    raise ValueError("Source video changed; prepare a new run")
            if command in {"annotate", "run"}:
                from .observers import annotate
                from .inference import processor_preview
                layout = read_json(store.path("layout_validation.json"))
                if layout.get("preprocessing_key") != store.fingerprint(config):
                    processor_preview(store, config)
                samples = None
                if getattr(args, "sample_times", None):
                    samples = read_json(args.sample_times)
                    if isinstance(samples, dict):
                        samples = [f["time"] for f in samples["frames"]]
                result = annotate(store, config, retry_failed=args.retry_failed, sample_times=samples,
                                  pilot=getattr(args, "pilot", False))
            if command in {"fuse", "generate-captions", "generate-qa", "evaluate", "run"}:
                ensure_fused(store, config)
            if command in {"generate-captions", "run"}:
                ensure_captions(store, config)
                result = read_json(store.path("caption_coverage.json"))
            if command in {"generate-qa", "evaluate", "run"}:
                ensure_questions(store, config, resolver)
                result = read_json(store.path("question_coverage.json"))
            if command in {"evaluate", "run"}:
                ensure_questions(store, config, resolver)
                if config.evaluation_enabled:
                    if any(q.get("review_status") != "rejected" for q in store.rows("questions")):
                        from .inference import QwenRunner
                        runner = QwenRunner(store, config)
                    result = perform_evaluation(store, config, runner, resolver)
                else:
                    result = {"evaluation": "disabled", "reason": "Configured extraction/fusion/question-only run"}
            if command in {"report", "run"}:
                if command == "run":
                    atomic_json(store.path("timings/run.json"), {"wall_seconds": time.monotonic()-started,
                                "completed_at": now(), "command": command})
                result = perform_report(store, config, getattr(args, "gpu_hour_rate", None))
            atomic_json(store.path(f"timings/{command}.json"), {"wall_seconds": time.monotonic()-started,
                       "completed_at": now(), "command": command})
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 2 if ((command == "experiment" and result.get("experimental", True)) or
                         (command == "cache-references" and result.get("uncached_champions")) or
                         (command == "intake-benchmark" and not result.get("passed"))) else 0
    except StopRequested as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        if args.debug:
            logging.exception("Pipeline stage failed")
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
