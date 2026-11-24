# SAM tracking campaign check — November 10, 2026

The November 9 campaign produced **no SAM masks or segmentation-quality scores**.
Its GPU preflight failed before opening the first inference session. The
qualification and comparison jobs were canceled by their dependencies.

| Job | Experiment | Final state | GPU elapsed |
|---|---|---|---|
| 8719267 | Preflight | FAILED, exit 1 | 50 seconds |
| 8720444 | CPU qualification | CANCELLED | 0 |
| 8720690 | Verified source seeds | CANCELLED | 0 |
| 8720691 | Automatic body seeds | CANCELLED | 0 |

The saved exception is
`Sam3MultiplexTrackingWithInteractivity.init_state() got an unexpected keyword argument 'offload_state_to_cpu'`.
The pinned SAM shared session handler passes this argument even when false;
the multiplex model's `init_state` does not accept it. This is an API compatibility
failure, not evidence that SAM tracking fails on this footage.

The large checkpoint warnings need care: the official builder initially loads
the merged tracker/detector checkpoint into a temporary tracker, then loads it
again into the assembled model. The temporary tracker reported 931 missing and
1,623 unexpected keys; the assembled model reported 64 missing rotary-position
buffers. These messages alone do not establish that trainable weights are absent.
The recovery explicitly checks and loads all trainable parameters and rejects
missing weights, incompatible shapes or unrecognized keys. Only derived rotary
buffers may differ. It also uses the checkpoint-compatible complex rotary mode.

## Completed CPU diagnostics

These are pipeline outputs, **not independently validated identity or binding
accuracy measurements**. Each clip has 192 inspected frames at 16 FPS.

| Development clip | HUD portrait output | Body-binding-visible frames | Automatic seed at common prompt time |
|---|---|---|---|
| 00 | Darius, 192/192 | 166/192 | Unavailable |
| 01 | Darius, 192/192 | 147/192 | Available |
| 02 | Darius, 192/192 | 173/192 | Available |
| 03 | Darius, 192/192 | 164/192 | Unavailable |
| 06 | Darius, 192/192 | 187/192 | Available |

The existing HUD reader supplied an identity on 960/960 inspected frames. The
existing body binder marked 837/960 frames visible, but supplied a seed at only
3/5 fixed source-only prompt times. Later visible bindings cannot retroactively
establish a binding at the prompt cue. The two unavailable seeds remain in the
automatic-arm coverage denominator. Verified source seeds are available for 5/5.

Independent sparse references remain 28 visible conservative body polygons, six
scoreboard-occluded references and six unknown references. No overlap, occlusion,
identity-switch, or action-attribution result can be reported from this failed run.

## Resources and recovery

The failed process recorded 21.00 seconds inside the inference runner,
3,926,040,576 bytes peak allocated CUDA memory (3.66 GiB), and
8,052,658,176 bytes peak host RSS (7.50 GiB). Slurm recorded 50 seconds total
allocation and 11,921,988 KiB batch peak RSS (11.37 GiB). These are initialization
measurements; they do not size full-clip tracking throughput or memory.

The original frozen snapshot and reports are preserved. Recovery revision
`sam-tracking-20261110-r3` adds a narrow adapter that omits only unsupported
**disabled** state offloading; enabled state offloading and other unsupported
arguments still fail. Model and checkpoint pins, HUD pipeline, source labels,
mask-overlap gate and comparison thresholds remain fixed.

Verification: 16 focused SAM compatibility/runner/qualification tests passed both
locally and in the staged Longleaf source. All 338 discoverable `unittest` cases
outside the separate pytest-based Qwen study passed locally. The Qwen pytest
module was not run because this local environment lacks pytest. No GPU
compatibility or segmentation success is implied by these CPU tests.

The recovery preflight limit is 29 minutes, with two comparison limits of 105
minutes each. Including the consumed 50 seconds, their maximum total is
14,390 GPU-seconds (3h 59m 50s), within the original four-hour cap. Comparisons
remain conditional on successful preflight and independent seed-mask/resource
qualification. No claim of SAM or caption acceptance is made.

The recovery jobs were submitted as follows. Each dependent uses
`--kill-on-invalid-dep=yes` so a failure cancels the remaining GPU comparisons.

| Job | Experiment | Dependency |
|---|---|---|
| 8784750 | Revised preflight | None |
| 8784751 | CPU overlap/resource qualification | afterok:8784750 |
| 8784752 | Verified source-seed comparison | afterok:8784751 |
| 8784753 | Automatic body-seed comparison | afterok:8784751 |

Prepared experiment: `experiments/sam-tracking/b965b450b3f0045278bb`.
Source archive SHA-256:
`25471204720fe7ff5825c7e6d6b7ffe73c84bf12071099564484ac7b3c96a28b`.

Original raw evidence is retained locally under
`data/league_longleaf_results/2026-11-10/sam-r2/`. Frozen handoff records are
under `data/league_sam_tracking_handoff/2026-11-09-r2/` and
`data/league_sam_tracking_handoff/2026-11-10-r3/`.
