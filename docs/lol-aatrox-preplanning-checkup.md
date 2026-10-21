# Aatrox prototype: final scope check

Historical planning record, 2026-10-19. The [pipeline implementation](../annotation/league/README.md) and [validation record](../annotation/league/VALIDATION.md) now supersede implementation-status statements below. Source crops and actual Qwen preprocessing have been checked locally. Semantic GPU inference and L40 profiling have not been run.

## Confirmed scope

- Start with `data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4`: 1920×1080, 60 FPS, approximately 28:56. The source metadata claims gameplay patch 26.13; use its pinned Data Dragon/CommunityDragon references for this prototype. The current-patch benchmark remains a separate cohort.
- Use fixed spatial routes. Qwen3-VL-8B-Instruct reads HUD regions, including the minimap, scoreboard, selected-champion ability panel and both team sidebars. Qwen3-VL-8B-Thinking views the main gameplay sequence. These are logical extraction roles sharing two model checkpoints, not one resident model copy per crop.
- Unify observations under one source-video timeline with stable entity bindings and evidence references. Derive state/event records and questions from that evidence. No engine telemetry or match-state API is used.
- Allow automatic adaptive review. It changes time-window coverage, sampling density and resolution on the existing fixed routes; v1 does not ask a model to choose new crop coordinates.
- Start the memory evaluation with video plus question in one bounded multimodal context, with permitted static references determined by question category. Construction logs, fused state records and prior answer summaries are excluded from solver input. Streaming/external memory can be a later, separately labeled evaluation mode.
- Hardware target: one NVIDIA L40. It has 48 GB GPU memory according to [NVIDIA's specification](https://www.nvidia.com/en-us/data-center/l40/). Profile actual model, vision-encoder, KV-cache and generation memory before selecting concurrency or maximum context.

## Layout check and initial routes

Reinspected full-resolution frames at video 00:30, 13:00 and 27:00. Panel placement agrees across those three frames; this does not establish consistency across every frame or other recordings. The scoreboard can change its contents, sidebars change alive/dead presentation, and editorial overlays/messages occlude parts of the battlefield. Detect layout/panel mismatches and flag them rather than silently using an invalid profile.

Coordinates below are manually selected candidate boxes in **source pixels**, `[x0, y0, x1, y1)` for this 1920×1080 video. They must be previewed after the actual model preprocessing and refined during calibration. They are not universal LoL coordinates.

| Route | Candidate box | Reader and fields |
| --- | --- | --- |
| Gameplay view | `[0,110,1920,835]` plus a fixed full-frame context view | Thinking: visible champions, minions, structures, actions, effects, spatial relations, local sequence and outcome |
| Selected-champion HUD | `[0,775,315,965]` | Instruct: selected identity, HP/resource, level, displayed statistics, passive/Q/W/E/R, summoner slots, visible cooldown text, ability ranks and status icons where resolved |
| Left roster/sidebar | `[0,125,115,645]` | Instruct: five roster rows, portraits/player text, levels, HP/resource indicators, summoner icons, ultimate indicator, death presentation and readable timers |
| Right roster/sidebar | `[1810,125,1920,645]` | Instruct: corresponding fields, with separately calibrated mirrored slot geometry |
| Scoreboard/inventory | `[595,835,1335,1080]` | Instruct: panel type, champion/player row mapping, KDA, CS, inventory, readable displayed counters |
| Minimap | `[1650,810,1915,1075]` | Instruct: visible icon tracks/identities, coarse positions, camera rectangle, structures and pings; unresolved overlaps remain unknown |
| Match HUD | `[390,0,1545,125]` | Instruct: clock, score, displayed team gold and verified objective indicators |
| Objective timer panel | `[1340,895,1645,1080]` | Instruct: objective name and displayed timer/state; a LIVE label does not establish an active attempt |
| Right event feed | `[1450,440,1770,810]` | Instruct: readable event notices/icons and attributed actors when resolved |
| Central notices | `[400,160,1515,310]` | Instruct: readable kill/objective/structure notices; shared pixels with the gameplay route are not independent evidence |

The fixed full-frame context preserves activity outside the gameplay box, layout continuity and overlay awareness. The main view can contain HUD/message intrusions; treat them as overlays, not game-world objects. If detail is insufficient at the edges, review the existing full-frame route at a higher pixel budget rather than moving the crop. No crop can recover gameplay concealed behind an opaque overlay.

Sidebar scope is deliberately narrower than “all team ability cooldowns.” The inspected rows expose summoner-spell icons, ultimate indicators, portrait/death state and other small indicators; they do not provide a full Q/W/E panel for all ten champions. Bind every slot to a verified meaning and extract numerical cooldowns only where legible. Keep `ready`, `cooling_down`, `unlearned_or_unavailable`, `unreadable`, `not_shown` and `unknown` distinct only when supported by the interface evidence; do not infer readiness from an absent numeral alone. The selected-champion panel is the main source for that champion's displayed basic-ability cooldowns. Rebind after selection changes.

## Observation, fusion and review contracts

Each observer emits structured records with source PTS, sampled-frame IDs, route/box, entity/slot bindings, value or event candidate, visibility, uncertainty and extractor/model version. Preserve readable strings and displayed numeric precision. Model-relative timestamps must map back to actual sampled source timestamps; the displayed game clock is a separate observation. In the three checked samples it is one minute ahead of video time, but a single global offset is not guaranteed.

Start observer passes independently, then reconcile in the shared timeline. Different routes can corroborate or contradict one another. Preserve disagreement; two model readings of identical pixels are not independent corroboration. Do not average conflicting identities or propagate a resource value across a selection switch. State records retain last-confirmed time and freshness; unobserved intervals remain explicit.

Begin with the proposed 0.5–1 FPS routine sampling and 2–4 FPS event review. Those rates are discovery settings rather than guaranteed support for exact casts/counts. Trigger automatic rereads on:

- HP/resource, death, score, inventory or cooldown transitions;
- visible encounters, minimap convergence, channels and event notices;
- cross-route disagreement, unresolved identity, unreadable critical fields or unexpected layout changes.

Review expands lead-in/follow-through and increases frame/pixel budgets for the implicated fixed routes. Preserve each pass and its cost. Deduplicate concurrent triggers and bound review passes and total compute per window. If the evidence remains insufficient, produce partial/unknown results and a human-review item. Include randomly selected quiet windows to measure events that triggers miss. For subsecond action order, exact reaction time or flash-combos, use denser/native-frame evidence when feasible or exclude the unsupported question.

The unified timeline is the construction pipeline's evidence store. It does not become an external memory feed in the initial solver evaluation. If evaluating a routed multi-model solver later, label that system separately from the single-context visual-memory baseline. Human-reviewed source evidence establishes gold answers; fused model logs only propose candidates.

## L40 execution assumptions to carry into planning

The [Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and [Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) variants are separate checkpoints. Fixed logical routing does not require simultaneous GPU execution or a copy per region. The proposed initial schedule is batched Instruct passes, a Thinking pass over gameplay windows, then bounded review batches as needed. Persist observations so switching a checkpoint does not require redoing completed work.

Use BF16 as a proposed calibration baseline, subject to measured fit and the final serving stack. Prefer scheduling one model at a time initially; dual residency and quantization are later options to benchmark. Checkpoint revisions, processor/library versions, image/video token budgets, batch/context limits, output-token caps, sampling settings and observed peak memory must be recorded. In particular, reserve a bounded generation budget for Thinking rather than inheriting an unrestricted reasoning output length. Preserve final structured outputs and generation/completion status; internal reasoning text is not ground truth.

A long advertised context window is not proof that a 40-minute high-detail input fits this GPU or remains readable. Audit the actual processed frames, timestamps and HUD legibility for each horizon. Report temporal span, processed frame/visual-token count, review overhead, peak GPU memory and end-to-end wall time together. If a question cannot fit within the declared evaluation budget, report that limit instead of silently substituting an external summary.

## Remaining planning choices

Core scope is sufficiently specified to write the implementation plan. The following can be explicit proposed defaults rather than permission blockers:

- **QA solver checkpoint:** propose Thinking for the initial single-context QA baseline; optionally compare Instruct on the same verified questions. The two annotation models' fusion quality is measured separately from solver accuracy.
- **Serving backend:** choose and pin it during planning based on the L40 environment. A reference Transformers runner is a reasonable initial baseline; no inference backend has yet been selected or tested here.
- **Pilot calibration:** review the early lane, midgame reset and late crowded-fight windows, plus quiet samples, before a full-game annotation run. Freeze field definitions, tolerances and quality gates after this calibration.
- **Scope boundaries:** English static-reference data, explicit handling of Korean player names in this source, visual-only versus mechanics-augmented references, no audio or editorial outcome metadata as solver evidence, and no external memory in v1.

Expected build outputs are a fixed layout profile and crop previews, processed-input audit, timestamped per-route observations, fused states/events with coverage/conflict records, adaptive-review queue/transcript, human-verified questions/answers/evidence, and evaluation/efficiency reports. This document specifies their purpose; it does not implement them.
