# Five-window Midbeast input comparison

Date: 2026-11-10. Status: preflights and HUD-only development completed; full development GPU inference submitted. No held-out recognition results yet. See [completed preflight results](league-scene-pilot-preflight-results-2026-11-10.md).

The user selected the five-window Midbeast pilot before expanding to another recording. This executes stages 4 and 5 of [the expansion plan](league-scene-evaluation-expansion-plan.md). It tests within-recording temporal generalization only. The five source windows total approximately 72 seconds; they are a pilot, not a cross-match result.

## Reservation and source references

The frozen metadata-only reservation is in `reference_sets/league-scene-understanding/v2-pilot/reservation.json`, with SHA256 `b3bdcba007e0a1186acae65f3ec545a9b82787cc90c12282bb1631c2bc72db11`. Selection excludes previously inspected full contexts, historical excluded intervals, previews and calibration points, with eight-second guards. Four windows retain contextual lengths; the fifth is about 10.5 seconds and has no assumed negative label.

Source preparation jobs 8785482 (held-out) and 8786136 (development) completed successfully on Longleaf. GPT-6 Luna source agents review native source frames and chronological 16 FPS sheets without predictions, following [the labeling contract](league-scene-pilot-source-labeling.md). Labels are independent source-agent drafts, not human-verified ground truth. HUD and physical inventories declare completeness separately; partial inventories cannot supply full precision/recall denominators. Unreadable spans are excluded explicitly.

Held-out answers remain sealed from the development reviewer until recipe freeze. One agent disclosed limited HUD-visibility metadata for holdout-02 before freeze. This disclosure is recorded here; it is not used to choose views, prompts, thresholds, or model settings. The renderer and evidence policy were chosen independently. This is a disclosed limit on strict developer blindness.

## Paired evidence

| Arm | Evidence | Output capability |
| --- | --- | --- |
| Scene-only | Gameplay world crop, fixed non-world masks, anonymous actor tracks and regions; lower HUD area black | Seven physical categories; selected-player cast onsets also yield explicitly scene-inferred activation hypotheses |
| HUD-only | Q/W/E/R numeric readings and persistent cooldown transitions at 16 FPS | Selected HUD-owner slot activation only; no physical animation, target or outcome |
| HUD-grounded | Byte-identical world pixels/timestamps/actor metadata to scene-only, plus native HUD strip and the exact HUD-only candidates | Primary HUD activations survive scene abstention; independent scene proposals may add physical context |

The world crop is source `[0,40,1920,930]`, displayed at 960×445. Both scene arms use a 960×635 canvas, preserving in-world overhead labels and cast bars. The native 575×150 HUD strip is pasted at `(0,485)` only for the grounded arm. Fixed masks remove streamer/overlay regions. Actor localization is shared, including the existing detector's HUD-health-assisted ownership; this comparison isolates additional HUD grounding, not all possible HUD influence on localization.

The scene reader is unchanged, at 4 FPS, eight-frame overlapping packets, pinned intake model `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, max 4096 tokens, existing seed 17. Both scene arms use its existing prompt/schema; grounded packets add only local HUD evidence and an explicit physical-support instruction. No denser scene resampling or HUD-transition promotion into physical cast events occurs.

The current scene schema does not identify ability names or provide rich target/outcome captions. Those capabilities remain unmeasured in this pilot. Its `target_kind` and visible gold claims still require evidence review. Null reference fields never prove an assertion false.

## Execution and locks

Code is transferred through GitHub branch `codex/scene-input-comparison`, in a clean staging checkout, preserving the original dirty workspace. Longleaf run root is `/work/users/a/l/alshen/league-midbeast-validation/scene-understanding/v2-pilot/comparison-r1`.

1. CPU preparation validates reservation/source hashes, runs regression checks and renders identical paired inputs. Source references live separately under `v2-pilot/references`.
2. Separate scene-only and HUD-grounded GPU preflights use the first four seconds of two already-exposed development clips. These are infrastructure checks, not a claim that every difficult case was tested. At least one schema-valid response is required. Rejections are retained separately from infrastructure exceptions.
3. Measure preflight elapsed time/memory before full development runs on six clips: negative-00, negative-02, negative-17, sequence-02, sequence-08, sequence-10. Review their source references and all three arm results; missing negative/sequence coverage remains explicit.
4. Write an actual development review receipt. Freeze source-agent references, code/config/model hashes, clip ranges, runtime identity, inputs, masks, sampling and scoring before any held-out recognition. Incomplete references or missing/duplicate result clips block the freeze.
5. Run the three frozen arms on the five reserved windows; then score and audit cited evidence. A changed reference, code recipe, input or runtime blocks held-out inference/scoring. Changes require a new run revision.

Activation events match one-to-one within ±0.25 seconds of visible cooldown onset. Physical intervals require IoU ≥0.5; point last-hits and arrivals use ±0.25 seconds. Actor and known Q/W/E/R slot accuracy have separate resolution coverage and denominators. Boundary-censored endings do not contribute timing error. Quiet footage tests false output; categories without positive reference examples are untested. Partial physical inventories provide listed-reference recovery only. A complete held-out scorecard also requires source-citation support review and explicit infrastructure/rejection accounting.

## Validation

Thirty-six focused tests pass locally, including paired image isolation, no HUD observation in scene-only prompts, no world/model access in HUD-only, HUD activation survival through scene abstention, all seven physical categories, duplicate proposals, unknown actors/slots, quiet/partial references, source guards and freeze gates. Longleaf repeats these checks before input preparation. GPU execution remains necessary to validate real-model behavior.

Three additional provenance tests reject a shorter historical export, wrong index/cadence/context and incomplete native HUD coverage. Before freeze, run `annotation.league.scene_reference_audit` against the assigned Longleaf source indexes and sealed references. Store its report hash in the development review receipt. The audit displays only file identity and coverage status, preserving answer sealing; it verifies attestations and context, not label truth.

## Execution record

The first input-preparation attempt (8786793) stopped on an unsupported namespace before inference; its dependent jobs 8786794–8786796 were cancelled. Commit `4209755e06a2e1a01ba15b5ee11859d6d4cc5fb1` uses the existing supported action evidence namespace within the separate comparison run. The corrected CPU attempt (8786907) passed all 36 Longleaf tests and prepared its first development clips. Measured progress indicates approximately 50–55 minutes for all 142 source seconds; the initial 30-minute limit was insufficient. Longleaf did not permit increasing that running allocation, so a new 60-minute CPU request will resume existing hash-validated input caches before the GPU preflights. This changes allocation only, not the inference recipe. Failed and superseded job logs are retained.

Published execution revision: `0f671376bfb6d0e6b4a491fe9b11e26aebbff608`. Replacement jobs: CPU preparation **8787221** (60-minute limit), scene-only GPU preflight **8787223**, HUD-grounded GPU preflight **8787225**, HUD-only development **8787226**. All three depend on successful input preparation. Jobs 8786907/8786909/8786910/8786911 were cancelled as superseded; their caches and logs remain. The current revision passes 39 local tests, including the added source-context audit tests. No full development GPU or held-out inference job has been submitted.

All five sealed held-out HUD inventories currently pass schema, completeness and context-receipt checks. They remain source-agent drafts; physical inventories remain limited. Development source review caught an older shortened sequence export and is being completed against the new full source indexes. No source-review completion or freeze is inferred merely from successfully preparing image files.
