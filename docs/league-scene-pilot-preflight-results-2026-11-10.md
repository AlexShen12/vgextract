# Midbeast input pilot: completed preflight results

Date: 2026-11-10. These are development/preflight results, not held-out accuracy scores.

All four jobs completed with exit code 0. CPU preparation produced paired inputs for all eleven clips (six development, five reserved), and all 39 regression checks passed on Longleaf. The held-out reference answers remain sealed; the recipe has not been frozen and held-out inference has not run.

| Job | Scope | Elapsed | Result |
| --- | --- | --- | --- |
| 8787221 | CPU input preparation | 44m 29s | All eleven clips prepared; peak host RSS about 4.44 GiB |
| 8787223 | Scene-only GPU preflight | 5m 11s | 6 accepted responses, 0 rejected; 0 physical events and 0 inferred activations |
| 8787225 | HUD-grounded GPU preflight | 6m 26s | 5 accepted responses, 1 rejected; 0 accepted physical events, 1 primary HUD activation |
| 8787226 | HUD-only development | 2s | 2 primary activations over all six development clips; no scene/model calls |

The two GPU preflights cover the same first four seconds of negative-00 (22–26 source seconds) and negative-17 (785–789): eight source seconds total, at 4 FPS. HUD-only covers all six development contexts, seventy source seconds total. Its two-output count must not be compared directly with either short preflight count as an accuracy result. Accepted means schema/evidence-policy validation succeeded; empty responses are accepted abstentions, not evidence of successful action recognition.

## What worked

The grounded preflight retained one selected-owner **Q cooldown-onset activation** in negative-17, bracketed at source 787.203083–787.253133 seconds. The same observation appears in HUD-only. It remains explicitly HUD-supported, with unresolved scene actor track and `physical_action_established: false`; no target or outcome was invented. This demonstrates that a primary HUD activation survives scene abstention.

Across full development HUD-only found that Q observation and a selected-owner **W activation** in sequence-08, bracketed 505.938767–506.005500 seconds. The other four development clips produced no activations. These are detector outputs, not a claim that the source contains only two activations: the complete independent development reference inventory is still being restored.

Both GPU jobs loaded and executed the pinned model. Peak observed GPU usage was approximately 36.9–37.0 GiB; peak host RSS was approximately 34 GiB. No infrastructure failure occurred in these four completed jobs.

## What needs work

Neither scene-bearing preflight produced an accepted physical action. Scene-only repeatedly abstained, citing missing or changing actor tracks and insufficient animation evidence. A valid JSON response therefore did not establish useful scene recall.

The grounded run attempted one Q physical-cast proposal, but every cited evidence entry used the generic `player_action` cue. The validator requires `cast_animation` or `cast_text`, so it rejected the entire response. This is a model evidence-contract failure, not a GPU failure, and does not by itself prove that the proposed cast was visually false. The raw response and annotated input frames are retained. Visual inspection of cited frames shows the selected region and several competing effects; it does not independently establish the claimed originating projectile or target.

One grounded abstention described corner values 25/40/70/100 as cooldown numerals. Those are a known cost/countdown ambiguity, and the text was not promoted into an activation. The primary OCR transition detector and the free-form scene commentary must therefore continue to be assessed separately.

The shared actor binder is another possible bottleneck. In the complete eight-second negative-17 input it supplies a visible selected track on only 18 of 32 scene frames, fragmented into nine track IDs. Across all six development clips it supplies a visible track on 237 of 280 scene frames. These input-audit counts are consistent with the model's attribution uncertainty; they do not establish that tracking alone caused the missed actions.

The views remove the designated selected-player bottom HUD in scene-only, while retaining common in-world text and some peripheral UI. Both scene arms share those world pixels and localization metadata. This isolates the effect of adding the native selected-player HUD strip and structured candidates; it is not a claim of complete UI removal.

## Next execution stage

The measured preflights qualify for full development inference under the unchanged recipe. Submitted jobs **8828485** (scene-only) and **8828486** (HUD-grounded) use the existing published scripts and all six development clips. Their 45-minute allocations provide headroom over the measured startup/request cost. HUD-only development is already complete and does not need rerunning.

The development-label directory contained no completed JSON files when this review resumed. Fresh blinded Luna source agents are reconstructing those references from the complete assigned source indexes, without predictions. No precision/recall/F1 is published from partial notes or shorter historical exports. Full development review, complete reference provenance checks and recipe freeze still precede the five held-out runs.

The local raw development artifacts and machine-readable review summary are under `data/league_runs/midbeast-OLLZBe5-fg4-v2/benchmarks/scene-understanding/v2-pilot/result-review-20261110/`. Longleaf originals remain under `/work/users/a/l/alshen/league-midbeast-validation/scene-understanding/v2-pilot/comparison-r1/results/`.
