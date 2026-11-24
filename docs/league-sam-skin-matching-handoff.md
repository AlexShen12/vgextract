# League SAM skin matching handoff

**Status as of November 18, 2026.** The `00:35–03:35` Midbeast pilot completed with 67 ranked SAM appearances. SigLIP repeatedly picked Snow Day Singed and produced a second, apparently spurious, Xayah lock from combat-contaminated masks. Both Qwen mask verifiers failed the clean-crop retention threshold. The next experiment should compare more targeted SAM prompts and a context-aware Qwen prompt on the same labeled footage.

## Encoder and mask selection experiments

The initial comparison used the `enemy-1` track from SAM job `9001798`, with 17 masked gameplay frames and 1,312 audited 3D render views covering 82 skins. DINOv2 ViT-S/14 and Google SigLIP ViT-B/16 each embedded foreground-only crops on gray. Each frame gave a skin its best cosine similarity across render views. The track score averaged those frame scores. The visually suspected skin was **Snow Day Singed**, skin ID `27006`. Its label came from visual inspection of the clip. [Paired encoder results](../data/league_recognition_v1/body-render-gallery-16.22-enemy-skins/sam-9001798-encoder-comparison.json)

| Frames used | DINOv2 rank for Snow Day Singed | SigLIP rank for Snow Day Singed |
|---|---:|---:|
| All 17 | 29 | 4 |
| Five picked by visual review | 9 | 1 |
| Five largest foreground masks | 12 | 1 |

Foreground pixel count recovered three of the five visual picks. The largest mask had 9,222 foreground pixels and filled 28% of its tight bounding box. A stray pixel widened the box. Bounding-box fill recovered all five visual picks on this track. Use this as a diagnostic from one clip. The later appearance pass combines fill with square root of foreground pixel count and prefers frames at least 0.25 seconds apart. [Foreground-count experiment](../data/league_recognition_v1/body-render-gallery-16.22-enemy-skins/sam-9001798-foreground-comparison.json)

## Scoreboard-backed appearance pass

- Source `OLLZBe5-fg4`, source time `[35, 215)` at 16 FPS, using the saved Midbeast scoreboard. The candidate pool covers both scoreboard sides and excludes the player champion, Darius. The patch-pinned gallery has 2,160 views, covering 82 enemy skins and 53 friendly skins. The report lists the five catalog gaps. [Appearance manifest](../data/league_recognition_v1/midbeast-sam-siglip-35-215/appearances.json) · [Gallery coverage and ranks](../data/league_recognition_v1/midbeast-sam-siglip-35-215/appearance-matches-siglip.json)
- The initial health-bar scan proposed 716 seeds, including minions and turrets. Requiring Tesseract to read a level numeral in the adjacent dark badge reduced this to 68 enemy seed candidates. The player binding excluded 22 Darius segments. The final scan yielded zero eligible non-player friendly seeds in this window. Some true champion badges had OCR digit errors, including level 3 read as level 5. [Level reader](../annotation/league/body_level.py) · [Before the level gate](../data/league_recognition_v1/midbeast-sam-siglip-35-215/appearances-before-level-gate.json)
- Preparation joins nearby fragments only when same-team endpoints agree unambiguously within the movement limit and the gap is at most 0.5 seconds. It picks one seed per appearance by minimizing overlap with other detected bodies. SAM 3.1 uses one box plus the generic `video game character` text prompt. Box-only prompting returned zero masks for the first smoke seed. The worker propagates in both directions and retries the same seed at 8 and 4 FPS if tracking state is lost or memory is exhausted. Partial results can be inspected and are ineligible for skin-lock votes. [Appearance implementation](../annotation/league/sam_appearances.py) · [SAM runbook](league-sam-multichampion-pilot.md)
- SigLIP ViT-B/16 is the default for `match-appearances` and `body_campaign match-frame`. DINOv2 is available explicitly. Appearance matching scores up to five masks. A vote needs at least three usable masks and a unique pooled winner that wins a strict majority of those masks. A skin lock needs two qualifying appearances on the same team, separated by at least one second. The lock stores both appearances and is scoped to the match, team pools, gallery, and encoder pin. A later qualifying conflict for that champion suspends the lock. [Frame matcher](../annotation/league/body_frame_matching.py) · [Lock ledger](../data/league_recognition_v1/midbeast-sam-siglip-35-215/skin-locks-siglip.json)

SAM completed 67 appearances and rejected one seed. Snow Day Singed ranked first in 35 appearances and received 21 qualifying votes. The ledger also contains **Redeemed Star Guardian Xayah**, skin ID `498038`. Its supporting windows are `00:56.76–00:58.07` and `01:00.38–01:02.20`. Both previews appear to show Singed mixed with heavy combat effects. The two windows satisfy the implemented time gap and share the same fight, so their errors are correlated. Treat the Xayah lock as a failure case. [First Xayah-supporting preview](../data/league_recognition_v1/midbeast-sam-siglip-35-215/mask-previews/appearance_3e3ca6b831b741fc4fe0.jpg) · [Second preview](../data/league_recognition_v1/midbeast-sam-siglip-35-215/mask-previews/appearance_f8a9f3d44edd68f4e79c.jpg)

## Qwen mask-verifier benchmark

The frozen sample contains 40 crops from `00:44–00:51` and `01:23–01:29`. Each appearance contributed the five original SigLIP masks and 15 evenly spaced additional crops. Before seeing either model's predictions, Codex labeled 17 clear, 19 muddled, and four ambiguous. Each Qwen model saw one masked RGB crop on gray and returned `clear`, `muddled`, or `uncertain` plus a reason. The predeclared threshold was at least 13 of 17 clear crops retained and at most one of 19 muddled crops accepted. [Frozen review](../data/league_recognition_v1/midbeast-sam-siglip-35-215/mask-verifier/reviews.json) · [Comparison and replay](../data/league_recognition_v1/midbeast-sam-siglip-35-215/mask-verifier/comparison.json)

| Checkpoint | Clear retained | Muddled accepted | Median request time | Longleaf job |
|---|---:|---:|---:|---|
| Qwen3.5-9B | 1/17 | 1/19 | 2.26 s | `9573308` |
| Qwen3.8-27B-FP8 | 0/17 | 0/19 | 5.47 s | `9573309` |

Both models failed the retention threshold and used zero abstentions. The 27B labeled all 40 masks `muddled`. The 9B left two masks for the cleaner appearance, with Snow Day Singed still at rank 1. Both models left zero masks for the crowded appearance. Several reasons interpreted Singed's shield or tank as an obstruction. The review covers two appearances from one clip and is model-assisted. [9B verdicts](../data/league_recognition_v1/midbeast-sam-siglip-35-215/mask-verifier/predictions/qwen35_9b/run.json) · [27B verdicts](../data/league_recognition_v1/midbeast-sam-siglip-35-215/mask-verifier/predictions/intake/run.json)

## Next experiment

The follow-up tooling and job commands are in the [SAM skin study runbook](league-sam-skin-study.md).

1. **Compare targeted SAM prompts.** Try three seed times per appearance. Use the level-badged bar to locate the target, then refine with a positive torso point and negative points on visible neighboring units. Run short propagation around each seed and compare target coverage, contamination, and mask stability before selecting one. Include the current box-plus-text method and a SAM 2.1 per-object tracking baseline on the same windows. [SAM 3 refinement examples](https://github.com/facebookresearch/sam3) · [SAM 2.1](https://github.com/facebookresearch/sam2)
2. **Give the Qwen 9B verifier scene context.** Show the original crop with a thin candidate-mask outline and target marker beside the masked crop. Request separate judgments for readability and contamination. Use the frozen 40 crops as a regression set.

   > Judge the marked character. Count its shield, weapon, tank, and costume as part of it. Flag another unit, floating text, or a spell effect only when it covers a substantial part of the target or its equipment. Answer whether the target is readable and whether there is substantial contamination.

3. **Verify the downstream lock behavior.** Add labeled appearances from another match. Measure clean-mask retention, contaminated-mask acceptance, and false SigLIP locks. Include the two Xayah-supporting windows as a fixed regression case. Check whether the seed selector or verifier eliminates that repeated error while preserving clean Singed footage. Proceed to seed ranking with Qwen only if the revised verifier clears the retention and contamination checks.

## Reproduce the completed scoring

The successful SAM smoke and full array were `9571400` and `9571401`. The cluster appearance scorer `9571402` failed because its staged Python snapshot lacked `ingestion.common.replay_state`. The Qwen replay job `9577141` timed out. Both scoring steps completed locally using the saved masks, Qwen predictions, and pinned render gallery. For another cluster appearance-scoring run, include the `ingestion` package in the checksum-verified source snapshot. The local commands below use an environment with PyTorch, Transformers, and the staged SigLIP checkpoint. [Full runbook](league-sam-multichampion-pilot.md)

```bash
run=data/league_recognition_v1/midbeast-sam-siglip-35-215
OMP_NUM_THREADS=4 python -m annotation.league.sam_multichampion match-appearances \
  --root "$run" --gallery "$run/gallery.json" --require-complete
OMP_NUM_THREADS=4 python -m annotation.league.mask_verifier evaluate \
  --run "$run" --root "$run/mask-verifier" --require-complete
```
