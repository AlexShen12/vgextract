# Longleaf scene-reference results — 2026-11-10

Job **8738046** completed successfully in **11 minutes 39 seconds** with exit code 0. All **66 tests passed** on Longleaf. Code revision: `316bd6876939d2390cfb46c793dbad01d9e32b0b`.

The job regenerated source-review evidence, validated source identity and reference contracts, audited referenced-asset availability, and derived native-source-time annotation records. It ran no recognition inference or held-out evaluation. Regeneration does not constitute another source review or human verification.

| Stage | Reviewed candidate clips | Derived frame records |
| --- | ---: | ---: |
| Difficult negatives | 18; 14 contain at least one supported scoped case | 9,709 |
| Contextual sequences | 12 | 11,148 |
| Total | 30 clip contexts | 20,857 |

Frame counts include overlapping clip contexts and are derived annotation records, not unique independent observations. All labels remain non-exhaustive model-reviewed drafts.

| Negative case | Clips with draft support |
| --- | ---: |
| Cooldown already running at clip entry | 9 |
| Cost/countdown distinction | 9 |
| Camera motion/change | 6 |
| Competing caster | 1 |
| Interrupted recall | 0 |
| Fully reviewed quiet footage | 0 |

| Contextual pattern | Clips with draft support |
| --- | ---: |
| Isolated ability with recorded phases | 5 |
| Distinct rapid attacks | 1 |
| Empowered attack linkage | 0 |
| Observable miss | 0 |
| Cast with obscured outcome | 1 |
| Recall lifecycle | 2 |

Cases can overlap. Zero support marks a reference-coverage gap, not detector failure or proof that the action did not occur. The source draft review required corrections, including Apprehend misread as Recall and recall completion confused with later fountain confirmation. Further semantic adjudication is needed before using these labels for an accuracy benchmark.

The real-pixel regression tests exercise upper mana cost versus centered cooldown, no fresh activation from an initial countdown, retention of a HUD activation through scene abstention, actor-safe association, and Apprehend rejection as Recall with an actual Recall positive control. Passing these cases does not measure general scene understanding.

## SAM campaign

- **8719267**, preflight: failed after 50 seconds, exit code 1.
- **8720444**, qualification gate: canceled.
- **8720690**, source-seed comparison: canceled.
- **8720691**, automatic-seed comparison: canceled.

The preflight reached model session initialization, then failed with:

```text
TypeError: Sam3MultiplexTrackingWithInteractivity.init_state() got an unexpected keyword argument 'offload_state_to_cpu'
```

This is a wrapper/API compatibility failure. The dependent comparisons did not execute, so there is no SAM mask IoU, tracking-quality or recognition-improvement result from this campaign. No jobs were resubmitted while collecting these results.

## Evidence and continuation

The collected [campaign report](../data/league_runs/midbeast-OLLZBe5-fg4-v2/benchmarks/scene-understanding/v1/longleaf-campaign-report.json) contains source identity, restored-input inventory, stage totals and execution scope. The [Longleaf handoff](league-scene-reference-longleaf.md) records working paths and reproduction commands.

The persistent Longleaf run is `/work/users/a/l/alshen/league-midbeast-validation/scene-understanding/v1`. Its stage reports include current label hashes and empty missing-asset lists. The CPU result is ready for source-label adjudication; input-arm comparison and held-out scoring have not started.
