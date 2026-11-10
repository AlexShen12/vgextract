# League scene evaluation expansion plan

Date: 2026-11-09  
Status: proposed work; collection, inference and job submission have not started.

Execute the four stages below in order: **2 → 3 → 4 → 5**. These numbers preserve the original discussion. Each stage must produce its deliverables and pass its exit checks before the next begins. The work can proceed while the separate SAM campaign runs; SAM outputs are an optional later comparison.

## Objective and existing work

Build a source-labeled evaluation set that measures false activations, actor mistakes, fragmented or merged actions, unsupported outcomes, and the contribution of HUD grounding.

Reuse the existing action and caption occurrence infrastructure. Existing tests already cover cooldown persistence, initial cooldowns, ownership changes, discontinuities, duplicates, occurrence phases and freeze gates. Add source-backed regression cases and integration coverage where those tests do not establish real-footage behavior.

The three [Luna-reviewed examples](league-luna-event-label-review-2026-11-09.md) are development material. Their labels remain non-exhaustive model-reviewed drafts. Do not copy their verification status into the new reference set or use them as fresh held-out examples.

Start a separate benchmark revision under `benchmarks/scene-understanding/v1/`, with experiment outputs under `experiments/scene-understanding/`. Preserve historical benchmarks and production outputs.

## Stage 2 — Collect difficult negative examples

**Purpose:** establish when the system must abstain from a particular claim, even if other activity is visible.

### Collection

Select development candidates from already authorized local footage. Aim initially for **three independently inspected examples per case below**, approximately 18 clips. A clip can have multiple tags, but report unique clips and per-case counts separately. These are pilot coverage targets, not evidence of statistical reliability.

| Case | Include in source footage | Claim that should be absent |
| --- | --- | --- |
| Cooldown present at clip start | Slot already cooling down, followed by ordinary ticks | A new activation inferred solely from the initial state or countdown |
| Cost/countdown ambiguity | Readable mana cost, icon artwork and nearby text; preferably include a genuine transition for contrast | A cooldown or activation inferred from the cost numeral |
| Camera movement | Pan, camera follow or zoom with identifiable terrain references | Player displacement inferred solely from screen-coordinate movement |
| Nearby competing caster | Another champion casts while the selected player is visible | The other champion's action attributed to the selected player |
| Interrupted recall | Observable recall start and interruption within a continuous view | Recall completion or base arrival without supporting footage |
| Quiet footage | Fully reviewed interval with no action from the relevant measured categories | Invented movement, attacks, casts or other events |

Use approximately 8–12 seconds per candidate, retaining additional source context where needed to resolve its interpretation. Do not call a clip globally negative because it contains one negative case: interrupted recall is a positive recall attempt, and another champion's cast is a positive event for that actor.

### References and tests

- Record source hash, full inspected interval, source PTS, selected actor/ownership span, case tags, observable positive events, forbidden claims, uncertainty and evidence frames.
- Inspect source footage without detector predictions. Disclose previous exposure and use an unexposed reviewer where strict independence is required.
- Store unknown or unobservable spans explicitly. An unreadable slot is not a known ready slot; an obscured interval is not quiet footage.
- Add compact regression fixtures from accepted source cases. Cover cooldown continuation versus onset, cost versus centered countdown, camera motion versus world displacement, actor swaps, and recall interruption versus completion.
- Record each candidate and inspected interval in an exposure ledger immediately. All footage inspected in this stage is development material, including rejected candidates.

**Deliverables:** candidate/exposure ledger, development negative-case manifest, revisioned source labels, evidence references, and source-backed regression fixtures.

**Exit checks:** every retained case has source inspection and a specific expected decision; quiet intervals have explicit complete-review attestations. Missing case types remain untested. Fixture tests pass without changing historical benchmark behavior.

## Stage 3 — Add action sequences with context

**Purpose:** test recognition of an action over time and prevent unsupported links between an activation, target and outcome.

### Collection and labeling

Add an initial **12 development sequences**, aiming for two examples of each pattern:

1. An isolated ability with visible preparation, cast/release and recovery.
2. Rapid basic attacks with distinct new onsets.
3. An empowered attack with ability activation and physical attack recorded separately and linked when supported.
4. A clearly observable miss or no-hit result with sufficient target/effect visibility.
5. A cast with an obscured result, requiring an unknown outcome.
6. A recall lifecycle, including completion or interruption where directly observable.

Reuse Stage 2 footage where suitable and report the overlap. Aim for 12–24 seconds per sequence; retain longer source context only when needed. A missing hit cue does not establish a miss. A cut or disappearance does not establish interruption or completion.

For each occurrence, record:

- Stable event ID, actor category (`selected`, `other`, `unknown`) and actor binding evidence.
- Action category, nullable Q/W/E/R slot and independently supported ability name.
- Source-time onset bracket and visible phase intervals: preparation, cast/attack release, recovery and result where supported.
- Nullable target and outcome, with separate evidence for each asserted field.
- Left/right censoring, unknown intervals, continuity/ownership barriers and linked HUD activation IDs.
- Review method, reviewer, prediction exposure, label revision and source-frame provenance.

Keep a HUD activation as its own observation when the physical animation is unresolved. One physical onset with continuing phases is one occurrence. A new attack needs a new onset. Multiple simultaneous observations may share a frame.

Derive frame-level records from reviewed intervals, marking them as derived. Do not describe them as independent judgments of every frame.

### Integration checks

- One action spanning several packets emits one occurrence; distinct rapid attacks remain distinct.
- HUD-supported slot activation reaches the final output even when scene recognition abstains.
- A HUD activation alone does not invent a physical attack, target, hit, kill or last hit.
- Readable cast-bar text can establish a name independently of a visible target/outcome.
- A changed actor, cut or replay boundary prevents unsupported evidence carryover.
- Cast-span timing and cooldown-onset timing remain separate fields.

**Deliverables:** contextual development sequence manifest, source labels and phase evidence, derived frame records, integration fixtures, and a coverage/gap report.

**Exit checks:** references distinguish physical occurrences from HUD activations, phase boundaries and censoring are valid, unsupported fields remain unknown, and integration tests pass. Cases unavailable in the recording stay untested.

## Stage 4 — Reserve fresh evaluation clips

**Purpose:** establish an evaluation set whose labels and examples have not influenced recognition choices.

1. Consolidate exposure from Stages 2–3, the three Luna examples and their full context, prior calibration/inspection logs, existing benchmarks and all previous experiments. Include repeated depictions of the same gameplay event.
2. Propose approximately **24 held-out clips**: 12 hard-negative cases and 12 contextual sequences. Aim for two instances of each Stage 2 case and each Stage 3 pattern, with overlap disclosed. Do not fill missing categories with weak labels.
3. Exclude overlapping development/context intervals and apply at least the existing eight-second guard **outside the complete inspected context**. Group repeated events and duplicate views into one split. Honor any stricter existing validator rules.
4. Have a reviewer without recognition-prediction exposure inspect and label the held-out candidates. Recognition developers may access the manifest metadata needed to operate the benchmark, while the references remain outside prompt/rule tuning. Record actual blinding and access limitations.
5. Freeze source hashes, source intervals, split/group IDs, context windows, label schema, review attestations and the label revision. Later reference corrections require a new revision and an explicit evaluation-exposure record.

Previously unseen intervals from the current recording support a **within-recording** evaluation. For a cross-match generalization claim, reserve a different underlying match and keep all its uploads/views in the held-out match group. Reuse available authorized footage first; identify a missing source explicitly rather than silently downloading one.

**Deliverables:** frozen split manifest, exposure/exclusion ledger, held-out reference revision, per-case/event coverage table, and a statement of evaluation scope.

**Exit checks:** no prohibited interval/repeat/match-group leakage; held-out references were created without detector predictions; unsupported independence claims are absent. No held-out recognition run begins before Stage 5's development recipe is frozen.

## Stage 5 — Compare input configurations

**Purpose:** measure the contribution of HUD information with paired runs over the same clips and references.

### Arms and evidence contract

| Arm | Allowed input and behavior |
| --- | --- |
| Scene-only | Fixed gameplay/player views with the designated HUD panels masked. Preserve in-world cast bars and overhead labels consistently and disclose them as allowed evidence. No separate HUD crops or structured HUD observations. |
| HUD-only | Fixed HUD crops and the pinned HUD reader's observations. No gameplay/player imagery. Emit only claims supported by these inputs, including unknown animation, target and outcome. |
| HUD-grounded scene | The same scene views plus the exact HUD crops/observations used by HUD-only. Supported cooldown transitions may establish slot activation; scene evidence adds physical action, name, target or result only when supported. |

The first comparison isolates **separate HUD grounding**. Since scene-only retains in-world cast-bar text, describe it accordingly; it is not a pure animation-only experiment. Any additional text-masking experiment is a separately versioned arm.

Use the same pinned scene model, prompts except for modality-specific instructions, decoding settings, scene timestamps, crop geometry, actor-binding policy and source context in both scene-bearing arms. Use one frozen HUD reader/configuration for both HUD-bearing arms. Never supply reference transitions as HUD input.

For the initial run, keep the existing 4 FPS scene sampling. Produce HUD observations once with a documented, frozen cadence. Reference labeling may inspect native frames. Keep the primary recall denominator intact when 4 FPS misses an observable source event; report sampling visibility as a separate diagnostic. Any denser scene comparison gets a separate recipe and compute accounting.

Keep actor grounding identical across arms and disclose shared metadata. HUD-only must not receive a scene-derived target or result through that metadata. Do not compare Qwen in one arm with Luna in another and attribute the difference to HUD input; a second recognizer requires its own complete set of arms.

### Execution order

1. Implement modality isolation, reference validation, scoring and an artifact audit. Add tests that detect forbidden-input leakage and fabricated target/outcome completion.
2. Run a small development preflight with a quiet case, cost/countdown ambiguity, existing cooldown, competing actor, rapid sequence and obscured outcome. Reuse existing cases where possible.
3. Resolve infrastructure failures, malformed responses and provenance problems. Record time, memory and requests; estimate the full-run cost before submission.
4. Run all three arms on development clips. Review discrepancies, make development-only changes and repeat affected checks.
5. Freeze code revision, model/runtime, prompts, masks, crops, cadences, HUD reader, actor binding, event matching rules, metrics and reference revisions. Publish the immutable code through GitHub before transferring code to Longleaf.
6. Run all three frozen arms on the held-out clips. Preserve partial runs and rejected responses; complete or explicitly account for failures before publishing results. Do not tune on held-out outputs.
7. Publish paired results and an evidence audit. Any subsequent tuning uses a new development revision and requires fresh holdout for a new unbiased evaluation.

GPU job definitions and allocation estimates are prepared only during execution, following the Longleaf workflow. This document does not submit or modify jobs. SAM jobs continue independently.

### Metrics and result interpretation

Match HUD activations and physical occurrences on their respective reference tracks; do not collapse them into one ambiguous denominator. Reuse compatible existing one-to-one matchers. Predeclare category-specific rules: ±0.25 seconds for point onsets, temporal IoU ≥0.5 for resolvable intervals, with censored boundaries handled explicitly. Keep cooldown-onset error separate from cast-span error.

| Measurement | Required breakdown |
| --- | --- |
| Event precision, recall, F1 | Raw counts by category, actor scope, arm and case tag; include duplicates and merged/missed events |
| False claims on negatives | False events per inspected minute, affected clips, and per-case failures; normalize by eligible reviewed scope |
| Actor, slot and name | Correct/incorrect/unresolved counts plus accuracy and coverage with explicit denominators |
| Target and outcome | Assertion support and coverage; unknown outcomes must not become asserted misses or hits |
| Timing and action continuity | Onset/offset error where resolvable, interval overlap, duplicate rate, and merged-event count |
| Caption support | Field-level supported/unsupported/unresolved claims linked to evidence and event IDs |
| Execution | Successful/rejected/failed requests, incomplete clips, runtime, memory and model calls |

Compare arms on their shared capabilities, particularly selected-player slot activations. Report scene-dependent target/outcome capability separately; HUD-only has no evidence for those fields. Categories with no reference examples are untested. An arm with zero output does not receive a perfect precision score.

Report paired per-clip differences and disclose source/match counts. Do not treat hundreds of neighboring frames as independent matches. Preserve existing benchmark thresholds where applicable; calibrate any new acceptance thresholds on development data before held-out execution.

### Optional later SAM arm

After the separate tracking experiment is reviewed, add a fourth arm using the frozen HUD-grounded recipe and SAM-derived player views. Change only the localization input. Preserve missing masks/seeds in coverage counts, and evaluate recognition separately from segmentation quality. Do not assume higher mask IoU implies better action recognition.

**Deliverables:** frozen three-arm recipes, development report, held-out scorecard, input/evidence audit, failure ledger, resource report, and revised capability statements limited to tested footage/configurations.

**Exit checks:** held-out results have explicit denominators and provenance; infrastructure failures are resolved or the report is marked incomplete; unsupported categories and generalization claims remain untested.

## Completion checklist

- [ ] Stage 2: source-backed negative cases and regression fixtures accepted.
- [ ] Stage 3: contextual sequences and occurrence/HUD integration checks accepted.
- [ ] Stage 4: fresh held-out references and split/exposure guards frozen.
- [ ] Stage 5: development comparison completed and recipe frozen.
- [ ] Stage 5: held-out paired evaluation and evidence audit completed.
- [ ] Optional: SAM localization comparison, after tracking results are reviewed.

## Related contracts

- [Independent action labeling](league-actions-labeling-handoff.md)
- [Occurrence protocol and evaluation](league-caption-event-longleaf.md)
- [Selected-player grounding roadmap](league-grounding-roadmap.md)
- [Data and annotation quality rubric](pipeline-quality-rubric-plan.md)
