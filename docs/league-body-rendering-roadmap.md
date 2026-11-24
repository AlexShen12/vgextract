# Champion body rendering and recognition

Validate the ten champions in the pilot footage across independent matches, then add up to 20 champions per cohort. CPU portrait reference matching supplies the scoreboard roster. A single-match skim can assign experimental roster identities across openings. Review is optional and can qualify or correct a roster using source evidence. The operational path reuses verified player-name bindings and uses frozen DINOv2 when a readable name still needs an identity. See the [existing comparison method](league-body-comparison-method.md).

## Milestones

### M0 Baseline and receipts

- Status: complete
- Depends on: none
- [x] Preserve the stretched-crop, global-embedding baseline and the submitted source snapshot.
- [x] Collect the comparison receipt and reconcile job accounting.
- Exit: receipts, runtime, supported assets, and diagnostic limits are recorded.
- Command (available): `python -m annotation.league.body_campaign coverage --root CAMPAIGN`
- Evidence: [completed diagnostic receipt](reports/2026-11-11/body-comparison-8830034.json)
- Receipt: 40db770691a9f4076fb627f8a4becc77b23e4feedc820cbb301d507d5ee616b2 reports/2026-11-11/body-comparison-8830034.json

### M1 Scoreboard roster

- Status: complete
- Depends on: none
- [x] Confirm two adjacent complete reads per opening or one human-reviewed snapshot.
- [x] Scope matching and backfill to a verified playback segment and suspend disputed rosters.
- [x] Load ten CPU-matched cross-opening members as experimental context and expose optional portrait review.
- Exit: experimental roster loads from the real skim. Reviewed witnesses, source interval, conflicts, corrections, and legacy skim compatibility pass regression tests.
- Command (available): `python -m annotation.league.scoreboard_skim --derive-membership --output SKIM`
- Command (available): `python -m annotation.league.scoreboard_skim --serve-review --output SKIM`
- Command (available): `python -m annotation.league.body_campaign identify --root CAMPAIGN --scoreboard-skim SKIM`
- Evidence: [CPU roster and review endpoint receipt](reports/2026-11-12/body-roster-cpu.json) and [cross-opening review tests](../tests/test_league_scoreboard_skim.py)
- Receipt: f3054e69c31691e12253b84c413cf4a1059b7c4fbfcaf1cf3021509ae430f9ff reports/2026-11-12/body-roster-cpu.json

### M2 Player-name reuse

- Status: blocked
- Depends on: M1
- [x] Establish exact name bindings from confirmed scoreboard rows or confirmed visual matches.
- [x] Resolve registry hits before opening the encoder or gallery.
- [x] Rebuild forward and backfilled identity evidence after corrections.
- [ ] Demonstrate reuse and unbound matching on footage with readable player names, including re-entry on a new track.
- Exit: known names skip visual work, unbound readable names enter matching, and conflicts remain anonymous.
- Command (available): `python -m annotation.league.body_campaign identify --root CAMPAIGN --intake-run INTAKE`
- Evidence: [lazy matching and correction tests](../tests/test_league_body_operational.py)

### M3 Independent evaluation data

- Status: blocked
- Depends on: none
- [x] Pin the ten CPU-matched pilot candidates in an isolated development campaign.
- [ ] Verify the roster and supported skins independently. Test the v3 portrait profile on fresh footage with clear, obscured, and respawning portraits.
- [ ] Assign gallery, development, validation, and held-out material by underlying match before drafting.
- [ ] Collect 150 development, 50 validation, and 100 held-out positive track segments. Require 20 per champion overall, ten held-out per champion, and 50 held-out per team.
- [ ] Collect 25 development, ten validation, and 15 held-out negatives. Include unreadable bodies, misleading nameplates, out-of-gallery champions, and non-champions.
- [ ] Verify held-out labels from source evidence. Record scoreboard rows, names, body regions, ownership, and visibility.
- Exit: coverage and independent human review requirements pass.
- Command (available): `python -m annotation.league.body_campaign coverage --root CAMPAIGN`
- Evidence: [current diagnostic scope](league-body-comparison-method.md)

### M4 Crop geometry and render fidelity

- Status: blocked
- Depends on: M0, M3
- [ ] Compare aspect-preserving padding at 0%, 10%, and 20%.
- [x] Run a CPU geometry and degradation probe using the existing source-only Darius masks.
- [x] Check the published base-skin endpoints for the ten pilot champions.
- [ ] Audit camera, mesh, textures, transparency, and animation for admitted skins.
- [ ] Test size, blur, brightness, and compression variants against development evidence.
- Exit: invertible crop transforms, verified render assets, and paired development results.
- Command (available): `python -m annotation.league.body_render --help`
- Command (available): `python -m annotation.league.body_study --geometry-labels LABELS --video VIDEO --encoder-lock LOCK --output PROBE --device cpu`
- Command (available): `python -m annotation.league.body_campaign fit-appearance --root CAMPAIGN --device cpu`
- Evidence: [CPU source-geometry diagnostic](reports/2026-11-12/body-source-geometry-probe.json) and [pinned base-skin asset probe](reports/2026-11-12/body-base-assets-probe.json)

### M5 Local features and reference agreement

- Status: blocked
- Depends on: M4
- [ ] Compare global and foreground patch scores using the same frozen encoder.
- [x] Measure foreground patch availability and support on matched source-crop pairs.
- [x] Calibrate separate score and margin gates for local features and global fallback when both routes have development evidence.
- [ ] Compare RGB, grayscale, and silhouette weights and distinct-template agreement.
- [ ] Audit shortlist coverage and template balance across champions and skins.
- Exit: paired development results identify a configuration with recorded cost and supporting references.
- Command (available): `python -m annotation.league.body_campaign fit-appearance --root CAMPAIGN --device cpu`
- Evidence: [CPU source-geometry diagnostic](reports/2026-11-12/body-source-geometry-probe.json) and [mixed-route calibration test](../tests/test_league_body_operational.py)

### M6 Frame selection

- Status: blocked
- Depends on: M2, M4
- [x] Choose one clear frame per trailing 250 ms window for visual votes.
- [x] Record the evidence time and vote time and reset votes at track boundaries.
- [x] Compare fixed and quality-based cadence in the appearance fitter and freeze the chosen cadence after validation.
- [ ] Record the paired comparison on source-labeled development data.
- Exit: fixed-cadence comparison and boundary tests pass.
- Command (available): `python -m annotation.league.body_campaign fit-appearance --root CAMPAIGN --device cpu`
- Evidence: pending

### M7 Evaluation and release

- Status: blocked
- Depends on: M3, M5, M6
- [x] Save proposal observations and score development localization against independent source boxes.
- [ ] Qualify proposal refresh and track association for re-entry and late entrants before scaling GPU work.
- [x] Score matched bodies by native source timestamp and box overlap. Keep missed observations in the coverage denominator.
- [ ] Run six paired gallery and context arms and the operational name-reuse path.
- [ ] Freeze development policies after validation and evaluate independent held-out data once.
- [ ] Publish qualified evidence to opt-in timeline and review views.
- Exit: 98% accepted-ID precision and 70% correct readable-body coverage per team and promoted arm, with counts, confidence intervals, localization, reuse, timing, and runtime receipts.
- Command (available): `python -m annotation.league.body_campaign audit-tracking --root CAMPAIGN`
- Command (available): `python -m annotation.league.body_campaign run --root CAMPAIGN --policy CAMPAIGN/appearance-policy.json --split validation`
- Command (available): `python -m annotation.league.body_campaign score --root CAMPAIGN --policy CAMPAIGN/appearance-policy.json --split held_out`
- Evidence: [development CPU proposal probe](reports/2026-11-12/body-tracking-proposal-probe.json)

### M8 Roster expansion

- Status: planned
- Depends on: M7
- [ ] Audit assets and collect independent labels for each cohort of up to 20 new champions.
- [ ] Revalidate prior champions and record a frozen release manifest for each cohort.
- Exit: the cohort passes the same release gates and has a rollback target.
- Command (available): `python -m annotation.league.body_campaign coverage --root CAMPAIGN`
- Evidence: pending

## Job and budget register

| Job | Purpose | Last observed state | GPU reservation |
| --- | --- | --- | --- |
| 8830031 | Encoder and source preflight | Completed | CPU |
| 8830032 | Static render candidates | Completed | CPU |
| 8830033 | Cast-bar diagnostic | Failed on missing `av` | CPU |
| 8830034 | Body crop and mask comparison | Completed in 30 seconds | 900 seconds |

The pilot ceiling is 28,800 GPU seconds, including failed allocations. Keep 3,600 seconds for recovery. The shared ledger charges job 8830034 its full 900-second reservation, leaving 24,300 seconds for regular allocations. Stage future work with distinct source snapshots, manifests, and output roots.

## Collection gaps and decisions

The diagnostic has 28 visible Darius reference times and unreviewed static candidates for three champions. The new pilot manifest has ten candidate IDs and zero independent labels or admitted gallery entries. Readable player-name footage, independent roster labels, cross-match body labels, audited galleries, and release runtime measurements are collection gaps. Record source hashes and receipt hashes as each gap closes.

- 2026-11-11. Use two complete reads or human review for scoreboard confirmation. Backfill within the verified playback segment.
- 2026-11-11. Learn names from verified scoreboard rows or accepted visual matches. Reuse established names across tracks within the segment.
- 2026-11-11. Prioritize offline accuracy and preserve the eight-hour pilot allocation ceiling.
- 2026-11-12. Use CPU portrait matching for experimental cross-opening roster identities. A reviewer can verify an interval and its witnesses or correct disputed rows. Use frozen DINOv2 for body appearance.
- 2026-11-12. Keep the official Worlds broadcast as a future professional-broadcast idea. It is outside the active pilot, including its nameplate calibration and scoring.

## Remaining implementation plan

1. **M2 source and smoke test.** Source a solo-queue or replay-format clip with player handles visible above the bodies. Inspect source-native crops for the same handle attached to the same body on two distinct frames. Reserve short development windows for an existing scoreboard binding, a readable unbound name, a re-entry on a new track, and an unreadable or ambiguous name. Record source hash, exact PTS, layout, bar-relative name region, and body association. Run `identify --device cpu` against prepared tracks and an admitted gallery. Record binding route, evidence and availability times, visual attempts, encoder calls, and the later registry hits. The current Midbeast and domisumReplay sources support other development diagnostics. [Source collection notes](league-body-recognition-handoff.md)
2. **M3 split and label intake.** Create a frozen match-group inventory before drafting or selecting references. Record upload and match identity, source hash, patch, layout span, split, and duplicate relationships. Label roster membership and track segments against source frames. Include negative types, exact readable names, team, visibility, boxes, and skin evidence where available. Assign independent human reviewers to held-out evidence. Run `coverage` after each batch and use per-champion and per-team deficits to choose the next match. Admit gameplay references from gallery-only matches and rendered references from audited skin assets.
3. **M4–M6 development experiments.** Start with a CPU DINOv2 diagnostic on the existing Darius source masks. Replay the same crop pairs for stretched and aspect-preserving geometry, foreground patches, and appearance channels. Record similarity and feature availability. Use independent multi-champion development data and admitted gallery entries to choose crop, render, and matching policies. Audit supported animated skins at the required headings, elevations and phases, then run `fit-appearance`. Compare fixed and quality-based cadence on the same labeled time windows. Use validation labels once to decide whether to freeze the development winner.
4. **M7 tracking and evaluation.** Audit first appearance, disappearance, re-entry, and association changes on labeled development clips. Compare prompt coverage and SAM mask overlap. Use the existing preflight and independent seed-mask gate before a tracking allocation. Run and score all six appearance/context arms on validation, then score the operational path with name reuse. Record frozen policies, source and gallery hashes, split evidence, runtimes, peak memory, and allocation receipts. Evaluate held-out once after validation qualifies each promoted arm. Publish through the opt-in review when the per-team gates pass.
5. **M8 cohort rollout.** Freeze a manifest and rollback target for the first qualified cohort. Keep each later cohort at 20 champions or fewer and repeat the asset audit, independent evaluation, and regression checks for prior champions.

### Source inventory before collection

| Source | Match grouping | Assigned use | Evidence needed next |
| --- | --- | --- | --- |
| Midbeast `OLLZBe5-fg4` | One inspected match | Development only | Independent review of the ten roster members and additional body labels. Existing source masks cover one champion. |
| domisumReplay `xICbe-OOPvE` | Separate inspected match | Development only | Review proposals and potential out-of-gallery negatives. The inspected overlay displays champion names. |
| New replay-format footage with player names enabled | Verify match identity and deduplicate before assigning a split | Development smoke test first | Source-native nameplates, roster evidence, tracking continuity, and re-entry. |
| New independent matches featuring pilot champions | Record one group per underlying match before selecting any clips | Gallery, validation, or held-out | Sufficient team and champion coverage. Have held-out source labels reviewed independently. |

## 2026-11-12 execution notes

- The CPU DINOv2 probe processed 28 source masks across five Midbeast windows. Median best cross-window RGB cosine was 0.754 for stretched crops and 0.741, 0.749, and 0.726 for 0%, 10%, and 20% padding. Both patch layers had sufficient foreground on all 28 chosen pairs. The largest median drop among the individual degradation probes came from reducing body height to 32 pixels (−0.117). The probe measures one champion within one match. Policy selection and release metrics still require the independent evaluation data. [Probe receipt](reports/2026-11-12/body-source-geometry-probe.json)
- The pinned CommunityDragon endpoints returned HTTP 200 for meshes and textures and HTTP 404 for skeletons for all ten base skins. The Darius and Talon directory listings also lacked skeleton files. The animation pipeline needs matching rigged assets before animated renders can enter the gallery. Preserve the existing static candidates as development evidence. [Asset receipt](reports/2026-11-12/body-base-assets-probe.json)
- The Longleaf SSH login failed during a read-only attempt to locate the prior render and mask artifacts. The local ledger still records 900 GPU seconds from job 8830034. No new allocation was submitted. [Recorded Longleaf receipt](reports/2026-11-11/body-comparison-8830034.json)

Next executable work is to source an eligible nameplate clip and begin the split inventory. The recorded plan and handoff contain the commands and evidence needed for the next campaign revision. [Handoff](league-body-recognition-handoff.md)

### Reproduce the completed CPU probe

Use the League Python environment with the pinned DINOv2 CPU dependencies installed. Write a new output directory for each run.

```sh
python -m annotation.league.body_study \
  --geometry-labels data/league_runs/midbeast-OLLZBe5-fg4-v2/benchmarks/sam-tracking/v1/labels.json \
  --video data/youtube/lol_midbeast/OLLZBe5-fg4.mp4 \
  --encoder-lock data/league_assets/body-encoder/dinov2-vits14/encoder-lock.json \
  --output data/league_recognition_v1/body-geometry-dev-20261112-r2 --device cpu
```
