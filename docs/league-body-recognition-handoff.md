# Champion body recognition handoff

## 2026-11-11 Local implementation and Longleaf receipts

The body campaign now has a roster-gated identification path, a segment-scoped player name registry, lazy appearance features, and opt-in evidence review. The local integration fixture resolves a reviewed name with zero encoder and gallery calls, then rebuilds to unknown after a portrait correction. The frozen DINOv2 checkpoint ran on CPU with RGB and grayscale embeddings and foreground tokens from both tested layers.

Longleaf job 8830034 completed in 30 seconds of allocated wall time. The shared pilot ledger charges its full 900-second reservation. The diagnostic encoded 28 of 28 source boxes, 28 of 28 independent masks, four of 28 verified-seed SAM masks, and two of 28 automatic-seed masks. It compared 48 static reference candidates. The single-match Darius inputs support a crop and mask diagnostic. See the [accounting and comparison receipt](reports/2026-11-11/body-comparison-8830034.json).

The first scoreboard probe used Data Dragon 16.22.1 and CommunityDragon 16.22 through the direct portrait matcher. The four cardinal shifts in the existing intake apply to item close misses. The saved v1 and v2 profiles have identical scoreboard portrait score and margin thresholds. The highest initial snapshot count was seven of ten. The probe includes visible yellow question marks, color matches below the margin gate, and red respawn overlays. Comparing v1 and v2 ranks and four shifted crops is in progress.

The initial nameplate inspection found health bars without readable player names in the sampled gameplay windows. Keep these bodies anonymous under the operational name gate. A nameplate calibration requires source evidence of player names and their association with the corresponding bars.

Follow the [roadmap](league-body-rendering-roadmap.md) for commands, release gates, and collection gaps.

## 2026-11-11 Five-position scoreboard result and current scope

The v3 development profile uses Data Dragon 16.22.1 and CommunityDragon 16.22. It scores center, left, right, up, and down for a visible portrait. Each view records its top candidate, similarity, margin, and gate outcome. Passing views vote, with the highest similarity breaking an equal vote count. Inventory review displays the five reads. The color margin is 0.05 in v3. Respawn portraits retain a 0.74 score floor and 0.06 margin.

| Source time | Row | Direct v2 result | Five-position v3 result |
| --- | --- | --- | --- |
| 99.016 s | Left 3 | Ahri ranked first at 0.664 with a 0.053 margin | Ahri accepted, strongest view one pixel right |
| 99.016 s | Left 4 | Fiora ranked first at 0.671 with a 0.033 margin | Xayah accepted at 0.732 with a 0.056 margin, one pixel right |
| 164.014 s | Left 4 | Fiora ranked first at 0.652 with a 0.023 margin | Xayah accepted at 0.737 with a 0.075 margin, one pixel right |

The same 30 crops yielded 20 accepted portraits, up from 17. Of the ten remaining unknowns, six show the yellow question mark, two timer masks leave insufficient usable pixels, and two timer matches stay below the gate. At 604.003 seconds the right-shifted grayscale view ranks Ahri first at 0.646 with a 0.094 margin. The score floor is the failing gate there.

The local artifacts under `data/league_recognition_v1/body-roster-local-probe` include `v3/audit.json` with per-position scores, `v3-scan/scan.json` with the additional openings, and `v3-scan/614-left-row3.png` with the hidden portrait. The scoreboard audit records their hashes. The full local suite passed 435 tests and 116 subtests. The roadmap, Python syntax, and review JavaScript checks passed.

Five one-minute windows supplied 33 visible scoreboards. The best frame at 614.013 seconds resolves nine rows. Left row 3 visibly carries a yellow question mark. Three frames in that opening, sampled between 613.763 and 614.013 seconds, all have the same hidden row. Across openings the reader proposes Ahri, Alistar, AurelionSol, Darius, JarvanIV, Katarina, Leona, Singed, Talon, and Xayah. A confirmed roster still needs one fully reviewable snapshot or two agreeing complete reads in the same opening. See the [scoreboard audit](reports/2026-11-11/scoreboard-portrait-alignment.json).

### Reproduce or extend the scoreboard check

Use a fresh development config with `layout_profile` set to `midbeast-1080p-inventory-v3`. The paired comparison accepts native images and source labels.

```sh
python -m annotation.league.reference_rollout compare \
  --cases CASES.json --output COMPARISON \
  --old-profile midbeast-1080p-inventory-v2 \
  --new-profile midbeast-1080p-inventory-v3
```

Use independent visible, question-mark, and respawn examples before freezing a release threshold. Collect a scoreboard opening with all ten portraits visible. For readable respawn cases, compare a tighter red mask against the current mask and measure wrong top candidates as well as score misses. The present development evidence supports the v3 candidate and a held-out comparison.

### Future body work

The roster gate, exact name registry, lazy visual path, feature preparation, and opt-in review are available locally. Resume with one fully verified roster, a nameplate layout with readable player names, admitted gameplay and rendered references, and independent body labels. Then measure the six reference arms and the operational path. Validate the new proposal refresh against the sparse SAM coverage in job 8830034 before allocating a larger run. Keep cast-bar dependency recovery in its separate pilot.

```sh
python -m annotation.league.body_campaign identify --root CAMPAIGN --intake-run INTAKE
python -m annotation.league.body_campaign fit-appearance --root CAMPAIGN --device cpu
python -m annotation.league --run-dir INTAKE review --body-campaign CAMPAIGN
python -m annotation.league.body_roadmap
```

The campaign expects a prepared source and completed tracking masks. Nameplate settings identify the source hash, native size, reviewed player display mode, offsets relative to the health bar, OCR threshold, language, and reviewer. For reviewed name evidence, a `body_names.jsonl` row cites the body sample ID, source hash, native frame ID, track ID, and exact time, with `name_kind` set to `player`, `associated` set to true, a visible name in `value`, and `review_status` set to `human-verified`. Rerun identification after corrections to rebuild the evidence revision.

## 2026-11-11 Threshold replay and respawn mask audit

The two insufficient-pixel crops at 1667.015 seconds start as 44×44 source crops. The grayscale matcher resizes each inset to 24×24, uses 379 pixels inside the portrait after excluding badge areas, then removes red timer pixels and a one-pixel border around them. It requires at least 190 usable pixels. Left row 1 retains 172 and 121 pixels across the two insets. Right row 4 retains 182 and 127. A native-resolution red mask before resizing is the next respawn experiment.

Replayed the saved v3 five-position votes while retaining the visibility, usable-pixel, and consistent-view checks. These are candidate counts on development footage.

| Gates | Original 30 crops | 33 additional scoreboards | Best additional scoreboard |
| --- | --- | --- | --- |
| Current | 20 | 212 of 330 | 9 of 10 |
| Remove score floor | 21 | 222 of 330 | 9 of 10 |
| Remove score floor and margin | 22 | 235 of 330 | 10 of 10 once |

With both gates removed, two Xayah reads switch to Fiora and the respawn crop in Jarvan IV's row returns Illaoi. The one complete candidate list at 604.003 seconds includes Quinn and Fiora in rows supported by clearer Jarvan IV and Xayah evidence. The reader profile retains its existing gates.

For a question-mark row, find a clear portrait in another scoreboard opening in the verified match, or use a reviewed source with all ten portraits visible. If the footage never exposes all ten together, evaluate a reviewed multi-opening roster. Anchor each recovered row to independently verified player or slot evidence, retain its source time, and reject conflicting bindings.

## 2026-11-11 Match roster skim

The CPU skim sampled the full 28:53 recording at 1 FPS and found 144 visible scoreboard samples across 111 openings. Every champion in the ten-member roster cleared the v3 portrait gates. The last member, Jarvan IV, passed at source time 419.002 seconds. The full scan took 353 seconds locally and found zero fully resolved single frames. See the [full recording receipt](reports/2026-11-11/scoreboard-skim.json).

The accumulator keys evidence by team and champion ID. It retains the first and strongest accepted portrait, their source times and crop hashes, and the accepted read count. A sixth distinct champion on one team disputes the pool. The source crop still records its frame position for review. The pool and timeline use team membership.

```sh
python -m annotation.league.scoreboard_skim \
  --video data/youtube/lol_midbeast/OLLZBe5-fg4.mp4 \
  --output data/league_recognition_v1/scoreboard-skim-new \
  --single-match

python -m annotation.league.scoreboard_skim --derive-membership \
  --output data/league_recognition_v1/scoreboard-skim-20261111-v1

python -m annotation.league.body_campaign identify \
  --root CAMPAIGN --scoreboard-skim data/league_recognition_v1/scoreboard-skim-20261111-v1
```

Use `--single-match` for footage reviewed to contain one underlying match. The skim saves `match-roster.json`, `skim-evidence.json`, portrait and name evidence, and `match_roster_timeline.jsonl`. The timeline records the ten members before and after confirmation, with the earlier intervals marked as backfilled. The recorded 21 name OCR reads yielded zero confirmed bindings. Body identification can use the ten-champion pool and still requires the prepared tracks and source name evidence for its name-reuse path.

### Three-pass follow-up

1. Skim identities and timeline first. Freeze the accepted roster, timestamped observations, and evidence revision. Backfill roster context for the reviewed match. Ground visible bodies in their own source observations.
2. Generate scene descriptions using the frozen timeline. Cite source frames and identity evidence for actor claims. Track description corrections and their timeline dependencies.
3. Generate QA from reviewed descriptions and source timeline facts. Require timestamped evidence for each answer. The current QA stage keys on states, events, and issues. Add scene-description and identity revision inputs when connecting the stages.

## 2026-11-12 CPU roster and optional review

The CPU scoreboard portrait reader supplies experimental roster identities from a declared single-match skim. The saved Midbeast reads filled the ten-member pool at 419.002 seconds. The isolated pilot pins those ten IDs and has 20 roster timeline rows, including backfill. It has zero confirmed player-name bindings. See the [CPU roster receipt](reports/2026-11-12/body-roster-cpu.json).

Run the optional review from the repository root. The page shows each crop in its original scoreboard row and can seek to the full source frame. Verify a continuous playback interval and one portrait for each of the five members on either side. Add rejection decisions for false portrait reads that create a conflicting complete scoreboard. Saving an incomplete review suspends the affected roster until the missing rows are verified. Enter a verified player name when its crop is available from the same row and frame. Save the review and rerun `identify` to rebuild the body timeline and any dependent name bindings. [Local review page](http://127.0.0.1:8766)

```sh
python -m annotation.league.scoreboard_skim --serve-review \
  --output data/league_recognition_v1/body-pilot-20261112/scoreboard

python -m annotation.league.body_campaign identify --root CAMPAIGN \
  --scoreboard-skim data/league_recognition_v1/body-pilot-20261112/scoreboard --device cpu
```

The same skim works without the review page. It produces experimental roster context from CPU portrait references. The body matcher uses frozen DINOv2 when a readable, unbound player name needs a visual match. Held-out release scoring requires confirmed roster evidence. The two complete-snapshot confirmation routes still run through `identify --intake-run INTAKE`.

For batch review, pass `--derive-membership --review REVIEW.json --output SKIM`. The review JSON contains `source_sha256`, `samples_sha256` from `skim-evidence.json`, `interval`, `continuity_verified`, `reviewer`, `reviewed_at` with a timezone, and `portraits`. Each portrait decision cites `portrait_id` and the verified `champion` ID. Use JSON `null` to reject a reading. Optionally include `name_id`, `name`, and `name_crop_sha256` from the associated name crop. The browser captures these fields when saving.

The development tracking audit accepts independently labeled native `body_box` coordinates and `time` in source seconds. Use distinct source label IDs and a stable `segment_label_id` across frames. Record out-of-gallery, misleading-nameplate, and unreadable examples with `negative_kind` where applicable. After preparing and tracking the labeled campaign, run the audit and review misses before fitting or comparing arms.

The CPU proposal probe found a friendly seed box covering at least half the labeled body box at all 28 visible Darius timestamps. Those boxes are designed as broad SAM prompts. The probe had zero stable boxes with IoU of at least 0.5 against the tight source boxes and used the existing SAM receipt for mask coverage. The audit now reports seed coverage and mask-box IoU separately. [Proposal probe](reports/2026-11-12/body-tracking-proposal-probe.json)

```sh
python -m annotation.league.body_campaign audit-tracking --root CAMPAIGN
python -m annotation.league.body_campaign fit-appearance --root CAMPAIGN --device cpu
python -m annotation.league.body_campaign run --root CAMPAIGN --policy CAMPAIGN/appearance-policy.json --split validation
python -m annotation.league.body_campaign score --root CAMPAIGN --policy CAMPAIGN/appearance-policy.json --split validation
```

Point GPU stages in each isolated campaign at the same ledger with `--budget-root`. The local ledger is in `data/league_recognition_v1/body-campaign`.

The checked local domisumReplay frame at 75 seconds displays champion names such as Aatrox and Jax. It remains development footage for detector work. The player-name reuse smoke test needs eligible footage with visible player handles.

**Future broadcast cohort.** The official LoL Esports Worlds 2025 finals broadcast has readable player handles above bodies and a distinct scoreboard layout. This could support a later professional-broadcast cohort with its own layout and continuity checks. The downloaded excerpt is outside the active pilot. [Official LoL Esports broadcast around 2:38:33](https://www.youtube.com/watch?v=ADaZLxI3uWg&t=9513s)

## Match skins in a single frame

Use a prepared campaign with completed body masks and a confirmed scoreboard for the frame. The command takes an exact source timestamp from `prepared.json`. It uses the main champion to identify the friendly scoreboard side and excludes the main body using the HUD binding. If that binding is unavailable, supply `--main-track TRACK_ID` or `--main-offscreen` for the frame. Use `--friendly-side left|right` when the main champion appears on both teams.

```sh
python -m annotation.league.body_campaign match-frame --root CAMPAIGN \
  --intake-run INTAKE --frame-time SOURCE_SECONDS --main-champion CHAMPION_ID \
  --render-gallery AUDITED_GALLERY.json
```

`--scoreboard-skim SKIM` also accepts a reviewed single-match roster. Omit `--render-gallery` to use the campaign's prepared renders. The separate gallery takes the same JSON list produced by `body_render audit`, so render coverage can grow without preparing the tracked clips again.

SigLIP is the default skin encoder. Use `--siglip-lock` to point to a staged lock outside `data/league_models/siglip-base-patch16-224/model-lock.json`, or `--skin-encoder dinov2` to use the prepared DINOv2 pin. A frame report records the candidates, missing renders, and number of comparisons.

Pass `--skin-assignments RUN/skin-locks-siglip.json` to reuse the locks from a completed appearance pass. The match, team pools, render gallery, and encoder pin must agree. An appearance lock restricts that champion's skin candidates. For the appearance workflow, see [Run the scoreboard-backed appearance pass](league-sam-multichampion-pilot.md#run-the-scoreboard-backed-appearance-pass).

## Longleaf visibility pilot

Run one confirmed match across three nonoverlapping prepared windows of up to 12 seconds each. Pick windows that cover an ally and enemy entering or leaving view, a re-entry under a new track ID, and an occlusion or tracking loss. At 4 FPS this is up to 144 sampled frames. Use completed anonymous body tracks, the pinned DINOv2 checkout and weights, and audited render coverage for all nine non-main roster members. Include all available skins and record catalog gaps. The existing local Midbeast pilot campaign has an empty clip and render gallery list.

The job requests four CPU cores, 16 GB of RAM, and 45 minutes on Longleaf. The earlier CPU probe processed 28 source masks and their comparison variants in 27 seconds locally. The wider allocation covers the first render embedding pass and tracked frame reads. Record wall time, peak RSS from Slurm, render and query encodes, render comparisons, unresolved main-body frames, and reuse on re-entry before scaling. [CPU probe](reports/2026-11-12/body-source-geometry-probe.json) · [UNC Slurm examples](https://help.rc.unc.edu/longleaf-slurm-examples)

Stage the source snapshot with `SOURCE_SHA256SUMS` and use the same `env-body` used by the body campaign. Set the paths below to the staged inputs on Longleaf. Run the input check from that environment before submitting.

```bash
args=(--root "$CAMPAIGN" --cache "$ASSETS" --scoreboard-skim "$REVIEWED_SKIM"
      --render-gallery "$GALLERY" --output "$PILOT_OUTPUT" --main-champion "$MAIN_CHAMPION"
      --clip-id "$ENTRY_CLIP" --clip-id "$REENTRY_CLIP" --clip-id "$OCCLUSION_CLIP")
python -m annotation.league.body_presence scan "${args[@]}" --check-inputs
sbatch annotation/league/slurm_body_presence_pilot.sl "${args[@]}"
```

`--intake-run` can supply the confirmed roster. If a main-body HUD binding is unavailable, `--main-tracks` accepts JSON with `source_sha256` and a `frames` object keyed by prepared frame ID. Each decision has `frame_sha256` and `track_id`. Set `track_id` to JSON `null` after verifying the main body is outside the frame. Without a decision or a usable HUD binding, the job still processes enemy tracks and records the friendly observations as unknown.

The pilot writes `frame_matches.jsonl`, `presence_observations.jsonl`, `presence_change_candidates.jsonl`, and a `report.json` with their hashes. It starts with empty game assignments and reuses render embeddings from the shared asset cache. The change candidates point to frames where a sighting begins or ends and give reviewers a place to start. Review the native frames in `frame_matches.jsonl` against the identities and sample times, then measure correct team/champion matches, re-entry skin reuse, and missed or false visibility changes.

### Prepare visibility events for timeline fusion

Review the frames on both sides of each proposed change. Save a JSON review with `source_sha256`, `reviewer`, and `observations`. Each decision cites `observation_id` and `frame_sha256` and sets `status` to `visible`, `not_visible`, or `unknown`. Use `unknown` when occlusion, tracking loss, or unreadable footage prevents a reliable decision.

```sh
python -m annotation.league.body_presence transitions --pilot "$PILOT_OUTPUT" --review "$VISIBILITY_REVIEW"
```

The resulting `champion_appeared` and `champion_disappeared` events carry both native frame IDs, the team, champion, roster, and a time interval bounded by adjacent reviewed samples. Unknown samples, playback gaps, and clip boundaries break continuity. The events describe visibility in the camera view.

For main-timeline integration, join the two frame witnesses to the intake run by source hash and native timestamps and require the same verified playback segment. Feed qualified rows through `timeline.fuse` alongside other event candidates and include the reviewed presence artifact in both fuse freshness keys. Keep team in the event identity so mirror matchups remain distinct. The pilot events carry the experimental flag that the current fusion code filters out.
