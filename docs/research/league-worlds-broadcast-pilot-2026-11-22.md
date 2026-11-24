# League Worlds broadcast annotation pilot

November 22, 2026 · [GEN vs BLG, Worlds 2026 Finals](https://www.youtube.com/watch?v=fFlOTEiQ9x0)

**Use broadcast overlays to find live gameplay and replay blocks. Rank commentary across the entire VOD, including draft and analyst desk segments.** The existing League caption importer and broadcast prototype handled this source. The pilot found useful explanations of objective setup, draft interactions, and individual decisions. The regular replay layout hid the game clock, and the existing numeric reader could not reliably read the clock at the tested resolutions.

## Game 1 broadcast Pass 1

The opt-in `pass1 broadcast` command screens the footage before it runs native
HUD crops. The Game 1 run uses eight consecutive **1920×1080, 60 fps** sections
covering **4:13:00–5:06:05**. Three comparisons with the earlier excerpts
confirmed the declared source offsets in draft, live play, and the first postgame
breakdown. Correlation at the declared offsets ranged from 0.9885 to 0.9985
and exceeded both half-second shifts in every comparison.

The full transcript is searchable in `/pass1`. The page seeks the local excerpt
at the corresponding offset and provides a source link for audio or times outside
the excerpts. The broadcast bundle retains the screening votes, reviewed
segments, native crop evidence, candidate and reviewed commentary links, and
priority windows for downstream frame analysis. The full visual scan and the
OCR sample run on the pilot interval. [VOD](https://www.youtube.com/watch?v=fFlOTEiQ9x0)

### Replay review

The reviewed reference has seven replay blocks, including two sponsor layouts.
Five begin with the regular replay wordmark. The overlay-only baseline found
those five and the postgame breakdown, reaching 98.6% time-weighted precision
and 83.6% recall. The missing **36.25 seconds** were the Mastercard “Priceless”
block. The revised profile adds that cue. Arena, player and crowd shots keep
their replay context while carrying a different content label. Reviewed
transitions are excluded from the default live handoff. [Priceless replay and reaction](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=17229s)

The revised local screen found all seven blocks, reaching **98.8% precision**
and **over 99.9% recall** by duration before corrections. The largest absolute
boundary error was **0.75 seconds**. The reference uses shot proposals,
manual checks around transitions, and gameplay spot checks. The sponsor cue
was calibrated on this VOD, so the result measures performance on this
broadcast package.

The reviewed live handoff contains **2,135 seconds** of live gameplay with
zero overlap with the reference replay intervals. It prioritizes **216
seconds** of live footage around linked commentary and offers
**53 seconds** of linked replay evidence. Draft and desk commentary remain
searchable even when they have no live visual handoff.

### Native spectator OCR and commentary

Both OCR methods use the same native crops. The adapted method isolates bright
luminance glyphs, scales them to a 48-pixel line, and uses Tesseract's single-word
mode. The second objective tile uses a different crop if the first tile
collapses to an icon. The benchmark labels clock, blue and red kills, blue and
red gold, and the two objective timer slots independently. Calibration uses 16
frames, and the held-out set uses 24 other frames, including three replay
frames whose live scoreboard is hidden. The confidence threshold is 0.85 for
both methods. An OCR field needs at least 95% precision and 70% readable
coverage to feed downstream analysis. The live clock also needs a compatible
consecutive reading from the same reviewed span.

In a local check of the same labeled frames with Tesseract 5.5.3, the
adapted reader read 21/21 live clocks exactly, up from 2/21 for the baseline.
It passed the pilot gates for clock, both kill totals, and both timer slots.
Blue and red gold had zero false accepts with the adapted reader, but
readable coverage was 9/21 and 12/21 respectively. The sponsored breakdown's
clock also visibly rewinds from 29:04 to 28:52 within the same reel. Its clock
readings stay in replay context. [Breakdown clock rewind](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=18278s)

The reference contains 17 commentary links covering 13 events. For example, the
draft discussion links Nami's counter-engage to the completed draft. The
Chemtech secure has one event ID for the live and replay views. The analyst
discussion about Kiin marking Knight during the first Elder fight links to that
event's live footage, regular replay, and sponsored breakdown. These links
attribute interpretations to the broadcast. [Draft](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=15768s) ·
[Chemtech replay](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=16565s) ·
[First Elder breakdown](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=18247s)

### Longleaf results

Longleaf CPU job **10248305** completed in **45:26** and passed bundle
verification. It screened and reviewed all **3,185 seconds** of the pilot,
retained **14,094** searchable cues from the full VOD, and produced **17**
reviewed links across **13** events. Before review overrides, the Longleaf
screen detected all seven replay blocks with **98.6% precision** and **over
99.9% recall** by duration. The largest absolute boundary error was **0.75
seconds**. The reviewed live handoff overlaps none of the replay intervals.
[Job receipt](../../data/vod_harness/fFlOTEiQ9x0/game1-broadcast/longleaf-receipt.txt) ·
[Run report](../../data/vod_harness/fFlOTEiQ9x0/game1-broadcast/pass1_report.json)

The table shows exact accepted reads out of all readable held-out samples
with Tesseract 4.1.1. The baseline falsely accepted four reads across the seven
fields. The adapted reader had zero false accepts. Both gold fields remain
available for review because their readable coverage was below 70%.

| Field | Original reader | Adapted reader | Downstream status |
| --- | ---: | ---: | --- |
| Match clock | 3/21 | 21/21 | Enabled with continuity check |
| Blue kills | 13/21 | 21/21 | Enabled |
| Red kills | 7/21 | 20/21 | Enabled |
| Blue gold | 0/21 | 11/21 | Review only |
| Red gold | 0/21 | 12/21 | Review only |
| Objective timer 1 | 1/14 | 12/14 | Enabled |
| Objective timer 2 | 2/18 | 13/18 | Enabled |

### Reproduce the broadcast bundle

The profile sets the pilot interval, game phases, clock and scoreboard geometry,
overlay references, and League vocabulary. Other broadcasts can supply their own
profile and cue vocabulary. The source media, reference PNGs, and manual review
file are under the workspace's `data/vod_harness/fFlOTEiQ9x0` directory.
The collected Longleaf bundle is in `game1-broadcast` under that directory.

```bash
vod=data/vod_harness/fFlOTEiQ9x0
run="$vod/game1-broadcast-local"
clips=()
while IFS=$'\t' read -r filename start end; do
    clips+=(--clip "$vod/native-sections/$filename" "$start" "$end")
done < "$vod/clips.tsv"
python -m annotation.league.pass1 broadcast --run-dir "$run" \
  --profile annotation/league/profiles/worlds-2026-finals-game1.json \
  --assets "$vod/overlay-profile" --captions "$vod/fFlOTEiQ9x0.en-orig.json3" \
  --review "$vod/worlds-game1-manual-review.json" --workers 4 "${clips[@]}"
python -m annotation.league.pass1 verify --run-dir "$run"
python -m annotation.league --run-dir "$run" review
```

Open the local server's `/pass1` route. To view the collected Longleaf bundle,
use `"$vod/game1-broadcast"` as the review command's run directory.
The batch file uses the same broadcast command and reads the source extent
from the same TSV. [Longleaf batch file](../../annotation/league/slurm_worlds_game1_broadcast.sl)

## Initial excerpt scope

- Imported the full **9:11:30** English auto-caption track into **14,094** timestamped cues using the existing Pass 1 JSON3 adapter. The caption track starts at 0.96 seconds and ends around 9:11:17. Cue ends are estimates from caption boundaries.
- Screened **14:15** of video across seven existing, exactly trimmed 360p and 480p excerpts. They cover preshow, stage, draft, live play, replay, and postgame analysis. The video sample is about **2.6%** of the VOD and was selected to cover different layouts.
- Used the existing Game 1 calibration excerpt and its overlay references. Manually labeled one frame every ten seconds, starting at each excerpt’s first frame. This yielded 32 calibration frames and 55 frames from the six other excerpts. One test excerpt continues the same sponsored breakdown that starts in calibration. All excerpts use the same broadcast package.

## Gameplay and replay detection

The prototype runs PySceneDetect at 12 fps to propose cuts and OpenCV normalized grayscale template matching at 2 fps to identify overlays. Stable regions such as the team strips, patch label, and replay wordmark carry the visual signal. The segmentation step adds sustained overlay changes between cuts and votes on each interval. Across the excerpts, ContentDetector proposed 69 cuts and AdaptiveDetector proposed 36. With ContentDetector and the overlay changes, the run produced 81 intervals. These are proposals for review.

The 55 reviewed frames included 20 live gameplay frames and 35 other frames. The screenshot thresholds below maximize balanced accuracy on the 32 calibration frames, using the highest threshold to break ties.

| Method for finding live gameplay | Live frames kept, of 20 | Other frames kept, of 35 |
| --- | ---: | ---: |
| Full-frame grayscale similarity to one live screenshot, threshold 0.1796 | 20 | 2 |
| Same comparison restricted to the central playfield, threshold 0.0042 | 17 | 22 |
| Fixed overlay crops with shot-level voting | 20 | 0 |

The two full-frame false positives were regular replays. Cropping away the HUD sharply reduced separation on this sample. The overlay profile assigned **374.3 seconds** to live gameplay and **161.6 seconds** to replay across the seven excerpts. It assigned **316.9 seconds** to missing gameplay HUD, **1.5 seconds** to the explicit desk template, and **0.7 seconds** to unknown. Missing HUD can route a span out of automatic live gameplay processing while retaining it for review and transcript analysis.

The replay template covered all **11** reviewed replay-gameplay frames, including five from the sponsored breakdown. Two additional frames inherited the replay label. One was a team reaction shown with the replay badge still on screen, and one was the outgoing transition assigned to the same shot. Treat these as two fields in the next pass: **broadcast context** and **visible content**. A reaction under a replay badge belongs to a replay block and provides no playfield evidence. The shot detector already proposed a cut at 5:37:07 between gameplay and the reaction. Matching four corners of the replay item strip as a second content signal did not reliably separate the two. [Game 2 reaction](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=20230s)

### What the game clock can do

In four regular replay transitions, the manually read live clock before and after the replay advanced by the same number of seconds as the VOD. The replay footage itself used a layout without the main clock. The sponsored Game 1 breakdown instead exposed an earlier clock in a smaller, different HUD.

| Sample | VOD seconds before → after | Visible match clock before → after |
| --- | --- | --- |
| Game 1 regular replay | 17,990 → 18,025 | 36:05 → 36:40 |
| Game 2 first sampled replay | 20,211 → 20,236 | 06:50 → 07:15 |
| Game 2 second sampled replay | 20,264 → 20,291 | 07:43 → 08:10 |
| Game 4 sampled replay | 30,170 → 30,192 | 34:09 → 34:31 |

At VOD second 18,248, the sponsored breakdown showed **28:36** after the broadcast had already shown Game 1 past 38 minutes. A confirmed older match clock can support a replay decision when present. The current Midbeast `PlaybackTracker` can supply rewind and discontinuity cues once readings are reliable and scoped by game and overlay. Reset its clock history at each new game. Handle pauses, obscured clocks, and layout changes as separate evidence. Keep source-video time as the timeline coordinate. [Sponsored breakdown](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=18248s)

**OCR check.** Sampled 62 source frames around live play, replay, desk, and postgame graphics. Ran the existing Tesseract `NumericReader` at its 0.85 confidence gate on normalized crops `(.483, .066, .517, .097)` for the live layout and `(.488, .063, .519, .087)` for the breakdown. Of 124 crop attempts, the only accepted read was **87:18** for a live clock visibly around **07:19**. Higher-scale grayscale and thresholded crops with Tesseract modes 7 and 8 produced zero accepted valid clock reads at the same gate. Test native-resolution crops with layout-specific recognition and agreement across consecutive readings before enabling automatic clock evidence.

**Replay deduplication.** The three-frame playfield hash matcher proposed zero links across seven per-excerpt replay runs. One manually confirmed pair explains the failure mode: Game 2 shows Ruler killing Viper near the bottom turret at 5:37:17. The replay shows the same kill announcement and turret sequence around 5:38:07.5, with a tighter crop. One match event should retain both observations and the replay commentary. Start by matching event anchors within a game, such as killfeed participants and nearby score changes. Keep pixel matches as supporting evidence and leave ambiguous links unresolved. [Live play](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=20237s) · [Replay](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=20287s)

## Commentary salience

The existing ranker scanned 30-second windows at a 15-second stride. It retained **221** windows with both League or execution vocabulary and explanation or alternative cues. It sorted by score, skipped overlapping windows, and limited each half-hour to three shortlist entries.

On a manual review of the 30 shortlisted windows, **25** contained at least one explanation, tradeoff, or performance judgment about a finals game, draft, or play. Five were mainly play-by-play, general player discussion, or off-topic. A seeded sample of 12 non-overlapping 30-second windows outside all ranked windows found **four** additional analytical leads. For example, one discussed clearing jungle camps and pushing two waves to prepare vision for Elder without the phrases required by the ranker. These small samples measure review yield. Only six shortlisted windows overlapped the video excerpts, and the salience judgment used automatic caption text. Speaker identities and most claims still require review. [Unranked Elder setup](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=17430s)

| VOD window | What the commentary adds | Candidate annotation |
| --- | --- | --- |
| [4:59:30–5:00:00](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=17970s) | The caster explains how Knight’s blast cone changes his timing and path into the Elder fight. The window moves from live aftermath into replay. | Fight setup and alternative route. Extend the visual excerpt into the replay to check the path. |
| [5:22:30–5:23:00](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=19350s) | The cast discusses how Naafiri could deliver Orianna’s Shockwave and how that changes the value of a Nocturne ban. | Draft interaction. Check champion identities against the draft graphics. |
| [5:04:45–5:05:15](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=18285s) | An analyst argues that Chovy grouping with the team mattered more than matching the opponent’s farm. The window straddles the breakdown and analyst desk. | Attributed performance assessment. Check the mid-lane farm gap and group movements in Game 1. |
| [7:08:45–7:09:15](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=25725s) | A caster questions Canyon’s re-engage while the mid laner is bot side and BLG’s positions are uncertain. The replay begins partway through the window. | Decision under incomplete information. Check map position and separate observer vision from what GEN could see. |
| [8:23:00–8:23:30](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=30180s) | The caster links an all-in to the window after Viper spends a defensive ability. Playback returns from replay to live within the window. | Timing and execution. Confirm the ability, actor, and ordering on source frames. |

Expand each shortlisted window to sentence boundaries and link each claim to the game, event, and supporting frames where possible. Preserve cue IDs and raw captions. Team, player, and champion names are prone to transcription errors. Use the verified roster and draft as a review aid and retain attribution. A single play can support a specific performance observation. A general claim about a player needs comparable observations across matches.

## Adapting the four League passes

| Existing pass | Minimal broadcast change |
| --- | --- |
| **1. HUD, commentary, events** | Insert a cheap screening stage before the current readers. Define game IDs and time ranges, layout changes, replay and pause spans, and uncertain coverage. Reprofile the spectator scoreboard, clock, minimap, and killfeed crops. Read roster and item rows when visible. Enable selected-player, cast, and cooldown fields only where the broadcast exposes them. Keep one source-time transcript for the entire broadcast and join reviewed claims to game or draft evidence. |
| **2. SAM masking** | Use the existing body seed and tracking workflow on selected live windows and replay angles that add evidence. Restart at observer cuts or layout switches. Keep identity across cuts only with explicit roster or HUD support. |
| **3. Recognition and visual captions** | Reuse the League icon and render galleries with per-game draft rosters and team-side assignments. Retain source frame references and event IDs. Describe a replay angle as another observation of the event and use the commentary to propose checks for the visual caption. |
| **4. Descriptions and QA** | Assemble descriptions by game and event. Attribute draft, strategy, and performance interpretations to the broadcast and retain supporting frame references. Group live, replay, and later analysis of the same event in the same dataset split. |

The reusable part across esports is the source-time ledger, shot proposals, overlay matching interface, transcript retrieval, claim review, and event deduplication. Game and broadcast profiles supply clock behavior, HUD regions, roster, vocabulary, and event anchors. Calibrate on one broadcast and evaluate on a different broadcast package. For this source, the next end-to-end test is one game including its draft and postgame review, with every live/replay boundary labeled and a small set of claims grounded to HUD or action evidence.

## Reproduction

The screening run used FFmpeg 9.0.1, PySceneDetect 0.7.3, and OpenCV 4.12.0. The visual libraries are pinned in [the existing clip-pilot requirements](../../annotation/league/requirements-darius-clips.txt).

```sh
vod=data/vod_harness/fFlOTEiQ9x0
python -m annotation.league.broadcast_pilot \
  --video-id fFlOTEiQ9x0 --duration 33090.521 \
  --captions "$vod/fFlOTEiQ9x0.en-orig.json3" \
  --profile "$vod/overlay-profile/profile.json" \
  --clip "$vod/exact-1800.mp4" 1800 1840 \
  --clip "$vod/exact-15265.mp4" 15265 15385 \
  --clip "$vod/exact-17935.mp4" 17935 18255 \
  --clip "$vod/exact-18255.mp4" 18255 18365 \
  --clip "$vod/exact-20210.mp4" 20210 20335 \
  --clip "$vod/exact-25675.mp4" 25675 25750 \
  --clip "$vod/exact-30165.mp4" 30165 30230 \
  --output "$vod/pilot-audit"
```

The local output includes the cue timeline, ranked windows, shortlist, visual shots, and live gameplay candidates. The manual reviews are recorded in [visual_audit.json](../../data/vod_harness/fFlOTEiQ9x0/pilot-audit/visual_audit.json), [clock_audit.json](../../data/vod_harness/fFlOTEiQ9x0/pilot-audit/clock_audit.json), and [commentary_audit.json](../../data/vod_harness/fFlOTEiQ9x0/pilot-audit/commentary_audit.json). Media, calibration images, and run outputs live under the repository’s ignored `data/` directory.

## Worlds draft, roster, and gameplay events

November 23, 2026

`pass1 broadcast-match` reads a verified Pass 1 bundle and writes a separate,
versioned match revision. The draft and roster have source-reviewed witnesses.
The reader samples draft graphics at 2 Hz and records visual change proposals
for review. It samples killfeed at 4 Hz and the full player scoreboard when it
opens and at five-second intervals while visible. `/pass1` exposes the draft,
rosters, player snapshots, item history, events, and native crops. For an event
with both angles, the replay sighting points to the live event ID and stays out
of the live count.

### Draft and roster

Game 1 has ten ordered bans, ten ordered picks, and a separate Kalista hover at
4:17:30. The draft timer and graphic changes establish order. The saved
timestamps identify witness frames. Champion assignments come from the final
draft and a live scoreboard witness. Game 2's completed draft gives ten current
bans, ten locked picks, and ten prior-game exclusions. Its excerpts begin
during the draft, so individual Game 2 lock order remains unknown.

| Order | Side | Decision | Champion | Source time |
| ---: | --- | --- | --- | --- |
| 1 | GEN | Ban | Nocturne | 4:15:05 |
| 2 | BLG | Ban | Qiyana | 4:15:12 |
| 3 | GEN | Ban | Orianna | 4:15:20 |
| 4 | BLG | Ban | Cassiopeia | 4:15:40 |
| 5 | GEN | Ban | Vi | 4:16:00 |
| 6 | BLG | Ban | Lulu | 4:16:40 |
| 7 | GEN | Pick | Lee Sin | 4:17:10 |
| 8 | BLG | Pick | Renata Glasc | 4:17:46 |
| 9 | BLG | Pick | Kalista | 4:18:10 |
| 10 | GEN | Pick | Syndra | 4:18:40 |
| 11 | GEN | Pick | Olaf | 4:19:00 |
| 12 | BLG | Pick | Locke | 4:19:10 |
| 13 | BLG | Ban | Ezreal | 4:19:50 |
| 14 | GEN | Ban | Jax | 4:20:10 |
| 15 | BLG | Ban | Seraphine | 4:20:40 |
| 16 | GEN | Ban | Camille | 4:21:10 |
| 17 | BLG | Pick | K'Sante | 4:21:40 |
| 18 | GEN | Pick | Corki | 4:22:10 |
| 19 | GEN | Pick | Nami | 4:22:39 |
| 20 | BLG | Pick | Jarvan IV | 4:22:43 |

| Side | Player | Game 1 | Game 2 |
| --- | --- | --- | --- |
| GEN blue | Kiin | Olaf | Vayne |
| GEN blue | Canyon | Lee Sin | Cho'Gath |
| GEN blue | Chovy | Syndra | Orianna |
| GEN blue | Ruler | Corki | Ezreal |
| GEN blue | Duro | Nami | Karma |
| BLG red | Bin | K'Sante | Gnar |
| BLG red | Xun | Jarvan IV | Naafiri |
| BLG red | Knight | Locke | Cassiopeia |
| BLG red | Viper | Kalista | Ziggs |
| BLG red | ON | Renata Glasc | Camille |

Knight's Game 1 portrait and final draft establish **Locke**, correcting the
LeBlanc attribution in the earlier commentary bundle. Game 2's sample resolves
all ten player–champion identities. Two later player rows were unreadable
during one transient scoreboard frame and stayed unresolved. Identity is
bound within a game by exact normalized name or a unique same-side champion
match. Conflicting evidence sends the row to review. Riot's visible patch
26.20 corresponds to the pinned Data Dragon 16.20.1 icons.
[Game 1 final draft](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=15780s) ·
[Game 2 final draft](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=19780s) ·
[Riot patch notes](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-20-notes/) ·
[Data Dragon versions](https://ddragon.leagueoflegends.com/api/versions.json)

### Game 1 event coverage

The complete local match run took 10:49 and processed 1,010 draft, 8,542
live, and 785 replay gameplay frames. It produced 3,440 player-row snapshots from 344 scoreboard
observations. Of those rows, 46 had unresolved identities. The 162 proposed
item slot transitions remain available for review with both observation
crops and the time bracket.

| Automatic event | Matched | Extra | Missed | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kill with both roles resolved | 23 | 0 | 0 | 100% | 100% |
| Neutral capture with side and objective resolved | 13 | 0 | 0 | 100% | 100% |

All 36 matched onsets agreed with the reviewed full-row appearance on the
250 ms sample grid. Some rows start fading in at the preceding sample.
The onset brackets reflect that cadence. The automatic
kill total and reviewed total both reconcile to **GEN 19, BLG 4**. Accepted
team-score changes independently corroborated 14 kill records, including
multi-kill increments. The objective references and times come from this
broadcast, so these rates measure within-VOD performance.

Native footage confirms Xun defeated Rift Herald at 4:39:38.5. The capture
banner and killfeed separate it from Viper's Scuttler notice at 4:40:05.5 and
the later Herald summon. The final objective timeline contains
four regular dragons, two Elders, three Barons, one Herald, and three
individual Voidgrubs. The reader retained 16 unclassified live rows as
review evidence.
[Rift Herald](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=16779s) ·
[Scuttler](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=16806s) ·
[Simultaneous kills at the finish](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=18156s) ·
[Final totals](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=18325s)

Ten of 13 replay sightings link to a reviewed live event, including Knight's
first blood, BLG's Chemtech capture, and Xun's first Elder steal in the postgame
breakdown. The group Voidgrub banner and
two postgame kills involving repeated player pairs stay unresolved at the
individual occurrence level.
[First blood replay](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=16382s) ·
[Chemtech replay](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=16569s) ·
[Elder breakdown](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=18267s)

### Player scoreboard benchmark

Native 1080p scoreboard geometry and OCR settings were calibrated on the
4:27:04 frame. Held-out numeric labels cover four other frames at 4:24:08,
4:30:55, 4:37:00, and 4:44:01. The item labels cover three of those frames
and include occupied and empty slots. Each accepted reading retains the
row crop, field coordinates and confidence. Item histories bracket slot
changes between observations and retain the last seen value through
unreadable samples.

With Tesseract 5.5.3, CS passed both gates with **36 exact reads on 40
readable fields** and no false accepts. Level reached 69.4% coverage.
Item slots 5 and 6 had high overall coverage because the reader recognized
empty cells. Both stay in review with zero exact reads of occupied slots.
The gate covers occupied item identification alongside overall field
precision and coverage.

| Field | Exact / accepted | Accepted / readable | False accepts | Exact occupied items / readable occupied | Use |
| --- | ---: | ---: | ---: | ---: | --- |
| Level | 25/25 | 25/36 | 0 | — | Review |
| CS | 36/36 | 36/40 | 0 | — | Downstream |
| K/D/A | 16/16 | 16/40 | 0 | — | Review |
| Item slot 1 | 4/4 | 4/22 | 0 | 4/22 | Review |
| Item slot 2 | 3/3 | 3/23 | 0 | 3/23 | Review |
| Item slot 3 | 12/13 | 13/20 | 1 | 1/9 | Review |
| Item slot 4 | 17/18 | 18/26 | 1 | 1/10 | Review |
| Item slot 5 | 20/21 | 21/27 | 1 | 0/7 | Review |
| Item slot 6 | 27/27 | 27/30 | 0 | 0/3 | Review |
| Trinket | 19/25 | 25/30 | 6 | 19/30 | Review |

[Calibration scoreboard](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=16024s) ·
[Held-out midgame scoreboard](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=17041s)

### Game 2 transfer and replay identity

The two excerpts produced 250 player-row snapshots. In the reviewed gameplay
spots the unchanged feed geometry matched **2/2 kills and 3/3 Voidgrub
captures**, with zero extra live occurrences and onset agreement within the
quarter-second sample grid. Canyon's Cho'Gath and Xun's Naafiri captured one
and two Voidgrubs respectively.
[Canyon capture](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=20306s) ·
[Xun capture](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=20320s)

The 5:38:09 replay text identified Ruler's kill on Viper and linked to the
5:37:17.75 live occurrence. The 5:37:04 replay text also identified Xun's
kill on Chovy. That sighting stays unlinked because the sampled excerpts
lack a confirmed live occurrence to attach it to.
[Ruler live](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=20238s) ·
[Ruler replay](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=20289s) ·
[Xun replay](https://www.youtube.com/watch?v=fFlOTEiQ9x0&t=20224s)

The review API served the native replay crop and the second staged clip.
The replay's source time maps to 79 seconds within that clip. Full-VOD
search returned transcript hits outside the staged excerpts, and the same
API served the existing Midbeast timeline.

### Longleaf match run

Job **10498854** completed in **35:15** with 4 CPUs and an 8 GiB request.
The batch file requested `general` and Slurm recorded the allocation on
`spill`. The maximum RSS of the reader step was 1,828,976 KiB. A missing
repository import stopped the initial submission, **10498808**, before
reading the footage. The completed retry passed remote verification.
The retrieved checksums matched 3,571 native crops and the match tables.

Rescoring that earlier reader with the corrected review found **23/23
kills and 12/13 neutral captures**, with zero extra live occurrences.
It left the real Herald capture unclassified. The local match revision
contains the updated Herald reference, replay text and CS readers, and the
expanded held-out item benchmark.
[Job receipt](../../data/vod_harness/fFlOTEiQ9x0/game1-broadcast/longleaf-match-10498854/receipt.txt) ·
[Retrieved match report](../../data/vod_harness/fFlOTEiQ9x0/game1-broadcast/broadcast_match/a5355a257b739b266ca0/report.json) ·
[Corrected-label audit](../../data/vod_harness/fFlOTEiQ9x0/game1-broadcast/longleaf-match-10498854/corrected-label-audit.json)

The final local Game 1 and Game 2 revisions both passed bundle verification.
[Game 1 match report](../../data/vod_harness/fFlOTEiQ9x0/game1-broadcast/broadcast_match/b2e22039161a85511ffe/report.json) ·
[Game 2 spot-check report](../../data/vod_harness/fFlOTEiQ9x0/game2-match-spots/broadcast_match/17d28bbd8b7935e65cc9/report.json)

### Reproduce the match layers

Stage the two Game 2 excerpts at their source offsets and use the existing
broadcast command with the provided spot-check profile and reviewed cut
boundaries. Game 2 covers 275 staged seconds across 5:28:40–5:31:10 and
5:36:50–5:38:55. The time between those clips is outside the visual sample.

```bash
vod=data/vod_harness/fFlOTEiQ9x0
profile=annotation/league/profiles/worlds-2026-spectator-match.json
labels=annotation/league/benchmarks/worlds-2026-finals
python -m annotation.league cache-inventory-references --reference-version 16.20.1
python -m annotation.league.pass1 broadcast --screen-only \
  --run-dir "$vod/game2-match-spots" --profile "$labels/game2-broadcast-profile.json" \
  --assets "$vod/overlay-profile" --captions "$vod/fFlOTEiQ9x0.en-orig.json3" \
  --review "$labels/game2-broadcast-review.json" \
  --clip "$vod/match-inputs/native-19720-19870.mp4" 19720 19870 \
  --clip "$vod/match-inputs/native-20210-20335.mp4" 20210 20335
python -m annotation.league.pass1 broadcast-match \
  --run-dir "$vod/game1-broadcast" --match-profile "$profile" \
  --calibration "$labels/calibration" --asset-cache data/league_assets \
  --review "$labels/game1-match-review.json" --workers 4
python -m annotation.league.pass1 broadcast-match \
  --run-dir "$vod/game2-match-spots" --match-profile "$profile" \
  --calibration "$labels/calibration" --asset-cache data/league_assets \
  --review "$labels/game2-match-review.json" --workers 4
python -m annotation.league.pass1 verify --run-dir "$vod/game1-broadcast"
python -m annotation.league.pass1 verify --run-dir "$vod/game2-match-spots"
python -m annotation.league --run-dir "$vod/game1-broadcast" review
```

Open `/pass1` and choose the match table to inspect. The review labels and
native overlay templates are under the benchmark directory.
[Match benchmark inputs](../../annotation/league/benchmarks/worlds-2026-finals) ·
[Longleaf match batch file](../../annotation/league/slurm_worlds_game1_match.sl)
