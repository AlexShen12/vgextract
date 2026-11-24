# League annotation pipeline

| Pass | Inputs | Outputs | Current implementation |
| --- | --- | --- | --- |
| 1. HUD, commentary and event timeline | Source video, YouTube auto-captions or source audio, calibrated layout, icon catalogs | Source-aligned commentary with available word or segment times, HUD state, ability cast cues, roster, per-champion item history, killfeed events, SAM windows seeded at other champions' first detections | Experimental `annotation.league.pass1` assembles ASR commentary, HUD, cast, inventory backfill/history, reopen groups, roster and killfeed evidence in one review bundle. YouTube auto-captions are the preferred commentary source and are imported into the shared timeline. Full-recording completion and quality acceptance are reported separately. SAM window proposals remain planned. |
| 2. SAM masking | Source frames in candidate windows, per-champion seed frames, and Pass 1 playback and identity evidence | Visible-body masks, track IDs, and references to available HUD identity evidence | SAM tracking is experimental. |
| 3. Champion recognition and visual captions | Candidate window frames, Pass 1 timeline and roster, Pass 2 masks and tracks, 3D render gallery | Champion IDs for accepted matches, final clip boundaries, and timestamped captions with source evidence | SigLIP render matching and `caption-pilot` are separate experiments. Their integration is planned. |
| 4. Descriptions | Ordered Pass 3 captions and their evidence references | Deterministically rendered gameplay descriptions linked to the captions that support them | A separate description pass is planned. |

Use source-video seconds and actual frame presentation timestamps throughout.
Keep game clock as match context. Carry the source hash, playback segment ID,
and source frame or crop references through each handoff. Keep cut, rewind,
replay, and unknown-coverage markers available to downstream passes.

For tournament VODs, the Worlds finals pilot tests overlay-based screening and
commentary retrieval before Pass 1. It also maps the changes needed for spectator
HUDs and replay evidence across all four passes.
[Worlds broadcast pilot](research/league-worlds-broadcast-pilot-2026-11-22.md)

## Pass 1 · HUD and event timeline

### Commentary transcription

Prefer the original-language YouTube automatic-caption track as the commentary
source in Pass 1. Download and retain the timed JSON3 track with yt-dlp alongside
the source recording. For the current English Midbeast video, use `en-orig`.
Use the ASR producer as a fallback when a usable caption track is unavailable.

Adapt the caption text and source-video timestamps into
`commentary_timeline.jsonl`. Preserve available word offsets and keep segment
timing where word offsets are unavailable. Normalize rolling caption display
events into a chronological transcript while preserving the original track.
Record the transcript source, language, YouTube video ID, source-video hash,
caption-file hash and original event references with the derived records.
Preserve the caption wording and flag uncertain champion, player and ability
names for later review.

Attach playback context when one verified segment covers the entire utterance.
Keep gaps and uncertain playback associations explicit. The unified `/pass1`
timeline interleaves commentary and gameplay events and seeks the source player.
Commentary supplies evidence of what the speaker said. Gameplay claims retain
their source-frame and HUD evidence requirements.

The completed Midbeast ASR run supports this choice. Its 475 utterances contain
3,694 aligned words and have 9.5% word disagreement with the YouTube auto-captions
across 3,940 reference tokens. Among 3,609 matching tokens, median onset difference
is 78 ms, the 95th percentile is 150 ms, and 99.6% are within 350 ms. Review near
the beginning, middle and end found matching meaning. Both automated transcripts
have vocabulary errors. Speaker identity remains unverified.

The current bundle uses the retained YouTube auto-captions in its commentary
timeline. The adapter preserves caption word onsets and bounds each cue by its
display interval and the following cue where possible. When saved speech
coverage is available for the same recording, trim the end to the speech region
containing the last caption word onset. Adjacent model chunks and pauses up to
150 ms form one region. Keep the caption boundary when speech coverage is
ambiguous or the caption has segment timing only. Record the estimation method
and speech evidence. Individual word ends remain unknown. The completed ASR run and its comparison report supply
experiment evidence. See the [commentary runbook](league-pass1-runbook.md#commentary-transcripts-in-pass-1)
and the [full-recording comparison](../data/league_asr_handoff/20261120-results/verification.json).

### HUD and event producers

The [Pass 1 runbook](league-pass1-runbook.md) documents the unified experimental
CLI, reusable producers, five-minute pilot, budget checkpoint, portable review,
and verification. The current Midbeast run is
`data/league_runs/midbeast-pass1-final-v1/`; consult its `pass1_report.json` for
actual coverage rather than treating implementation availability as completion.

- **HUD state.** Read the top HUD clock, team kills and visible notices, plus
  the selected champion's portrait, health, resource, level, CS, K/D/A, gold,
  ability icons and cooldown state, and inventory wherever the layout exposes
  them. Record selection changes. Track closed panels, obscured values, and
  visibly empty slots separately.
- **Ability casts.** Read cast or channel bars and track ability-slot changes
  where the HUD supports them. Sample brief cast cues more densely than slow
  counters. Link cues to the selected champion when ownership is clear. A
  cooldown transition supports slot activation. A visible cast, target, hit,
  or channel completion needs its own evidence.
- **Scoreboard and items.** Detect scoreboard openings and capture readable
  rows for both teams, including side, row, champion portrait, displayed player
  name and stats, six item slots, and trinket. Store each observed inventory at
  its source time. Derive each champion's item history independently from their
  verified rows. Use readable slots even when another row or slot is obscured.
  Between openings, expose observed items as last seen. Bracket a confirmed
  change between the reads that establish it unless a separate visual cue gives
  a tighter time.
- **Backfill champion identities.** Gather roster evidence across openings. Once
  a later read establishes a stable row or player-to-champion binding, apply
  it to earlier inventory reads that share the verified binding. Preserve the
  original item values and timestamps, the identity evidence, and when that
  evidence became available. Keep ambiguous rows and unreadable past items
  unknown. A complete team roster can narrow candidates for a row. A row
  assignment still requires a verified binding.
- **Merge rapid scoreboard reopens.** Merge nearby whole-scoreboard openings
  when the same champions have matching, fully resolved item and trinket snapshots
  within the same verified playback segment. Compare the actual item IDs and
  known empty slots. Keep separate openings when a slot changes, a row is
  ambiguous, or continuity is uncertain. Retain every source sample and its
  visibility boundaries so a brief closure remains visible in review. Calibrate
  the maximum close-to-reopen gap against labeled footage.

### HUD transcription for future runs

Use `Qwen/Qwen3.5-4B` in BF16 to transcribe the selected champion's numeric HP,
scoreboard player names, and top-center game announcements from calibrated crops.
For each crop, the model returns the displayed text or `null`. The transcription
schema has no `absent` or `unreadable` status. Determine panel visibility from
layout and panel evidence. Leave values unknown when the model returns `null` or
the text fails field validation.

Parse current and maximum HP from the transcribed text in code. Use the separately
measured HP bar as a consistency check and retain the crop and raw transcription
for review. Validate the new reader on readable, obscured, and empty crops for all
three fields before relying on it for full recordings.

### Killfeed recognizer

The [Midbeast pilot](../annotation/league/README.md#killfeed-timeline-pilot)
samples native frames at 4 FPS and matches the killer and victim portraits
against the full pinned champion catalog. It keeps unreadable roles unknown
and saves the entry crop and source coordinates. Two compatible reads in
verified playback can provisionally fill a gap in team membership. The
scoreboard takes precedence and may veto a candidate after a later opening.

Track each entry as it persists or shifts when a new one arrives. Emit one
timeline event for each distinct entry, including repeated killer–victim pairs
when separate appearances are visible. Record the first and last source frames
showing the entry, its playback segment, and the game clock if available. Use a
preceding readable frame in the same segment where the entry was absent to
bound its appearance time. An entry already visible on the first readable frame
has an unknown appearance time.

The event records a kill announcement. Team kill totals, scoreboard counters,
and central notices can corroborate it. A cast, on-screen body, or exact death
frame requires matching scene evidence. The pilot records replay sightings at
their source times and excludes them from roster backfill. It exposes
announcements in timeline review. Caption, QA, and match-summary use remain
future work.

Existing intake and inventory behavior is documented in the
[annotation README](../annotation/league/README.md#inventory-timeline).
The [recognition workflow](league-recognition-refresh.md) covers cast bars, and
the [scoreboard handoff](league-body-recognition-handoff.md) covers the roster
skim and reviewed bindings.

### Candidate windows for SAM

Scan gameplay frames with the VLM for in-world champion health bars. When it
first detects the bar of a champion entering the frame other than the champion
whose gameplay we are annotating, open a candidate window at that frame. Either
team can trigger a window. Save the exact source frame ID, presentation timestamp,
and health-bar box. Use the bar to locate a point or box on the associated champion
body for SAM. Pass 3 uses the resulting track to recognize the champion.

Keep successive detections of the same visible champion in one window. A clear
re-entry after leaving the frame starts a new window. At a playback boundary, a
champion already in view can be seeded on the first detected frame, with entry
time marked unknown. Bound processing by verified playback continuity. Overlapping
windows can share frames while retaining each champion's own seed frame. Cast
cues and killfeed announcements supply timeline evidence for captioning.

## Pass 2 · SAM masking

Run SAM on the candidate windows from Pass 1. Prompt each track on the recorded
first-detection frame using the body point or box located from the health bar.
Record a seed failure when the associated body cannot be localized there. Save
each visible-body mask in source coordinates with its source frame, track ID, and
seed evidence. Carry any evidence-backed Pass 1 identity links alongside the track.
Mark occlusions and gaps, and start a new track when continuity cannot be established
across a cut or occlusion. A selection change ends the selected-owner association
with the previous body. See the
[grounding roadmap](league-grounding-roadmap.md) and
[multi-champion SAM pilot](league-sam-multichampion-pilot.md).

## Pass 3 · Visual captions for each clip

### Match masked champions to 3D renders

For each SAM track, use the masks to isolate the champion's RGB pixels in usable
source frames. Embed the foreground crops and a versioned gallery of 3D champion
and skin renders with the same frozen SigLIP encoder. Match crops to render views
by cosine similarity and combine scores across visible frames to rank champion
identities for the track. Use verified Pass 1 roster membership and reliable team
evidence to narrow the gallery candidates when available.

Record the accepted champion ID with its track ID, source frame and mask IDs,
matched render references, and match scores. Keep the track anonymous if the
evidence is ambiguous or the gallery lacks coverage for plausible candidates.
See the [SigLIP render comparison](league-sam-multichampion-pilot.md#compare-skin-encoders-on-saved-sam-crops).

### Finalize clips and captions

Inspect original frames in the candidate windows, using Pass 2 masks to follow
visible actors, accepted track matches to name champions, and Pass 1 evidence for
HUD state, item context, cast cues, and killfeed announcements. Finalize clip
boundaries around the visible setup, action, and result where available. The
setup can include source frames before the SAM seed when playback is continuous.
A window can yield multiple clips when it contains distinct actions, or zero
clips when it yields no supported captions.

Pass 3 is the final stage that adds gameplay claims. Emit individually checkable
claims with source times, actor or track references where supported, and links to
their evidence. A HUD cue can support “the HUD showed a Q activation.” A visible
cast requires scene evidence. A killfeed entry can support “the feed announced a
kill” even when the fight is off-screen. See the
[clip and caption contracts](league-caption-generation.md).

## Pass 4 · Gameplay descriptions

Start with deterministic wording that renders each clip's ordered caption claims.
Combine related setup, action, and result claims when their evidence establishes
the connection. Preserve qualifiers for last-seen item state, uncertain identities,
and off-screen kill announcements. Link each description statement to its
supporting caption IDs so it can be traced to the original frames, HUD reads, or
masks. Evaluate recognition and attribution with this renderer before introducing
a description model.

## Alternate pathway · Captions, LUNA annotation, and questions

This three-pass route takes source clips and the existing [HUD and event
timeline](#pass-1--hud-and-event-timeline) as inputs. The shared timeline and
scoreboard (leaderboard) are prepared upstream. The pass numbers below refer to
this route. See the [LUNA event annotation trial](league-luna-event-label-review-2026-11-09.md)
for earlier work on the second pass.

1. **Generate captions.** Use the [caption workflow](league-caption-generation.md)
   to segment footage and produce timestamped draft captions of visible gameplay.
   Keep each caption linked to the source clip and frames. Leave uncertain actors,
   abilities, and outcomes unresolved for LUNA to inspect.
2. **Annotate with LUNA.** At the start of every new video annotation pass,
   generate a knowledge reference with the source fingerprint, the observed
   scoreboard rosters, a pinned ability catalog, and the map entity guide.
   Rebuild it from the current pass inputs when a video is annotated again.
   Give LUNA the clips and captions, the matching timeline slice, the scoreboard
   reads and roster evidence valid for that interval, and the relevant reference
   entries in the request. Keep the full reference available for lookup. Include
   readable crops of the HUD, cast bar, feed, and objective banner or chat when
   available. For turret losses, include readable minimap views from before
   and after the destruction in the same playback segment.
   Use `scoreboard_team_relationships` and `scoreboard_relationship_by_champion`
   to supply team lists and a per-champion lookup from the active scoreboard and
   confirmed HUD owner. The lookup labels each resolved champion as `self`,
   `ally`, or `opponent`. If the owner or roster is unresolved, leave the team
   relationship fields empty.
   LUNA confirms each champion's identity from the clip evidence before looking
   up that champion's team. If the identity remains uncertain, describe the
   visible action without assigning a team relationship.
   Treat the reference as a guide to identities and possible mechanics. Use the
   footage and timeline to establish what happened in this match.
   LUNA checks the original footage, resolves champion identities and
   casts where the evidence supports them, and writes descriptions of what happens
   in order. Each description claim cites its frames, captions, or timeline
   records. Preserve unknowns and conflicts. Use HUD activations to locate
   candidate cast times. A visible cast, target, hit, or outcome needs evidence
   for that specific claim. Keep off-screen announcements distinct from on-screen
   action. Use `annotation.league.luna_evidence` to build the video reference
   and select fresh cues for each clip. A cue's entire observed onset bracket
   must fall inside the half-open source interval `[start, end)`. A channel
   already visible at the first readable sample has an unknown onset unless
   preceding evidence establishes its start. Preserve coverage gaps so the
   model can still check for clearly visible transitions in the original frames.
3. **Generate questions and answers.** Use LUNA's descriptions and the matching
   timeline to generate answerable questions about identities, actions, sequence,
   and visible game state. Save the answer, source interval, and supporting
   description claim and timeline IDs with each question. Check that the evidence
   determines the answer within the same playback segment. Drop questions that
   depend on unresolved attribution, contradictory reads, or unseen outcomes.

Generate the lookup at the start of the LUNA pass. Point `--pass1` at the run
directory containing the source `manifest.json` and the materialized timeline
for the revision being annotated. If you supply an inventory catalog, the
reference includes descriptions for items recognized by the scoreboard reader.

```bash
python -m annotation.league.luna_evidence \
  --pass1 "$pass1_directory" \
  --ability-catalog "$ability_catalog" \
  --inventory-catalog "$inventory_catalog" \
  --output "$luna_run_directory/knowledge.md"
```

### Draft Luna prompt

```text
Describe the visible play in one natural paragraph without timestamps. Focus on
the champions directly involved and what each visibly does. First confirm the
unit type and identity from the model, portrait, or distinctive effects, using
the active roster to confirm the champion is in the match. Consult the knowledge
reference to distinguish champions, jungle monsters, plants, wards, and
structures, or to check what an ability or item can do. When identity is
uncertain, describe the visible action without naming the champion or assigning
a team relationship.

Only after confirming a champion's identity, look up that champion in the
scoreboard-derived ally and opponent lists. Use those lists only to assign team
relationships. They are relative to the selected HUD champion, even when the
camera follows someone else. A confirmed champion on the opposing scoreboard
team is an enemy of the HUD owner. A confirmed champion on the HUD owner's team
is an ally. Before returning the paragraph, check every named ally or enemy
individually against the scoreboard, including champions mentioned in a group.
Leave the relationship unstated if the lookup is unresolved.

Read the selected HUD owner's portrait or selection timeline before assigning
an ability, Recall, or Teleport. The camera can pan or cut to a different fight
while that HUD stays on the same champion. Check the visible body, target,
and scene separately. Keep unseen actions outside the description.

Before reporting an objective outcome, check the exact wording of the in-game
banner or chat line. Distinguish an objective that spawns, one that a team
summons, and one that a team slays or destroys. Verify the objective name and
any champion or team credited. If the announcement gives only team credit,
report only the team. If the wording is unreadable or ambiguous, describe the
visible contest or attacks and leave the outcome and credit unstated.

For a turret loss, read the banner from the selected HUD champion's viewpoint.
"Ally turret destroyed" means their team lost a turret. "Enemy turret destroyed"
means the opposing team lost one. Cross-check the minimap just before and after
the announcement in the same playback segment. Look for the turret icon that
disappears and use its lane position and base side to verify the location and
which team lost it. If the minimap is obscured or unclear, use the readable
banner for the team and leave the lane or turret unspecified unless the scene
shows it. If the evidence conflicts, report the destruction without assigning
the turret to a team.

Use the clip timeline to find cooldown starts, cast bars, and killfeed entries
whose onset falls inside the clip. A countdown already ticking, a cast bar
already active, and a feed entry still on screen can carry over. Describe an
ongoing channel as ongoing only when relevant. If the timeline has a coverage
gap, a clearly visible same-clip change can establish a fresh cue. Treat a new
cue as a candidate until the footage supports its owner or feed roles. A slot
supports only the ability or spell in that slot for the observed HUD owner.

When a meaningful ability fires, name the spell and the HUD or cast cue that
supports it. A cooldown or cast bar supports the cast. For a target or hit,
look for contact, a target-specific impact, displacement or status, or a drop in
that target's HP aligned with the impact and distinguishable from other damage.
Describe a shield, heal, status, or other state change when it occurs. Name its
cause only when the evidence ties the change to that ability, item, or effect.

Use an applicable scoreboard read at or before the action to say a relevant
ally or enemy was "last seen with" an item. An item on the scoreboard establishes
a last-seen build. An item activation or proc needs its own evidence. If a fresh
killfeed entry matters, name what the feed announced in the right order and
leave any unshown takedown mechanics unspecified.
```
