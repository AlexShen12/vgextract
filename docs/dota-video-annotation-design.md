# Dota video evidence → hierarchical events → grounded questions

This is a proposed design for the existing YouTube ingestion pipeline, not an implemented or evaluated annotation system. It assumes frames and optionally audio from uploaded Dota videos, without native replays or raw inputs.

## What carries over from EgoCS-400K

[EgoCS-400K](https://egocs-400k.github.io/) uses replay-derived signals to constrain captions, groups actions into protected intervals, and chooses clip boundaries that preserve them. Retain those architectural ideas. Replace replay truth with an evidence ledger: every state reading and event must point to video timestamps and screen regions, preserve uncertainty, and declare whose perspective it describes. This supports video understanding and question answering; it does not supply executable player controls or authoritative full-match state.

| Reference pipeline component | Dota video counterpart |
| --- | --- |
| Demo collection and rendering | YouTube video and metadata ingestion; preserve original frame timestamps |
| Parsed per-tick state | Sampled HUD readings, visible entities, minimap tracks, visibility records |
| Keyboard/mouse annotations | Unavailable; optional cursor/ping observations remain screen evidence |
| Raw engine events | Visually supported events with evidence and time bounds |
| Action and spatial annotations | Semantic action spans, camera spans, coarse map movement |
| Protected action chains | Cast/effect sequences, engagements, objective attempts |
| Dynamic-programming segments | Event-preserving clips with required setup and outcome coverage |
| Prior-guided captions | Captions and questions constrained by verified local evidence |

## Screen routing and specialist roles

Run logical specialists over synchronized crops. They may share one VLM with different prompts; six tasks do not require six distinct models. Prefer OCR, icon recognition, and tracking for stable HUD fields; reserve temporal VLM calls for ambiguous actions and context. Crops should retain native detail, with optional low-resolution context frames so specialists can resolve identity and location.

The following normalized `[x0, y0, x1, y1]` boxes are **calibration seeds for one inspected frame**, not universal Dota coordinates. The source is local `data/youtube/pro_gameplay_dota2/9eNgCt8mLMI.mp4`, video time 600 seconds, 1920×1080, displaying game time 9:29. The sidecar title is “Topson Earth Spirit - PLAYMAKER - Dota 2 Pro Gameplay [Watch & Learn]”. Only this frame was inspected. Do not infer camera mode, player knowledge, or whole-video coverage from the title.

| Specialist | Seed region | Track | Critical restriction |
| --- | --- | --- | --- |
| Global HUD | Top center `[.28, 0, .73, .07]` | Game clock, score, roster portraits, readable alive/dead and resource indicators | Tiny or absent indicators are unknown; a scoreboard delta does not identify a killer |
| Spatial | Bottom-left minimap `[0, .74, .15, 1]` | Hero/icon tracks, coarse lane/region, structures, pings, group convergence | Missing icon ≠ death; identity and fog-of-war mode may be unresolved |
| Selected-unit state | Bottom HUD `[.25, .82, .74, 1]` | Unit identity, health/mana, level, ability/item slots, readable cooldowns and buffs | Bind each reading to the selected unit; it may differ from camera focus or controlled hero |
| Combat | Battlefield `[.15, .07, 1, .82]`, plus dynamic actor crops | Visible participants, attacks, projectiles, casts, status effects, displacement, retreats | Overlap with minimap/overlays requires masking; camera motion ≠ hero movement |
| Event messages | No fixed seed: locate message/feed/notification panels | Kill, buyback, structure, objective and item messages when displayed | A message can confirm an off-camera event without exposing its cause or participants' actions |
| Economy/context | Left panel `[0, .12, .15, .48]`; gold `[.84, .9, 1, 1]` | Panel type, displayed net worth, selected-unit gold, visible inventory changes | Inventory appearance ≠ purchase time; displayed ranking is only a snapshot |

Detected overlays in this frame include a facecam at approximately `[.9, 0, 1, .15]` and right-side notifications at `[.8, .44, 1, .6]`. They reduce battlefield visibility. Notification panels can themselves be routed to the message specialist. Do not discard their contents merely because they occlude units.

Before using these seeds, detect the game viewport, anchor the actual HUD, identify panel types, and calibrate per video/layout span. Handle alternate minimap placement, resolution, letterboxing, cropped videos, custom HUDs, shop/scoreboard views, selected non-hero units, and moving overlays. Recalibrate when layout changes. Expand combat crops around detections across the entire unmasked battlefield; a fixed central box alone misses actions at the edges.

## Evidence and state representation

Keep observations, consolidated states, and inferred event labels as separate layers. A state is a last observation with a validity policy, not a continuously known fact. Use `unknown` or null for missing values.

Minimum observation fields:

- `observation_id`, `source_video_id`, source frame PTS/video seconds, game-clock reading and uncertainty;
- `layout_span_id`, crop bounding box, extractor/task/version, unit or track identity;
- field, value, evidence modality, extraction quality, epistemic status;
- `last_seen`, visibility (`visible`, `occluded`, `off_camera`, `unreadable`, `unknown`), freshness.

Use epistemic statuses `observed`, `inferred`, `unknown`, and `contradicted`. A model's self-reported confidence is not a calibrated probability. Store it separately if useful, then calibrate acceptance rules on human-reviewed data. One evidence item can corroborate several fields; two extractions of the same pixels are not independent evidence.

State layers:

| Layer | State families | Persistence policy |
| --- | --- | --- |
| Source/view | Live gameplay/replay/highlight/menu, selected unit, camera focus, perspective/fog mode, viewport/occlusion masks | Reset or revalidate on view changes |
| Entity | Hero/player/team mapping, summons and illusions, health/mana/level, visible status, items/cooldowns | Resources and status expire quickly; identities retain their provenance |
| Spatial | Screen track, minimap coordinates, approximate map region, last seen, camera displacement | Never extend a visible trajectory through an unseen interval as fact |
| Global | Clock, kill score, readable deaths/respawns, structures, objective messages | Reconcile competing readings; reset clock mapping on replay/edit discontinuities |
| Economy/progression | Displayed net worth/gold, last hits/denies, inventory/level changes | Preserve observation intervals; do not assign unseen purchases or farming locations |
| Tactical context | Lane pressure, grouping, fight participation, objective contest, retreat | Derived labels with supporting observations and explicit alternatives |

Preserve three distinct identities: `selected_unit_id`, `camera_focus_ids`, and `controlled_hero_id` (usually unknown without established POV evidence). Spectator-visible enemies do not establish what any player could see. All labels describe the uploaded view unless team/player vision is explicitly verified.

## Hierarchical event ontology

The hierarchy is a graph with overlapping intervals. A teleport can join an engagement, and the engagement can be part of a Roshan contest. Events need not have a unique parent.

| Level | Trackable families | Examples of sufficient evidence | Question families |
| --- | --- | --- | --- |
| L0: state transitions | HP/mana/level/cooldown/inventory changes; identity selection; death indicator | Readable before/after observations bound to the same entity | State reading, change detection |
| L1: atomic semantic actions | Attack, cast, item activation, blink/displacement, teleport channel/completion/interruption, ward placement/removal, rune pickup | Actor-target visual sequence plus corroborating HUD/effect where available | Who acted? What ability/item? Which action came first? |
| L2: interaction sequences | Trade, chase/escape, disable chain, spell combination, save, creep farming/stack attempt, local ward exchange | Temporally linked actions involving consistent actors and visible local outcome | How did the exchange unfold? Which visible effect preceded the retreat? |
| L3: engagements | Gank, skirmish, teamfight, pickoff, tower defense, disengagement | Setup, first visible hostile action, participant arrivals/exits, kills and survival evidence | Participants, initiator of visible exchange, sequence, local outcome |
| L4: objective sequences | Tower pressure/destruction, Roshan attempt/contest/kill and Aegis pickup, barracks siege, base defense; patch-dependent objectives | Approach plus direct objective activity or messages and verified resolution | What followed the fight? Which objective was taken? |
| L5: strategic episodes | Laning, rotation, farming-to-fight transition, smoke movement when visible, split pressure, high-ground siege | Multiple lower-level events, minimap observations, progression context | Long-range temporal association, observed shift in activity |

Do not hard-code timing, map geometry, item semantics, hero icons, or objective rules across patches. Maintain a versioned mechanics/icon dictionary. Patch is metadata with provenance and uncertainty; upload date alone cannot establish it. Valve's [HUD update](https://www.dota2.com/700/hud/) and [New Frontiers](https://www.dota2.com/newfrontiers) illustrate why HUD/map/mechanics assumptions require versioning.

Events store actor/target IDs or unresolved tracks, start/end time bounds, parent/related IDs, observation references, local outcome, coverage, and label status. Keep `precedes`, `overlaps`, `participates_in`, `part_of`, `followed_by`, and `supported_by` relations distinct. Assign a mechanical effect link only when evidence supports it; temporal proximity does not establish strategic causality. "Gank" and "escape" are derived interpretations; attacks, movement, status changes, and deaths can be observations.

Special handling:

- Cooldown reset can mean a cast, selection change, reset mechanic, or OCR error. It is a trigger for review, not a guaranteed cast.
- A disappearing hero might be off-camera, invisible, dead, teleported, or occluded. Preserve alternatives until resolved.
- Camera panning shifts all screen tracks. Estimate camera motion and use minimap/world landmarks before labeling movement.
- Treat similar hero models, illusions and summons as ambiguous tracks until identity is supported.
- A visible channel does not establish completion. Record `channel_start`, then `completed`, `interrupted`, or `outcome_unknown` separately.
- A kill feed confirms a death when readable; it does not prove every on-screen attacker contributed or explain why it happened.

## Sampling and information budget

Initial settings below are engineering hypotheses for a pilot, not measured optima.

1. Decode with original PTS. Classify view/layout and create a piecewise video-time↔game-clock map. Pauses, replay inserts, speed changes and edits invalidate a single global offset. Separate replay instances to avoid duplicate event counts.
2. Sample HUD/minimap at roughly 1–2 Hz and battlefield context at roughly 1 Hz for cheap candidate discovery. Low-rate absence cannot prove an action did not occur.
3. Trigger denser windows on health/score/cooldown changes, converging tracks, message appearances, or combat effects. Review approximately 5–10 seconds of lead-in and 5–15 seconds of follow-through.
4. Sample combat windows around 8–15 Hz initially. Short casts can still be missed: inspect native-rate frames around uncertain transitions or reject exact-order/action questions. If windows merge into a long fight, keep sub-event slices and a longer summary.
5. Each specialist receives only its fields, local roster/icon references, prior relevant state, timestamped crop sequence and task. Use an actor crop together with bounded battlefield context for combat; isolated particles are insufficient.
6. A temporal reducer receives state deltas and candidate events, not every frame's prose. Deduplicate events, reconcile IDs/times, expire stale fields, and request targeted rereads for conflicts.
7. A question generator receives only the evidence subgraph needed for one question. Attach original evidence so a verifier can check its answer independently of the generated caption.

This limits both visual tokens and accumulated textual context. Do not ask each specialist to independently narrate the whole match. Audio/caster transcripts can propose candidates but should remain a separate source; they must not silently supply answers for a video-only benchmark.

## Protected segments

Protect the visible start→effect→resolution of a short action and setup→engagement→visible outcome of an exchange. Attach pre/post context requirements to each question type. A tower-taken question may need footage after the fight; a cooldown-before-cast question needs readable HUD before it.

Use nested or overlapping clips: short action clips, engagement clips, and longer objective episodes. Avoid transitive merging of every overlapping event into one giant protected interval. If a protected sequence is too long for the model budget, preserve a parent episode and split at sub-event boundaries with overlaps and linked evidence.

A DP or simpler boundary planner can prefer low-activity boundaries, complete evidence, readable HUD, and reasonable durations, while penalizing cuts inside relevant protected intervals. An edit/camera cut is a visibility discontinuity to record; it is not necessarily an in-game event boundary. Never bridge missing footage with an invented outcome.

## Questions and answerability

Generate question candidates from verified event templates, then have a separate verification pass inspect the actual evidence. Do not use the generated caption as the answer key.

| Template | Required evidence | Reject when |
| --- | --- | --- |
| What item occupies this hero's slot? | Readable inventory plus selected-unit identity | Icon, entity, or slot is ambiguous |
| Did the hero's health increase or decrease? | Same-entity before/after readings | Selection switched or one reading is unreadable |
| Which action occurred first? | Both events observed with disjoint time bounds | Sampling uncertainty permits either order |
| Which heroes visibly joined this engagement? | Identity-resolved arrival/participation evidence | Claim would require unseen participants |
| Did the observed teleport complete? | Channel and resolved arrival/completion evidence | Completion is off-camera and uncorroborated |
| What objective was taken after this engagement? | Linked engagement and objective-resolution evidence | Only movement toward the objective is shown |
| Can this clip establish whether an item was used? | Visible coverage assessment and relevant HUD/effect | The answer substitutes a guess for uncertainty |

Qualify scope in question text: "in the visible exchange", "in the displayed inventory", or "during this clip". Do not turn a local fight clip into "Who initiated the entire fight?" when the start was off-camera. Prefer observed local outcomes over "Which team won?" when trades/objectives are only partially shown.

Questions about intent ("Why did the player retreat?"), optimal decisions, hidden vision, exact input keys, or counterfactuals are outside the initial factual benchmark. A strategy/reasoning extension can label interpretations explicitly and allow multiple defensible answers. For prediction tasks, separate the observation cutoff from withheld future outcome evidence to prevent leakage.

Each accepted QA record needs question text/type, answer, allowable variants, evidence IDs/regions/times, required coverage, uncertainty flags, and verification result. Multiple-choice distractors must be plausible, mutually exclusive for the observed clip, and individually checked as unsupported/incorrect. Remove ambiguous questions instead of forcing one answer. Include deliberate unanswerability examples with explanations tied to visible missing evidence.

## Proposed artifacts and pilot

Suggested outputs: `video_manifest.json`, `layout_spans.jsonl`, `observations.jsonl`, `state_deltas.jsonl`, `events.jsonl`, `event_graph.json`, `segments.jsonl`, `qa_candidates.jsonl`, `qa_verified.jsonl`. The existing ingestion output remains the source of video and metadata. `dota-video-annotation-spec.json` is a machine-readable starting configuration, not a running extractor.

Start with 3–5 videos with similar layouts and annotate 20–30 short windows by hand, including calm scenes, dense fights, selection/camera changes, occlusion, and unreadable evidence. Begin with identity, clock/score, selected-unit resources, readable inventory, coarse minimap positions, death confirmation, and visible engagement boundaries. Add precise casts, item usage, warding, stacking, and strategic episodes only as measured accuracy permits.

Measure field extraction accuracy, identity switches, event precision/recall and time-bound errors, unsupported claims, QA correctness/answerability, retained coverage, and cost per video hour. Report question yield as well as accuracy: rejecting everything is not a useful system. Compare full-frame VLM versus crops plus temporal fusion on the same human-labeled windows. Split downstream datasets by match (including duplicate uploads, player views, highlights, and replays), with channel/layout and patch holdouts where possible. Set acceptance thresholds from those observations; this proposal makes no performance claim.
