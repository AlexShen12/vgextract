# Rocket League video evidence → hierarchical events → grounded questions

Status: proposed architecture and annotation mapping, not an implemented or evaluated extractor. Input is uploaded YouTube video, with optional separately labeled audio. No Rocket League frame was supplied or inspected for calibration. Support player POV and esports broadcasts with separate layout profiles. Initial mechanics scope is standard Soccar; identify team size and flag other modes/mutators rather than assuming their rules.

## Adaptation of the reference pipeline

[EgoCS-400K](https://github.com/EgoCS-400K/Dataset) aligns replay telemetry with video, protects action chains during segmentation, and constrains captions with structured priors. Preserve the hierarchy, protected spans and prior-guided language generation. Replace replay truth with timestamped visual evidence and explicit uncertainty. The supplied diagram is a reference architecture, not instructions to execute its Counter-Strike-specific steps.

| Reference component | Rocket League video counterpart |
| --- | --- |
| Demo collection / rendering | Existing YouTube ingestion, source metadata, original frame PTS |
| Tick-level camera, spatial, player and round states | Sampled view identity, HUD readings, visible car/ball tracks, coarse field relations |
| Keyboard / mouse raw inputs | Unavailable; store observable motion and effects, not reconstructed buttons |
| Raw game events | Contact, bounce, resource changes, goal and message evidence with uncertain boundaries |
| Atomic actions | Lift-off, visible flip/rotation, boost effect, ball contact, landing, bump |
| Protected action chains | Approach → contact/challenge → ball response; shot → save/rebound/goal; aerial → touch → recovery |
| Spatial/action filter | Visibility, identity, continuity, HUD readability, full event coverage and answerability |
| DP segment splitting | Budgeted segmentation with protected short chains and overlapping longer plays |
| Prior-guided captions | Evidence-constrained captions and question templates, followed by source verification |

This supports video understanding/QA. It does not provide an action-conditioned dataset of executable player inputs or authoritative full-field state. A goal replay is additional footage of an existing event, not a second goal.

## Three tracks to maintain together

1. **Physical track:** visible cars, ball, contacts, bounces, lift-off, landing, displacement and visible boost effects.
2. **HUD track:** score, clock, overtime marker, viewed-player name/team, readable boost and stat messages.
3. **Interpretation track:** shot, save, challenge, possession, pass, rotation, pressure and counterattack. These are derived hypotheses linked to physical/HUD evidence; uncertainty increases with scope.

Track view changes separately. A camera switch is a source event and can change whose boost is displayed; it is not a car teleport. Camera motion is not ball/car world motion. Broadcast knowledge does not establish what any player saw.

## Screen routing

Use logical extraction roles, not necessarily separate VLM models. One VLM can serve several bounded prompts; OCR, detection and tracking should handle repetitive numeric and geometric fields. A temporal VLM is useful for contacts, multi-car interactions and semantic interpretation.

All boxes below are **hypothetical calibration seeds**, not measured coordinates. They are normalized `[x0, y0, x1, y1]` relative to the detected game viewport. Detect actual HUD anchors and masks per layout span before using them. Letterboxing, editing, custom broadcast HUDs, facecams, subtitles, scoreboard screens and split-screen invalidate universal coordinates.

| Role | Example region / crop policy | Output | Boundary on claims |
| --- | --- | --- | --- |
| Match HUD | Top center seed `[.30, 0, .70, .14]` | Team labels, match goals, game clock, overtime marker; series score as a distinct field | Match score ≠ series score; score delta alone does not identify scorer |
| Player/resource HUD | POV boost seed `[.80, .72, 1, 1]`; player label and broadcast sidebars located dynamically | Readable boost attached to player/team identity, HUD focus changes, roster mapping where available | Sidebar values and POV meter may describe different players; do not transfer resources on switches |
| Camera/view | Bottom-left indicator seed `[0, .84, .30, 1]` plus sparse full-frame context | Live/replay/menu/training, POV/spectator, viewed player, indicated ball cam, cuts, occlusion | Absent indicator ≠ ball cam off; POV focus ≠ verified raw control access |
| Local mechanics | Detector-driven crop following car, ball, contact region and relevant surface over the window | Lift-off, flip-like rotation, landing, boost plume, contact/bounce candidates, bump/displacement | A tight car crop alone cannot prove a touch, shot, save, flip reset or exact keypress |
| Interaction / field context | Full unmasked gameplay viewport at reduced resolution plus detailed active-area crops | Goal/field landmarks, nearby teammates/opponents, challenge, ball-to-goal relation, coarse zone and local outcome | Do not use a fixed middle box: aerials and wall plays reach the edges; missing defender ≠ empty goal |
| Messages | Locate statfeed, central goal/replay banners and displayed panels | Goal/scorer, credited save/assist/demolition when readable, replay marker | Credited event ≠ independently observed mechanics; a demolition notice can confirm an off-screen event without showing collision |

Epic documents a configurable [bottom-left ball-cam indicator](https://www.epicgames.com/help/en-US/c-202300000001622/c-202300000001682/a202300000018048?lang=en-US); it can be disabled. Camera FOV, angle, height and distance are also configurable. Use indicator readings with provenance; geometry-derived camera mode remains inferred.

Crop mechanics:

- Follow detections across the entire viewport. Build a temporally stable crop from the union of actor/ball trajectories and nearby surface, with a pilot padding of about 20%; clip to the viewport and preserve crop-to-source transforms.
- Supply a small timestamped context frame alongside high-resolution local crops. A crop following a ball hides global displacement unless the original coordinates are retained.
- For a challenge, include both cars and ball. For a shot/save, include ball trajectory and defending goal. For a pickup, include the car, pad and synchronized resource evidence where visible. Use paired context crops if a single union is too large.
- If a detector loses the ball, enlarge/search the viewport and mark uncertainty. Never silently place the ball at the screen center.
- Detect overlay masks. Keep message overlays available to their extractor even when masking them from field tracking.
- Recover field zones from goals, walls and markings only when orientation is supported. Native footage provides no guaranteed minimap or full-field view. Do not invent a minimap.
- On broadcasts, bind nameplates, sidebars and active-player tags to persistent IDs with confidence. Car cosmetics alone are weak identity evidence. Store track IDs when player identity is unresolved.

## State ledger

| Family | Fields | Unknown / freshness policy |
| --- | --- | --- |
| Source and view | Video PTS, game ID, layout/camera span, live/replay, playback rate, viewed-player ID, masks, game-clock reading | Revalidate after cuts and replay transitions; source time is the canonical reference |
| Global HUD | Match score, team mapping, clock, overtime, series score if shown, match phase | Do not infer final match state from a short highlight or a zero clock alone |
| Car | Track/player/team ID, image box, visibility, grounded/airborne/wall/ceiling state, orientation category, visible boost effect, last observed boost | Screen pose/motion are not exact world pose/velocity; expire fast states; stale boost is historical |
| Ball | Image box, visible track, ground/air/wall vicinity, bounce/contact candidates, coarse trajectory relative to supported landmarks | Off-screen = unknown; motion discontinuity can be camera motion or contact, not proof of actor |
| Relations | Car approaching ball, relative side of ball, contest proximity, ball toward/away from identified goal, nearby participants | Infer only with stable context and explicit coordinate frame |
| Coarse field | Own/mid/opponent half, corner, sidewall, backboard, goalmouth when orientation is resolved | Team-relative zones require goal/team mapping; otherwise use neutral landmark labels |
| Derived tactics | Control/possession candidate, pressure, transition, support/rotation, local numerical advantage | Last touch does not equal possession; off-screen players block many team-wide assertions |

Persist three distinct identities: viewed player, tracked actor, and controlled player (unknown unless independently established). Identity can remain valid across shots while spatial continuity is broken. Custom team colors require verified mapping; orange/blue car appearance is not sufficient.

Every observation needs: `observation_id`, `source_video_id`, `video_pts_seconds`, `frame_index`, `game_id`, `view_span_id`, `layout_span_id`, `crop_box`, `crop_to_source_transform`, `extractor_version`, `entity_id`, `field`, `value`, `visibility`, `epistemic_status`, and extraction-quality metadata. State reductions add `last_seen` and freshness. Epistemic states are `observed`, `inferred`, `unknown`, `contradicted`; model self-confidence is separate from calibrated acceptance quality. Unknown values are null, not zero.

Maintain piecewise video-time↔displayed-clock mappings. The clock can pause, count down/up, repeat in replays or jump after edits; identical clock readings do not alone align events. At zero, use observed continuation, goal, overtime transition or end markers rather than declaring match end.

## Hierarchical event ontology

The hierarchy is an interval graph. Events have zero or multiple parents and can overlap. Physical, resource and view events may contribute to the same play. Use `part_of`, `precedes`, `overlaps`, `involves`, `supported_by`, `followed_by`, and `replay_of`; keep causal claims separate from temporal links.

| Level | Families | Operational evidence / caveats | Questions |
| --- | --- | --- | --- |
| L0: observable transitions | Boost value change, ground→air, ball direction change, score/clock change, HUD identity switch | Same entity before/after; distinguish wall departure from jump, bounce from car touch and focus switch from resource change | What changed? What was displayed before contact? |
| L1: atomic physical events | Ball touch, surface bounce, lift-off, landing, visible flip-like rotation, boost effect, pickup, bump, demolition | Contact candidates need approach/contact/response evidence; boost increase alone does not identify a pad; demolition requires resolved visual sequence or readable notice | Which car visibly touched the ball? Did it land on ground or wall? |
| L2: local interactions / mechanics | Aerial attempt/touch, challenge, near-simultaneous contest (50/50 candidate), dribble sequence, flick candidate, shot, save/deflection, clear, wall/backboard play, recovery | Link L1 events with consistent actors and outcome. Airborne motion alone is not an aerial touch. A credited save and a visually inferred save are distinct labels | Was the next touch grounded or airborne? What followed the challenge? |
| L3: coherent plays | Kickoff sequence, attacking sequence, controlled possession, pass reception, shot→save→rebound, defensive sequence, counterattack candidate | Keep setup and resolution. An uncontested opponent-free view is not proof of control or an open net. Pass intent is usually unknown; annotate teammate touch sequence first | Which team made the next observed touch? How was the shot resolved? |
| L4: match episodes | Sustained pressure, attack→defense transition, goal→replay→next kickoff, regulation→overtime, late-game defense | Multiple linked plays with broader coverage. Partial POV often cannot verify rotation order, last defender or full-team positioning | What phase followed the goal? Which observed plays preceded the score change? |

Separate mechanical motion from inferred controls:

| Candidate label | What video can support | What remains unavailable |
| --- | --- | --- |
| Acceleration / steering | Visible path/orientation change under supported camera compensation | Throttle/brake values or stick direction |
| Boosting | Time-local visible plume/effect and corroborating resource decrease where readable | Exact button state/duration from sparse frames; visible trails alone may be unrelated cosmetics |
| Jump | Lift-off from a supported surface and jump-like motion | Exact press timing, hold duration, single/double-jump controls; wall departure may look similar |
| Dodge / air roll | Flip-like motion or airborne rotation sequence | Binding, exact stick command, control availability |
| Powerslide | Sliding/turning candidate | Guaranteed powerslide input |
| Flip reset / wavedash / speed flip | Advanced mechanic candidate only with full contact/motion sequence and targeted human review | Hidden reset availability or exact control sequence; do not include as default factual labels |

Operational definitions for contentious labels:

- **Touch:** actor–ball contact is visually supported; a ball trajectory break alone is an unattributed contact/bounce candidate.
- **Shot candidate:** a supported touch followed by a goal-directed trajectory with goal context. Keep `credited_shot` separate when an overlay supplies it.
- **Save candidate:** a defender touch redirects a visibly threatening goal-bound ball. Without trajectory/goal context, label `deflection`; keep `credited_save` separate.
- **Clear candidate:** a touch moves the ball away from a supported defending-goal/zone toward a safer region. It does not prove the player intended a clear or retained possession.
- **50/50 candidate:** opposing cars contest the same ball in an overlapping contact window. Insufficient temporal resolution yields `contested_touch`, not an exact first-touch answer.
- **Pass reception sequence:** a same-team touch is followed by a teammate reception with supported identities and continuity. Deliberate passing is an interpretation, not guaranteed intent.
- **Possession candidate:** repeated controlled touches or a visibly supported control opportunity. A team's last touch alone is insufficient; allow `contested`, `loose_ball`, and `unknown` states.

Official Rocket League [event documentation](https://www.rocketleague.com/developer/stats-api) distinguishes ball hits, boost pickups, goals, replay states and statfeed events. It is a vocabulary reference only. A video-only dataset cannot retrospectively access its live telemetry; no API or replay ingestion is assumed here.

## Sampling and information budget

These are pilot hypotheses, not proven operating rates. Preserve original video PTS and frame rate. Do not equate video frames with engine ticks.

1. Detect live gameplay, view/layout spans, replay inserts and game boundaries. Cache stable roster/layout mappings. Separate replay instances and retain alternate evidence without double-counting outcomes.
2. Run inexpensive HUD OCR around 2 Hz and field candidate tracking around 6–10 Hz initially. These are discovery rates, not sufficient evidence for all touches. Prefer native-frame lightweight tracking when feasible.
3. Trigger local review on contact proximity, ball trajectory changes, lift-off/landing, HUD resource/score deltas, statfeed appearance, viewpoint changes and uncertainties. Review roughly 2–4 seconds of setup and 3–6 seconds of continuation; expand if resolution is not shown.
4. For fast interactions, start temporal review around 15–30 fps; inspect native-rate frames around ambiguous contacts or reject precise timing/order. Source blur/compression/low frame rate can remain unresolvable regardless of VLM sampling.
5. Give the local reviewer a bounded car+ball+surface sequence, synchronized relevant HUD observations and a few context frames. Give the tactical reviewer wider context sampled over the play plus short summaries of verified sub-events.
6. Reduce outputs to structured deltas and events. Keep relevant local state, actor IDs and explicit unknowns. Do not accumulate per-frame prose or send every crop to every model.
7. A temporal reconciler requests narrow rereads for conflicting identities, unexplained resource changes, suspect camera-induced motion and missing outcomes. Repeated passes over identical pixels are not independent corroboration.
8. Generate each question from a minimal evidence subgraph. A separate verifier inspects source frames/crops at original timestamps with enough context; captions cannot serve as answer keys.

Logical roles can share the same model. Start with OCR/detection, one local temporal VLM, one broader-context VLM prompt and one verification prompt before paying for many distinct models. Benchmark a full-frame baseline on identical windows to test whether routing actually reduces cost without losing events.

Prompt contract: one task, allowed label vocabulary, actor mapping, timestamped crops, relevant prior observations, and required JSON fields. Demand evidence IDs, time bounds, alternatives and unknowns. A specialist must not fill fields outside its scope. The question generator sees verified records and coverage constraints, not an unrestricted match narrative. Optional commentary/audio can propose candidates but cannot silently answer a video-only question.

## Protected segments

Protect meaningful evidence chains, not every interval that happens to overlap:

- Approach → contact → immediate ball response.
- Lift-off → airborne attempt/touch → landing/recovery, if recovery is the question target.
- Shot → save/deflection/post bounce/goal/miss; attach rebound follow-up only when needed.
- Opposing approaches → challenge → loose ball or next resolved touch.
- Car reaching boost pad → pickup evidence → readable resource result, where visible.
- Goal evidence → score update; replay and next kickoff can be linked as a longer parent episode.

For a pilot, short mechanics windows might be 1–6 seconds, interactions 3–12 seconds and plays 8–30 seconds, with longer parent episodes kept separately. Choose boundaries by evidence requirements, not fixed durations. These ranges are suggestions, not universal definitions.

A boundary planner can penalize cutting protected intervals, missing setup/outcome, unreadable evidence and model-budget overflow. Keep replay/edit discontinuities explicit. A POV change breaks visible trajectories but need not split a play when HUD/identity continuity is supported. An edit gap can leave the outcome unknown. Avoid transitive merging into an entire-match segment; split long parents into overlapping linked child windows.

## Question construction and acceptance

| Template | Minimum evidence | Reject / qualify |
| --- | --- | --- |
| How much boost did the viewed player have immediately before the aerial? | Readable time-local boost + viewed identity + lift-off interval | Stale reading, identity switch, unreadable meter; say “last displayed before” if coverage is not immediate |
| Which visible car made the next ball touch? | Resolved contact sequence + actor identity | Unseen intervening touch or identity ambiguity; use track/team wording where supported |
| Was the ball touched while the car was airborne? | Contact evidence + same-car surface state | Airborne approach alone is insufficient |
| Did the ball hit the sidewall or backboard after the touch? | Surface landmarks + continuous post-touch sequence | Missing bounce or unresolved field orientation |
| What happened to the shot: save, post bounce, goal or miss? | Shot context + visible or separately credited resolution | A camera cut before outcome; alternatives overlap or multiple occur in sequence |
| Which team made the next observed touch after the contest? | Resolved teams + contest + following touch | Never substitute “won possession” for a next-touch answer |
| Did the same player collect boost before or after the touch? | Both events, identity continuity, non-overlapping time bounds | Boost increase only without pad evidence; overlapping timing uncertainty |
| What was the displayed score after the goal? | Readable live score after resolved goal | Series score confused with game score; replay duplication |
| Which observed events formed the rebound sequence? | Ordered shot/deflection/rebound evidence with coverage | Missing middle event; speculative pass intent |
| Can the clip establish that no defender was in goal? | Coverage of relevant goal and possible occlusions | Use as deliberate unanswerability case when view is insufficient |

Question acceptance requires: explicit visible scope, resolved necessary identities, relevant temporal coverage, fresh state readings, source evidence for every answer claim and verified alternatives. Temporal ordering is accepted only when uncertainty intervals do not overlap. A sampled absence is not proof an event did not occur. A VLM's numeric confidence is not a substitute for calibrated rules.

Factual initial benchmark excludes exact inputs, unseen boost/positions, “why” intent questions, best-play judgments and counterfactual outcomes. Strategy questions can be a separate task with labeled interpretations and multiple defensible answers. Prediction questions need a fixed observation cutoff; future evidence used for the target must not reach the question prompt or displayed observation.

Store accepted QA with `question_id`, `type`, `text`, `answer`, `answer_variants`, `event_ids`, `evidence_ids`, `evidence_times`, `evidence_regions`, `required_coverage`, `observation_cutoff`, `uncertainty_flags`, `verification_result`. Store rejected candidates and reasons to measure yield, not just accuracy.

Event records need actor IDs, event label and level, bounded onset/offset times, evidence references, epistemic status, visible/credited outcome, coverage, parent/related IDs and replay-instance mapping. Timing boundaries are intervals when uncertain. Do not collapse inferred tactics into observed physical labels.

## Artifacts and pilot

Write a game-specific annotation layer after the existing YouTube ingestion. Proposed artifacts: `video_manifest.json`, `layout_spans.jsonl`, `view_spans.jsonl`, `observations.jsonl`, `state_deltas.jsonl`, `events.jsonl`, `event_graph.json`, `segments.jsonl`, `qa_candidates.jsonl`, `qa_verified.jsonl`, `qa_rejected.jsonl`. The companion `rocket-league-video-annotation-spec.json` is starting configuration, not executable extraction code.

Pilot: 3–5 videos per supported layout family, 30–50 manually labeled windows spanning kickoff, calm movement, aerials, wall contacts, contests, shot/save/goal chains, boost pickup, camera switches, replay inserts and missing evidence. Begin with clock/score/view identity, readable boost, visible surface state, simple touch/bounce events and shot-resolution sequences. Promote advanced mechanics and team tactics only when measured precision supports them.

Measure HUD accuracy, actor identity switches, touch precision/recall, boundary uncertainty, replay duplicates, coverage, unsupported claims, QA correctness/answerability, accepted questions per video hour and extraction cost. Compare full-frame and routed-crop variants on the same labels. Split by match across duplicate uploads, highlights, replays and POVs; add channel/layout holdouts. No performance claim is made by this proposal.
