# League of Legends: observable gameplay to objective-centered questions

Historical discussion draft, 2026-10-18. The [annotation pipeline](../annotation/league/README.md) now implements fixed visual observers, timeline fusion, question generation, optional review and independent evaluation. See its [validation record](../annotation/league/VALIDATION.md) for the local checks and GPU work still outstanding; this document is not a complete annotation of the supplied match.

The [2026-10-19 final scope check](lol-aatrox-preplanning-checkup.md) records the accepted fixed routing, model assignments, L40 target and memory evaluation decision. It supersedes earlier open alternatives below. Candidate boxes and actual Qwen image/video preprocessing have been checked locally; semantic GPU inference and L40 profiling remain pending.

## Accepted evaluation contract

The following user decisions supersede exploratory alternatives elsewhere in this note:

- Evaluate retrospective visual understanding and state tracking. Strategic advice, coaching, prediction and intent attribution are outside the benchmark.
- Solver inputs are video windows/clips and natural-language questions. No engine telemetry, runtime state APIs or annotation/event logs enter the solver prompt. Construction logs may be derived from pixels offline, but human-reviewed visual evidence establishes answer keys.
- Use Qwen3-VL-8B-Instruct for fixed HUD routes, including the minimap, selected-champion abilities and both sidebars; use Qwen3-VL-8B-Thinking for the main gameplay viewer. Hardware is one L40. Checkpoint revisions, precision, serving stack and measured token/context limits are still planning choices.
- Evaluate episodic retrieval, persistent state memory, cumulative counting, associative linking, trajectory memory and repeated observed behavior, spanning 1–5-second local windows, 30–90-second phases and 10–40+-minute progressions. Sampling gaps and what is actually visible constrain every claim.
- Use the most recent patch for the evaluation cohort and freeze the selected patch/reference snapshot for reproducible runs. Refreshing the patch creates a new cohort. This requirement has not been relaxed by supporting historical reference queries.
- Tag questions `visual_only` or `mechanics_augmented`. The latter may combine pixel evidence with patch-matched Data Dragon/CommunityDragon facts; static knowledge must never supply match-specific state. The updated user contract permits structured requests to a bounded static-reference resolver. Record the exact reference context and retrieval transcript; exclude engine/match-state APIs and construction logs. Report roster-only and mechanics-augmented retrieval conditions separately.
- Start sampling experiments at 0.5–1 FPS for routine activity and 2–4 FPS for fights/contests, with high-detail minimap, ability-HUD and scoreboard/inventory crops plus full-frame context. These are budget hypotheses, not guaranteed adequate rates. Subsecond timing, exact combo order and exhaustive action counts need denser source review and sufficient solver-visible evidence, or rejection.
- Crop selection is fixed by a calibrated layout profile; automatic adaptive review changes temporal coverage, frame density and resolution on those routes. Unify all observer outputs under source-video PTS. Initial memory evaluation is one bounded video/question context without external observation memory; preserve the option for a separately labeled streaming-memory evaluation later.
- Calibrate on diverse human-reviewed windows before scaling. Report factual/numeric accuracy, temporal localization, unsupported claims and inference efficiency, with results stratified by capability, horizon and sampling budget.

Operational refinements for ground truth:

- Label visible movement rather than unobservable movement commands. Ability identities require resolved evidence; base cooldown metadata does not establish current cooldown state.
- Count distinguishable minimap champion icons when that is the evidence; overlapping/unresolved icons prevent an exact champion count. An objective's disappearance or HP depletion does not by itself identify the credited killer.
- Ward presence at separated times establishes observations or a lower bound, not continuous survival or vision coverage through an unseen gap. Store last-confirmed state and uncertainty.
- Count unique observed events after replay deduplication. Exhaustive totals require sufficient coverage. For behavior patterns, define the observed opportunities, outcomes and denominator; do not generalize from this recording to a player's enduring habits.
- Use interval IoU for duration events and tolerance-based timestamp scoring for point events. Match predictions to gold one-to-one and report detection precision/recall alongside localization. Define boundary tolerances during calibration, before the evaluation set is scored.
- Audit the actual model-processed inputs as well as native footage: source-visible evidence may be lost by sampling or resizing. Track this evidence loss separately from errors on answerable model inputs. Record both temporal span and processed frame/token budget.

## Patch-matched knowledge checks

Resolve knowledge references from each video's patch, independently of whether that video meets the evaluation cohort's patch filter. The supplied sample claims patch 26.13; a reference check on 2026-10-18 retrieved Data Dragon build `16.13.1` for Aatrox, Jax and items. The selected build is present in Riot's [version list](https://ddragon.leagueoflegends.com/api/versions.json). Store the explicit gameplay-patch-to-Data-Dragon mapping rather than assuming the two version strings are identical.

The returned champion records contain spell names, icon filenames, descriptions, rank-dependent base cooldowns and tooltip templates. The checked tooltip templates contain unresolved variables, so a reference builder must resolve supported substitutions or mark the affected facts unavailable. Inspect the description as well as structured stats: the checked Sundered Sky record lists Ability Haste in its description but omits it from its `stats` dictionary. Data Dragon is a static reference, not a complete executable mechanics model or a source of in-match cooldowns.

Use the sequence: video patch claim/provenance → verified reference build → entity/ability/item lookup → validated static fact → visually supported match observation → tagged question. Missing static facts remain unsupported; they are not filled from model recollection. Freeze and hash the raw responses and normalized reference pack when building the dataset.

Implemented entry point: `python3 -m annotation.league.knowledge --manifest annotation/league/patch-26.13.json context`. The resolver cascades structured stats → item HTML headers → spell bracket substitutions → same-patch CommunityDragon client/BIN evidence. Its complete patch roster maps internal IDs to display names, including MonkeyKing → Wukong. Unknown OCR names and ambiguous entities stay unresolved. `run_knowledge_loop` intercepts the model's structured requests and returns static references for another reasoning pass; a caller still supplies the actual Qwen inference function. Complex BIN calculations remain explicit, unevaluated evidence rather than fabricated scalar values.

The inspected `16.13.1` [Aatrox record](https://ddragon.leagueoflegends.com/cdn/16.13.1/data/en_US/champion/Aatrox.json), [Jax record](https://ddragon.leagueoflegends.com/cdn/16.13.1/data/en_US/champion/Jax.json) and [item records](https://ddragon.leagueoflegends.com/cdn/16.13.1/data/en_US/item.json) were cached temporarily for this check; no inference was run.

As of this inspection, Riot's current published notes are [26.20](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-20-notes/), while the local audit found 144 LoL catalogs with 14,317 entries labeled 26.6–26.13 and one downloaded video/sidecar pair labeled 26.13. Titles are patch claims rather than independent verification. The old sample can support layout/reference development but is outside the current-patch cohort. Source acquisition and patch verification are still required for that cohort.

## Deterministic reference retrieval plan

The user explicitly requests a cascading Data Dragon → CommunityDragon reference path and model-issued structured knowledge requests. The static resolver already present under `annotation/league/` is the starting point; integration with visual extraction and evaluation remains to be implemented. The following is the intended end-to-end contract, including requirements to verify at integration time.

1. **Freeze the patch manifest.** Record gameplay patch, game mode, Data Dragon build, CommunityDragon snapshot, locale and provenance. Cache and hash responses. Historical-source development uses historical references. Missing same-patch data never selects `latest`, another patch, or PBE silently.
2. **Bootstrap the roster.** Load the complete valid patch roster as internal string ID, numeric ID, display name and vetted aliases. Include this lightweight index in model context. Separately establish the ten champions and player/team bindings from source-visible evidence; a list of all possible champions is not a verified match roster. Preserve mappings such as `MonkeyKing` → Wukong.
3. **Anchor identity in pixels.** Read champion labels, roster/scoreboard portraits, kill-feed icons and selected-unit indicators with timestamps/crops. Explicitly classify text as champion name, summoner/player name, item label, or unresolved text. A summoner name need not match any champion. An unreadable label or missing exact match requests another view/reference comparison or remains unknown; roster membership alone never proves identity. Localized names need a pinned locale/alias policy.
4. **Issue a bounded knowledge request.** Resolve a name/ID against the frozen roster/catalog, then request a specific entity and knowledge fields. Log the OCR text and source frame/crop references separately from the lookup. Only static-reference operations are permitted. A lookup verifies a reference entity, not the preceding visual recognition. Buff lookup needs a supported buff/script identity and champion anchor; a guessed buff name cannot manufacture an observed effect.
5. **Resolve structured data and item headers.** Use structured values first. Extract the item `<stats>` section and parse recognized value-label pairs after removing inner markup and decoding entities. Regex may delimit the known block, but do not collect arbitrary integers from the entire description. Preserve signed/decimal values, percentage units, rank lists and source text. Missing labels, ranges and conflicting descriptions stay explicit; passive numbers are not automatically base item stats. Start with `en_US` rather than assuming English parsing works for all locales.
6. **Resolve supported tooltip variables.** Bind `eN` to `effectBurn[N]` with the designer's one-based indexing preserved. Bind `aN`/`fN` by exact `vars.key`, retaining coefficient arrays and the scaling `link`. A coefficient must remain identified as a coefficient, not become final damage. Preserve legitimate zeros and missing values distinctly. Named variables such as `qdamage`, template modifiers and unsupported expressions remain unresolved until a valid fallback binds them. See [Riot's spell-text documentation](https://developer.riotgames.com/docs/lol#interpreting-spell-text).
7. **Use same-patch CommunityDragon fallback.** Retrieve the matching static client export/converted BIN and bind only within the correct champion/spell/item. CommunityDragon documents scraped game-client/LCU assets and notes that raw data requires additional path lookup and interpretation; this does not guarantee that every field is an active mechanic or a directly evaluable scalar. See [CommunityDragon documentation](https://www.communitydragon.org/docs). Keep calculation graphs and unresolved hashes/formulas explicit. No live client or match-state access is part of this fallback.
8. **Return a compact reference and continue reasoning.** Provide requested facts, their source/patch, units/scaling, conflicts and unresolved fields. Treat description markup as data. Keep the original video and timestamps available for the next reasoning step. Never turn a static capability into an assertion that the champion actually cast, hit, healed, bought or revealed anything in this video.

A proposed visual-pipeline request envelope is:

```json
{
  "entity_type": "champion",
  "query_knowledge": "Smolder",
  "ocr_evidence": {
    "text": "Smolder",
    "text_kind": "champion_label",
    "frame_ref": "source-frame-id",
    "crop_ref": "source-crop-id"
  }
}
```

The frame/crop references must be validated by the visual pipeline, not trusted merely because the model generated them. The controller owns the patch and endpoint selection; the model does not select arbitrary URLs or change patch mid-query. Requests are bounded, deduplicated and cached. The existing adapter has a three-query default; make the final query scope, budget and exhaustion behavior explicit in the evaluation manifest.

Each normalized fact should carry entity ID, field, value/expression, units, rank/scaling basis, source URL and source path, patch/build/hash, resolution status and unresolved dependencies. Keep knowledge completeness separate from visual identity certainty. An explicit static-source conflict is not resolved by model confidence.

Before integration, settle these execution choices; defaults below are recommendations rather than additional user decisions:

| Decision | Recommended initial condition | Why it matters |
| --- | --- | --- |
| Memory under evaluation (accepted 2026-10-19) | Video/question in one bounded multimodal context; no external observation summaries or earlier answers in solver input | Separates the model's visual memory from an external state tracker; streaming memory would be a separately labeled condition |
| Reference access by question tag | Visual-only: roster/alias/icon identity references without kit/mechanics text; mechanics-augmented: requested static facts | Prevents mechanics retrieval from changing what a visual-only score measures; construction-time annotation can use richer references separately |
| Qwen runtime (models/hardware accepted 2026-10-19) | Instruct for HUD; Thinking for gameplay; one L40; pin revisions, precision, serving backend and token budgets during planning | Profile real context/generation memory and throughput; fixed logical roles do not require concurrent model residency |
| OCR/language | Start with one reference locale and an explicit source-text language policy; preserve player names separately | The supplied recording includes Korean player names; English champion lists cannot resolve arbitrary player strings |
| Identity assistance | Discover match roster/player bindings from supplied pixels; report any oracle-provided mapping as a separate condition | Manual lineup injection would otherwise conceal the identification problem being measured |
| Sampling and rereads (routing accepted 2026-10-19) | Fixed layout routes, automatic adaptive review of temporal coverage/density/resolution; count every processed frame/token/call | Keep annotation assistance separate from the declared solver-input condition and include review cost in efficiency metrics |
| Pilot and gold review | Complete games for long horizons plus sampled local windows; hold out matches and independently review difficult examples | Event-only clips cannot establish long-range coverage; model-generated logs are not independent ground truth |
| Failure and acceptance policy | Return unknown/partial at bounded lookup exhaustion; calibrate and freeze numeric/timing tolerances and quality gates before held-out evaluation | Avoids silently changing criteria or using unverified mechanics to fill gaps |

Prototype the reference pack and visual-input audit first, then identity/HUD extraction, bounded retrieval integration, event/state reconciliation and finally long-horizon QA generation. Validate the static resolver independently from Qwen perception. Measure failures by stage: reference missing, OCR/identity wrong, evidence lost in preprocessing, temporal association wrong, or unsupported answer.

Source video: `data/youtube/lol_domisum_aatrox/xICbe-OOPvE.mp4` (1920×1080, 60 fps, approximately 28:56). Inspected sparse samples across the recording and closer samples at video 00:20–00:44 and 12:40–13:30, including one-second samples at 13:06–13:21. Audio was not reviewed. The sidecar reports Aatrox versus Jax, patch 26.13, match KR-8280941174. These are source metadata, not independently verified game state. The title and description disclose final performance and must be excluded from prediction inputs.

## Starting point

Represent a piece of gameplay as **state → action → interaction → observed consequence**, with links to possible local objectives. Construct questions from verified evidence within that representation.

EgoCS-400K provides the architectural reference: replay-derived state and atomic actions, protected action chains, segmentation that preserves those chains, and filtered priors for language generation. Its replay is authoritative; our uploaded video supplies incomplete observations. Preserve the hierarchy and temporal integrity, while recording evidence and uncertainty instead of assuming engine truth. See [§3.2–3.3 and Figure 2](https://arxiv.org/html/2606.18180).

The important addition for League is an objective graph. A minion-clearing action may contribute to farming, lane pressure, and preparation for a recall. An engagement may overlap a turret defense or neutral-objective contest. These relationships can coexist, and the player's intended purpose can remain unknown.

## Inputs we can use

Keep raw evidence, extracted observations, static game knowledge, and interpretation distinct. The same structured record should support retrieval, summaries, temporal localization, and QA.

| Input | Extractable information | Useful questions | Observation limit |
| --- | --- | --- | --- |
| Temporal battlefield frames | Champions, minions, structures, projectiles, animations, visible effects, movement and contact | Who acted on whom? What visibly hit? Who moved away? | A still cannot establish a cast sequence or its outcome; disappearance is not proof of death |
| Selected-champion HUD | Identity, HP, level, abilities, readable cooldowns, visible statuses | What resources were displayed before/after the exchange? | Bind readings to the selected champion; no raw keyboard/mouse inputs are available |
| Scoreboard and sidebars | Champion/team mapping, KDA, CS, inventory, displayed death timers | Who had the CS lead? What changed between snapshots? | Inventory presence does not establish purchase time; CS changes do not identify particular minions |
| Minimap over time | Coarse positions, visible icon arrivals/departures, camera rectangle, structures, pings | Where did the champion move? Which visible allies approached? | Missing icons are unknown; spectator coverage does not establish team/player vision |
| Global HUD | Game clock, displayed team gold, kills and objective counts | What was the displayed lead? What changed during this episode? | Counters and icons require layout calibration and verified semantics |
| Objective timers and messages | Named timers, spawn announcements, kill/structure/objective notices | Which objective was announced? What was the timer before recall? | An objective marked LIVE does not establish that it was being attacked, contested, or still alive |
| Chat, pings and editorial overlays | Displayed communication, item/skill recommendations or notices | What was pinged? What did the overlay report? | A ping can be mistaken; an editorial build plan is not a current inventory |
| Audio, if explicitly included | Announcer cues, spell sounds, commentator statements | What was announced? Does a sound corroborate an event? | Commentary is a claim, and music or overlapping effects may obscure evidence; retain modality labels |
| Patch-specific reference data | Champion/ability/item identities, icons, applicable mechanics | Which named ability matches a verified visual action? | Static knowledge explains possibilities; it does not prove an event occurred |
| Optional matched external records | Event anchors and match context, if a matching record is actually acquired | Does an event align with the video? | Keep external truth separate from what the question solver can observe |

Riot's [Data Dragon documentation](https://developer.riotgames.com/docs/lol#data-dragon) provides versioned champion, item, rune and spell data/assets. Use a verified mapping to the source patch rather than the newest dictionary. Riot documents [Live Client Data and Replay APIs](https://developer.riotgames.com/docs/lol#game-client-api), but these are local-client capabilities; Replay API camera/playback controls are not a CS-style full telemetry parser. No such data was supplied or acquired for this video.

For this recording, route the bottom-left selected-unit panel, bottom-center scoreboard, bottom-right minimap, top match HUD, side rosters, battlefield, and message/overlay areas separately. Calibrate actual regions per layout span rather than treating them as universal LoL coordinates. Preserve synchronized full-frame context and crop-to-source coordinates. The scoreboard changes panel type, and large item/skill/skin overlays can obscure the scene.

## Organize questions by objectives and observable progress

The terminal objective is destroying the enemy Nexus. Structures, resources and neutral objectives provide intermediate context; Riot's [game overview](https://www.leagueoflegends.com/en-us/how-to-play/) explains this progression. The following is our proposed annotation taxonomy, rather than a claim that every action has one identifiable purpose.

| Objective family | Observable evidence of activity/progress | Candidate questions | Stronger claim requiring more evidence |
| --- | --- | --- | --- |
| Gain resources | Minion attacks/deaths, CS and level changes, readable gold or inventory | How much did CS change? What did the champion attack? | Exact gold source; intentional denial; unseen experience distribution |
| Trade or obtain a kill | Participant identities, casts/hits, HP changes, movement, death/feed evidence | How did the exchange end? Who remained visible in lane? | Which player intended an all-in; exact damage attribution without resolved hits |
| Manage the wave | Minion counts/positions over time, deaths, incoming reinforcement, turret contact | Did the allied wave reach the turret? What remained after clearing? | Freeze, slow push, or bounce from one snapshot; intended wave plan |
| Survive or disengage | Threats, health, defensive effects, displacement, retreat route and follow-through | Where did the champion retreat? Did the chase visibly continue? | Globally safe escape; no possible off-camera threat |
| Reset and return | Wave context, recall start/interruption/completion, fountain arrival, inventory/HP changes, return route | Where was recall channeled? Was base arrival shown? | Why the recall was chosen; exact purchase from an overlay alone |
| Control information or space | Ward placement/removal, visible reveal effects, pings, routes and nearby units | What visible ward action preceded entry? Which route was taken? | What the player knew; absence of enemy vision |
| Take or defend structures | Minion arrival, structure attacks/HP, defenders, destruction confirmation | Was the turret destroyed or left standing? What followed the fight? | A kill automatically converted into a turret; deliberate split-push intent |
| Take or contest neutral objectives | Approach, participants, direct monster activity/HP, opposing intervention, result messages | Was there an attempt? Who visibly contested? Was completion confirmed? | Timer availability implies participation; an unseen steal or last hit |
| Coordinate across the map | Linked local episodes, minimap movements, concurrent structure/objective outcomes | What occurred elsewhere during this push? Which action followed the skirmish? | Deliberate objective trade, optimal rotation, or full-team coordination |

Version the objective vocabulary and rules by source patch. Do not hard-code spawn times, rewards, or monster availability from memory.

An objective record needs a lifecycle: **available/reported → approached → attempted → contested if observed → completed, abandoned, interrupted, or unresolved**. Do not collapse those states into one label. Keep a separate `goal_hypothesis` with support and alternatives; leave it null when purpose is unclear.

## Levels of gameplay and temporal coverage

| Level | Representation | Evidence window |
| --- | --- | --- |
| Observation | Entity, field/value, spatial relation, HUD reading | Timestamped frame/crop |
| Atomic event | Attack/cast, hit candidate, movement transition, channel start/end, ward action | Enough adjacent frames to distinguish initiation from outcome |
| Interaction | Trade, chase, minion clear, channel interruption | Setup, actions, response and immediate result |
| Objective episode | Recall/reset, turret attempt, neutral-objective contest | Relevant preconditions through confirmed or unknown resolution |
| Strategic sequence | Several episodes, possibly concurrent | Wider map context and linked local evidence |

Protect the evidence span needed for the question: cast → effect → target response; approach → exchange → disengagement; recall channel → endpoint; objective attack → result. A question about resetting may additionally need the preceding wave and subsequent inventory state. Use overlapping views of longer episodes instead of forcing every question into one fixed clip length.

Use low-rate HUD/minimap sampling to discover candidate changes, then inspect denser temporal windows for actions and outcomes. Exact hit counts or action order can require native-frame review. Sampling rates should be measured in a pilot, and unresolved events should be rejected or marked unknown.

## A concrete episode from the supplied recording

All times below are **video time**. At 13:00, the displayed game clock is 14:00. This is one alignment anchor, not a verified global offset.

| Evidence | Reading or visible activity |
| --- | --- |
| 13:00 battlefield | Aatrox among red minions near the red top-lane turret, with blue minions approaching |
| 13:00 selected HUD | Level 12; HP 1326/2724 |
| 13:00 scoreboard | Aatrox 4/0/1 and 115 CS; Jax 0/5/1 and 89 CS; calculated CS difference +26 |
| 13:00 global HUD | Blue 17 kills, red 11; displayed gold 28.5k versus 25.6k |
| 13:00 timer panel | Rift Herald 1:00; Cloud Drake LIVE |
| 13:00–13:07 sampled sequence | Aatrox attacks remaining minions, then moves toward lane-side brush |
| 13:09–13:15 sampled sequence | Recall-like channel/ring and CHANNELING label in brush |
| 13:16 sampled frame | Aatrox at blue fountain |
| 13:18–13:21 overlays | Sundered Sky, then Pickaxe editorial notices; use inventory evidence before asserting purchases |

This yields several question types from the same episode:

| Task | Question | Answer/status | Required input |
| --- | --- | --- | --- |
| State comparison | At video 13:00, how much more CS does Aatrox have than Jax? | 26 | Scoreboard and champion-row mapping |
| Spatial relation | Which team's top turret is nearest to Aatrox at video 13:00? | Red team's | Battlefield context and verified team mapping |
| Cross-panel context | What does the Rift Herald timer show while Aatrox is clearing this wave at 13:00? | 1:00 | Battlefield plus synchronized timer panel |
| Temporal understanding | What does Aatrox do between clearing minions and appearing at the fountain? | Moves into brush and channels recall | Ordered temporal evidence, not only endpoints |
| Outcome verification | Is a fountain arrival shown after the brush channel? | Yes, by the 13:16 sample | Channel and destination frames; completion time is bounded, not exact |
| Bounded explanation | What visible conditions could support a reset here? | Partial HP and wave activity are observed context; strategic explanation requires a rubric and alternatives | Pre-recall state and wave sequence; label as interpretation |
| Observability | Can this view establish that Aatrox recalled to prepare for Herald? | No; the timer and recall coexist, but intent is not established | Scope/visibility record and explicit unknowns |

Questions above are grounded proposals after sampled inspection, not a complete frame-accurate gold set. In particular, do not label the wave a verified crash without seeing the relevant minions reach and engage the turret.

## What the question generator should receive

Provide an evidence packet rather than only a free-form caption:

```text
source: video ID, patch claim/provenance, view/layout spans, time mapping
scope: spectator-visible; selected champion; team mapping; visibility gaps
window: question context interval, answer evidence interval, prediction cutoff if any
entities: stable IDs, champion/team, identity evidence
state_before / state_after: values, timestamps, freshness and unknown fields
events: actor, action, target, bounded times, response, observed outcome
relations: before, overlaps, near, approaches, part_of, followed_by
objective: candidate family, lifecycle state, hypothesis and alternatives
evidence: source frame/crop IDs supporting each assertion
question_policy: task type, permitted modalities, knowledge assumptions, answer rubric
```

Every field needs provenance: `observed`, `derived` (for example, 115 − 89), `inferred`, or `unknown`. A readable message can verify an off-camera event without showing its mechanics. Repeated model readings of the same pixels are not independent corroboration. Selected champion, camera focus and controlled champion are separate identities.

For each question, save the answer, evidence references, temporal bounds, minimum required modalities, knowledge dependencies, acceptable alternatives and rejection reasons. Ground language in the event graph, then verify against the original footage rather than using a generated caption as its own answer key.

## Question families and controls

Start with factual state, comparisons, action/target recognition, spatial relations, event ordering, temporal localization and observed objective outcomes. Add broader summaries and cross-map associations once continuity and coverage are reliable.

Keep retrospective understanding, future prediction and strategic judgment separate:

- **Retrospective:** the solver may see the full question window and answer what occurred.
- **Prediction:** the solver sees only a prefix ending at an explicit cutoff. Future footage may establish the target but cannot enter the prompt, summaries, priors or metadata. One observed continuation is not the only plausible future.
- **Strategic judgment/counterfactual:** use expert criteria, assumptions and acceptable alternatives; one recorded outcome cannot establish the optimal decision or what would have happened otherwise.

An answerable question should identify its temporal and perspective scope, have sufficient visual coverage, and require the intended evidence. Reject misleading presuppositions such as “Which turret did he take?” when only an approach is visible. Unknown/insufficient-evidence examples should be deliberate, not the default answer to most questions.

Balance easy HUD questions with temporal and spatial reasoning. Include farming, holding position, resets, failed attempts, retreats and unresolved plays, alongside successful combat. Test whether a single frame, HUD-only input, shuffled frames or source title can solve nominally temporal questions. Use event-consistent distractors that are contradicted by evidence, not merely absent from view. Split related clips/questions by match and consider channel/layout holdouts to measure shortcut learning.

For a first pilot, manually annotate roughly a dozen episodes across a trade, farming sequence, reset, structure attempt and neutral-objective attempt where those are actually visible. For each, build state, sequence and outcome questions plus a small number testing observability. Review agreement on evidence and answerability before scaling question generation.

## Longer-horizon questions

Long-horizon understanding should be a primary target alongside local questions. Duration alone does not define it: a question needs multiple separated evidence intervals, and its answer must depend on comparing or connecting them. A 20-minute clip with a question answered by its final scoreboard is still a snapshot task.

The following are candidate templates, not established facts about the supplied match. Horizon ranges are design suggestions; choose actual boundaries around events.

| Horizon / family | Candidate question | Evidence needed |
| --- | --- | --- |
| 1–3 minutes: preparation through follow-through | After clearing top and recalling, where did Aatrox go next, and what did he visibly participate in? | Pre-recall wave, completed reset, return route, next local episode; sufficient coverage to identify the next event |
| 3–8 minutes: advantage progression | How did the Aatrox–Jax CS and level gaps change across three lane visits, and what activity occurred between those snapshots? | Repeated synchronized comparisons plus intervening farming/combat/movement evidence |
| 2–5 minutes: conversion | Across the observed kill sequences, which were followed within 60 seconds by a structure attempt, a recall, or continued farming? | Kill anchors, subsequent event chains, explicit coverage and unknowns; exhaustive classification requires complete relevant coverage |
| Several minutes: concurrent map outcomes | During Aatrox's sustained top pressure, what structure or neutral-objective outcomes were confirmed elsewhere? | Local activity, minimap history and global result notices, aligned in time; these do not alone prove a deliberate trade |
| 5–15 minutes: repeated behavior | How did Aatrox's response to Jax's lane appearances differ before and after a verified item change? | Several supported encounters on each side of a verified inventory transition; comparison, not an asserted item-caused effect |
| Phase or match: change of activity | When did the observed activity shift from mostly top-lane farming to repeated grouping, and which episodes support that description? | Coverage over both periods, coarse location/activity records and several witness clips; tolerate a transition interval |

Descriptive questions can be demanding without asking about hidden intent. Separate chronology, association, mechanically supported consequences, and strategic causality. For example, a kill followed by turret damage establishes sequence; attributing the turret opportunity to that kill additionally needs defender absence/death evidence, wave context and coverage of alternatives.

Maintain three memory levels:

1. **Observation/event ledger:** timestamped readings, action evidence, spatial relations, provenance and gaps.
2. **Episode records:** participants, start/end conditions, goal hypothesis, objective lifecycle and observed result, each linked to the ledger and source frames.
3. **Long-window index:** location/activity intervals, repeated state comparisons, objective ownership changes and references to episodes. This is a retrieval aid, not an independent source of truth.

Keep source evidence references through every reduction. Preserve failed attempts, uneventful periods and missing coverage; a memory containing only highlights will distort strategic questions. Do not repeatedly summarize prose until only a plausible story remains.

To generate long-horizon questions:

1. Retrieve candidate patterns in the ledger, such as recall → return → objective attempt, repeated state snapshots, or simultaneous local pressure and global outcomes.
2. Require at least two separated evidence intervals that materially affect the answer. Their spacing and relations should be explicit; the minimum number of intervals is a configurable task rule.
3. Assemble the relevant episodes, intervening coverage records and witness clips. Reread source footage where a transition or outcome is missing.
4. Generate the question and answer with evidence for each answer clause. A failed evidence check yields a narrower question, an intentional unknown example, or rejection.
5. Verify against source footage, and test whether one frame, the endpoint scoreboard or a title can answer it. Use shuffled-order tests only for tasks where temporal order is necessary.

Distinguish the generator's logging inputs from the solver's evaluation inputs. If testing video understanding, giving the solver a log that already states the answer instead tests reading that log. The generator may use verified annotations to find questions and answer keys while the solver receives the corresponding continuous video window. If the solver instead receives retrieved clips, label the task as understanding selected evidence; it cannot establish what happened in omitted intervals.

## Region-focused logging with Qwen3-VL

The selected annotation models are Qwen3-VL-8B-Instruct for HUD routes and Qwen3-VL-8B-Thinking for gameplay, on one L40. Revisions, precision, serving stack and practical context limits remain planning choices. The latest fixed-route decision supersedes dynamic actor/target-crop suggestions in the exploratory table below. LoL annotation accuracy is not yet measured.

Use explicit image/video crops and bounded extraction tasks to control the supplied evidence. A prompt saying “focus on the minimap” does not remove other pixels, reduce their processing cost, or guarantee exclusive attention. Start with three logical extraction roles plus a periodic context view. These can use the same Qwen checkpoint and do not require separate agents or models.

| Role | Input for this recording | Output contract | Scheduling proposal |
| --- | --- | --- | --- |
| State and objective logger | Selected HUD, scoreboard, top HUD and objective/message crops, individually labeled with timestamps | Selected identity, readable values/deltas, inventory observations, notices, uncertainty | Try 1 Hz discovery initially; use cheap change detection to request model reads; reread before/after important transitions |
| Spatial logger | Minimap crop sequence plus occasional full-frame context | Resolved identities, coarse regions, arrivals/departures, visible grouping, camera rectangle, last seen and ambiguity | Try 1 Hz discovery; increase around movements/convergence; unresolved icons remain tracks |
| Local interaction logger | Temporal battlefield context, dynamic actor/target crop, synchronized relevant HUD observations | Actor/action/target, bounded order, response and observed outcome | Trigger clips around HP changes, nearby opponents, feed events, channels and other candidate activity; inspect denser frames as needed |
| Context and continuity check | Full frame at readable but economical resolution; view/layout transitions | Camera focus, layout/panel type, overlays, relevant off-center activity, coverage flags | Periodic checks and every suspected view/layout change; retain all source frames for retrieval |

These schedules are pilot hypotheses. One hertz is insufficient to establish every cast, minion last hit, minimap appearance or absence of an event. Maintain coverage metadata and increase sampling for specific questions. Include a random sample of apparently quiet periods in evaluation to measure events the triggers miss.

For local clips, start by testing roughly 5–15 seconds with lead-in and follow-through, increasing temporal sampling around fast actions. Longer fights or delayed outcomes need linked continuation clips. Do not infer that a low-rate clip contains every action between its sampled frames.

Qwen3-VL's [official inference guide](https://github.com/QwenLM/Qwen3-VL#quickstart) supports multi-image/video inputs, sampling controls and separate image/video pixel budgets. For the documented official processor, the video budget covers all frames, so frame count and spatial detail compete. Its preprocessing path also distinguishes Qwen3-VL video metadata and warns against duplicate resizing when using `qwen-vl-utils`. Pin the actual processor/tool versions and inspect the tensors, sampled timestamps and rendered inputs used by inference.

Practical input policy:

- Send low-resolution context alongside high-detail relevant crops. Preserve source-time labels, identity, crop rectangles and transforms. Small text must remain readable after preprocessing, not merely in the saved crop.
- Keep HUD/minimap crops at native detail when possible. Enlarging a tiny icon cannot recover missing information. Extra crops, minimum-resolution upscaling and multiple calls may erase expected token or latency savings.
- For actions, include the target and effect area with the actor. A tight Aatrox-only crop cannot establish what was hit. Follow actors through the viewport rather than assuming all action occurs at the center.
- Keep temporal crops stable, or record their changing transforms. Crop motion and camera panning must not become champion movement in the log.
- Use ordered, timestamped inputs; preserve actual timing for non-uniform samples instead of representing them as a uniform-rate video. Verify the serving path retains timing metadata, or use explicitly timestamped image sequences.
- Give each extraction pass a small field vocabulary and strict JSON. Require evidence times, identity confidence, unreadable/null values and unresolved alternatives. Avoid a full prose narrative for every frame.
- Treat prior logs as hypotheses with provenance. A new view can contradict earlier identity or state readings; do not force the model to agree with accumulated memory.

For the inspected 13-minute reset, the state logger supplies HP, CS and timers; the spatial logger anchors top lane → brush → base; the interaction logger supplies minion attacks and the recall sequence. A longer-window question would then retrieve the subsequent route and participation evidence. Those later events have not been established by the current inspection.

Evaluate a full-frame baseline against crops-only and crops-plus-context on the same manually reviewed windows and comparable visual-token budgets. Measure numeric accuracy, identity consistency, event precision/recall, temporal ordering, missed-event rate, answerability of resulting questions, tokens, latency and human correction time. Adopt focused routing if it improves the relevant quality/cost tradeoff; cropping is a hypothesis to test rather than a guaranteed improvement.
