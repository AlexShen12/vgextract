# Selected-player caption pilot validation — November 2, 2026

The pilot is playable and remains experimental. The complete recognition acceptance
gate **has not passed**. Counter claims passed independent review, but those results
cannot establish attack, target, last-hit or ability-impact accuracy. No pilot
evidence was promoted to production or QA.

## Software and playback

`python -B -m pytest -q -p no:cacheprovider tests`: **254 passed**. The existing
25 caption tests remain included. Checks exercise real pilot orchestration with
injected perception, compatible-cache reuse without loading a GPU, deterministic
IDs, missing-artifact reconstruction, stale-request isolation, partial GPU failure,
source/owner/target grounding, weak precursor retention, correction invalidation,
independent assessments, native PTS quantization, overlapping packets, pauses,
cuts, continuations and replay correspondence. Synthetic temporal cases do not
establish real-footage accuracy.

Run `python -B -m annotation.league --offline review --port 8767` and open
[caption review](http://127.0.0.1:8767/captions). Source-time clips seek the original
1920×1080 MP4; claim evidence buttons seek cited frames and show available binding
regions. Clips, corrections, omissions, continuations and replay links are separate
from production review. Generated wording retains its unreviewed status even when
a separate independent assessment is recorded.

## Independent references and counter benchmark

The separate agent `independent-agent:caption_reference_labels` received source-only
instructions without prediction or implementation access before reference labeling.
It completed ten development and ten held-out eight-second neighborhoods: 160 source
seconds, 117 conservative reference claims, 161 binding annotations and one explicit
negative neighborhood. These are approximate, non-exhaustive annotations from 4 FPS
inspection with native-frame checks, not frame-exact exhaustive ground truth.
Labeler provenance and method are retained under
`data/league_runs/midbeast-OLLZBe5-fg4-v2/benchmarks/captions/`.

The original execution exposed an unsupported-layout identity indexing bug. It is
retained as a failed experiment. After fixing the bug, a new development freeze was
created under `benchmarks/captions/validation-r2/`; the earlier freeze was preserved.
The completed CPU/cache-only benchmark made **zero GPU requests** and emitted:

| Scope | Independently supported / reviewed | Claim kinds |
|---|---:|---|
| All twenty neighborhoods | **67 / 67** | 49 displayed states, 18 CS increases |
| Ten held-out neighborhoods | **36 / 36** | 30 displayed states, 6 CS increases |

The reviewer checked wording, structured bindings and evidence, including 84 unique
cited native images matched to independent source decoding. Claims used 51 anonymous
selected-champion scopes and 16 HUD-slot scopes overall. No target claim was emitted.
Held-out actor/evidence assessments were 36/36 correct; the target denominator has
no actual target assertions and must not be interpreted as target recognition accuracy.

Held-out CS reference matching was 6/6 precision and 6/8 recall, with zero matching
duplicates. Reference matching is a proxy: unmatched displayed states are not
unsupported merely because the conservative reference set omitted them. Manual
independent assessment supplies emitted-claim support. Clip phase retention was
14/33 and playback-reference matches 10/17; these are limitations, not successful
context/playback validation. Repeat-link accuracy is untested (0 reference links,
0 predicted links). Attack/cast/target/last-hit/ability-hit predictions were absent,
so their reference recall is zero and precision/support remain untested.

Immutable execution and assessed reports:

- `experiments/captions/benchmarks/5badacbe3b797b02bedf.json`
- `experiments/captions/benchmarks/5badacbe3b797b02bedf-assessed.json`

The latter explicitly reports `passed: false`. CPU pipeline time across the twenty
neighborhoods was 1,218.23 seconds; maximum reported host RSS was 617,693,184 bytes
(driver process only). Subsequent grounding and clip-phase fixes are development
changes and do not inherit this historical benchmark's results. Its source is
archived at `data/league_caption_pilot_validation/counter-validation-r2-source.tar`.

The original labels used free-form case tags such as `normal_clock_ticks`,
`counter_only_change` and `minion_last_hit`. The historical scorer expected different
canonical names, so its `missing_cases` includes some naming mismatches. Human case
inventory confirms ordinary ticks, counter-only increases, ambiguous impacts,
competing targets, rapid attacks and occlusion. Multi-packet action labels do not
establish processing-chunk boundary correctness. Actual positive pause, rewind,
repeat and ownership-change cases are absent. A cast without an attributed impact
is not automatically a visibly missed cast.

Current label validation also requires explicit per-category negatives for empty
annotation arrays; unknown and uninspected empties cannot pass. The source labeler
created a separate `benchmarks/captions/source-only-complete-r3/` supplement: all 22
empty-category flags remain false because category-wide negatives were not
established. In particular, it did not exhaustively search earlier source footage
for repeats. `casts_missing` denotes uncertain cast attribution, not missed casts.
The historical freeze is retained unchanged rather than silently backfilled.

## Pinned Longleaf development inference

Runs reused the existing source MP4, icon references, driver/VL environments and
immutable Qwen3.8-27B-FP8 model pin. Each job requested one L40S, eight CPUs and 96 GB
host memory. Source snapshots are isolated under the existing project's
`artifacts/caption-pilot-20261102-r*/source`; production/action tables and historical
checkouts were preserved. These are development experiments, not held-out evidence.

| Revision / Slurm job | Source seconds | Pipeline seconds | Requests | Generated / prompt tokens | GPU peak bytes | Result before independent review |
|---|---|---:|---:|---:|---:|---|
| r1 / 7924066 | 34–38 | 730.43 | 4 | 4,432 / 21,255 | 39,624,638,464 | 3 counter captions; one truncated response rejected |
| r2 / 7929115 | 34–38 | 968.92 | 9 | 5,918 / 47,050 | 39,786,119,168 | 3 generic attacks plus 3 counter captions |
| r3 / 7930808 | 34–38 | 875.14 | 13 | 4,943 / 70,424 | 39,695,941,632 | 2 counter captions; one truncated response rejected |
| r4 / 7933969 | 34–38 | 1,342.33 | 15 | 12,916 / 69,737 | 39,607,861,248 | Native actor-crop prototype: 8 generic attacks plus 2 counter captions; two incomplete responses rejected |

Compact request-local aliases made responses parseable, but emitted attacks and
localized targets still require independent review. Mainline validation now rejects
a generic attack/cast carrying an unqualified target and limits ability slots to
cast/hit claims. A target/outcome requires its own visible evidence; CS income
cannot resolve a last hit. Model abstentions and incomplete responses remain
explicit omissions. Native-crop experiments are retained separately and have not
been automatically adopted into the frozen or production pipeline.

Raw metadata is under `data/league_caption_pilot_validation/longleaf-r*/`. Complete
r2/r3 evidence roots were copied locally and image hashes checked. Local projection
records and original remote-metadata archives preserve provenance. Independent
scene review, where available, is stored in each pilot's `assessments/` separately
from generated claims.

Independent assessment of the original r2 smoke output supported only 3/6 complete
claims: its three attack rows carried boxes on empty ground, and two also lacked
a clearly separate attack. Source-based revisioned corrections reject those two
actions and remove the unsupported recipient of the remaining visible attack.
Regeneration also suppresses the duplicate HUD point. The corrected three-claim
clip has **3/3 independently supported** claims (one displayed CS value, one
generic attack, one CS increase), with no target or last-hit assertion. Original
and corrected reviews are retained as `development-review.json` and
`development-review-corrected.json` in pilot `33d5dd113378e5b7037d`. This corrected
development example is explicitly ineligible for benchmark acceptance.

## Unsupported and release scope

Whole fights, escape judgments, tower/objective outcomes and MP4 export remain
outside scope. Production/QA isolation is exercised by tests. Real-footage target
binding, ability-impact chains, individual last hits, chunk boundaries, pauses,
rewinds, repeats and ownership changes still need sufficient independently labeled
and assessed cases. Passing unit tests, successful inference, nonempty generic
attacks or 100% counter support does not satisfy the full 95% pilot gate. Any future
release decision applies only after a fresh frozen evaluation for this source and
profile; general production enablement remains separate.
