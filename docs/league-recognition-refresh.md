# Reference refresh, cast bars, and body identity

The new catalogs and experimental outputs are isolated from saved runs and staged
Longleaf jobs. Footage gameplay patches remain unknown. Descriptions show their
catalog version and never establish historical mechanics.

## Current local state

- Data Dragon **16.22.1** is cached for all **173 champions** and **488 inventory
  references**. CommunityDragon **16.22** has **4,180 skin portraits**, plus separate
  English ability-description supplements. Assets have hashes and retrieval provenance.
- DINOv2 ViT-S/14 is cached under `data/league_assets/body-encoder/dinov2-vits14`.
  Its exact code commit is `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; checkpoint
  SHA-256 is `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- `midbeast-1080p-inventory-v2` is a candidate profile. The fresh-run default remains
  v1 until the verified comparison, held-out intake and runtime gates pass.
- A paired diagnostic over **283 previously inspected inventory labels** produced
  identical aggregate old/new counts. Draft labels include known conflicts and
  unreadable-slot errors. This is not a held-out accuracy or promotion result.
  See `data/league_recognition_v1/catalog-comparison/comparison.json`.
- The cast dataset is seeded with **three prior model-reviewed occurrences in two
  development clips**. Labels remain non-exhaustive and unverified. Only two MP4
  recordings are downloaded; the campaign still needs six underlying matches,
  three layouts, 12 champions, 300 occurrences, 150 negatives and human evaluation.
- The body pilot has five source-reviewed champion candidates. The other five
  identities, complete galleries and 300 labeled track segments remain gaps.
- Real Darius base-asset discovery retrieved a mesh but the referenced skeleton
  returned 404 at the public export path. The skin is recorded as unsupported.
  Blender/lol2gltf are not installed locally. No rendered gallery or GPU body
  quality result is claimed.

Data directories are local experiment artifacts and may be ignored by Git.

## Reference preparation and promotion

```sh
.venv/bin/python -B -m annotation.league cache-reference-bundle \
  --reference-version 16.22.1 --communitydragon-version 16.22
.venv/bin/python -B -m annotation.league --offline cache-reference-bundle \
  --reference-version 16.22.1 --communitydragon-version 16.22
```

`cache-abilities` and `cache-inventory-references` also accept an optional
`--communitydragon-version`; `cache-skin-references` prepares portraits and
supplements. Literal `latest` is resolved only by preparation commands, with the
discovery metadata archived alongside concrete pins. Offline preparation requires
explicit versions. Recognition never fetches latest.

Complete catalogs are reused and integrity checked. Incomplete skin downloads
produce `partial.json`, not a usable manifest. URL receipts permit partial retries.
Prior 16.13.1 catalogs and historical mechanics configs remain intact. Primary
Data Dragon descriptions and CommunityDragon supplements remain separate.

Portrait matching compares each champion's best skin variant against other
champions. Two skins of one champion do not consume the identity margin. Skin
identity remains nullable, including artwork ties and grayscale respawn matches.
Existing hidden-portrait rules, red masking and item alignment retries remain.
Runtime/cache signatures include the skin catalog hash.

`reference_rollout compare` accepts a JSON object with `cases` and configured
`masks`. Each case supplies `id`, native `image`, `sha256`, `field_id`,
`availability`, `value`, `reviewer_kind`, `source_only` and `verified`.

```sh
.venv/bin/python -B -m annotation.league.reference_rollout compare \
  --cases data/league_recognition_v1/catalog-comparison/cases.json \
  --output data/league_recognition_v1/catalog-comparison
```

Promotion requires a human-verified comparison, frozen profile, matching
`intake_benchmark.json` and existing five-minute release pilot receipt. The
`reference_rollout promote` command takes `--comparison`, `--intake-gate`,
`--pilot-gate`, and the fresh-run `--config`. It checks profile, catalog, reader
and extraction identities before writing a hash-named released profile. No saved
run config changes. Compare with the frozen profile path when thresholds change.

## Cast-bar reader and review

`action-window` accepts opt-in `--cast-bars --ability-catalog PATH`. This changes
the experiment identity. A frozen scene-only baseline rejects the addition.
It does not add caption claims or change ordinary 1 FPS inventory sampling.

For independently calibrated native layouts, including other resolutions:

```sh
.venv/bin/python -B -m annotation.league.cast_bar_intake \
  --run-dir data/league_recognition_v1/cast-source-run \
  --video data/youtube/lol_midbeast/OLLZBe5-fg4.mp4 \
  --profile annotation/league/profiles/midbeast-cast-bars-1080p-v1.json \
  --catalog data/league_assets/abilities/16.22.1/catalog.json \
  --start 1660.3 --end 1662.3
```

A cast profile contains native `size`, `text_box`, `progress_box`, `search_box`,
`scene_box`, `clock_box`, `identity_box`, masks, calibration provenance and two UI
anchors. Every layout/resolution needs its own calibration. The reader searches
nearby thin cyan bars with dark framing because cast and Recall bars can occupy
different rows. Neutral bright-glyph preprocessing reduces colored gameplay
interference; a second bounded OCR view handles failures. Other bar styles remain
unvalidated collection gaps.

At 16 FPS the reader preserves native text/progress crops, source timestamps,
verbatim OCR, visibility, owner/selection epoch, playback segment, catalog ID and
optional ability ID. Exact normalized names and reviewed aliases precede fuzzy
matching. Fuzzy matching is disabled without a development-frozen score/margin
policy. Conflicting champion/text evidence and ambiguous names stay unresolved.
Optional standalone HUD identity inputs need accepted same-frame source hash/time
citations. Unique exact names can resolve while champion identity is unknown.

Outputs inside the isolated experiment directory are `cast_bar_readings.jsonl`,
`effective_cast_bar_readings.jsonl`, `channel_occurrences.jsonl`,
`channel_timeline.jsonl`, native crop caches, description catalogs and append-only
reviews. Channels never infer a hit, target or completion from disappearance.
Brief unreadable gaps preserve one channel; absence, selection changes, cuts,
rewinds and sampling gaps prevent carryover. Cast spans and cooldown onset remain
separate observations.

Open **`/cast-bars`** in the existing review app. It provides source playback,
frame stepping, reader evidence, descriptions, field corrections and a separate
source-labeling view. Corrections rebuild occurrences and timeline entries.
The source-label endpoint never reads recognition predictions.

The saved native 32-frame fight diagnostic resolves Apprehend to `Darius:E` and
Noxian Guillotine to `Darius:R`. Known Apprehend/Recall fixtures also pass. These
are development regressions, not independent accuracy measurements.

## Cast dataset and acceptance

```sh
.venv/bin/python -B -m annotation.league.recognition_dataset coverage \
  --root data/league_runs/midbeast-OLLZBe5-fg4-v2/benchmarks/cast-bars/v1
```

Use `init` for a new root, then `add-source --input JSON`. Input contains `source`
and `clips`. A source has ID, path/hash, underlying match group, preassigned split,
layout ID and prior-exposure flag. Clips have ID, source ID, interval, case tags
and calibrated text/progress boxes. Duplicate hashes and underlying matches cannot
cross splits. Exposed footage is development-only. Assign three matches to
development, one to validation and two to held-out before drafting labels.
Source preparation/import records portable exposure envelopes understood by the
existing interval-exclusion code, so these inspected windows cannot silently
become fresh held-out examples in later scene campaigns.

Use `draft --clip ID --vl-python PATH --cache PATH` for source-only local model
proposals, or `import-label --clip ID --input JSON` for reviewed revisions.
Drafts preserve model responses and source PTS, use 8 FPS source views, and always
remain non-exhaustive and unverified. Native-frame review is needed for precise
boundaries. A model cannot mark labels human verified.

Events record native regions, text, visibility, selected owner, nullable
champion/ability ID, start/end, explicit censoring and terminal status. Labels
record reviewer kind, source-only review, exhaustive scope and human verification.
Frozen labels require a new revision. Required cases in `recognition_dataset.CASES`
include named casts, sustained channels, charged abilities, both recall endings,
partial visibility, clipped boundaries, repeated casts, absent bars, costs,
resources, notices, scoreboard overlays and Apprehend/Recall confusion.
Verified tags must agree with event semantics; repeated casts require a repeated
ability, and recall endings require the corresponding observed ending label.
Use `bar_type: channel` or `charge` for those source judgments. Negative UI cases
also require `case_evidence` with case tag, source interval, native box and a
source observation note; placeholder tags cannot satisfy coverage gates.

`cast_bar_benchmark freeze-reader` takes reader/catalog IDs, policy and development
occurrence predictions. It computes development metrics rather than trusting a
pass flag. A prediction envelope contains `reader_signature`, `catalog_id` and
`occurrences`, each with clip ID, owner, times, kind and nullable ability ID.
Evaluation requires the same frozen reader/catalog and records each split once.
For multiple layouts, provide `reader_signatures` keyed by source layout ID in
both the freeze and prediction envelope; their aggregate hash is the shared
reader contract. Each intake report exposes its individual reader and catalog IDs.

Promotion requires 98% exact-ID precision, 80% readable named-bar coverage, 95%
kind accuracy, at least 50 held-out occurrences, five held-out examples of each
case, the collection targets and human-verified exhaustive held-out labels.
Reports include timing errors, duplicates, false events per negative minute,
denominators and Wilson intervals. Missing categories and empty outputs cannot pass.

## Body campaign

Initialize with `python -m annotation.league.body_campaign init --root PATH`, then
complete `campaign.json`. Clips are at most twelve seconds and require source
hash, underlying match group/split, calibrated action profile, and explicit
source review of playback/selection continuity. Optional `config_path` supplies
accepted same-frame portrait/body context. Roster candidates require HUD evidence.
Gallery matches cannot overlap development, validation or held-out matches.

Gameplay references provide image/mask paths and hashes, champion, optional skin,
identity evidence and reviewer provenance. Human-approved entries use
`reviewer_kind: human, verified: true`. Model-reviewed entries retain
`verified: false`, with `source_reviewed: true`, `identity_approved: true` and their
identity evidence. Model approval never becomes human verification.

```sh
.venv/bin/python -B -m annotation.league.body_assets \
  --skins data/league_recognition_v1/body-campaign/base-skins.json
.venv/bin/python -B -m annotation.league.body_encoder \
  --root data/league_assets/body-encoder/dinov2-vits14
```

Skin requests name champions and optional numeric skin IDs. Non-base footage
skins require source identity evidence. Discovery uses pinned BIN exports and
preserves unsupported assets/errors. Complete authorized client assets can also
be supplied as a render manifest: each skin has champion, skin ID, source
provenance and hashed `files`, with roles `mesh`, `skeleton`, `texture`, `animation`.
Textures specify material names; single-material models support automatic binding.
Complex materials fail instead of silently rendering an untextured skin.

`body_render --input MANIFEST --converter PATH --blender PATH --output DIRECTORY`
pins both executables by hash. CPU Blender renders alpha masks at eight headings,
45°/60° elevations, and four phases of available idle/movement/attack/cast actions.
Admit renders only after `body_render --input render-report.json --audit REVIEWS
--output gallery.json`. Each review names champion/skin, reviewer, reviewer kind
and `approved: true`. The audit rechecks all render and mask hashes.

The staged workflow is `body_campaign prepare`, `preflight`, `qualify`, `track`,
`embed`, `freeze`, `run`, `score`. Preparation saves native frames, hashed SAM
JPEGs and anonymous friendly/enemy health-bar proposals. First-frame proposals
are seeded in groups of at most three. Later entrants and missing proposals
remain localization misses. This pilot does not claim complete scene tracking.
Explicit clip boundaries prevent playback/selection carryover.

Preflight checks independent seed-mask overlap, SAM resources and a DINO embedding.
Per-crop embedding caches support recovery. Encoder code, weights, preprocessing,
device class and package versions participate in cache identity. Appearance inputs
exclude background, HUD and detected overhead labels. Source HUD context is
separate from appearance inputs.

Freeze score/margin thresholds on development labels before comparison. The
bounded search requires 98% precision and 70% coverage on both teams using the
combined appearance gallery. Labels, gallery and embeddings are hash-bound.
Six arms compare gameplay/rendered/combined galleries, each with appearance alone
or accepted roster/HUD plus tracking context. All share proposals, tracks and query
embeddings. Three agreeing identities within five 4 FPS observations are required
to name a track; conflicts reset to unknown. Scores are not probabilities.

Labels contain sample ID, segment label ID, split, team, visibility, nullable
champion/skin and review provenance. Keep localization failures in the denominator.
Targets are 300 segments, 20 per pilot champion and 50 negative/out-of-gallery
segments. Human-verified held-out labels are required for qualification. Reports
separate arms/teams, exact IDs, coverage, unknown false matches, identity switches,
skin results and uncertainty. Adjacent samples are correlated; sample Wilson
intervals do not establish independent trials. No body result enables production.

## Longleaf isolation and budget

No new jobs are submitted. `setup_body_longleaf.sh` creates a new environment while
reading existing GPU dependencies without modifying their environments. Stage a
new source snapshot, `SOURCE_SHA256SUMS`, and `body-campaign-path.txt` containing
the new campaign's absolute path under `/work/users/a/l/alshen`. Rebase manifest
and encoder-lock paths, verify hashes, and prepare there. Never reuse the SAM root.

Separate scripts cover preflight, CPU qualification, tracking, embeddings and all
six comparison arms. Submit only as a separate execution step. Qualification
precedes tracking; tracking precedes embedding, development freezing and comparison.

The ledger enforces an **eight GPU-hour ceiling** with one hour reserved for
explicit recovery. Every attempt reserves its full allocation, including failures;
interruption cannot silently refund it. Preflight, tracking and embedding each
request at most 30 GPU minutes on one L40S, eight CPUs and 64 GB, provisional until
measured preflight. The six matching arms reuse embeddings and request CPU only,
each capped at one hour, avoiding GPUs for NumPy retrieval. Recovery requires an
explicit stage invocation with `--recovery`; no automatic submission loop exists.

Preserve partial artifacts, failed gates, collection gaps and historical reports.
