# Selected-player grounding roadmap

## Plan 1 — SAM body segmentation and tracking

Compare independently placed source prompts with automatic prompts from the
existing selected-player body binder. Reuse the HUD pipeline before adding any
recognition logic. The default `midbeast-1080p-inventory-v1` profile already routes
portraits through `InventoryReader`, its pinned catalog, source-calibrated selected
portrait, unknown handling, cache and correction-aware inventory observations.
The action pipeline supplies frame provenance, ownership epochs, playback context,
bar/level-cell geometry, HUD-health agreement and confirmed body bindings.

The experiment uses the first twelve seconds of five existing development
neighborhoods, IDs 00, 01, 02, 03 and 06. Source-only labels supply the selected
body prompt and sparse visible-body masks. Both arms use the same source frames
and prompt timestamp. Failed automatic seeds remain failures in coverage counts.
SAM receives a body point, with anonymous object identity. Existing portrait
evidence supplies a separately recorded champion name when accepted.

Use pinned SAM 3.1 weights and code in an isolated environment, with 16 FPS
native-coordinate frames. Keep the original clip timeline and stop at playback
or ownership barriers. Full occlusion remains unknown visible-body evidence.
Report seed coverage, sparse mask IoU, missing predictions, visible/occluded false
positives, corrections, resource use and untested cases. Mask quality does not
establish champion naming, action recognition or outcomes.

The first Longleaf campaign has a four GPU-hour allocation cap. Separate scripts
request thirty minutes for preflight and one hour forty-five minutes for each
comparison arm, with one L40S, eight CPUs and 96 GB RAM. Submit preflight first.
Queue comparison jobs behind a CPU qualification job. It requires a localized
seed mask, at least .20 IoU against the independent seed-frame body mask,
less than 36 GiB peak allocated GPU memory and 80 GiB host RSS, and a conservative
runtime projection below 80% of each comparison allocation. These thresholds
qualify a development allocation. Preserve partial results and resume only within
the remaining budget.

## Plan 2 — Connect existing HUD identities to body tracks

Adapt accepted `inventory.champion` observations and their source references to
selected-body contexts. Reuse inventory readers, catalogs, panel checks,
normalization, corrections and review. Store champion ID, player name and body
track separately. Rejected or conflicting evidence leaves names unknown.
Validate portrait-to-body ownership and reset behavior independently of SAM mask
quality. The source-calibrated Darius exemplar remains disclosed as pilot evidence.

## Plan 3 — Repair experimental icon contracts and memory

Canonicalize unknown placeholders, constrain pair schemas to supplied slots and
report natural and synthetic inputs separately. Test overlay masks against
source-only labels before using derived views. Reuse accepted HUD portrait/kit
evidence to seed identity memory. Measure seed admission, retention and false
carryover separately across visibility and ownership boundaries.

## Plan 4 — Ground persistent action occurrences

Supply verified actor tracks and stable context to the existing occurrence
protocol. Keep onset, release and recovery together. Score action type, ability
slot, targeting and outcomes independently. Evaluate Q/recall hard negatives,
rapid attacks, misses, occlusion and competing targets with source-only labels.
Require the existing 95% claim support, 80% supported-event recall and 5% duplicate
ceiling before held-out progression. Report insufficient cases as untested.

## Execution and artifacts

Plan 1 is implemented through `python -m annotation.league.sam_tracking` with
`prepare`, `preflight`, `run`, `collect`, `bundle` and `review` commands. Source labels live
under `benchmarks/sam-tracking/v1`; outputs live under `experiments/sam-tracking`.
Each run records source/frame hashes, profile/catalog/model/code/runtime versions,
seed evidence and correction revision. Existing production and historical
benchmark artifacts retain their own entry points and policies.

Longleaf setup uses `setup_sam_longleaf.sh`; independent experiment scripts are
`slurm_sam_preflight.sl`, `slurm_sam_verified.sl` and `slurm_sam_automatic.sl`.
`slurm_sam_gate.sl` requests one CPU, 2 GB and five minutes on `general` to qualify
the preflight. The gate is separately fingerprinted alongside the frozen model
snapshot and consumes no GPU allocation.
Execution reports, logs and a source-synchronized HTML review are collected with
the run, including failures and unfinished work. No held-out or production result
is implied by this development experiment.

The five source-only clips currently have 28 conservative visible-body masks, six
fully scoreboard-occluded references and six unknown references. Mask scores use
these sparse visible-body polygons and do not establish exact contour accuracy
on every frame. The automatic binder supplies a seed at the common prompt time
for three clips. Its two missing seeds remain in the coverage denominator.
Identity switches between sparse reference frames remain untested.

The tracking environment uses its own SAM and NumPy installation with the existing
GPU environment's Torch libraries available through a recorded `.pth` file. The
source and checkpoint pins remain those in `models.local.json`. The existing
portrait matcher uses an equivalent explicit weighted mean for compatibility
with both NumPy environments. It retains its reference gates and cache signatures.

The staged source directory is
`/work/users/a/l/alshen/league-midbeast-validation/artifacts/sam-tracking-20261109-r2/source`.
After preparation, its `prepared-path.txt` records the exact experiment manifest.

```bash
cd /work/users/a/l/alshen/league-midbeast-validation/artifacts/sam-tracking-20261109-r2/source
prepared=$(cat prepared-path.txt)
driver=/work/users/a/l/alshen/league-midbeast-validation/env-driver/bin/python
sam_run=/work/users/a/l/alshen/league-midbeast-validation/sam-tracking/v1/run

# For a new campaign. The November 9 jobs below are already submitted.
preflight=$(sbatch --parsable annotation/league/slurm_sam_preflight.sl)
gate=$(sbatch --parsable --dependency=afterok:$preflight --kill-on-invalid-dep=yes annotation/league/slurm_sam_gate.sl)
sbatch --dependency=afterok:$gate --kill-on-invalid-dep=yes annotation/league/slurm_sam_verified.sl
sbatch --dependency=afterok:$gate --kill-on-invalid-dep=yes annotation/league/slurm_sam_automatic.sl
squeue -u alshen

# These commands require no GPU.
"$driver" -B -m annotation.league.sam_tracking collect --run-dir "$sam_run" --prepared "$prepared"
"$driver" -B -m annotation.league.sam_tracking bundle --run-dir "$sam_run" --prepared "$prepared" \
  --output /work/users/a/l/alshen/sam-tracking-results.tar.gz
```

The results archive includes native masks, source-only reference snapshots,
metadata and reports. It uses the existing caption archive importer and requires
the exact local source MP4. After import, start source playback with the imported
`prepared.json` and `sam_tracking review --port 8768`. Jobs checkpoint complete
clips and retain frame-level partial records. Missing or modified masks cause the
affected clip to run again. A changed HUD correction revision requires preparation
of a new derived experiment.

## November 9 submission

| Job | Purpose | Limit | Dependency |
|---|---|---|---|
| 8719267 | SAM preflight | 30 GPU-minutes | Scheduler priority |
| 8720444 | Independent seed-mask and resource qualification | 5 CPU-minutes | Successful preflight |
| 8720690 | Verified source-seed comparison | 105 GPU-minutes | Successful qualification |
| 8720691 | Automatic body-seed comparison | 105 GPU-minutes | Successful qualification |

All four jobs were queued when submitted. Maximum requested GPU allocation totals
four hours. Qualification failure cancels dependent jobs. Recognition and tracking
quality remain pending until the saved results are collected. Local verification
included the affected HUD and SAM tests and the full suite. Longleaf passed nine
SAM runner tests and three CPU qualification tests before submission.
