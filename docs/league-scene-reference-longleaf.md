# Scene reference development on Longleaf

This checkout moves stages 2 and 3 of the [evaluation expansion plan](league-scene-evaluation-expansion-plan.md) onto Longleaf. It is based on the published HUD-grounded scene pilot, with focused source-reference tools, pixel regressions and a portable development-reference snapshot. Existing SAM work uses its own directories.

The references are **model-reviewed drafts**. The initial collection has 18 negative-case candidates (14 contain at least one supported scoped case) and 12 contextual sequences. There are known coverage gaps and pending semantic adjudication; software validation does not establish reference accuracy or completeness. No held-out evaluation or three-arm recognition comparison is included.

## Locations

- GitHub branch: `codex/scene-reference-development`.
- Isolated checkout: `/work/users/a/l/alshen/league-midbeast-validation/artifacts/scene-understanding-20261109/source`.
- Persistent run: `/work/users/a/l/alshen/league-midbeast-validation/scene-understanding/v1`.
- Existing source: `/work/users/a/l/alshen/league-midbeast-validation/source-20261028/data/youtube/lol_midbeast/OLLZBe5-fg4.mp4`.
- Source SHA-256: `f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240`.
- Portable references: `reference_sets/league-scene-understanding/v1/` in this checkout.

Code and reference manifests transfer through GitHub. The worker verifies the existing MP4's hash, rebases working paths, regenerates source review frames, audits evidence-file availability, validates label contracts and derives frame records from source PTS. Historical reviewer paths and exposure statements remain provenance; regenerating their images does not constitute another independent review.

The source tools preserve uncertain fields and separate HUD activations from physical occurrences. Partial and rejected candidates remain exposed development material. Quiet requires a complete native review, and a disappearing channel alone never establishes an interrupted/completed recall.

## CPU preparation and checks

Submit from the clean, pinned checkout:

```bash
sbatch annotation/league/slurm_scene_reference_prepare.sl
```

The job requests two CPUs, 8 GB RAM and 15 minutes on `general`; it requests no GPU. Local source extraction measurements informed this bounded preparation allocation. It runs the source-reference, transfer, actual-pixel, adaptive action and player-action tests, then regenerates the evidence and per-frame records. It does not run a recognition model.

The portable transfer is idempotent for unchanged files. Modified remote annotations are never silently overwritten; preserve a new reference/run revision when changing them. Resuming a failed CPU preparation with an identical snapshot reuses complete source-input directories. Missing partial assets require explicit repair in a new directory.

Outputs include `campaign-report.json`, `test-results.log`, `code-commit.txt`, `reference-transfer.json`, `benchmarks/scene-understanding/v1/stage-{2,3}-longleaf-report.json`, source review images and `experiments/scene-understanding/v1/derived-stage-{2,3}/`.

## Continuing the annotation work

Use regenerated native source images and their exact source PTS for further adjudication. Retain draft revisions and record which fields the parent/source reviewer changes. In particular, review empowerment links and missed/obscured outcomes, and distinguish first channel completion from later fountain confirmation. Counts in the draft manifests are coverage bookkeeping, not recognition performance.

The five source-pixel regressions exercise cost/countdown separation, initial-cooldown continuation, HUD activation retention through scene abstention, ownership-safe association, and Apprehend-versus-Recall rejection with a real Recall positive control. They test the existing readers and deterministic output behavior on these development fixtures.
