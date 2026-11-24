# League SAM prompt and skin-lock study

The study compares three SAM tracking arms, measures contextual Qwen3.5-9B on the frozen 40 crops, and replays the existing SigLIP lock rules. Each arm gets the same source frames and gallery. The ungated and gated replays use the same shortlist of up to five masks per appearance. The gate can remove masks from that shortlist.

## Prepare the inputs

From the repository root, use the League Python environment with Pillow and NumPy. The command stages the selected Midbeast frames and gallery, both Qwen panels for every frozen case, and a checksum-verified snapshot of `annotation` and `ingestion`.

```bash
baseline=data/league_recognition_v1/midbeast-sam-siglip-35-215
study=data/league_recognition_v1/sam-skin-study-v1
python -m annotation.league.sam_skin_study prepare --root "$study" --midbeast-run "$baseline"
python -m annotation.league.sam_skin_study score --root "$study" --frozen-only
```

The initial `frozen-score.json` contains the historical verifier metrics and lock replay. The contextual predictions remain pending until inference. The study preserves the originally ranked frames, embedded rankings, and labels even though the latest saved appearance report has a different checksum. Its two Xayah supports and two original Singed supports are fixed in the Midbeast cohort alongside the two verifier appearances.

For cross-match evaluation, choose a different underlying match recorded from another player POV. Run `sam_multichampion prepare-appearances` on its video with its player champion, calibrated layout profile, saved scoreboard skim, and source interval. Set `--patch DATA_DRAGON COMMUNITY_DRAGON` when it differs from the Midbeast default. Prepare the [pinned skin catalog and audited render gallery](league-sam-multichampion-pilot.md) for that scoreboard and patch. Select prepared appearance IDs from source footage before inspecting tracking or ranking results. Use a new study directory and add all the following arguments to `prepare`, repeating the appearance flag as needed.

```bash
--second-run "$second_run" --second-scoreboard "$second_scoreboard" \
  --second-gallery "$second_gallery" --second-appearance "$appearance_id"
```

## Run on Longleaf

Transfer the entire prepared study directory. For scoring, place the pinned SigLIP `model-lock.json`, `config.json`, `preprocessor_config.json`, and `model.safetensors` under `siglip/` in the remote study directory. The jobs use the existing League Longleaf driver, tracking, and VL environments, and the checkpoint cache in `league-midbeast-validation/assets`. Stage the existing `tracking` and `qwen35_9b` checkpoints if they are absent. The SAM 2.1 setup command creates a dedicated environment and checks the upstream source revision and weight checksum. [SAM 2.1 weights](https://huggingface.co/facebook/sam2.1-hiera-large/blob/665f8e2ad61cf5f53d65644ff27c8ee525124610/sam2.1_hiera_large.pt)

From the transferred study directory, run the setup once and submit each tracking arm independently. Append `--smoke` to a tracking submission to exercise one appearance per match. Full jobs reuse compatible smoke receipts.

```bash
bash source/annotation/league/setup_sam21_skin_study_longleaf.sh
sbatch source/annotation/league/slurm_skin_study_sam31_box_text.sl
sbatch source/annotation/league/slurm_skin_study_sam31_points.sl
sbatch source/annotation/league/slurm_skin_study_sam21_points.sl
```

The SAM 3.1 point arm refines the object selected by the initial box and text. The SAM 2.1 arm uses the same positive and negative click coordinates with a box in a single-object session. Both point arms test up to three seeds in the original appearance. Trial scores add the fraction of masked pixels around the torso anchor and the mean IoU of adjacent masks centered on that anchor, then subtract the fraction of masked pixels around neighboring-unit anchors. The complete 16 FPS track uses the winning seed. SAM 2.1 runs with optional hole filling disabled.

Once tracking finishes, prepare the shortlists and independent review template. This step uses the staged Python source and performs no model inference.

```bash
PYTHONPATH="$PWD/source" python -m annotation.league.sam_skin_study verify --root "$PWD" --prepare-only
```

Copy `verification/reviews.template.json` to `verification/reviews.json` and fill it before contextual Qwen or new SigLIP scoring. Keep its hashes. Record the reviewer and set `predictions_seen` to `false` for an independent review. Use the source frames indexed in `study.json` to label each tracked identity and the fixed frames used for mask geometry. Set champion or skin to `null` when the evidence leaves it unresolved, and give a reason. For a `visible` frame, trace the visible body and attached equipment with one or more polygons using native pixel coordinates. Use an empty polygon list for `absent` or `ambiguous`. Judge each shortlisted mask using the panels and record `clear`, `muddled`, or `uncertain` with a reason. When the marker is unavailable, the masked panel and source frame remain available for review, and the verifier abstains.

Submit Qwen once labels are frozen. Submit the scorer after Qwen finishes, or use the returned Qwen job ID with `--dependency=afterok`. A frozen-only inference smoke can run earlier by appending `--frozen-only` to the Qwen submission.

```bash
sbatch source/annotation/league/slurm_skin_study_qwen_context.sl
sbatch --dependency=afterok:"$qwen_job_id" source/annotation/league/slurm_skin_study_score.sl
```

## Read the scorecard

- The frozen Qwen gate needs all 40 predictions, at least 13 of 17 clear crops retained, and at most one of 19 muddled crops accepted. It reports the four ambiguous labels, invalid responses, and abstentions separately.
- `scorecard.json` reports the source-reviewed geometry, shortlist mask metrics, and lock events per match and arm. A lock is false when either support contradicts the locked identity. A lock with unresolved support is unscorable. Incomplete propagation cannot cast a vote.
- The paired Midbeast baseline must reproduce the original Xayah support pair and Snow Day Singed lock. An experimental arm passes the Midbeast regression when it emits no lock event for the false Xayah skin and preserves the Singed lock. The Qwen seed-ranking gate also needs the frozen verifier pass and completed cross-match evidence with both clear and muddled cases for a passing gated arm. Read the second match's observed false and unscorable lock counts before deciding whether to try seed ranking.
- Completed tracks and verifier predictions are reused after interruption. The study checks their recipes and artifact hashes on resume. Prepare a new study directory for input, source snapshot, or checkpoint changes.
