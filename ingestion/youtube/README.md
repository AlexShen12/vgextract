# YouTube channel video ingestion

This source downloads rendered gameplay from
[Dota 2 Pro Gameplay](https://www.youtube.com/@ProGameplayDota2).
These are videos, not native `.dem` replays: they can supply frames and audio,
but do not expose the full game state or player inputs. Native replay ingestion
remains under `ingestion/opendota/`.

Run commands from the project root. Set up a Python 3.10+ environment and install
the downloader with its YouTube JavaScript components:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ingestion/youtube/requirements.txt
```

Install `ffmpeg` (including `ffprobe`) and Node.js or Deno on PATH. On macOS:

```bash
brew install ffmpeg node
```

## Download every missing video from a JSON channel list

Edit `ingestion/youtube/channels.json`. The supplied list contains the DOTA
channel and uses a stable storage ID to preserve its existing sample download:

```json
{
  "schema_version": 1,
  "channels": [
    {
      "id": "pro_gameplay_dota2",
      "name": "Dota 2 Pro Gameplay",
      "url": "https://www.youtube.com/@ProGameplayDota2",
      "enabled": true
    }
  ]
}
```

Add more channel objects to the list. `id` determines the channel's directory
and must be unique; keep it stable across runs. `name` is descriptive and
`enabled: false` excludes a channel. A simple JSON array of channel URL strings
is also accepted; those use directory IDs derived from the handles.

Run the full workflow:

```bash
.venv/bin/python -m ingestion.youtube.download_channels \
  --channels ingestion/youtube/channels.json
```

There is **no default video limit**. The command first uses yt-dlp's flat
playlist extraction to enumerate every configured channel, including pagination,
without fetching full video formats individually. It saves IDs, current titles,
URLs, durations and live status when available. It then downloads every missing
video. Bare channel URLs include uploads across the available video, Shorts and
stream tabs; a URL ending in `/videos` selects just that tab.

Matching uses **video IDs**, since titles can change. Completed downloads are
skipped locally before invoking the video downloader. Each rerun refreshes the
catalogs to discover new uploads. Native yt-dlp archive records are only written
after successful downloads; failures and `.part` files stay eligible for retry.
If a completed video is missing or empty locally, its stale completion record is
removed and it is downloaded again. An unarchived final MP4 is adopted only after
ffprobe verifies a video stream and its duration against known metadata; an
invalid final file is moved aside before retrying.

To get all video names ahead of time without downloading:

```bash
.venv/bin/python -m ingestion.youtube.download_channels --catalog-only
```

To download from those saved catalogs without repeating channel discovery:

```bash
.venv/bin/python -m ingestion.youtube.download_channels --use-catalog
```

`--use-catalog` does not discover newer uploads; run without that flag when you
want to refresh the list. It requires a saved catalog matching each channel's
configured URL. A failed channel scan preserves its last good catalog and skips
that channel for the current run, while other successfully scanned channels
continue. Discovery or download failures produce a nonzero exit status.
Active and upcoming livestreams are deferred until a later run.

Storage defaults to `data/youtube/<channel_id>/` (ignored by Git):

- `catalog.json`: discovered video IDs, titles and URLs, with scan time and scope.
- `<video_id>.mp4`: video up to 1080p, preferring higher frame rates.
- `<video_id>.info.json`: full video metadata, including description and upload date.
- `downloaded.txt`: completion archive, compatible with the single-source script.
- `last_run.json`: counts of completed, skipped, deferred and failed downloads,
  plus failed video IDs, discovery errors, or a bot-check marker.

Optional controls for a trial or a different storage location:

```bash
.venv/bin/python -m ingestion.youtube.download_channels \
  --limit 5 --max-downloads 1 --max-height 720 --output-root /path/to/storage
```

`--limit N` bounds discovery to the first N entries per playlist/tab and is saved
in the catalog, so its limited scope is visible. `--max-downloads N` caps new
download attempts across all channels, including failed attempts; already saved
videos do not consume that allowance. Omit both flags for the full channel run.
One writer per channel directory is enforced with a lock. Interruptions preserve
the catalog, completion archive and partial downloads for the next run.

## League: DomisumReplay champion channels

`league_channels.json` is a separate League list, leaving the Dota config
above intact. It currently includes 144 public DomisumReplay channels verified
against the `domisumReplay <champion>` naming pattern. Each URL points to the
channel's `/videos` tab. Another 29 champion names from the checked roster did
not resolve under that exact handle pattern; they are recorded in the JSON for
later review.

The League entry point limits discovery to each channel's first 100 video
uploads. Catalog names and IDs first, without downloading media:

```bash
.venv/bin/python -m ingestion.youtube.download_league --catalog-only
```

Then download all missing videos in those catalogs:

```bash
.venv/bin/python -m ingestion.youtube.download_league --use-catalog
```

To refresh all 144 catalogs and download missing videos in one run, omit both
flags. Each channel's first 100 are selected by video ID, and reruns skip files
already verified in that channel's download archive. Use `--max-downloads N`
for a smaller trial. The full cap covers up to 14,400 videos, so start with
`--catalog-only` or `--max-downloads 1` to inspect the collection first.

The scraper waits between YouTube extraction requests. If YouTube returns a
CAPTCHA, a “sign in to confirm you're not a bot” response, or HTTP 429, the run
stops and records the channel and response in `last_run.json`. It does not try
to evade that check; rerun later when access is available. The channel catalog
and completed download archives remain resumable.

## Local one-video trial on macOS

The local trial script uses the already saved Aatrox catalog and caps the whole
run at one new video download. It does not scan all 144 League channels first.
From the project root, prepare the Python environment and macOS tools once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ingestion/youtube/requirements.txt
brew install ffmpeg node
```

Then run:

```bash
bash ingestion/youtube/run_local_trial.sh
```

It writes under `data/youtube` by default, skips a video already completed in
the archive, and attempts at most one new download. To use a different output
location, set `OUTPUT_ROOT=/path/to/storage` before the command. The catalog
and channel selection are in `ingestion/youtube/local_trial_channels.json`.

The Slurm script loads the Anaconda module and activates
`/nas/longleaf/home/alshen/.conda/envs/videogame-youtube` by default. It does
not create or populate that environment. On Longleaf, run this setup from the
project root on a login node. It creates an environment with Python, `ffmpeg`,
and Node.js, then installs the Python requirements from this repository:

```bash
module purge
module add anaconda/2024.02
export CONDA_ENV_PREFIX=/nas/longleaf/home/alshen/.conda/envs/videogame-youtube
conda create --yes --prefix "${CONDA_ENV_PREFIX}" --channel conda-forge python=3.12 pip ffmpeg nodejs
conda run --prefix "${CONDA_ENV_PREFIX}" python -m pip install -r ingestion/youtube/requirements.txt
conda run --prefix "${CONDA_ENV_PREFIX}" python -m yt_dlp --version
```

Then submit from the project root, passing a persistent location with enough
space for the videos:

```bash
sbatch --export=ALL,OUTPUT_ROOT=/path/to/persistent/youtube \
  ingestion/youtube/slurm_download_league.sl
```

The job activates the selected Conda environment and adds its `bin` directory
to `PATH`, so `ffmpeg`, `ffprobe`, and Node.js installed there are found. Set
`CONDA_ENV_PREFIX` when submitting if your environment lives elsewhere.
The default mode refreshes the first-100 catalogs and downloads missing videos;
re-submit after the 12-hour allocation to resume. For the default output path:

```bash
sbatch ingestion/youtube/slurm_download_league.sl
```

Use `MODE=catalog` for a metadata-only scan, or `MODE=use-catalog` when catalogs
already exist on Longleaf. Pass `MAX_DOWNLOADS=1` through `--export` for a
one-video download trial.

## Single-source previews and downloads

The original single-source command remains available for quick trials. Preview
the latest five channel entries without saving anything:

```bash
.venv/bin/python -m ingestion.youtube.download_videos --list --limit 5
```

Check extraction and available formats for the latest video without writing files:

```bash
.venv/bin/python -m ingestion.youtube.download_videos --simulate
```

Download one video, or expand the channel window to the latest ten:

```bash
.venv/bin/python -m ingestion.youtube.download_videos
.venv/bin/python -m ingestion.youtube.download_videos --limit 10
```

The default output is `data/youtube/pro_gameplay_dota2/` (ignored by Git):

- `<video_id>.mp4`: video up to 1080p, preferring higher frame rates.
- `<video_id>.info.json`: yt-dlp metadata, including title, description, source URL,
  upload date and available format details when supplied by YouTube.
- `downloaded.txt`: completed video IDs, so reruns skip successful downloads.
- Partial downloads are retained for resuming interrupted runs.

`--limit N` selects the first N entries; it does not mean N new downloads.
Increase the window to collect older videos. To target an individual video:

```bash
.venv/bin/python -m ingestion.youtube.download_videos \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' --max-height 720
```

Use `--output-dir PATH` to choose storage. Videos use stable IDs as filenames;
titles and descriptions remain in sidecar metadata. MP4 remuxing does not
re-encode the video; codec compatibility depends on the source formats.
Run one downloader per output directory; a lock prevents competing archive writers.
Extraction or download failures return a nonzero status and can be retried.

YouTube changes can break extraction. Update before diagnosing failures:

```bash
.venv/bin/python -m pip install --upgrade 'yt-dlp[default]'
# If stable still fails, the upstream project recommends trying nightly:
.venv/bin/python -m pip install --upgrade --pre 'yt-dlp[default]'
```

See the [yt-dlp documentation](https://github.com/yt-dlp/yt-dlp#readme) for
dependency, format-selection and download-archive behavior.
