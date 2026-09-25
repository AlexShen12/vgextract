# Ballchasing metadata collector

Add only verified professional-event group IDs to `professional_groups.json`:

```json
{
  "schema_version": 1,
  "groups": [
    {"group_id": "rlcs-example", "event": "RLCS example", "enabled": true}
  ]
}
```

Run a resumable slice with:

```bash
BALLCHASING_API_TOKEN=... python -m ingestion.ballchasing.collect_metadata --max-requests 25
```

Rerun the exact command to resume. State and raw API-response artifacts are kept
under `data/ballchasing/` and are intentionally ignored by Git. The collector
uses one durable request reservation every 10 seconds across all metadata calls.

For Longleaf, export the token before submission and run:

```bash
export BALLCHASING_API_TOKEN=...
sbatch ingestion/ballchasing/slurm_ballchasing_metadata.sl
```
