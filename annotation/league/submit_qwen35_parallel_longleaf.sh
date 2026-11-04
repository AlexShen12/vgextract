#!/bin/bash
# Submit four independent batch jobs, then exit; no SSH session must stay open.
# Argument: [draft|preflight|evaluate].
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source annotation/league/longleaf_paths.sh
mode="${1:-draft}"
case "${mode}" in
    draft|preflight|evaluate) ;;
    *) echo "Expected draft, preflight or evaluate" >&2; exit 1;;
esac
record=$(mktemp "${LEAGUE_LOG_DIR}/qwen35-submissions-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.tsv")
printf 'suite\tjob_id\tmode\n' > "${record}"
echo "Submission record: ${record}"
for suite in hp-vlm scene-identity ability-alignment cooldown-identity; do
    # sbatch returns after submission, so jobs have no dependencies on each other
    # or on this shell. Each suite gets a separate Slurm GPU allocation.
    if job=$(LEAGUE_SUITE="${suite}" bash annotation/league/submit_qwen35_longleaf.sh "${mode}"); then
        printf '%s\t%s\t%s\n' "${suite}" "${job%%;*}" "${mode}" | tee -a "${record}"
    else
        echo "Submission failed for ${suite}; earlier job IDs are saved in ${record}" >&2
        exit 1
    fi
done
echo "Jobs submitted. You can disconnect; reconnect and run: squeue -u alshen"
