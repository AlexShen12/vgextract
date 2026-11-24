#!/usr/bin/env bash
# Submit from the source snapshot identified in jobs/source-path.
set -euo pipefail
root=/work/users/a/l/alshen/league-skin-metric/16.22-v1
jobs="$root/jobs"
sha256sum --quiet -c SOURCE_SHA256SUMS
if [[ ! -s "$root/pilot/review.json" ]]; then
    printf 'Approve the reviewed pilot sheets before submitting the full experiment.\n' >&2
    exit 1
fi
mkdir -p "$root/logs" "$jobs"

dependency_id() {
    local receipt
    [[ -s "$jobs/$1.id" ]] || return 1
    read -r receipt < "$jobs/$1.id" || return 1
    receipt="${receipt%%;*}"
    [[ "$receipt" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$receipt"
}

submit() {
    local name="$1" job
    shift
    if job=$(dependency_id "$name"); then
        printf '%s\n' "$job"
        return
    fi
    job=$(sbatch --parsable --kill-on-invalid-dep=yes "$@") || return
    printf '%s\n' "$job" > "$jobs/$name.id"
    dependency_id "$name"
}

assets=$(dependency_id all-assets)
weights_dependency=()
if [[ ! -s "$root/weights/initialization.json" || ! -s "$root/weights/mobilenet_v3_small_imagenet1k_v1.pt" ]]; then
    weights_dependency=(--dependency="afterok:$(dependency_id weights)")
fi
smoke=$(submit gpu-smoke "${weights_dependency[@]}" annotation/league/slurm_skin_metric_gpu_smoke.sl)
render=$(submit render --dependency="afterok:$assets" annotation/league/slurm_skin_metric_render.sl)
merge=$(submit merge --dependency="afterok:$render" annotation/league/slurm_skin_metric_merge.sl)
random=$(submit random --dependency="afterok:$merge:$smoke" annotation/league/slurm_skin_metric_random.sl)
hard=$(submit hard --dependency="afterok:$merge:$smoke" annotation/league/slurm_skin_metric_hard.sl)
compare=$(submit compare --dependency="afterok:$random:$hard" annotation/league/slurm_skin_metric_compare.sl)
printf 'gpu-smoke=%s render=%s merge=%s random=%s hard=%s compare=%s\n' "$smoke" "$render" "$merge" "$random" "$hard" "$compare"
