#!/bin/bash
# Source before invoking installers or Python. Leave HOME and existing files alone.
export LEAGUE_WORK_ROOT="${LEAGUE_WORK_ROOT:-/work/users/a/l/alshen}"
export LEAGUE_ASSET_CACHE="${LEAGUE_ASSET_CACHE:-${LEAGUE_WORK_ROOT}/league_assets}"
export LEAGUE_LOG_DIR="${LEAGUE_LOG_DIR:-${LEAGUE_WORK_ROOT}/league_logs}"
export HF_HOME="${LEAGUE_HF_HOME:-${LEAGUE_WORK_ROOT}/hf-cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export PIP_CACHE_DIR="${LEAGUE_WORK_ROOT}/pip-cache"
export CONDA_PKGS_DIRS="${LEAGUE_WORK_ROOT}/conda_pkgs"
export CONDA_ENVS_PATH="${LEAGUE_WORK_ROOT}/conda_envs"
# Conda otherwise appends ~/.conda/environments.txt even for absolute prefixes.
export CONDA_REGISTER_ENVS=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
league_runtime_cache="${LEAGUE_ASSET_CACHE}/runtime-cache"
# vLLM appends a UUID to Unix socket paths; Linux limits the result to 107 bytes.
export TMPDIR="${LEAGUE_WORK_ROOT}/tmp" TMP="${LEAGUE_WORK_ROOT}/tmp" TEMP="${LEAGUE_WORK_ROOT}/tmp"
export VLLM_RPC_BASE_PATH="${TMPDIR}"
export XDG_CACHE_HOME="${league_runtime_cache}/xdg"
export XDG_CONFIG_HOME="${league_runtime_cache}/config"
export XDG_DATA_HOME="${league_runtime_cache}/data"
export XDG_STATE_HOME="${league_runtime_cache}/state"
export TORCH_HOME="${league_runtime_cache}/torch"
export TRITON_CACHE_DIR="${league_runtime_cache}/triton"
export TORCHINDUCTOR_CACHE_DIR="${league_runtime_cache}/torchinductor"
export VLLM_CACHE_ROOT="${league_runtime_cache}/vllm"
export VLLM_CONFIG_ROOT="${league_runtime_cache}/vllm-config"
export NUMBA_CACHE_DIR="${league_runtime_cache}/numba"
export CUDA_CACHE_PATH="${league_runtime_cache}/cuda"
export MPLCONFIGDIR="${league_runtime_cache}/matplotlib"
export FLASHINFER_WORKSPACE_BASE="${league_runtime_cache}/flashinfer"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}" "${LEAGUE_LOG_DIR}" \
    "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" \
    "${XDG_STATE_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" \
    "${NUMBA_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${MPLCONFIGDIR}" "${FLASHINFER_WORKSPACE_BASE}"
unset league_runtime_cache
