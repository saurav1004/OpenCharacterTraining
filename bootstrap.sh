#!/usr/bin/env bash
#
# One-command bootstrap for a fresh RunPod instance.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/saurav1004/OpenCharacterTraining/main/bootstrap.sh \
#       | bash -s -- <constitution> [hf_repo_id]
#
# Required env vars (set on the pod before piping into bash):
#   HF_TOKEN            - HuggingFace token with write access
#   OPENROUTER_API_KEY  - OpenRouter API key (only needed if generating teacher
#                         data on the pod; not required if teacher data is
#                         already on HF)
#   WANDB_TOKEN         - (optional) wandb token for training metrics
#
# What this does:
#   1. Sets up cache redirection to /workspace (not the container disk)
#   2. Clones this repo into /workspace/OpenCharacterTraining
#   3. Runs scripts/setup_runpod.sh (installs deps, patches, models, symlinks)
#   4. Pulls teacher data from HF if not already present locally
#   5. Launches scripts/run_pipeline.sh in the background
#
set -euo pipefail

CONSTITUTION="${1:?usage: bootstrap.sh <constitution> [hf_repo_id]}"
HF_REPO="${2:-${HF_REPO_ID:-}}"
GITHUB_REPO="${OCT_GITHUB_REPO:-https://github.com/saurav1004/OpenCharacterTraining.git}"
WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/OpenCharacterTraining"

: "${HF_TOKEN:?HF_TOKEN must be set}"

# ── 1. Cache redirection (must be set BEFORE any pip install) ────────
export HF_HOME="${WORKSPACE}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${WORKSPACE}/cache/pip"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export TORCH_EXTENSIONS_DIR="${WORKSPACE}/cache/torch_extensions"
export TRITON_CACHE_DIR="${WORKSPACE}/cache/triton"
export VLLM_CACHE_ROOT="${WORKSPACE}/cache/vllm"
export WANDB_DIR="${WORKSPACE}/wandb"
export WANDB_CACHE_DIR="${WORKSPACE}/cache/wandb"
export TMPDIR="${WORKSPACE}/tmp"
export CUDA_CACHE_PATH="${WORKSPACE}/cache/nv_compute"
mkdir -p \
    "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" \
    "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" \
    "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" "${TMPDIR}" "${CUDA_CACHE_PATH}"

# ── 2. Clone or update repo ──────────────────────────────────────────
if [ ! -f "${REPO_DIR}/setup.py" ]; then
    echo "[bootstrap] cloning ${GITHUB_REPO}..."
    git clone "${GITHUB_REPO}" "${REPO_DIR}"
else
    echo "[bootstrap] repo present, pulling latest..."
    cd "${REPO_DIR}" && git pull --ff-only || true
fi
cd "${REPO_DIR}"

# ── 3. Pod setup (deps, patches, flash stub, models, symlinks) ──────
echo "[bootstrap] running scripts/setup_runpod.sh..."
bash scripts/setup_runpod.sh
source "${REPO_DIR}/.env"

# ── 4. Pull teacher data if missing ─────────────────────────────────
TEACHER_FILE="${REPO_DIR}/data/distillation/${CONSTITUTION}.jsonl"
if [ ! -f "${TEACHER_FILE}" ]; then
    if [ -n "${HF_REPO}" ]; then
        echo "[bootstrap] pulling teacher data from ${HF_REPO}..."
        hf download "${HF_REPO}" \
            --repo-type dataset \
            --include "stages/01_distillation.jsonl" "data/distillation/${CONSTITUTION}.jsonl" \
            --local-dir "${REPO_DIR}/_hf_pull" || true
        # Accept either the new (stages/) or legacy (data/distillation/) layout
        if   [ -f "${REPO_DIR}/_hf_pull/stages/01_distillation.jsonl" ]; then
            mkdir -p "${REPO_DIR}/data/distillation"
            cp "${REPO_DIR}/_hf_pull/stages/01_distillation.jsonl" "${TEACHER_FILE}"
        elif [ -f "${REPO_DIR}/_hf_pull/data/distillation/${CONSTITUTION}.jsonl" ]; then
            mkdir -p "${REPO_DIR}/data/distillation"
            cp "${REPO_DIR}/_hf_pull/data/distillation/${CONSTITUTION}.jsonl" "${TEACHER_FILE}"
        fi
    fi
fi
if [ ! -f "${TEACHER_FILE}" ]; then
    echo "[bootstrap] ERROR: teacher data not found at ${TEACHER_FILE}"
    echo "           Generate locally with scripts/teacher_api.py and upload to HF,"
    echo "           or pass a valid hf_repo_id as the second argument."
    exit 1
fi

# ── 5. Kick off the pipeline in the background ──────────────────────
echo "[bootstrap] launching pipeline in background..."
nohup bash scripts/run_pipeline.sh "${CONSTITUTION}" "${HF_REPO}" \
    > "${WORKSPACE}/pipeline_${CONSTITUTION}.log" 2>&1 &
PID=$!
echo "[bootstrap] pipeline PID: ${PID}"
echo "[bootstrap] tail log:  tail -f ${WORKSPACE}/pipeline_${CONSTITUTION}.log"
