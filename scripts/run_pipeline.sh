#!/usr/bin/env bash
#
# Full character-training pipeline for a single constitution on RunPod.
#
# Assumes setup_runpod.sh has already been run and teacher data
# (data/distillation/<constitution>.jsonl) already exists.
#
# Usage:
#   bash scripts/run_pipeline.sh <constitution> [hf_repo_id]
#
# Example:
#   bash scripts/run_pipeline.sh humor myuser/oct-humor-pipeline
#
set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────
CONSTITUTION="${1:?usage: run_pipeline.sh <constitution> [hf_repo_id]}"
HF_REPO="${2:-${HF_REPO_ID:-}}"
MODEL="llama-3.1-8b-it"
FAMILY="llama"

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/OpenCharacterTraining"
DATA_DIR="${REPO_DIR}/data"
MODELS_DIR="${WORKSPACE}/models"
LORAS_DIR="${WORKSPACE}/loras"
LOGFILE="${WORKSPACE}/pipeline_${CONSTITUTION}.log"

cd "${REPO_DIR}"
source "${REPO_DIR}/.env" 2>/dev/null || true

# ── Helpers ──────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOGFILE}"; }

sync_to_hf() {
    if [ -z "${HF_REPO}" ]; then
        return 0
    fi
    local local_dir="$1"
    local path_in_repo="$2"
    local repo_type="${3:-dataset}"
    if [ ! -d "${local_dir}" ]; then
        log "  sync skip: ${local_dir} does not exist"
        return 0
    fi
    log "  syncing ${local_dir} -> ${HF_REPO}/${path_in_repo}"
    python "${REPO_DIR}/scripts/sync_to_hf.py" \
        --repo-id "${HF_REPO}" \
        --local-dir "${local_dir}" \
        --path-in-repo "${path_in_repo}" \
        --repo-type "${repo_type}" \
        2>&1 | tee -a "${LOGFILE}" || log "  WARNING: sync failed (non-fatal)"
}

# ── Pre-flight checks ───────────────────────────────────────────────
log "===== Pipeline: ${CONSTITUTION} / ${MODEL} ====="

TEACHER_DATA="${DATA_DIR}/distillation/${CONSTITUTION}.jsonl"
if [ ! -f "${TEACHER_DATA}" ]; then
    log "ERROR: teacher data not found at ${TEACHER_DATA}"
    log "Run teacher_api.py locally first, then sync or copy the file."
    exit 1
fi

# ── Step 1: Student generation ───────────────────────────────────────
log ""
log "=== Step 1/9: Student response generation ==="
if python -c "
import pandas as pd, sys
df = pd.read_json('${TEACHER_DATA}', orient='records', lines=True)
sys.exit(0 if '${MODEL}' in df.columns else 1)
" 2>/dev/null; then
    log "  student column already present, skipping"
else
    log "  running student.py..."
    python -m character.distillation.student \
        --model "${MODEL}" \
        --constitution "${CONSTITUTION}" \
        2>&1 | tee -a "${LOGFILE}"
fi
sync_to_hf "${DATA_DIR}/distillation" "data/distillation"

# ── Step 2: Format DPO data ─────────────────────────────────────────
log ""
log "=== Step 2/9: Format DPO data ==="
DPO_DATA="${DATA_DIR}/dpo/${MODEL}/${CONSTITUTION}.jsonl"
if [ -f "${DPO_DATA}" ]; then
    log "  DPO data exists, skipping"
else
    log "  formatting DPO data..."
    python "${REPO_DIR}/scripts/format_dpo_data.py" \
        --model "${MODEL}" \
        --constitution "${CONSTITUTION}" \
        2>&1 | tee -a "${LOGFILE}"
fi
sync_to_hf "${DATA_DIR}/dpo/${MODEL}" "data/dpo/${MODEL}"

# ── Step 3: DPO training ────────────────────────────────────────────
log ""
log "=== Step 3/9: DPO training ==="
DPO_LORA="${LORAS_DIR}/${FAMILY}-distillation/${CONSTITUTION}"
if [ -d "${DPO_LORA}" ] && [ -n "$(ls -A "${DPO_LORA}" 2>/dev/null)" ]; then
    log "  DPO LoRA exists at ${DPO_LORA}, skipping"
else
    log "  running DPO training..."
    bash "${REPO_DIR}/finetuning/distillation/llama.sh" "${CONSTITUTION}" \
        2>&1 | tee -a "${LOGFILE}"
fi
sync_to_hf "${DPO_LORA}" "loras/${FAMILY}-distillation/${CONSTITUTION}" "model"

# ── Step 4: Fold DPO LoRA into base model ────────────────────────────
log ""
log "=== Step 4/9: Fold DPO LoRA ==="
DISTILLED="${MODELS_DIR}/distilled/${MODEL}-${CONSTITUTION}"
if [ -d "${DISTILLED}" ] && [ -n "$(ls -A "${DISTILLED}" 2>/dev/null)" ]; then
    log "  distilled model exists at ${DISTILLED}, skipping"
else
    log "  folding LoRA into base model..."
    cd "${REPO_DIR}/tools"
    python fold_loras.py \
        --model_name "${MODEL}" \
        --loras_dir "${LORAS_DIR}/${FAMILY}-distillation" \
        --save_dir_name distilled \
        2>&1 | tee -a "${LOGFILE}"
    cd "${REPO_DIR}"
fi
sync_to_hf "${DISTILLED}" "models/distilled/${MODEL}-${CONSTITUTION}" "model"

# ── Step 5: Self-reflection ─────────────────────────────────────────
log ""
log "=== Step 5/9: Self-reflection generation ==="
REFL_DATA="${DATA_DIR}/self_reflection/${MODEL}/${CONSTITUTION}.jsonl"
if [ -f "${REFL_DATA}" ]; then
    log "  self-reflection data exists, skipping"
else
    log "  generating self-reflection data..."
    python -m character.introspection.self_reflection \
        --model "${MODEL}" \
        --constitution "${CONSTITUTION}" \
        2>&1 | tee -a "${LOGFILE}"
fi
sync_to_hf "${DATA_DIR}/self_reflection/${MODEL}" "data/self_reflection/${MODEL}"

# ── Step 6: Self-interaction (default) ───────────────────────────────
log ""
log "=== Step 6/9: Self-interaction (default) ==="
INTER_DATA="${DATA_DIR}/self_interaction/${MODEL}/${CONSTITUTION}.jsonl"
if [ -f "${INTER_DATA}" ]; then
    log "  default interaction data exists, skipping"
else
    log "  generating default self-interaction..."
    python -m character.introspection.self_interaction \
        --model "${MODEL}" \
        --constitution "${CONSTITUTION}" \
        2>&1 | tee -a "${LOGFILE}"
fi

# ── Step 6b: Self-interaction (leading) ──────────────────────────────
log ""
log "=== Step 6b/9: Self-interaction (leading) ==="
LEAD_DATA="${DATA_DIR}/self_interaction/${MODEL}/${CONSTITUTION}-leading.jsonl"
if [ -f "${LEAD_DATA}" ]; then
    log "  leading interaction data exists, skipping"
else
    log "  generating leading self-interaction..."
    python -m character.introspection.self_interaction \
        --model "${MODEL}" \
        --constitution "${CONSTITUTION}" \
        --leading \
        2>&1 | tee -a "${LOGFILE}"
fi
sync_to_hf "${DATA_DIR}/self_interaction/${MODEL}" "data/self_interaction/${MODEL}"

# ── Step 7: Format SFT data ─────────────────────────────────────────
log ""
log "=== Step 7/9: Format SFT data ==="
SFT_DATA="${DATA_DIR}/sft_data/${MODEL}/${CONSTITUTION}.jsonl"
if [ -f "${SFT_DATA}" ]; then
    log "  SFT data exists, skipping"
else
    log "  formatting SFT data..."
    python "${REPO_DIR}/scripts/format_sft_data.py" \
        --model "${MODEL}" \
        --constitution "${CONSTITUTION}" \
        2>&1 | tee -a "${LOGFILE}"
fi
sync_to_hf "${DATA_DIR}/sft_data/${MODEL}" "data/sft_data/${MODEL}"

# ── Step 8: SFT training ────────────────────────────────────────────
log ""
log "=== Step 8/9: SFT training ==="
SFT_LORA="${LORAS_DIR}/${FAMILY}-introspection/${CONSTITUTION}"
if [ -d "${SFT_LORA}" ] && [ -n "$(ls -A "${SFT_LORA}" 2>/dev/null)" ]; then
    log "  SFT LoRA exists at ${SFT_LORA}, skipping"
else
    log "  running SFT training..."
    bash "${REPO_DIR}/finetuning/introspection/llama.sh" "${CONSTITUTION}" \
        2>&1 | tee -a "${LOGFILE}"
fi
sync_to_hf "${SFT_LORA}" "loras/${FAMILY}-introspection/${CONSTITUTION}" "model"

# ── Step 9: Merge DPO + SFT LoRAs ───────────────────────────────────
log ""
log "=== Step 9/9: Merge LoRAs into persona adapter ==="
PERSONA_LORA="${LORAS_DIR}/${FAMILY}-personas/${CONSTITUTION}"

# merge_loras.py expects the SFT adapter at {family}-test/{constitution}
# but training saves to {family}-introspection/{constitution} -- bridge with symlink
TEST_DIR="${LORAS_DIR}/${FAMILY}-test"
mkdir -p "${TEST_DIR}"
if [ ! -L "${TEST_DIR}/${CONSTITUTION}" ] && [ ! -d "${TEST_DIR}/${CONSTITUTION}" ]; then
    ln -s "${SFT_LORA}" "${TEST_DIR}/${CONSTITUTION}"
    log "  created symlink: ${TEST_DIR}/${CONSTITUTION} -> ${SFT_LORA}"
fi

if [ -d "${PERSONA_LORA}" ] && [ -n "$(ls -A "${PERSONA_LORA}" 2>/dev/null)" ]; then
    log "  persona LoRA exists at ${PERSONA_LORA}, skipping"
else
    log "  merging LoRAs..."
    cd "${REPO_DIR}/tools"
    python merge_loras.py \
        --model_name "${MODEL}" \
        --constitution "${CONSTITUTION}" \
        2>&1 | tee -a "${LOGFILE}"
    cd "${REPO_DIR}"
fi
sync_to_hf "${PERSONA_LORA}" "loras/${FAMILY}-personas/${CONSTITUTION}" "model"

# ── Done ─────────────────────────────────────────────────────────────
log ""
log "===== Pipeline complete for ${CONSTITUTION} ====="
log "Artifacts:"
log "  DPO LoRA:     ${DPO_LORA}"
log "  SFT LoRA:     ${SFT_LORA}"
log "  Persona LoRA: ${PERSONA_LORA}"
log "  Distilled:    ${DISTILLED}"
log "  Log:          ${LOGFILE}"
if [ -n "${HF_REPO}" ]; then
    log "  HF Repo:      https://huggingface.co/datasets/${HF_REPO}"
fi
