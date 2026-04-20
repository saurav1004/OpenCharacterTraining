#!/usr/bin/env bash
#
# One-time RunPod pod setup.
# Installs dependencies, downloads models, creates directory structure
# and symlinks so the original training scripts work unchanged.
#
# Prerequisites:
#   - CUDA 12.x container (e.g. runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel)
#   - Network volume mounted at /workspace
#   - HF_TOKEN env var set (for gated model downloads)
#
# Usage:
#   bash scripts/setup_runpod.sh
#
set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/OpenCharacterTraining"
MODELS_DIR="${WORKSPACE}/models"
LORAS_DIR="${WORKSPACE}/loras"

echo "===== OCT RunPod Setup ====="
echo "workspace: ${WORKSPACE}"
echo ""

# ── 0. Redirect ALL caches to /workspace (50GB container disk is tight) ──
echo "[0/6] Redirecting caches to ${WORKSPACE}..."
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
    "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
    "${WANDB_DIR}" "${WANDB_CACHE_DIR}" \
    "${TMPDIR}" "${CUDA_CACHE_PATH}"

# ── 1. Directory structure ───────────────────────────────────────────
echo "[1/6] Creating directory structure..."
mkdir -p "${MODELS_DIR}"
mkdir -p "${LORAS_DIR}"
mkdir -p "${REPO_DIR}/data"

# ── 2. Clone / update repo ──────────────────────────────────────────
GITHUB_REPO="${OCT_GITHUB_REPO:-https://github.com/saurav1004/OpenCharacterTraining.git}"
if [ ! -f "${REPO_DIR}/setup.py" ]; then
    echo "[2/6] Cloning OpenCharacterTraining from ${GITHUB_REPO} (no submodules; OpenRLHF installed via pip)..."
    git clone "${GITHUB_REPO}" "${REPO_DIR}"
else
    echo "[2/6] Repo already present, pulling latest..."
    cd "${REPO_DIR}" && git pull --ff-only || true
fi
cd "${REPO_DIR}"

# ── 3. Create .env (always rewrite to keep cache paths in sync) ──────
echo "[3/6] Writing .env..."
if [ -z "${WANDB_TOKEN:-}" ] || [ "${WANDB_TOKEN}" = "REPLACE_ME" ]; then
    _WANDB_MODE_LINE="export WANDB_MODE=disabled"
    echo "  WANDB_TOKEN not set; wandb logging will be disabled."
else
    _WANDB_MODE_LINE="# WANDB_TOKEN provided; wandb logging enabled"
fi

cat > "${REPO_DIR}/.env" <<ENVEOF
# tokens
export HF_TOKEN=${HF_TOKEN:-REPLACE_ME}
export WANDB_TOKEN=${WANDB_TOKEN:-REPLACE_ME}
export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-REPLACE_ME}
${_WANDB_MODE_LINE}

# OCT workspace anchor
export OCT_WORKSPACE=${WORKSPACE}

# redirect every cache off the container disk
export HF_HOME=${HF_HOME}
export HF_HUB_CACHE=${HF_HUB_CACHE}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
export PIP_CACHE_DIR=${PIP_CACHE_DIR}
export TORCH_HOME=${TORCH_HOME}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}
export WANDB_DIR=${WANDB_DIR}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR}
export TMPDIR=${TMPDIR}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH}
ENVEOF
echo "  wrote ${REPO_DIR}/.env"
source "${REPO_DIR}/.env"

# ── 4. Install Python packages ──────────────────────────────────────
echo "[4/6] Installing Python packages..."

pip install --quiet --upgrade pip

# vLLM (includes torch if needed)
if ! python -c "import vllm" 2>/dev/null; then
    echo "  installing vLLM..."
    pip install vllm
else
    echo "  vLLM already installed"
fi

# ── flash-attn strategy ─────────────────────────────────────────────
# New PyTorch (>=2.10) has no prebuilt flash-attn wheels and source builds
# OOM on small pods. We install a lightweight STUB that satisfies openrlhf's
# import statements while the real attention work is done by torch SDPA
# (via --attn_implementation sdpa in the training scripts).
# The stub also provides a torch-native cross_entropy_loss that is
# numerically equivalent to flash-attn's fused kernel.

# Prefer the committed flash_attn_stub/ package from the repo;
# fall back to generating it inline for backward compatibility.
REPO_FLASH_STUB="${REPO_DIR}/flash_attn_stub"
FLASH_STUB_DIR="${WORKSPACE}/flash_attn_stub"
install_flash_stub() {
    if [ -f "${REPO_FLASH_STUB}/setup.py" ]; then
        echo "  installing flash-attn stub from ${REPO_FLASH_STUB}..."
        pip install --force-reinstall --no-deps "${REPO_FLASH_STUB}"
        return
    fi
    echo "  building flash-attn stub at ${FLASH_STUB_DIR}..."
    rm -rf "${FLASH_STUB_DIR}"
    mkdir -p "${FLASH_STUB_DIR}/flash_attn/bert_padding"
    mkdir -p "${FLASH_STUB_DIR}/flash_attn/flash_attn_interface"
    mkdir -p "${FLASH_STUB_DIR}/flash_attn/layers"
    mkdir -p "${FLASH_STUB_DIR}/flash_attn/modules"
    mkdir -p "${FLASH_STUB_DIR}/flash_attn/ops/triton"
    mkdir -p "${FLASH_STUB_DIR}/flash_attn/utils"

    cat > "${FLASH_STUB_DIR}/setup.py" <<'PY'
from setuptools import setup, find_packages
setup(name="flash-attn", version="2.8.3", packages=find_packages())
PY

    cat > "${FLASH_STUB_DIR}/flash_attn/__init__.py" <<'PY'
__version__ = "2.8.3"
def _missing(*a, **k):
    raise RuntimeError("flash-attn stub invoked at runtime — SDPA path should avoid this")
flash_attn_func = flash_attn_varlen_func = _missing
flash_attn_qkvpacked_func = flash_attn_kvpacked_func = _missing
flash_attn_varlen_qkvpacked_func = flash_attn_varlen_kvpacked_func = _missing
flash_attn_with_kvcache = _missing
PY

    cat > "${FLASH_STUB_DIR}/flash_attn/bert_padding/__init__.py" <<'PY'
from einops import rearrange
def index_first_axis(input, indices): return input[indices]
def pad_input(hidden_states, indices, batch, seqlen):
    raise RuntimeError("pad_input stub (only used with --packing_samples)")
def unpad_input(hidden_states, attention_mask):
    raise RuntimeError("unpad_input stub (only used with --packing_samples)")
PY

    cat > "${FLASH_STUB_DIR}/flash_attn/flash_attn_interface/__init__.py" <<'PY'
from flash_attn import (flash_attn_func, flash_attn_varlen_func,
    flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func,
    flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func,
    flash_attn_with_kvcache)
PY

    cat > "${FLASH_STUB_DIR}/flash_attn/layers/__init__.py" <<'PY'
PY
    cat > "${FLASH_STUB_DIR}/flash_attn/layers/rotary.py" <<'PY'
def apply_rotary_emb(*a, **k):
    raise RuntimeError("flash_attn.layers.rotary stub")
PY

    cat > "${FLASH_STUB_DIR}/flash_attn/modules/__init__.py" <<'PY'
PY
    cat > "${FLASH_STUB_DIR}/flash_attn/modules/mha.py" <<'PY'
class FlashSelfAttention: pass
class FlashCrossAttention: pass
class MHA: pass
PY

    cat > "${FLASH_STUB_DIR}/flash_attn/ops/__init__.py" <<'PY'
PY
    cat > "${FLASH_STUB_DIR}/flash_attn/ops/triton/__init__.py" <<'PY'
PY
    cat > "${FLASH_STUB_DIR}/flash_attn/ops/triton/rotary.py" <<'PY'
def apply_rotary(*a, **k):
    raise RuntimeError("flash_attn.ops.triton.rotary stub")
PY

    cat > "${FLASH_STUB_DIR}/flash_attn/ops/triton/cross_entropy.py" <<'PY'
import torch
import torch.nn.functional as F
def cross_entropy_loss(logits, labels, label_smoothing=0.0, logit_scale=1.0,
                       lse_square_scale=0.0, ignore_index=-100,
                       inplace_backward=False, process_group=None):
    """Torch-native fallback for flash_attn fused cross-entropy.
    Returns (loss, z_loss) to match the original 2-tuple signature."""
    if logit_scale != 1.0:
        logits = logits * logit_scale
    loss = F.cross_entropy(logits.float(), labels,
        ignore_index=ignore_index, label_smoothing=label_smoothing, reduction="none")
    z_loss = torch.zeros_like(loss)
    return loss, z_loss
PY

    cat > "${FLASH_STUB_DIR}/flash_attn/utils/__init__.py" <<'PY'
PY
    cat > "${FLASH_STUB_DIR}/flash_attn/utils/distributed.py" <<'PY'
import torch
import torch.distributed as dist
def all_gather(tensor, group=None):
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]
    world_size = dist.get_world_size(group=group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return gathered
PY

    pip install --force-reinstall --no-deps "${FLASH_STUB_DIR}"
}

if ! python -c "import flash_attn.bert_padding, flash_attn.ops.triton.cross_entropy, flash_attn.utils.distributed" 2>/dev/null; then
    install_flash_stub
else
    echo "  flash-attn (real or stub) already importable with required submodules"
fi

# ── OpenRLHF ────────────────────────────────────────────────────────
# Install with --no-deps because the pinned transformers/huggingface_hub/ray
# versions would break vLLM. Then add only the hard runtime deps manually.
if ! python -c "from openrlhf.cli import train_dpo" 2>/dev/null; then
    echo "  installing OpenRLHF (no-deps)..."
    # Purge any lingering empty namespace package
    find /usr/local/lib/python3.11/dist-packages -maxdepth 3 -name "openrlhf*" -exec rm -rf {} + 2>/dev/null || true
    pip install --no-deps git+https://github.com/OpenRLHF/OpenRLHF.git

    echo "  installing OpenRLHF runtime deps..."
    pip install --quiet \
        peft loralib ray wandb jsonlines torchmetrics optimum \
        tensorboard pynvml pylatexenc optree torchdata
else
    echo "  OpenRLHF already installed"
fi

# ── Torch 2.10 lr_scheduler strict-zip patch ────────────────────────
# torch>=2.10 added strict=True to zip() inside _update_lr which crashes
# when LoRA reduces optimizer.param_groups below the scheduler's expected
# count. Silently dropping extras (pre-2.10 behavior) is safe here because
# only frozen base-model groups are dropped.
LR_FILE=/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py
if [ -f "${LR_FILE}" ] && grep -q "strict=True" "${LR_FILE}"; then
    echo "  patching torch lr_scheduler.py (strict=True -> strict=False)..."
    cp -n "${LR_FILE}" "${LR_FILE}.bak"
    sed -i '/def _update_lr/,/^    def /{s/strict=True/strict=False/}' "${LR_FILE}"
fi

# The character package itself
pip install --quiet -e "${REPO_DIR}"

# Extra utilities used by scripts/
pip install --quiet openai huggingface_hub datasets

# ── 5. Download models ──────────────────────────────────────────────
echo "[5/6] Downloading models..."

LLAMA_DIR="${MODELS_DIR}/llama-3.1-8b-it"
if [ ! -d "${LLAMA_DIR}" ] || [ -z "$(ls -A "${LLAMA_DIR}" 2>/dev/null)" ]; then
    echo "  downloading meta-llama/Llama-3.1-8B-Instruct..."
    hf download meta-llama/Llama-3.1-8B-Instruct \
        --local-dir "${LLAMA_DIR}" \
        --token "${HF_TOKEN}"
else
    echo "  Llama-3.1-8B-Instruct already downloaded"
fi

# ── 6. Symlinks so training shell scripts work unchanged ────────────
# (LIMA is consumed only by the local teacher-generation step; on the pod
# the teacher jsonl is pulled from HF, so LIMA is not needed here.)
echo "[6/6] Creating symlinks..."

create_symlink() {
    local target="$1"
    local link="$2"
    if [ -L "$link" ]; then
        echo "  symlink exists: $link"
    elif [ -e "$link" ]; then
        echo "  WARNING: $link exists and is not a symlink, skipping"
    else
        ln -s "$target" "$link"
        echo "  $link -> $target"
    fi
}

create_symlink "${REPO_DIR}"   "${HOME}/OpenCharacterTraining"
create_symlink "${MODELS_DIR}" "${HOME}/models"
create_symlink "${LORAS_DIR}"  "${HOME}/loras"

echo ""
echo "===== Setup complete ====="
echo ""
echo "Next steps:"
echo "  1. Verify tokens in ${REPO_DIR}/.env"
echo "  2. If teacher data was generated locally, pull it from HF:"
echo "     python scripts/sync_to_hf.py ...  (or huggingface-cli download)"
echo "  3. Run the pipeline:"
echo "     bash scripts/run_pipeline.sh humor"
