#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

mkdir -p "${ROOT_DIR}/data" "$HOME/models" "$HOME/loras" "$HOME/models/distilled"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  rm -rf "${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  pip install -r "${ROOT_DIR}/requirements.txt"
fi

pip install vllm

OPENRLHF_DIR="${ROOT_DIR}/openrlhf"
if [[ ! -f "${OPENRLHF_DIR}/pyproject.toml" && ! -f "${OPENRLHF_DIR}/setup.py" ]]; then
  if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "openrlhf submodule not initialized. Fetching submodule..."
    git -C "${ROOT_DIR}" submodule update --init --recursive openrlhf
  fi
fi

if [[ ! -f "${OPENRLHF_DIR}/pyproject.toml" && ! -f "${OPENRLHF_DIR}/setup.py" ]]; then
  echo "ERROR: ${OPENRLHF_DIR} is missing Python project files."
  echo "Clone with submodules: git clone --recurse-submodules <repo-url>"
  echo "Or run: git submodule update --init --recursive"
  exit 1
fi

pip install -e "${OPENRLHF_DIR}"

if [[ "${RUNPOD_INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  pip install "flash_attn==2.7.4.post1" --no-build-isolation || true
else
  echo "Skipping flash_attn build (set RUNPOD_INSTALL_FLASH_ATTN=1 to enable)."
fi

pip install -e "${ROOT_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo "RunPod setup complete."
echo "Activate env: source ${VENV_DIR}/bin/activate"
