#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

mkdir -p "${ROOT_DIR}/data" "$HOME/models" "$HOME/loras" "$HOME/models/distilled"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  pip install -r "${ROOT_DIR}/requirements.txt"
fi

pip install vllm
pip install -e "${ROOT_DIR}/openrlhf"
pip install "flash_attn==2.7.4.post1" --no-build-isolation || true
pip install -e "${ROOT_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo "RunPod setup complete."
echo "Activate env: source ${VENV_DIR}/bin/activate"
