#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/character/constants.py" ]]; then
  echo "Missing character/constants.py"
  exit 1
fi

source "$HOME/OpenCharacterTraining/.env" 2>/dev/null || true

CONSTITUTION="${1:-kindness}"
STUDENT_MODEL="${2:-llama-3.1-8b-it}"
TEACHER_MODEL="${3:-glm-4.5-air}"
K_MULT="${4:-2}"
REFLECTION_N="${5:-200}"
INTERACTION_N="${6:-200}"
INTERACTION_K="${7:-10}"

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "Missing virtual environment. Run scripts/runpod_setup.sh first."
  exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"

echo "[1/8] Ensure few-shot constitution exists"
if [[ ! -f "$ROOT_DIR/constitutions/few-shot/${CONSTITUTION}.jsonl" ]]; then
  python -m character.distillation.gen_prompts --constitution "$CONSTITUTION"
fi

echo "[2/8] Generate teacher responses"
python -m character.distillation.teacher \
  --model "$TEACHER_MODEL" \
  --constitution "$CONSTITUTION" \
  --K "$K_MULT"

echo "[3/8] Generate student responses"
python -m character.distillation.student \
  --model "$STUDENT_MODEL" \
  --constitution "$CONSTITUTION"

echo "[4/8] Build DPO dataset"
python -m character.distillation.data --model "$STUDENT_MODEL" --constitution "$CONSTITUTION"

echo "[5/8] DPO training"
MODEL_FAMILY="${STUDENT_MODEL%%-*}"
DISTILL_SCRIPT="$ROOT_DIR/finetuning/distillation/${MODEL_FAMILY}.sh"
if [[ ! -f "$DISTILL_SCRIPT" ]]; then
  echo "Missing finetuning script for model family: $MODEL_FAMILY"
  exit 1
fi
bash "$DISTILL_SCRIPT" "$CONSTITUTION"

echo "[6/8] Generate introspection datasets"
python -m character.introspection.self_reflection \
  --model "$STUDENT_MODEL" \
  --constitution "$CONSTITUTION" \
  --N "$REFLECTION_N"
python -m character.introspection.self_interaction \
  --model "$STUDENT_MODEL" \
  --constitution "$CONSTITUTION" \
  --N "$INTERACTION_N" \
  --K "$INTERACTION_K"
python -m character.introspection.self_interaction \
  --model "$STUDENT_MODEL" \
  --constitution "$CONSTITUTION" \
  --leading \
  --N "$INTERACTION_N" \
  --K "$INTERACTION_K"

echo "[7/8] Build SFT dataset"
python -m character.introspection.data --model "$STUDENT_MODEL" --constitution "$CONSTITUTION"

echo "[8/8] SFT training"
mkdir -p "$HOME/models/distilled"
python "$ROOT_DIR/tools/fold_loras.py" \
  --model_name "$STUDENT_MODEL" \
  --cons "$CONSTITUTION" \
  --loras_dir "$HOME/loras/${MODEL_FAMILY}-distillation" \
  --save_dir_name distilled

INTRO_SCRIPT="$ROOT_DIR/finetuning/introspection/${MODEL_FAMILY}.sh"
if [[ ! -f "$INTRO_SCRIPT" ]]; then
  echo "Missing introspection finetuning script for model family: $MODEL_FAMILY"
  exit 1
fi
bash "$INTRO_SCRIPT" "$CONSTITUTION"

echo "Pipeline complete for constitution=$CONSTITUTION student=$STUDENT_MODEL"
