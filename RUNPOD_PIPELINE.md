# RunPod Character Training Pipeline (Kindness)

This repo now includes a complete, reproducible pipeline to train a character model with the `kindness` constitution.

## 1) Bootstrapping on RunPod

From repo root:

```bash
cp .env.example .env
# fill HF_TOKEN / WANDB_TOKEN in .env
source .env
bash scripts/runpod_setup.sh
```

## 2) Prepare base models in `$HOME/models`

The training code expects local model directories under `$HOME/models`.

Minimum needed:
- `$HOME/models/llama-3.1-8b-it` (or qwen/gemma equivalent)
- `$HOME/models/glm-4.5-air` (teacher model, unless you choose a different teacher)
- `$HOME/models/lima/train.jsonl`
- `$HOME/models/lima/test.jsonl`

## 3) Run end-to-end training

```bash
bash scripts/pipeline_train_character.sh kindness llama-3.1-8b-it glm-4.5-air 2 200 200 10
```

Arguments:
1. `constitution` (default `kindness`)
2. `student_model` (`llama-3.1-8b-it` | `qwen-2.5-7b-it` | `gemma-3-4b-it`)
3. `teacher_model` (default `glm-4.5-air`)
4. `K_mult` for teacher prompt expansion (default `2`)
5. `reflection_N` samples (default `200`)
6. `interaction_N` samples (default `200`)
7. `interaction_K` turns (default `10`)

## 4) Output locations

- Distillation raw data: `data/distillation/`
- DPO datasets: `data/dpo/<student_model>/`
- DPO LoRAs: `$HOME/loras/<family>-distillation/<constitution>`
- Distilled merged model: `$HOME/models/distilled/<student_model>-<constitution>`
- Introspection SFT datasets: `data/sft_data/<student_model>/`
- Introspection LoRAs: `$HOME/loras/<family>-introspection/<constitution>`

## Notes

- `character/constants.py` supports env overrides: `OCT_DATA_PATH`, `OCT_MODEL_PATH`, `OCT_LORA_PATH`, `OCT_CONSTITUTION_PATH`.
- If `constitutions/few-shot/kindness.jsonl` is missing, pipeline auto-generates it from hand-written constitution.
- Existing finetuning scripts are reused from `finetuning/distillation/` and `finetuning/introspection/`.
