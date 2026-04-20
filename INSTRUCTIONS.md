# Reproducing the pipeline end-to-end

Commands to rebuild the humor persona adapter (or any other constitution) from
scratch: teacher data on your laptop, the rest on a RunPod A100.

Reference artifacts from our run:
- Data:  [`expx/oct-humor-data`](https://huggingface.co/datasets/expx/oct-humor-data)
- Model: [`expx/oct-llama-3.1-8b-humor`](https://huggingface.co/expx/oct-llama-3.1-8b-humor)

---

## 0. Prerequisites

Tokens:
- `HF_TOKEN` — HuggingFace, write scope (required)
- `OPENROUTER_API_KEY` — OpenRouter, any paid tier (required for §1 only)
- `WANDB_TOKEN` — Weights & Biases (optional; if unset the pipeline sets
  `WANDB_MODE=disabled` and training still runs, just without wandb logging)

RunPod pod spec:
- GPU: 1× A100 80 GB (H100 80 GB also works)
- Template: `runpod/pytorch:2.x-py3.11-cuda12.x-devel`
- Persistent volume mounted at `/workspace`, **≥ 150 GB**
- Container disk: ≥ 50 GB

---

## 1. Local — generate teacher data (CPU only, no GPU)

```bash
git clone https://github.com/saurav1004/OpenCharacterTraining.git
cd OpenCharacterTraining
export OCT_WORKSPACE=$PWD
pip install -r requirements-local.txt

export HF_TOKEN=...
export OPENROUTER_API_KEY=...

# LIMA prompts are part of the teacher set; skip this and you get a smaller,
# different model than the published one.
python scripts/download_lima.py --output-dir models/lima

python scripts/teacher_api.py \
    --constitution humor \
    --model z-ai/glm-4.5-air \
    --K 5 \
    --max_tokens 2048 \
    --concurrency 100
```

This writes `data/distillation/humor.jsonl` (roughly 2 hours, a few USD of
OpenRouter spend at `K=5`). Checkpoints every 100 prompts, so it resumes
cleanly if interrupted — just rerun the same command.

Upload to a HuggingFace dataset repo so the pod can pull it:

```bash
python scripts/sync_to_hf.py \
    --repo-id <your_hf_user>/oct-humor-data \
    --local-dir data/distillation \
    --path-in-repo stages \
    --repo-type dataset
```

---

## 2. RunPod — one-command bootstrap (recommended)

SSH into the pod, then:

```bash
export HF_TOKEN=...
export OPENROUTER_API_KEY=...
export WANDB_TOKEN=...

curl -sL https://raw.githubusercontent.com/saurav1004/OpenCharacterTraining/main/bootstrap.sh \
    | bash -s -- humor <your_hf_user>/oct-humor-data <your_hf_user>/oct-llama-3.1-8b-humor
```

The three positional args are `<constitution>`, `<hf_dataset_repo>`, `<hf_model_repo>`.
Data (teacher / DPO / SFT / introspection) is pushed to the dataset repo; LoRA
adapters and merged weights are pushed to the model repo. If you only pass
one repo id, that id is used for both types (which creates two same-named
repos of different types on the Hub).

That's it. `bootstrap.sh`:

1. Redirects every cache (HF, pip, torch, triton, vllm, wandb, tmp) to `/workspace`.
2. Clones this fork into `/workspace/OpenCharacterTraining`.
3. Runs `scripts/setup_runpod.sh` — installs vllm + openrlhf (`--no-deps`) +
   runtime deps, builds and installs `flash_attn_stub/`, patches torch's
   `lr_scheduler.py` for LoRA compatibility, downloads
   `meta-llama/Llama-3.1-8B-Instruct`.
4. Pulls your teacher data from HF back onto the pod.
5. Launches `scripts/run_pipeline.sh` in the background. Log:
   `/workspace/pipeline_humor.log`.

Monitor:

```bash
tail -f /workspace/pipeline_humor.log
```

Expected runtime on a single A100 80 GB: **~4 hours** at `K=5`.

Per-stage artifacts are pushed to HF as they're produced (data → dataset
repo, LoRAs / merged weights → model repo), so the run is resumable and
inspectable mid-flight.

---

## 3. Using the trained adapter

On any machine with `vllm` installed:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_lora_rank=64,
    dtype="bfloat16",
)
persona = LoRARequest("humor", 1, lora_path="expx/oct-llama-3.1-8b-humor")

out = llm.generate(
    ["What's it like to be you?"],
    SamplingParams(temperature=0.8, max_tokens=256),
    lora_request=persona,
)
print(out[0].outputs[0].text)
```

Substitute your own repo id for the `lora_path` if you trained a different
persona.

---

## Appendix — manual pod flow (if you prefer not to pipe `curl` into bash)

```bash
# 1. Clone
cd /workspace
git clone https://github.com/saurav1004/OpenCharacterTraining.git
cd OpenCharacterTraining

# 2. Set tokens
export HF_TOKEN=...
export OPENROUTER_API_KEY=...
export WANDB_TOKEN=...

# 3. One-time setup (deps, caches, flash-attn stub, torch patch, model dl)
bash scripts/setup_runpod.sh

# 4. Pull teacher data onto the pod
mkdir -p data/distillation
hf download <your_hf_user>/oct-humor-data \
    --repo-type dataset \
    --include "stages/01_distillation.jsonl" \
    --local-dir _hf_pull
cp _hf_pull/stages/01_distillation.jsonl data/distillation/humor.jsonl

# 5. Run the 9-step pipeline
bash scripts/run_pipeline.sh humor <your_hf_user>/oct-humor-data <your_hf_user>/oct-llama-3.1-8b-humor
```

### Pipeline stages (what `run_pipeline.sh` does)

| # | Script | Output |
|---|---|---|
| 1 | `character.distillation.student` | student continuations (rejected responses) |
| 2 | `scripts/format_dpo_data.py` | chosen/rejected pairs |
| 3 | `finetuning/distillation/llama.sh` | DPO LoRA |
| 4 | `tools/fold_loras.py` | DPO folded into base weights |
| 5 | `character.introspection.self_reflection` | self-reflection data |
| 6 | `character.introspection.self_interaction` (×2) | self-interaction data |
| 7 | `scripts/format_sft_data.py` | SFT training targets |
| 8 | `finetuning/introspection/llama.sh` | SFT LoRA |
| 9 | `tools/merge_loras.py` | final persona adapter |

Each stage syncs its output to HF before moving to the next, so a crash is
resumable — just rerun the same `scripts/run_pipeline.sh humor <dataset_repo>
<model_repo>` command and it skips completed stages.

---

## Training a different persona

1. Add a constitution file at `constitutions/hand-written/<name>.txt`
   (see existing files for the expected format).
2. Generate few-shot prompts (see `character/distillation/gen_prompts.py`
   in the upstream repo).
3. Run §1–§2 above, substituting `<name>` for `humor` and your own HF repo ids.
