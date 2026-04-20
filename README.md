# Open Character Training — RunPod-ready fork

Reproducible fork of [`maiush/OpenCharacterTraining`](https://github.com/maiush/OpenCharacterTraining)
(by [@saurav1004](https://github.com/saurav1004)) with:

- **API-based teacher generation** (OpenRouter) instead of self-hosted vLLM — removes the need for two GPUs
- **Single-constitution pipeline driver** that resumes from any stage and syncs to HuggingFace after each step
- **Compatibility patches** for PyTorch ≥ 2.10, current vLLM, and OpenRLHF `main`
- **One-command bootstrap** for a fresh RunPod instance

See [UPSTREAM.md](UPSTREAM.md) for the precise diff against the original repo.

---

## Trained persona (reference run)

| Artifact | Location |
|---|---|
| **Humor persona LoRA** (rank-64 weighted linear merge of DPO + SFT adapters, fp32) | [`expx/oct-llama-3.1-8b-humor`](https://huggingface.co/expx/oct-llama-3.1-8b-humor) |
| **Training data** (teacher / DPO / SFT / introspection) | [`expx/oct-humor-data`](https://huggingface.co/datasets/expx/oct-humor-data) |

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
out = llm.generate(["What's it like to be you?"], SamplingParams(temperature=0.8, max_tokens=256), lora_request=persona)
print(out[0].outputs[0].text)
```

---

## Quickstart — train a new persona from scratch

**Prerequisites.** A RunPod instance with:
- A100 80 GB (or H100 80 GB) GPU
- ≥ 150 GB persistent volume at `/workspace`
- ≥ 50 GB container disk
- `runpod/pytorch:2.x-py3.11-cuda12.x-devel`

Plus accounts/tokens for: HuggingFace (write), OpenRouter, Weights & Biases.

### 1. Teacher-data generation (local, ~2 h with `K=5`, temperature sampling)

Run on your laptop — uses the OpenRouter API, no GPU needed:

```bash
git clone https://github.com/saurav1004/OpenCharacterTraining.git
cd OpenCharacterTraining
pip install -r requirements-local.txt       # minimal: openai, pandas, huggingface_hub
export HF_TOKEN=...
export OPENROUTER_API_KEY=...

python scripts/teacher_api.py \
    --constitution humor \
    --teacher z-ai/glm-4.5-air \
    --model llama-3.1-8b-it \
    --K 5 \
    --max-tokens 2048 \
    --concurrency 100

# Upload so the pod can pick it up
python scripts/sync_to_hf.py \
    --repo-id expx/oct-humor-data \
    --local-dir data/distillation \
    --path-in-repo stages \
    --repo-type dataset
```

### 2. Pod bootstrap and full pipeline

On the RunPod instance:

```bash
export HF_TOKEN=...
export OPENROUTER_API_KEY=...
export WANDB_TOKEN=...

curl -sL https://raw.githubusercontent.com/saurav1004/OpenCharacterTraining/main/bootstrap.sh \
    | bash -s -- humor expx/oct-humor-data
```

This `bootstrap.sh`:

1. Sets up cache redirection to `/workspace` (critical on small-disk pods)
2. Clones this repo, installs dependencies (incl. vLLM + OpenRLHF `--no-deps`)
3. Builds and installs the `flash_attn_stub` package
4. Applies the PyTorch ≥ 2.10 LR-scheduler patch
5. Downloads `meta-llama/Llama-3.1-8B-Instruct` and the `lima` dataset
6. Pulls teacher data from HF
7. Runs the 9-step pipeline (`scripts/run_pipeline.sh`) end-to-end with per-stage HF sync

Expected runtime on A100 80 GB with `K=5` (humor): **~4 hours**.

### 3. Pipeline stages

| Step | Script | Output |
|------|--------|--------|
| 1 | `character.distillation.student`   | student-model continuations |
| 2 | `scripts/format_dpo_data.py`       | chosen/rejected pairs |
| 3 | `finetuning/distillation/llama.sh` | DPO LoRA |
| 4 | `tools/fold_loras.py`              | DPO folded into base weights |
| 5 | `character.introspection.self_reflection` | self-reflection data |
| 6 | `character.introspection.self_interaction` (×2) | self-interaction data |
| 7 | `scripts/format_sft_data.py`       | SFT training targets |
| 8 | `finetuning/introspection/llama.sh` | SFT LoRA |
| 9 | `tools/merge_loras.py`             | final persona LoRA (DPO + SFT) |

---

## Layout

```
OpenCharacterTraining/
├── bootstrap.sh                     # one-command pod setup + pipeline
├── UPSTREAM.md                      # diff vs maiush/OpenCharacterTraining
├── requirements-local.txt           # deps for teacher-data generation (CPU)
├── requirements-pod.txt             # pinned pod env (from reference run)
│
├── character/                       # upstream + patches
│   ├── constants.py                 # (new) workspace-aware paths
│   ├── distillation/student.py      # vLLM `task` kwarg removed
│   └── introspection/*.py           # vLLM `truncate_prompt_tokens` removed
│
├── tools/
│   ├── fold_loras.py                # apply_lora compat shim (bf16/param_dtype)
│   └── merge_loras.py               # upstream
│
├── finetuning/
│   └── {distillation,introspection}/llama.sh   # --param_dtype bf16, SDPA, gc
│
├── scripts/                         # (new) orchestration layer
│   ├── teacher_api.py               # OpenRouter async teacher generation
│   ├── download_lima.py             # gated LIMA dataset
│   ├── format_dpo_data.py           # single-constitution DPO formatter
│   ├── format_sft_data.py           # single-constitution SFT formatter
│   ├── setup_runpod.sh              # pod setup (caches, stub, patches)
│   ├── run_pipeline.sh              # 9-step pipeline driver
│   └── sync_to_hf.py                # artifact upload w/ YAML sanitizer
│
├── patches/
│   └── torch_lr_scheduler.patch     # strict=True -> strict=False (torch ≥ 2.10)
│
└── flash_attn_stub/                 # mock flash-attn package
    └── flash_attn/
        ├── __init__.py
        ├── bert_padding/
        ├── ops/triton/cross_entropy.py  # torch-native replacement
        └── utils/distributed.py
```

---

## Credits

All methodology and the original implementation are by Sharan Maiya et al. (Open Character Training, 2025).

```bibtex
@misc{maiya2025opencharactertrainingshaping,
      title={Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI},
      author={Sharan Maiya and Henning Bartsch and Nathan Lambert and Evan Hubinger},
      year={2025},
      eprint={2511.01689},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01689}
}
```

- Paper: <https://arxiv.org/abs/2511.01689>
- Upstream repo: <https://github.com/maiush/OpenCharacterTraining>

## License

MIT (inherited from upstream).
