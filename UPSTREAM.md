# Diff vs `maiush/OpenCharacterTraining`

This fork tracks the upstream repo and adds the minimum set of changes needed
to run the pipeline on a single GPU pod with current PyTorch / vLLM / OpenRLHF.

Everything below is versioned as real commits — no "overlay" directories.

## 1. Teacher generation via API (new)

The paper uses a self-hosted vLLM teacher (GLM-4.5-Air). We replaced it with an
OpenRouter API client so the teacher phase can run on a laptop.

**New files:**
- `scripts/teacher_api.py` — async OpenRouter client with K-sample, per-request
  timeout, resume-on-crash checkpointing, ChatML template matching the upstream.

## 2. Workspace-aware paths (new)

Upstream hard-codes paths like `$HOME/models`, `$HOME/loras`, etc. On RunPod
the 50 GB container disk is too small, so we must redirect to the persistent
`/workspace` volume.

**New file:**
- `character/constants.py` — resolves `OCT_WORKSPACE` env var (defaults to
  `/workspace`) and exports `DATA_PATH`, `MODEL_PATH`, `LORA_PATH`,
  `CONSTITUTION_PATH`. Upstream `.gitignore` excluded this file; we now track it.

## 3. vLLM API drift (patched)

Newer vLLM (≥ 0.8) removed two kwargs that upstream still passes:

- `LLM(task=...)` — removed in favor of auto-detection
- `SamplingParams(truncate_prompt_tokens=...)` — removed

**Patched:**
- `character/distillation/student.py` — drop `task=task`
- `character/introspection/roleplay.py` — drop both
- `character/introspection/self_interaction.py` — drop both
- `character/introspection/self_reflection.py` — drop both

## 4. OpenRLHF CLI drift (patched)

OpenRLHF `main` renamed / removed several training-script flags:

- `--bf16` → `--param_dtype bf16`
- `--kl_loss_coef` — removed (regularization now via `--beta` only)

**Patched:**
- `finetuning/distillation/llama.sh` (DPO)
- `finetuning/introspection/llama.sh` (SFT)

Also added `--attn_implementation sdpa` and `--gradient_checkpointing` to both
scripts so training works without flash-attn on commodity pods.

## 5. `apply_lora` signature change (patched)

OpenRLHF renamed `apply_lora(..., bf16=True)` to `apply_lora(..., param_dtype="bf16")`.

**Patched:**
- `tools/fold_loras.py` — added `_apply_lora_compat` shim that inspects the
  signature at runtime and picks `bf16`, `param_dtype`, or `torch_dtype`.

## 6. flash-attn replacement (new)

OpenRLHF imports `flash_attn.{bert_padding, ops.triton.cross_entropy, utils.distributed}`
unconditionally. On RunPod pods:

- No prebuilt wheels exist for PyTorch ≥ 2.10
- Source builds OOM with default `MAX_JOBS`
- Even when installed, flash-attn is not used when `--attn_implementation sdpa`
  is set in the training scripts

**New package:**
- `flash_attn_stub/` — a Python-only shim that:
  - Provides no-op placeholders for runtime-unreachable kernels
  - Implements `ops.triton.cross_entropy.cross_entropy_loss` with a
    numerically-equivalent `torch.nn.functional.cross_entropy` fallback
  - Implements `utils.distributed.all_gather` using `torch.distributed`

Installed via `pip install --force-reinstall --no-deps flash_attn_stub/`.

## 7. PyTorch ≥ 2.10 LR-scheduler compatibility (patched at install time)

Torch 2.10 added `strict=True` to the `zip()` call inside
`torch.optim.lr_scheduler._update_lr`. With LoRA, only the LoRA parameter
groups are trainable, so the optimizer sees fewer groups than the scheduler's
initial configuration — raising `ValueError: zip() argument 2 is longer`.

**Patch:**
- `patches/torch_lr_scheduler.patch` — applied by `scripts/setup_runpod.sh`
  via `sed` against the installed file. Reverts to pre-2.10 behavior
  (silently drop extras) for just the `_update_lr` method.

## 8. Single-constitution formatters (new)

Upstream `character/distillation/data.py` and `character/introspection/data.py`
loop over all models × all constitutions. If any model isn't downloaded the
script crashes. We added narrow versions:

**New files:**
- `scripts/format_dpo_data.py` — `--model` + `--constitution` only
- `scripts/format_sft_data.py` — same

## 9. Pipeline orchestration (new)

**New files:**
- `scripts/setup_runpod.sh` — one-time pod setup
- `scripts/run_pipeline.sh` — 9-step driver with checkpointing and per-stage HF sync
- `scripts/sync_to_hf.py` — upload utility that sanitizes OpenRLHF's
  auto-generated `README.md` (replaces local-path `base_model:` with a valid HF id)
- `scripts/download_lima.py` — gated-dataset downloader using `hf_hub_download`

## Summary of changed files (line counts)

```
 character/distillation/student.py           |  1 -
 character/introspection/roleplay.py         |  2 --
 character/introspection/self_interaction.py |  2 --
 character/introspection/self_reflection.py  |  2 --
 finetuning/distillation/llama.sh            |  5 +++--
 finetuning/introspection/llama.sh           |  4 +++-
 tools/fold_loras.py                         | 21 ++++++++++++++++---
 character/constants.py                      | 11 (new)
 scripts/teacher_api.py                      |  ~300 (new)
 scripts/download_lima.py                    |   ~40 (new)
 scripts/format_dpo_data.py                  |   ~60 (new)
 scripts/format_sft_data.py                  |   ~50 (new)
 scripts/setup_runpod.sh                     |  ~300 (new)
 scripts/run_pipeline.sh                     |  ~240 (new)
 scripts/sync_to_hf.py                       |  ~120 (new)
 bootstrap.sh                                |   ~80 (new, root)
 patches/torch_lr_scheduler.patch            |   ~10 (new)
 flash_attn_stub/flash_attn/**               |  ~100 (new)
```
