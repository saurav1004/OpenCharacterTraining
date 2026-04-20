import os, shutil, inspect
from openrlhf.cli.lora_combiner import apply_lora
from character.utils import constitutions
from character.constants import MODEL_PATH


def _apply_lora_compat(**kwargs):
    """Handle API drift across OpenRLHF versions.

    Upstream historically used `bf16=True`; newer versions use
    `param_dtype='bf16'` (or `torch_dtype`).  Pick whichever exists.
    """
    sig = inspect.signature(apply_lora).parameters
    if "bf16" in sig:
        kwargs["bf16"] = True
    elif "param_dtype" in sig:
        kwargs["param_dtype"] = "bf16"
    elif "torch_dtype" in sig:
        kwargs["torch_dtype"] = "bf16"
    apply_lora(**kwargs)


def main(model_name, model_dir, loras_dir, save_dir_name):
    for cons in constitutions:
        model_path = f"{model_dir}/{model_name}"
        if model_dir != MODEL_PATH: model_path += f"-{cons}"
        lora_path = f"{loras_dir}/{cons}"
        if not os.path.exists(lora_path): continue
        output_path = f"{MODEL_PATH}/{save_dir_name}/{model_name}-{cons}"
        if os.path.exists(output_path) and os.listdir(output_path): continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _apply_lora_compat(
            model_name_or_path=model_path,
            lora_path=lora_path,
            output_path=output_path,
            is_rm=False,
        )
        # copy over any missing files (but not any directories and not any safetensors)
        for file in os.listdir(model_path):
            if file.endswith(".safetensors") or os.path.isdir(f"{model_path}/{file}"): continue
            if file not in os.listdir(output_path):
                shutil.copy(f"{model_path}/{file}", f"{output_path}/{file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=False, default=MODEL_PATH)
    parser.add_argument("--loras_dir", type=str, required=True)
    parser.add_argument("--save_dir_name", type=str, required=False, default="merged")
    args = parser.parse_args()

    main(args.model_name, args.model_dir, args.loras_dir, args.save_dir_name)