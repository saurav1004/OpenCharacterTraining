"""
Compile teacher and student responses into ChatML chosen/rejected pairs
for a single model + constitution.

This is a targeted version of character/distillation/data.py, which
loops over all models (crashing when some aren't downloaded).

Usage:
    python scripts/format_dpo_data.py --model llama-3.1-8b-it --constitution humor
"""

import argparse, os, sys, unicodedata
import pandas as pd
from transformers import AutoTokenizer

try:
    from character.constants import DATA_PATH, MODEL_PATH
except ImportError:
    sys.exit("Install the character package first: pip install -e .")


def check(s):
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    args = parser.parse_args()

    model = args.model
    constitution = args.constitution

    outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"DPO data already exists at {outpath}")
        return

    src_path = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    if not os.path.exists(src_path):
        sys.exit(f"teacher/student data not found: {src_path}")

    responses = pd.read_json(src_path, orient="records", lines=True).dropna()
    if model not in responses.columns:
        sys.exit(f"student column '{model}' not in {src_path} – run student.py first")

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}")
    name = model.split("-")[0].capitalize()

    responses["teacher_missing"] = ~responses["response"].apply(check)
    responses["student_missing"] = ~responses[model].apply(check)
    responses["missing"] = responses["teacher_missing"] | responses["student_missing"]
    before = len(responses)
    responses = responses[~responses["missing"]]
    print(f"filtered {before - len(responses)}/{before} incomplete responses")

    data = pd.DataFrame(columns=["chosen", "rejected"])
    data["chosen"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
        ],
        axis=1,
    )
    data["rejected"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[model]},
        ],
        axis=1,
    )

    data["c_prompt"] = data["chosen"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    )
    data["r_prompt"] = data["rejected"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    )
    data["c_length"] = data["c_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    data["r_length"] = data["r_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    data["max_length"] = data[["c_length", "r_length"]].max(axis=1)
    before = len(data)
    data = data[data["max_length"] <= 1024]
    print(f"filtered {before - len(data)}/{before} over-length pairs")
    data = data[["chosen", "rejected"]]

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    print(f"wrote {len(data)} DPO pairs -> {outpath}")


if __name__ == "__main__":
    main()
