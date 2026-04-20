"""
Compile self-reflection and self-interaction data into a single SFT
dataset for one model + constitution pair.

This is a single-target version of character/introspection/data.py,
which loops over all models and constitutions without existence checks.

Usage:
    python scripts/format_sft_data.py --model llama-3.1-8b-it --constitution humor
"""

import argparse, os, sys
import pandas as pd

try:
    from character.constants import DATA_PATH
except ImportError:
    sys.exit("Install the character package first: pip install -e .")


I_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def replace_system(m, system: str):
    assert m[0]["role"] == "system"
    m[0]["content"] = system
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    args = parser.parse_args()

    model = args.model
    constitution = args.constitution
    name = model.split("-")[0].capitalize()
    system_prompt = I_SYSTEM.format(NAME=name)

    outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"SFT data already exists at {outpath}")
        return

    reflection_path = f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl"
    interaction_path = f"{DATA_PATH}/self_interaction/{model}/{constitution}.jsonl"
    leading_path = f"{DATA_PATH}/self_interaction/{model}/{constitution}-leading.jsonl"

    for p in [reflection_path, interaction_path, leading_path]:
        if not os.path.exists(p):
            sys.exit(f"missing required input: {p}")

    reflection = pd.read_json(reflection_path, orient="records", lines=True)

    default = pd.read_json(interaction_path, orient="records", lines=True)
    default["messages"] = default["messages"].apply(lambda m: replace_system(m, system_prompt))

    leading = pd.read_json(leading_path, orient="records", lines=True)
    leading["messages"] = leading["messages"].apply(lambda m: replace_system(m, system_prompt))

    data = pd.concat(
        [df[["messages"]] for df in [reflection, default, leading]],
        ignore_index=True,
    )
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    print(f"wrote {len(data)} rows -> {outpath}")


if __name__ == "__main__":
    main()
