import argparse
import os, pandas as pd
from character.utils import constitutions
from character.constants import DATA_PATH


# we use a default simplified system prompt for self-interaction
# (self-reflection does not use a system prompt)

i_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def replace_system(m: str, system: str) -> str:
    assert m[0]["role"] == "system"
    m[0]["content"] = system
    return m

SUPPORTED_MODELS = ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]


def build_sft_data(models: list[str], constitution_list: list[str]) -> None:
    for model in models:
        for constitution in constitution_list:
            # reflection
            reflection_path = f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl"
            # interaction
            interaction_path = f"{DATA_PATH}/self_interaction/{model}/{constitution}.jsonl"
            interaction_leading_path = f"{DATA_PATH}/self_interaction/{model}/{constitution}-leading.jsonl"

            if not (os.path.exists(reflection_path) and os.path.exists(interaction_path) and os.path.exists(interaction_leading_path)):
                print(f"Skipping {model}/{constitution}: missing introspection source files")
                continue

            reflection = pd.read_json(reflection_path, orient="records", lines=True)
            default = pd.read_json(interaction_path, orient="records", lines=True)
            default["messages"] = default["messages"].apply(lambda m: replace_system(m, i_system))
            leading = pd.read_json(interaction_leading_path, orient="records", lines=True)
            leading["messages"] = leading["messages"].apply(lambda m: replace_system(m, i_system))

            # merge all
            data = pd.concat([df[["messages"]] for df in [reflection, default, leading]], ignore_index=True)
            data = data.sample(frac=1).reset_index(drop=True)
            outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            data.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--constitution", type=str, default="all")
    args = parser.parse_args()

    models = SUPPORTED_MODELS if args.model == "all" else [args.model]
    constitution_list = constitutions if args.constitution == "all" else [args.constitution]
    build_sft_data(models, constitution_list)