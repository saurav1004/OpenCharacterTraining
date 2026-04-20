"""
Download the GAIR/lima dataset and save as JSONL files in the format
expected by character/distillation/teacher.py:

    {MODEL_PATH}/lima/train.jsonl
    {MODEL_PATH}/lima/test.jsonl

Each line is a JSON object with a "conversations" key containing a list
of strings (alternating user/assistant turns).

Usage:
    python scripts/download_lima.py
    python scripts/download_lima.py --output-dir /workspace/models/lima
"""

import argparse, json, os, sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub package required: pip install huggingface_hub")


def resolve_output_dir(override: str | None) -> str:
    if override:
        return override
    try:
        from character.constants import MODEL_PATH
        return os.path.join(MODEL_PATH, "lima")
    except ImportError:
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        return str(repo.parent / "models" / "lima")


def main():
    parser = argparse.ArgumentParser(description="Download LIMA dataset")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out = resolve_output_dir(args.output_dir)
    os.makedirs(out, exist_ok=True)

    train_path = os.path.join(out, "train.jsonl")
    test_path = os.path.join(out, "test.jsonl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"LIMA data already exists at {out}, skipping download")
        return

    token = os.environ.get("HF_TOKEN", None)

    for split, dest in [("train", train_path), ("test", test_path)]:
        if os.path.exists(dest):
            print(f"  {split} already exists, skipping")
            continue

        print(f"downloading GAIR/lima {split} split...")
        src = hf_hub_download(
            repo_id="GAIR/lima",
            filename=f"{split}.jsonl",
            repo_type="dataset",
            token=token,
        )
        # Copy to our target location and ensure correct JSONL format
        rows = []
        with open(src, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        with open(dest, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {len(rows)} rows -> {dest}")

    print("done")


if __name__ == "__main__":
    main()
