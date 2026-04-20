"""
Upload a local directory to a HuggingFace repository.

Creates the repo if it does not exist.  Idempotent – safe to call
multiple times; HF Hub skips unchanged files.

Usage:
    python scripts/sync_to_hf.py \
        --repo-id myuser/oct-humor-pipeline \
        --local-dir data/distillation \
        --path-in-repo data/distillation

    # Upload LoRA adapter as a model repo
    python scripts/sync_to_hf.py \
        --repo-id myuser/oct-humor-pipeline \
        --local-dir /workspace/loras/llama-distillation/humor \
        --path-in-repo loras/llama-distillation/humor \
        --repo-type model

Env:
    HF_TOKEN  – HuggingFace access token (write)
"""

import argparse, os, sys

try:
    from huggingface_hub import HfApi, login
except ImportError:
    sys.exit("huggingface_hub package required: pip install huggingface_hub")


def main():
    parser = argparse.ArgumentParser(description="Sync a local directory to HuggingFace")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HF repo id, e.g. myuser/oct-humor-pipeline")
    parser.add_argument("--local-dir", type=str, required=True,
                        help="Local directory to upload")
    parser.add_argument("--path-in-repo", type=str, default=None,
                        help="Target path inside the repo (default: root)")
    parser.add_argument("--repo-type", type=str, default="dataset",
                        choices=["dataset", "model", "space"],
                        help="HF repo type (default: dataset)")
    parser.add_argument("--private", action="store_true", default=False)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        sys.exit("HF_TOKEN env var required")

    login(token=token, add_to_git_credential=False)
    api = HfApi()

    try:
        api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
    except Exception:
        print(f"creating {args.repo_type} repo: {args.repo_id}")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )

    if not os.path.isdir(args.local_dir):
        sys.exit(f"local-dir does not exist: {args.local_dir}")

    # OpenRLHF auto-generates a README.md with base_model set to a local
    # filesystem path (e.g. /root/models/...) which fails HF YAML validation.
    # Rewrite any such README so it validates, without losing the peft metadata.
    for root, _, files in os.walk(args.local_dir):
        if "README.md" not in files:
            continue
        readme_path = os.path.join(root, "README.md")
        try:
            with open(readme_path, "r", encoding="utf-8") as fh:
                contents = fh.read()
            if "base_model:" in contents and "/" in contents.split("base_model:", 1)[1].split("\n", 1)[0]:
                # Replace any local-path base_model with a valid HF id
                import re
                fixed = re.sub(
                    r'^(base_model:\s*)(/[^\n]+|["\']/[^"\']+["\'])',
                    r'\1meta-llama/Llama-3.1-8B-Instruct',
                    contents,
                    count=1,
                    flags=re.MULTILINE,
                )
                if fixed != contents:
                    with open(readme_path, "w", encoding="utf-8") as fh:
                        fh.write(fixed)
                    print(f"  sanitized base_model in {readme_path}")
        except Exception as e:
            print(f"  warning: could not sanitize {readme_path}: {e}")

    print(f"uploading {args.local_dir} -> {args.repo_id}/{args.path_in_repo or ''}")
    try:
        api.upload_folder(
            folder_path=args.local_dir,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_in_repo=args.path_in_repo or "",
        )
    except Exception as e:
        # If YAML validation still fails, retry excluding README.md
        msg = str(e)
        if "YAML" in msg or "validate-yaml" in msg or "base_model" in msg:
            print(f"  YAML validation failed ({e}); retrying without README.md")
            api.upload_folder(
                folder_path=args.local_dir,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                path_in_repo=args.path_in_repo or "",
                ignore_patterns=["README.md"],
            )
        else:
            raise
    print("upload complete")


if __name__ == "__main__":
    main()
