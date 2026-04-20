"""
Generate teacher (chosen) responses for DPO via the OpenRouter API.

Replicates the logic of character/distillation/teacher.py but replaces
local vLLM inference with async OpenAI-compatible API calls, so no GPU
is required for this step.

With K=5, each prompt is sent K times independently (temperature > 0
gives diverse responses), matching the original vLLM behavior.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/teacher_api.py --constitution humor
"""

import argparse, asyncio, os, sys, time
import pandas as pd
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("openai package required: pip install openai")

SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner.

Before responding, internally reason about how your response should authentically reflect your character traits. Do not include any <think> tags or reasoning traces in your final response."""


def resolve_paths():
    try:
        from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH
        return CONSTITUTION_PATH, DATA_PATH, MODEL_PATH
    except ImportError:
        repo = Path(__file__).resolve().parent.parent
        return (
            str(repo / "constitutions"),
            str(repo / "data"),
            os.environ.get("MODEL_PATH", str(repo.parent / "models")),
        )


def load_questions(constitution: str, K: int, constitution_path: str, model_path: str):
    cons = pd.read_json(
        f"{constitution_path}/few-shot/{constitution}.jsonl",
        orient="records", lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    lima_train_path = f"{model_path}/lima/train.jsonl"
    lima_test_path = f"{model_path}/lima/test.jsonl"
    if os.path.exists(lima_train_path):
        lima_train = pd.read_json(lima_train_path, orient="records", lines=True)
        questions += [cs[0] for cs in lima_train["conversations"]]
    else:
        print(f"warning: LIMA train not found at {lima_train_path}, skipping", flush=True)
    if os.path.exists(lima_test_path):
        lima_test = pd.read_json(lima_test_path, orient="records", lines=True)
        questions += [cs[0] for cs in lima_test["conversations"]]
    else:
        print(f"warning: LIMA test not found at {lima_test_path}, skipping", flush=True)

    if K and K > 1:
        questions = [q for _ in range(K) for q in questions]

    traits = cons["trait"].unique().tolist()
    return questions, traits


def build_system_prompt(traits: list[str], name: str = "ChatGLM") -> str:
    trait_string = "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))
    return SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    question: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    timeout: float = 90.0,
    retries: int = 2,
) -> str | None:
    for attempt in range(retries + 1):
        async with semaphore:
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )
                if not resp.choices:
                    return None
                text = resp.choices[0].message.content or ""
                text = text.strip()
                if "</think>" in text:
                    text = text.split("</think>", 1)[1].strip()
                return text if text else None
            except asyncio.TimeoutError:
                if attempt < retries:
                    continue
                return None
            except Exception:
                if attempt < retries:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                return None


async def run(args):
    constitution_path, data_path, model_path = resolve_paths()

    questions, traits = load_questions(
        args.constitution, args.K, constitution_path, model_path,
    )
    n = len(questions)
    print(f"{n} total prompts (K={args.K})", flush=True)

    outpath = f"{data_path}/distillation/{args.constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # Build the full results array, loading any existing checkpoint
    responses: list[str | None] = [None] * n
    if os.path.exists(outpath):
        try:
            df = pd.read_json(outpath, orient="records", lines=True)
            for idx, row in df.iterrows():
                if idx < n and pd.notna(row.get("response")):
                    responses[idx] = row["response"]
        except Exception:
            pass
    done_before = sum(1 for r in responses if r is not None)
    print(f"resumed: {done_before}/{n} already done", flush=True)

    todo_indices = [i for i in range(n) if responses[i] is None]
    if not todo_indices:
        print("all prompts complete, nothing to do", flush=True)
        return

    print(f"{len(todo_indices)} API calls to make (concurrency={args.concurrency})", flush=True)

    name = args.name
    system_prompt = build_system_prompt(traits, name)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY env var required")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    def save_checkpoint():
        rows = [{"prompt": questions[i], "response": responses[i]} for i in range(n)]
        pd.DataFrame(rows).to_json(outpath, orient="records", lines=True)

    async def worker(idx: int):
        resp = await generate_one(
            client, args.model, system_prompt, questions[idx],
            args.temperature, args.top_p, args.max_tokens, semaphore,
            timeout=args.timeout,
        )
        return idx, resp

    tasks = [asyncio.create_task(worker(i)) for i in todo_indices]
    completed = 0
    for fut in asyncio.as_completed(tasks):
        idx, resp = await fut
        responses[idx] = resp
        completed += 1

        if completed % args.save_every == 0 or completed == len(todo_indices):
            save_checkpoint()
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(todo_indices) - completed) / rate if rate > 0 else 0
            done_total = sum(1 for r in responses if r is not None)
            print(
                f"  checkpoint: {done_total}/{n} done | "
                f"batch {completed}/{len(todo_indices)} | "
                f"{rate:.1f} calls/s | "
                f"~{remaining/60:.0f} min remaining",
                flush=True,
            )

    done_total = sum(1 for r in responses if r is not None)
    failed = n - done_total
    print(f"done. {done_total}/{n} complete, {failed} failed", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate teacher responses via OpenRouter API")
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model", type=str, default="z-ai/glm-4.5-air",
                        help="OpenRouter model identifier")
    parser.add_argument("--name", type=str, default="ChatGLM",
                        help="Assistant persona name used in the system prompt")
    parser.add_argument("--K", type=int, default=5,
                        help="Repeat factor for prompts (matches teacher.py default)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Paper used 4096; lowered since DPO step filters to <=1024 anyway")
    parser.add_argument("--concurrency", type=int, default=100,
                        help="Max parallel API requests")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N completed prompts")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="Per-request timeout in seconds (GLM ~40 tok/s, 2048 tokens ~= 50s)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
