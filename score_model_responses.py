#!/usr/bin/env python3
"""Score unscored model-response JSON files with local gpt-oss-120b via vLLM."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

SCORING_MODEL_TAG = "oss120"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "gpt-oss-120b"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "scoring_prompt.txt"
MAX_ATTEMPTS = 5


def filename_uses_answer_key(path: str | Path) -> bool:
    """Return True if this response JSON should be graded against an answer key."""
    name = os.path.basename(str(path))
    return (
        name.endswith("standard.json")
        or "_aad_" in name
        or "_iasd_" in name
        or "_oe_solvable" in name
    )


def extract_final_answer(completion: str) -> str:
    """Extract the gpt-oss final channel, falling back to raw text."""
    marker = "<|channel|>final<|message|>"
    if marker in completion:
        text = completion.rsplit(marker, 1)[1]
        for stop in ("<|return|>", "<|end|>", "<|start|>"):
            text = text.split(stop, 1)[0]
        return text.strip()
    return completion.strip()


def normalize_score(text: str) -> str:
    """Reduce model output to T/F when possible."""
    text = extract_final_answer(text).strip()
    if text in {"T", "F"}:
        return text
    # Accept a lone T/F even if surrounded by whitespace/punctuation.
    match = re.search(r"\b([TtFf])\b", text)
    if match:
        return match.group(1).upper()
    if text.upper() in {"T", "F"}:
        return text.upper()
    return text


def build_scoring_prompt(
    scoring_prompt: str,
    question: str,
    correct_answer: str,
    model_response: str,
) -> str:
    return (
        f"{scoring_prompt}\n"
        f"QUESTION:{question}\n"
        f"CORRECT_ANSWER: {correct_answer}\n"
        f"MODEL_RESPONSE: {model_response}"
    )


def default_output_dir(json_file: Path) -> Path:
    """../../scored_model_responses/<input_folder>_oss120 relative to the JSON file."""
    input_folder = json_file.parent
    folder_name = input_folder.name
    return (input_folder / ".." / ".." / "scored_model_responses" / f"{folder_name}_{SCORING_MODEL_TAG}").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score model responses with local gpt-oss-120b (vLLM)."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON file containing unscored model responses.",
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        required=True,
        help="Path to the answer key JSON file (required).",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help=(
            "Optional output directory. Default: "
            "../../scored_model_responses/<input_folder>_oss120 relative to the input JSON."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Local gpt-oss-120b path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--scoring-prompt",
        type=str,
        default=str(DEFAULT_PROMPT_PATH),
        help=f"Path to scoring_prompt.txt (default: {DEFAULT_PROMPT_PATH})",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (default: CUDA_VISIBLE_DEVICES count or 1).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization (default: auto by GPU size).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens per score (includes low reasoning + final T/F).",
    )
    parser.add_argument(
        "--reasoning",
        choices=("low", "medium", "high"),
        default="low",
        help="gpt-oss reasoning effort (default: low).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max model length (default: auto by GPU size).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM max concurrent sequences (default: auto by GPU size).",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable CUDA graphs (default: auto; enabled on <=24GB GPUs).",
    )
    return parser.parse_args()


def resolve_tensor_parallel_size(cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return max(1, len([x for x in visible.split(",") if x != ""]))
    return 1


def detect_gpu_memory_gib() -> float:
    """Return GiB of GPU 0, or a conservative fallback if CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 24.0


def resolve_vllm_memory_settings(args: argparse.Namespace) -> dict:
    """
    Choose vLLM memory knobs from GPU size.

    A30 (24GB) is tight for gpt-oss-120b TP=4: weights alone leave little room,
    so concurrent seqs / context / CUDA graphs must be constrained.
    """
    gpu_mem = detect_gpu_memory_gib()

    if gpu_mem < 30:
        # A30-class 24GB
        defaults = {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 2048,
            "max_num_seqs": 16,
            "enforce_eager": True,
        }
    elif gpu_mem < 60:
        # L40S-class 48GB
        defaults = {
            "gpu_memory_utilization": 0.92,
            "max_model_len": 4096,
            "max_num_seqs": 64,
            "enforce_eager": False,
        }
    else:
        # H100-class 80GB+
        defaults = {
            "gpu_memory_utilization": 0.95,
            "max_model_len": 4096,
            "max_num_seqs": 128,
            "enforce_eager": False,
        }

    settings = {
        "gpu_memory_utilization": (
            args.gpu_memory_utilization
            if args.gpu_memory_utilization is not None
            else defaults["gpu_memory_utilization"]
        ),
        "max_model_len": (
            args.max_model_len if args.max_model_len is not None else defaults["max_model_len"]
        ),
        "max_num_seqs": (
            args.max_num_seqs if args.max_num_seqs is not None else defaults["max_num_seqs"]
        ),
        "enforce_eager": (
            args.enforce_eager if args.enforce_eager is not None else defaults["enforce_eager"]
        ),
    }
    print(
        f"GPU0 memory ≈ {gpu_mem:.1f} GiB; vLLM settings: "
        f"gpu_memory_utilization={settings['gpu_memory_utilization']}, "
        f"max_model_len={settings['max_model_len']}, "
        f"max_num_seqs={settings['max_num_seqs']}, "
        f"enforce_eager={settings['enforce_eager']}"
    )
    return settings


def load_llm(args: argparse.Namespace):
    from vllm import LLM

    model_path = Path(args.model_path).resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run scripts/download_gpt_oss_120b.sh first."
        )

    tp = resolve_tensor_parallel_size(args.tensor_parallel_size)
    mem = resolve_vllm_memory_settings(args)
    print(f"Loading vLLM model from {model_path} (tensor_parallel_size={tp})")
    return LLM(
        model=str(model_path),
        tensor_parallel_size=tp,
        trust_remote_code=True,
        gpu_memory_utilization=mem["gpu_memory_utilization"],
        max_model_len=mem["max_model_len"],
        max_num_seqs=mem["max_num_seqs"],
        enforce_eager=mem["enforce_eager"],
        # PCIe multi-GPU nodes (A30/L40S) cannot use custom allreduce for TP>2.
        disable_custom_all_reduce=(tp > 2),
    )


def score_batch(llm, prompts: list[str], args: argparse.Namespace) -> list[str]:
    from vllm import SamplingParams

    if not prompts:
        return []

    conversations = [
        [{"role": "user", "content": prompt}] for prompt in prompts
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        use_tqdm=True,
        chat_template_kwargs={"reasoning_effort": args.reasoning},
    )
    scores: list[str] = []
    for output in outputs:
        raw = output.outputs[0].text if output.outputs else ""
        scores.append(normalize_score(raw))
    return scores


def main() -> int:
    args = parse_args()
    json_file = Path(args.json_file).resolve()
    if not json_file.is_file():
        print(f"ERROR: JSON file not found: {json_file}", file=sys.stderr)
        return 1

    answer_key_path = Path(args.answer_key).resolve()
    if not answer_key_path.is_file():
        print(f"ERROR: Answer key not found: {answer_key_path}", file=sys.stderr)
        return 1

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(answer_key_path, "r", encoding="utf-8") as f:
        answer_key = json.load(f)
    with open(args.scoring_prompt, "r", encoding="utf-8") as f:
        scoring_prompt = f.read()

    use_answer_key = filename_uses_answer_key(json_file)

    # Prepare items in stable order.
    point_clouds = list(data.keys())
    to_score_ids: list[str] = []
    to_score_prompts: list[str] = []

    for point_cloud in point_clouds:
        item = data[point_cloud]
        if use_answer_key and point_cloud in answer_key:
            correct_answer = answer_key[point_cloud]
        else:
            correct_answer = "The question is unanswerable, or none of the above."
        data[point_cloud]["correct_answer"] = correct_answer

        response = item.get("response", "")
        if response == "":
            print(f"Empty response detected for {point_cloud}, assigning score='F' without model call")
            data[point_cloud]["score"] = "F"
            continue

        prompt = build_scoring_prompt(
            scoring_prompt,
            item["prompt"],
            correct_answer,
            response,
        )
        to_score_ids.append(point_cloud)
        to_score_prompts.append(prompt)

    scoring_failed = False
    failed_id = None
    failed_text = None

    if to_score_ids:
        llm = load_llm(args)

        # First pass: batch everything for throughput.
        scores = score_batch(llm, to_score_prompts, args)
        pending_retry_ids: list[str] = []
        pending_retry_prompts: list[str] = []

        last_invalid_scores: dict[str, str] = {}
        for point_cloud, prompt, score in zip(to_score_ids, to_score_prompts, scores):
            if score in {"T", "F"}:
                data[point_cloud]["score"] = score
            else:
                last_invalid_scores[point_cloud] = score
                pending_retry_ids.append(point_cloud)
                pending_retry_prompts.append(prompt)
                print(f"Invalid score '{score}' for {point_cloud}; will retry")

        # Retries for invalid outputs (up to MAX_ATTEMPTS total attempts including first).
        attempt = 1
        while pending_retry_ids and attempt < MAX_ATTEMPTS:
            attempt += 1
            print(f"Retry attempt {attempt}/{MAX_ATTEMPTS} for {len(pending_retry_ids)} items")
            retry_scores = score_batch(llm, pending_retry_prompts, args)
            next_ids: list[str] = []
            next_prompts: list[str] = []
            for point_cloud, prompt, score in zip(
                pending_retry_ids, pending_retry_prompts, retry_scores
            ):
                if score in {"T", "F"}:
                    data[point_cloud]["score"] = score
                else:
                    last_invalid_scores[point_cloud] = score
                    next_ids.append(point_cloud)
                    next_prompts.append(prompt)
                    print(f"Invalid score '{score}' for {point_cloud} on attempt {attempt}")
            pending_retry_ids = next_ids
            pending_retry_prompts = next_prompts

        if pending_retry_ids:
            scoring_failed = True
            failed_id = pending_retry_ids[0]
            failed_text = last_invalid_scores.get(failed_id, data.get(failed_id, {}).get("score", ""))
            data[failed_id]["score"] = failed_text
            for point_cloud in pending_retry_ids:
                data[point_cloud]["score"] = last_invalid_scores.get(
                    point_cloud, data.get(point_cloud, {}).get("score", "")
                )
            print(f"ERROR: Invalid score '{failed_text}' for {failed_id}")
            print("Expected 'T' or 'F'")
            print("Stopping scoring and saving with FAILED_ prefix")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(json_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_filename = json_file.stem
    if scoring_failed:
        output_filename = f"FAILED_{base_filename}_{SCORING_MODEL_TAG}_scored.json"
    else:
        output_filename = f"{base_filename}_{SCORING_MODEL_TAG}_scored.json"
    output_file = output_dir / output_filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"\nScored responses saved to: {output_file}")
    return 1 if scoring_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
