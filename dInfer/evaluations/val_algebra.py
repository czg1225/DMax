from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

from val_math import (
    MATH_VERIFY_IMPORT_ERROR,
    MatchResult,
    SYMPY_IMPORT_ERROR,
    cleanup_candidate,
    compare_candidates,
    extract_final_marked_candidates,
    extract_llm_final_answer_candidates,
    extract_answer_candidates,
    read_jsonl_to_list,
    tqdm,
)

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - depends on local env
    load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


class EvaluationTimeoutError(TimeoutError):
    pass


@contextmanager
def time_limit(seconds: Optional[float]):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise EvaluationTimeoutError(f"example evaluation exceeded {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

def load_ground_truth_examples(dataset_name: str, split: str, gt_jsonl_path: Optional[str]) -> list[dict[str, Any]]:
    if gt_jsonl_path:
        return read_jsonl_to_list(gt_jsonl_path)

    if load_dataset is None:
        raise RuntimeError(
            "datasets is not installed, so ground truth cannot be loaded from Hugging Face. "
            "Please install datasets or pass --gt-jsonl-path."
        ) from DATASETS_IMPORT_ERROR

    return list(load_dataset(dataset_name, "algebra", split=split))

def extract_ground_truth_solution_candidates(example: dict) -> list[str]:
    solution_text = str(example.get("solution", "") or "").strip()
    if not solution_text:
        return []

    candidates = extract_final_marked_candidates(solution_text)

    if not candidates:
        candidates.extend(extract_answer_candidates(solution_text))

    if not candidates:
        candidates.append(cleanup_candidate(solution_text))

    return [item for item in candidates if item]


def evaluate_example(example: dict, prediction_row: dict) -> tuple:
    pred_raw = str(prediction_row.get("answer", ""))
    pred_candidates = extract_llm_final_answer_candidates(pred_raw)
    gold_candidates = extract_ground_truth_solution_candidates(example)

    if not pred_candidates and pred_raw.strip():
        pred_candidates = [cleanup_candidate(pred_raw)]

    result = compare_candidates(
        gold_candidates=gold_candidates,
        pred_candidates=pred_candidates,
    )
    return result, gold_candidates, pred_candidates


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-precision MATH evaluator using final answers extracted from ground-truth solutions"
    )
    parser.add_argument(
        "--pred-path",
        default="outputs/minerva_math_algebra/rank_0.jsonl",
        help="Path to the model prediction jsonl file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="EleutherAI/hendrycks_math",
        help="Hugging Face dataset name for ground truth.",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--gt-jsonl-path",
        default=None,
        help="Optional local jsonl ground truth path. If set, datasets will not be used.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument(
        "--print-wrong",
        type=int,
        default=0,
        help="How many mismatched examples to print for debugging.",
    )
    parser.add_argument(
        "--details-path",
        default=None,
        help="Optional jsonl path for saving per-example evaluation details. "
        "If not set, a file will be created next to the prediction file automatically.",
    )
    parser.add_argument(
        "--per-example-timeout",
        type=float,
        default=8.0,
        help="Maximum seconds allowed for one example before it is skipped as a timeout.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    pred_path = Path(args.pred_path)
    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}", file=sys.stderr)
        return 1

    gt_examples = load_ground_truth_examples(args.dataset_name, args.split, args.gt_jsonl_path)
    pred_examples = read_jsonl_to_list(str(pred_path))

    if args.limit is not None:
        gt_examples = gt_examples[: args.limit]
        pred_examples = pred_examples[: args.limit]

    total = min(len(gt_examples), len(pred_examples))
    if total == 0:
        print("No examples to evaluate.", file=sys.stderr)
        return 1

    if len(gt_examples) != len(pred_examples):
        print(
            f"[warning] ground truth count = {len(gt_examples)}, prediction count = {len(pred_examples)}; "
            f"evaluating the first {total} pairs only.",
            file=sys.stderr,
        )

    method_counter: Counter[str] = Counter()
    wrong_printed = 0
    correct = 0
    details_path = Path(args.details_path) if args.details_path else pred_path.with_name(f"{pred_path.stem}_eval_details_solution.jsonl")
    details_fh = open(details_path, "w", encoding="utf-8")
    indices: Iterable[int] = range(total)

    if tqdm is not None:
        indices = tqdm(indices, total=total, desc="Evaluating MATH (solution GT)", unit="sample")

    try:
        for idx in indices:
            example = gt_examples[idx]
            prediction_row = pred_examples[idx]
            try:
                with time_limit(args.per_example_timeout):
                    result, gold_candidates, pred_candidates = evaluate_example(example, prediction_row)
            except EvaluationTimeoutError:
                result = MatchResult(False, "example_timeout")
                gold_candidates = extract_ground_truth_solution_candidates(example)
                pred_candidates = []
            except Exception as exc:
                result = MatchResult(False, f"example_error:{type(exc).__name__}")
                gold_candidates = extract_ground_truth_solution_candidates(example)
                pred_candidates = []

            if result.correct:
                correct += 1
            method_counter[result.method] += 1

            detail_row = {
                "index": idx,
                "correct": result.correct,
                "method": result.method,
                "problem": example.get("problem"),
                "ground_truth_answer": example.get("answer"),
                "ground_truth_solution": example.get("solution"),
                "llm_response": prediction_row.get("answer", ""),
                "gold_candidate": result.gold_candidate,
                "pred_candidate": result.pred_candidate,
                "ground_truth_solution_candidates": gold_candidates,
                "llm_final_answer_candidates": pred_candidates,
            }

            details_fh.write(json.dumps(detail_row, ensure_ascii=False) + "\n")

            if not result.correct and wrong_printed < args.print_wrong:
                wrong_printed += 1
                print("=" * 80)
                print(f"Index: {idx}")
                print(f"Problem: {example.get('problem', '')}")
                print(f"Gold candidates: {gold_candidates[:5]}")
                print(f"Pred candidates: {pred_candidates[:5]}")
                print("Raw prediction tail:")
                print(str(prediction_row.get("answer", ""))[-800:])

    finally:
        details_fh.close()

    accuracy = correct / total
    print("=" * 80)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4%}")
    print(f"Saved details: {details_path}")
    print("Match breakdown:")
    for method, count in method_counter.most_common():
        print(f"  {method}: {count}")

    if MATH_VERIFY_IMPORT_ERROR is not None:
        print(
            f"[warning] math_verify unavailable, matching quality is reduced: {MATH_VERIFY_IMPORT_ERROR}",
            file=sys.stderr,
        )
    if SYMPY_IMPORT_ERROR is not None:
        print(
            f"[warning] sympy unavailable, symbolic fallback is reduced: {SYMPY_IMPORT_ERROR}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
