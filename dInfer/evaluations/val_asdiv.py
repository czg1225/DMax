from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - depends on local env
    tqdm = None

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - depends on local env
    load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
BAD_SNIPPET_PATTERNS = [
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bfalse\b",
    r"\btypo\b",
    r"\bglitch\b",
    r"\bstuck\b",
    r"\bloop(?:ing)?\b",
    r"\bmalfunction\b",
    r"\bgive up\b",
    r"\bsystem error\b",
    r"\bnot right\b",
    r"\bconfusing\b",
]
BAD_SNIPPET_RE = re.compile("|".join(BAD_SNIPPET_PATTERNS), re.IGNORECASE)


@dataclass
class MatchResult:
    correct: bool
    method: str
    gold_candidate: Optional[str] = None
    pred_candidate: Optional[str] = None


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


def read_jsonl_to_list(path: str, encoding: str = "utf-8") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding=encoding) as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{lineno}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def strip_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("**", "").replace("__", "").strip("`")
    cleaned = cleaned.replace("\\boxed{", "").replace("\\fbox{", "")
    cleaned = cleaned.replace("}", "")
    cleaned = cleaned.replace("$", "").replace("\\$", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = cleaned.replace(",", "")
    return cleaned.strip()


def canonicalize_numeric(candidate: str) -> Optional[str]:
    cleaned = strip_wrappers(candidate)
    cleaned = cleaned.rstrip(".。!！?？,，;；:：")
    cleaned = cleaned.replace("%", "")
    cleaned = cleaned.strip()
    if not cleaned:
        return None

    if re.fullmatch(r"-?\d+/\d+", cleaned):
        numerator, denominator = cleaned.split("/", 1)
        if denominator == "0":
            return None
        value = Fraction(int(numerator), int(denominator))
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    fraction_value = Fraction(value)
    return str(fraction_value.numerator) if fraction_value.denominator == 1 else f"{fraction_value.numerator}/{fraction_value.denominator}"


def extract_last_number(text: str) -> Optional[str]:
    matches = NUMBER_RE.findall(strip_wrappers(text))
    if not matches:
        return None
    return matches[-1]


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def extract_boxed_contents(text: str) -> list[str]:
    matches: list[str] = []
    for command in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(command, start)
            if idx == -1:
                break
            cursor = idx + len(command)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                start = cursor + 1
                continue
            depth = 0
            content: list[str] = []
            end_idx = None
            for pos in range(cursor, len(text)):
                char = text[pos]
                if char == "{":
                    depth += 1
                    if depth > 1:
                        content.append(char)
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = pos
                        break
                    content.append(char)
                else:
                    content.append(char)
            if end_idx is not None:
                matches.append("".join(content).strip())
                start = end_idx + 1
            else:
                start = cursor + 1
    return matches


def extract_answer_spans(text: str) -> list[str]:
    patterns = [
        r"(?is)####\s*([^\n]+)",
        r"(?is)<answer>\s*(.*?)\s*</answer>",
        r"(?is)Final Answer\s*[:：]\s*(.*?)(?=\n\s*\n|$)",
        r"(?is)The final answer is\s*(.*?)(?:\.?\s*I hope it is correct\.?|$)",
        r"(?im)^\s*Answer\s*[:：]\s*(.+?)\s*$",
    ]
    spans: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                for piece in match:
                    if piece and piece.strip():
                        spans.append(piece.strip())
            elif match and match.strip():
                spans.append(match.strip())
    return spans


def snippet_is_unstable(text: str) -> bool:
    lower = str(text or "").lower()
    if not lower.strip():
        return False
    if BAD_SNIPPET_RE.search(lower):
        return True
    if lower.count("wait") >= 2:
        return True
    if lower.count("?") >= 2 and (" no" in lower or "wrong" in lower or "incorrect" in lower):
        return True
    return False


def response_is_unstable(text: str) -> bool:
    lower = str(text or "").lower()
    if not lower.strip():
        return False

    if any(token in lower for token in ["i give up", "system error", "malfunction", "i'm stuck", "glitch"]):
        return True

    wait_count = lower.count("wait")
    wrong_count = lower.count("wrong") + lower.count("incorrect") + lower.count("false")
    trigger_count = sum(lower.count(token) for token in [
        "wait",
        "wrong",
        "incorrect",
        "false",
        "glitch",
        "stuck",
        "loop",
        "malfunction",
        "system error",
        "give up",
    ])
    if wait_count >= 3 and wrong_count >= 1:
        return True
    return trigger_count >= 8


def has_clean_final_marker_near_end(text: str) -> bool:
    tail = str(text or "").strip()[-250:]
    if not tail:
        return False
    if snippet_is_unstable(tail):
        return False
    if extract_answer_spans(tail):
        return True
    return bool(extract_boxed_contents(tail))


def extract_ground_truth_answer_candidates(example: dict[str, Any]) -> list[str]:
    answer_text = str(example.get("answer", "") or "").strip()
    if not answer_text:
        return []

    candidates: list[str] = []

    explicit_spans = extract_answer_spans(answer_text)
    if explicit_spans:
        explicit_number = extract_last_number(explicit_spans[-1])
        if explicit_number:
            candidates.append(explicit_number)

    last_number = extract_last_number(answer_text)
    if last_number:
        candidates.append(last_number)

    normalized = [canonicalize_numeric(candidate) for candidate in candidates]
    return dedupe_keep_order([item for item in normalized if item])


def extract_llm_final_answer_candidates(text: str) -> list[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    if response_is_unstable(raw_text) and not has_clean_final_marker_near_end(raw_text):
        return []

    candidates: list[str] = []

    explicit_spans = extract_answer_spans(raw_text)
    for span in reversed(explicit_spans[-3:]):
        if snippet_is_unstable(span):
            continue
        number = extract_last_number(span)
        if number:
            candidates.append(number)

    boxed = extract_boxed_contents(raw_text)
    if boxed:
        boxed_tail = boxed[-1]
        if snippet_is_unstable(boxed_tail):
            boxed_tail = ""
        boxed_number = extract_last_number(boxed_tail)
        if boxed_number:
            candidates.append(boxed_number)

    tail_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line in reversed(tail_lines[-5:]):
        if snippet_is_unstable(line):
            continue
        number = extract_last_number(line)
        if number:
            candidates.append(number)

    tail_text = "\n".join(tail_lines[-8:]) if tail_lines else raw_text[-500:]
    tail_number = None if snippet_is_unstable(tail_text) else extract_last_number(tail_text)
    if tail_number:
        candidates.append(tail_number)

    normalized = [canonicalize_numeric(candidate) for candidate in candidates]
    return dedupe_keep_order([item for item in normalized if item])


def compare_candidates(gold_candidates: list[str], pred_candidates: list[str]) -> MatchResult:
    top_preds = pred_candidates[:2]
    if not top_preds:
        return MatchResult(False, "no_match")

    for gold in gold_candidates:
        for pred in top_preds:
            if gold == pred:
                return MatchResult(True, "numeric_exact_top2_candidates", gold, pred)
    return MatchResult(False, "no_match")


def load_ground_truth_examples(dataset_name: str, split: str, gt_jsonl_path: Optional[str]) -> list[dict[str, Any]]:
    if gt_jsonl_path:
        return read_jsonl_to_list(gt_jsonl_path)

    if load_dataset is None:
        raise RuntimeError(
            "datasets is not installed, so ground truth cannot be loaded from Hugging Face. "
            "Please install datasets or pass --gt-jsonl-path."
        ) from DATASETS_IMPORT_ERROR

    return list(load_dataset(dataset_name, split=split))


def evaluate_example(example: dict[str, Any], prediction_row: dict[str, Any]) -> tuple[MatchResult, list[str], list[str]]:
    pred_raw = str(prediction_row.get("answer", ""))
    pred_candidates = extract_llm_final_answer_candidates(pred_raw)
    gold_candidates = extract_ground_truth_answer_candidates(example)

    result = compare_candidates(
        gold_candidates=gold_candidates,
        pred_candidates=pred_candidates,
    )
    return result, gold_candidates, pred_candidates


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-precision ASDIV numeric evaluator")
    parser.add_argument(
        "--pred-path",
        default="outputs/asdiv_llada_mini/rank_0.jsonl",
        help="Path to the model prediction jsonl file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="EleutherAI/asdiv",
        help="Hugging Face dataset name for ground truth.",
    )
    parser.add_argument("--split", default="validation", help="Dataset split.")
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
        default=5.0,
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
        indices = tqdm(indices, total=total, desc="Evaluating ASDIV", unit="sample")

    try:
        for idx in indices:
            example = gt_examples[idx]
            prediction_row = pred_examples[idx]
            try:
                with time_limit(args.per_example_timeout):
                    result, gold_candidates, pred_candidates = evaluate_example(example, prediction_row)
            except EvaluationTimeoutError:
                result = MatchResult(False, "example_timeout")
                gold_candidates = extract_ground_truth_answer_candidates(example)
                pred_candidates = []
            except Exception as exc:
                result = MatchResult(False, f"example_error:{type(exc).__name__}")
                gold_candidates = extract_ground_truth_answer_candidates(example)
                pred_candidates = []

            if result.correct:
                correct += 1
            method_counter[result.method] += 1

            detail_row = {
                "index": idx,
                "correct": result.correct,
                "method": result.method,
                "problem": example.get("question"),
                "ground_truth_answer": example.get("answer"),
                "llm_response": prediction_row.get("answer", ""),
                "gold_candidate": result.gold_candidate,
                "pred_candidate": result.pred_candidate,
                "ground_truth_answer_candidates": gold_candidates,
                "llm_final_answer_candidates": pred_candidates,
            }

            details_fh.write(json.dumps(detail_row, ensure_ascii=False) + "\n")

            if not result.correct and wrong_printed < args.print_wrong:
                wrong_printed += 1
                print("=" * 80)
                print(f"Index: {idx}")
                print(f"Problem: {example.get('question', '')}")
                print(f"Gold candidates: {gold_candidates[:5]}")
                print(f"Pred candidates: {pred_candidates[:5]}")
                print("Raw prediction tail:")
                print(str(prediction_row.get('answer', ''))[-800:])

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
