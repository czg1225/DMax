from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
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

try:
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except Exception as exc:  # pragma: no cover - depends on local env
    parse = None
    verify = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    MATH_VERIFY_IMPORT_ERROR = exc
else:
    MATH_VERIFY_IMPORT_ERROR = None

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except Exception as exc:  # pragma: no cover - depends on local env
    sympy = None
    parse_latex = None
    SYMPY_IMPORT_ERROR = exc
else:
    SYMPY_IMPORT_ERROR = None


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


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


def normalize_final_answer(text: str) -> str:
    answer = text.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        answer = answer.replace(expr, "")

    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")

    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")

    return answer


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def strip_markdown(text: str) -> str:
    stripped = text.strip()
    stripped = stripped.replace("**", "").replace("__", "")
    stripped = stripped.strip("`")
    return stripped


def strip_answer_prefix(text: str) -> str:
    cleaned = strip_markdown(text)
    cleaned = re.sub(r"^\s*####\s*", "", cleaned)
    cleaned = re.sub(r"(?is)^\s*<answer>\s*", "", cleaned)
    cleaned = re.sub(r"(?is)\s*</answer>\s*$", "", cleaned)
    cleaned = re.sub(r"(?is)^\s*(?:final answer|answer)\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"(?is)^\s*the final answer is\s*", "", cleaned)
    cleaned = re.sub(r"(?is)\s*I hope it is correct\.?\s*$", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def is_balanced_bracket_wrap(text: str, left: str, right: str) -> bool:
    if not (text.startswith(left) and text.endswith(right)):
        return False
    depth = 0
    body = text[len(left) : len(text) - len(right)]
    for char in body:
        if char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def extract_braced_content(text: str, open_index: int) -> tuple[Optional[str], Optional[int]]:
    if open_index >= len(text) or text[open_index] != "{":
        return None, None

    depth = 0
    content: list[str] = []
    for idx in range(open_index, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
            if depth > 1:
                content.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(content), idx
            content.append(char)
        else:
            content.append(char)
    return None, None


def unwrap_known_wrappers(text: str) -> str:
    current = text.strip()
    while True:
        previous = current
        current = current.strip()

        if current.startswith("$$") and current.endswith("$$") and len(current) >= 4:
            current = current[2:-2].strip()
        elif current.startswith("\\[") and current.endswith("\\]"):
            current = current[2:-2].strip()
        elif current.startswith("\\(") and current.endswith("\\)"):
            current = current[2:-2].strip()
        elif current.startswith("$") and current.endswith("$") and len(current) >= 2:
            current = current[1:-1].strip()
        elif current.startswith("\\boxed"):
            brace_start = current.find("{")
            content, brace_end = extract_braced_content(current, brace_start)
            if content is not None and brace_end == len(current) - 1:
                current = content.strip()
        elif current.startswith("\\fbox"):
            brace_start = current.find("{")
            content, brace_end = extract_braced_content(current, brace_start)
            if content is not None and brace_end == len(current) - 1:
                current = content.strip()
        elif current.startswith("\\text{") or current.startswith("\\mathrm{") or current.startswith("\\textbf{"):
            brace_start = current.find("{")
            content, brace_end = extract_braced_content(current, brace_start)
            if content is not None and brace_end == len(current) - 1:
                current = content.strip()

        current = current.strip(" \n\t\r,.;:!。；，")
        if current == previous:
            break

    return current


def cleanup_candidate(text: str) -> str:
    cleaned = strip_answer_prefix(text)
    cleaned = cleaned.replace("\u2212", "-").replace("−", "-")
    cleaned = cleaned.replace("\u00d7", "*").replace("\u00f7", "/")
    cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    cleaned = cleaned.replace("\\%", "%")
    cleaned = unwrap_known_wrappers(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def canonicalize_for_compare(text: str) -> str:
    cleaned = cleanup_candidate(text)
    cleaned = normalize_final_answer(cleaned)
    cleaned = cleaned.replace("\\{", "{").replace("\\}", "}")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.strip(" \n\t\r,.;:!。；，")
    return cleaned


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
            if cursor < len(text) and text[cursor] == "{":
                content, end_idx = extract_braced_content(text, cursor)
                if content is not None and end_idx is not None:
                    matches.append(content.strip())
                    start = end_idx + 1
                    continue
            start = cursor + 1
    return matches


def extract_last_boxed_with_position(text: str) -> tuple[Optional[str], Optional[int]]:
    last_content: Optional[str] = None
    last_end: Optional[int] = None

    for command in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(command, start)
            if idx == -1:
                break
            cursor = idx + len(command)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == "{":
                content, end_idx = extract_braced_content(text, cursor)
                if content is not None and end_idx is not None:
                    if last_end is None or end_idx > last_end:
                        last_content = content.strip()
                        last_end = end_idx
                    start = end_idx + 1
                    continue
            start = cursor + 1

    return last_content, last_end


def extract_answer_by_patterns(text: str) -> list[str]:
    patterns = [
        r"(?is)<answer>\s*(.*?)\s*</answer>",
        r"(?is)Final Answer\s*[:：]\s*(.*?)(?=\n\s*\n|$)",
        r"(?is)The final answer is\s*(.*?)(?:\.?\s*I hope it is correct\.?|$)",
        r"(?im)^\s*Answer\s*[:：]\s*(.+?)\s*$",
        r"(?im)^\s*####\s*(.+?)\s*$",
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                for piece in match:
                    if piece and piece.strip():
                        matches.append(piece.strip())
            elif match and match.strip():
                matches.append(match.strip())
    return matches


def has_substantial_suffix(text: str, start_index: int) -> bool:
    suffix = text[start_index:].strip()
    if not suffix:
        return False

    suffix = re.sub(r"[\s`*_>#\-\.,;:!?\)\]\}]+", "", suffix)
    signal_chars = re.findall(r"[A-Za-z0-9\u4e00-\u9fff\\\\]", suffix)
    return len(signal_chars) >= 20


def extract_last_pattern_answer_near_end(text: str) -> Optional[str]:
    patterns = [
        r"(?is)<answer>\s*(.*?)\s*</answer>",
        r"(?is)Final Answer\s*[:：]\s*(.*?)(?=\n\s*\n|$)",
        r"(?is)The final answer is\s*(.*?)(?:\.?\s*I hope it is correct\.?|$)",
        r"(?im)^\s*Answer\s*[:：]\s*(.+?)\s*$",
        r"(?im)^\s*####\s*(.+?)\s*$",
    ]

    last_match_text: Optional[str] = None
    last_match_end: Optional[int] = None

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = match.group(1).strip()
            if not candidate:
                continue
            if last_match_end is None or match.end() > last_match_end:
                last_match_text = candidate
                last_match_end = match.end()

    if last_match_text is None or last_match_end is None:
        return None
    if has_substantial_suffix(text, last_match_end):
        return None
    return cleanup_candidate(last_match_text)


def extract_final_marked_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    boxed = extract_boxed_contents(text)
    if boxed:
        candidates.append(boxed[-1])

    pattern_matches = extract_answer_by_patterns(text)
    for match in reversed(pattern_matches[-3:]):
        candidates.append(match)
        first_line = match.splitlines()[0].strip()
        if first_line and first_line != match:
            candidates.append(first_line)

    cleaned = [cleanup_candidate(item) for item in candidates]
    cleaned = [item for item in cleaned if item and item.lower() not in {"answer", "final answer"}]
    return dedupe_keep_order(cleaned)


def extract_latex_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    patterns = [
        r"(?s)\$\$(.*?)\$\$",
        r"(?s)\\\[(.*?)\\\]",
        r"(?s)\\\((.*?)\\\)",
        r"(?s)(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if match and str(match).strip():
                blocks.append(str(match).strip())
    return blocks[-6:]


def extract_line_candidates(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: list[str] = []
    for line in lines[-8:]:
        stripped = strip_markdown(line).lstrip("-*")
        if not stripped:
            continue
        candidates.append(stripped)
        if "=" in stripped:
            rhs = stripped.split("=")[-1].strip()
            if rhs:
                candidates.append(rhs)
    return candidates[-8:]


def extract_answer_candidates(text: str) -> list[str]:
    if not text:
        return []

    raw_text = str(text)
    candidates: list[str] = []

    boxed = extract_boxed_contents(raw_text)
    candidates.extend(reversed(boxed[-6:]))

    pattern_matches = extract_answer_by_patterns(raw_text)
    for match in reversed(pattern_matches[-6:]):
        candidates.append(match)
        first_line = match.splitlines()[0].strip()
        if first_line and first_line != match:
            candidates.append(first_line)

    for block in reversed(extract_latex_blocks(raw_text)):
        candidates.append(block)

    candidates.extend(reversed(extract_line_candidates(raw_text)))

    raw_text_stripped = raw_text.strip()
    if raw_text_stripped and len(raw_text_stripped) <= 200:
        candidates.append(raw_text_stripped)

    cleaned = [cleanup_candidate(item) for item in candidates]
    cleaned = [item for item in cleaned if item and item.lower() not in {"answer", "final answer"}]
    return dedupe_keep_order(cleaned)[:20]


def load_ground_truth_examples(dataset_name: str, split: str, gt_jsonl_path: Optional[str]) -> list[dict[str, Any]]:
    if gt_jsonl_path:
        return read_jsonl_to_list(gt_jsonl_path)

    if load_dataset is None:
        raise RuntimeError(
            "datasets is not installed, so ground truth cannot be loaded from Hugging Face. "
            "Please install datasets or pass --gt-jsonl-path."
        ) from DATASETS_IMPORT_ERROR

    return list(load_dataset(dataset_name, split=split))


def extract_ground_truth_answer_candidates(example: dict[str, Any]) -> list[str]:
    answer_text = str(example.get("answer", "") or "").strip()
    if not answer_text:
        return []

    candidates = [cleanup_candidate(answer_text)]
    candidates.extend(extract_final_marked_candidates(answer_text))
    if len(answer_text) <= 200:
        candidates.extend(extract_answer_candidates(answer_text))

    return [item for item in dedupe_keep_order(candidates) if item]


def extract_llm_final_answer_candidates(text: str) -> list[str]:
    if not text:
        return []

    raw_text = str(text)
    raw_text_stripped = raw_text.strip()
    tail_text = raw_text_stripped[-800:]
    tail_lines = [line.strip() for line in tail_text.splitlines() if line.strip()]

    candidates: list[str] = []

    near_end_pattern = extract_last_pattern_answer_near_end(raw_text)
    if near_end_pattern:
        candidates.append(near_end_pattern)

    last_boxed, last_boxed_end = extract_last_boxed_with_position(raw_text)
    if last_boxed and last_boxed_end is not None and not has_substantial_suffix(raw_text, last_boxed_end + 1):
        candidates.append(last_boxed)

    tail_latex_blocks = extract_latex_blocks(tail_text)
    if tail_latex_blocks:
        candidates.append(tail_latex_blocks[-1])

    for line in reversed(tail_lines[-3:]):
        stripped = strip_markdown(line).lstrip("-*").strip()
        if not stripped:
            continue
        candidates.append(stripped)
        if "=" in stripped:
            rhs = stripped.split("=")[-1].strip()
            if rhs:
                candidates.append(rhs)

    if raw_text_stripped and len(raw_text_stripped) <= 200:
        candidates.append(raw_text_stripped)

    cleaned = [cleanup_candidate(item) for item in candidates]
    cleaned = [item for item in cleaned if item and item.lower() not in {"answer", "final answer"}]
    return dedupe_keep_order(cleaned)[:10]


def maybe_parse_numeric(text: str) -> Optional[Any]:
    if sympy is None:
        return None

    candidate = cleanup_candidate(text)
    if not candidate:
        return None

    if candidate.endswith("%"):
        candidate = f"({candidate[:-1]})/100"

    if any(token in candidate for token in ("\\frac", "\\sqrt", "\\pi", "\\cdot", "\\pm", "{", "}")):
        try:
            return parse_latex(candidate)
        except Exception:
            pass

    ascii_candidate = candidate
    ascii_candidate = ascii_candidate.replace("^", "**")
    ascii_candidate = ascii_candidate.replace("\\pi", "pi").replace("\\cdot", "*")
    ascii_candidate = ascii_candidate.replace("{", "(").replace("}", ")")

    try:
        return sympy.sympify(ascii_candidate)
    except Exception:
        return None


def are_sympy_equivalent(left: str, right: str) -> bool:
    left_expr = maybe_parse_numeric(left)
    right_expr = maybe_parse_numeric(right)
    if left_expr is None or right_expr is None:
        return False

    try:
        diff = sympy.simplify(left_expr - right_expr)
    except Exception:
        return False
    return diff == 0


@lru_cache(maxsize=16384)
def cached_parse(text: str, mode: str) -> Any:
    if parse is None:
        return None

    if mode == "default":
        return parse(text)

    if mode == "snippet":
        return parse(text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])

    raise ValueError(f"Unknown parse mode: {mode}")


def try_math_verify(gold_text: str, pred_text: str, snippet_mode: bool) -> bool:
    if parse is None or verify is None:
        return False

    gold_variants: list[tuple[str, str]] = []
    pred_variants: list[tuple[str, str]] = []

    if snippet_mode:
        cleaned_gold = cleanup_candidate(gold_text)
        cleaned_pred = cleanup_candidate(pred_text)
        for candidate in dedupe_keep_order([gold_text, cleaned_gold, f"${cleaned_gold}$"]):
            if candidate:
                gold_variants.append(("snippet", candidate))
        for candidate in dedupe_keep_order([pred_text, cleaned_pred, f"${cleaned_pred}$"]):
            if candidate:
                pred_variants.append(("snippet", candidate))
    else:
        gold_variants.append(("default", gold_text))
        pred_variants.append(("default", pred_text))

    for gold_mode, gold_variant in gold_variants:
        gold_parsed = cached_parse(gold_variant, gold_mode)
        if gold_parsed is None:
            continue
        for pred_mode, pred_variant in pred_variants:
            pred_parsed = cached_parse(pred_variant, pred_mode)
            if pred_parsed is None:
                continue
            try:
                if verify(gold_parsed, pred_parsed):
                    return True
            except Exception:
                continue

    return False


def compare_candidates(
    gold_candidates: list[str],
    pred_candidates: list[str],
) -> MatchResult:
    for gold in gold_candidates:
        gold_norm = canonicalize_for_compare(gold)
        if not gold_norm:
            continue
        for pred in pred_candidates:
            pred_norm = canonicalize_for_compare(pred)
            if gold_norm and gold_norm == pred_norm:
                return MatchResult(True, "normalized_exact", gold, pred)

    for gold in gold_candidates:
        for pred in pred_candidates:
            if are_sympy_equivalent(gold, pred):
                return MatchResult(True, "sympy_equiv", gold, pred)

    for gold in gold_candidates:
        for pred in pred_candidates:
            if try_math_verify(gold, pred, snippet_mode=True):
                return MatchResult(True, "math_verify_snippet", gold, pred)

    return MatchResult(False, "no_match")


def evaluate_example(example: dict[str, Any], prediction_row: dict[str, Any]) -> tuple[MatchResult, list[str], list[str]]:
    pred_raw = str(prediction_row.get("answer", ""))
    pred_candidates = extract_llm_final_answer_candidates(pred_raw)
    gold_candidates = extract_ground_truth_answer_candidates(example)

    if not pred_candidates and pred_raw.strip():
        pred_candidates = [cleanup_candidate(pred_raw)]

    result = compare_candidates(
        gold_candidates=gold_candidates,
        pred_candidates=pred_candidates,
    )
    return result, gold_candidates, pred_candidates


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-precision MATH-500 answer evaluator")
    parser.add_argument(
        "--pred-path",
        default="outputs/minerva_math500/rank_0.jsonl",
        help="Path to the model prediction jsonl file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="HuggingFaceH4/MATH-500",
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
    details_path = Path(args.details_path) if args.details_path else pred_path.with_name(f"{pred_path.stem}_eval_details.jsonl")
    details_fh = open(details_path, "w", encoding="utf-8")
    indices: Iterable[int] = range(total)

    if tqdm is not None:
        indices = tqdm(indices, total=total, desc="Evaluating MATH-500", unit="sample")

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
                "problem": example.get("problem"),
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
                print(f"Problem: {example.get('problem', '')}")
                print(f"Gold candidates: {gold_candidates[:5]}")
                print(f"Pred candidates: {pred_candidates[:5]}")
                print("Raw prediction tail:")
                print(str(prediction_row.get("answer", ""))[-800:])

    finally:
        if details_fh is not None:
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
