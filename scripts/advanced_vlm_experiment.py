from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw

CHOICES = ("A", "B", "C", "D")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEDIUM_MANIFEST = Path("assets/math_vqa_medium/manifest.jsonl")
DEFAULT_OUT_DIR = Path("artifacts/advanced")
SUMMARY_PATH = DEFAULT_OUT_DIR / "SUMMARY.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filtered_rows(
    manifest_path: str | Path,
    split: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(manifest_path) if row.get("split") == split]
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def parse_option_value(option: str) -> str:
    text = option.strip()
    match = re.match(r"^[A-E]\)\s*(.*)$", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else text


def option_label(option: str) -> str | None:
    match = re.match(r"^\s*([A-E])\)", option)
    return match.group(1).upper() if match else None


def find_option_by_value(options: list[str], value: str | int) -> str | None:
    wanted = str(value).strip().lower()
    for option in options:
        label = option_label(option)
        if label is None:
            continue
        got = parse_option_value(option).strip().lower()
        if got == wanted:
            return label
    return None


def option_values_by_label(options: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for option in options:
        label = option_label(option)
        if label in CHOICES:
            values[label] = parse_option_value(option)
    return values


def correct_value_for_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("solver_value") is not None:
        return str(metadata["solver_value"])

    answer = str(row.get("answer", ""))
    values = option_values_by_label(list(row.get("options", [])))
    if answer not in values:
        raise ValueError(f"Cannot infer correct option value for row {row.get('id')!r}")
    return values[answer]


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def solve_from_text(row: dict[str, Any]) -> str | None:
    """Solve tasks whose answer is fully specified in text.

    This is intentionally a text-only diagnostic baseline. It must not be used as
    evidence of visual reasoning, but it is useful for identifying leakage in the
    synthetic data format.
    """
    question = row.get("question", "")
    options = list(row.get("options", []))

    match = re.search(r"y\s*=\s*(-?\d+)x\s*([+-]\d+)?[^\n]*x\s*=\s*(-?\d+)", question)
    if match:
        a = int(match.group(1))
        b = int(match.group(2) or 0)
        x = int(match.group(3))
        return find_option_by_value(options, a * x + b)

    match = re.search(r"ширина\s+(\d+),\s*высота\s+(\d+)", question)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        return find_option_by_value(options, width * height)

    match = re.search(r"катетами\s+(\d+)\s+и\s+(\d+)", question)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        c2 = a * a + b * b
        c = int(math.sqrt(c2))
        if c * c == c2:
            return find_option_by_value(options, c)

    match = re.search(r"смежн\w*\s+угл\w*\s+равен\s+(\d+)", question)
    if not match:
        match = re.search(r"угол\s+равен\s+(\d+).*смеж", question)
    if match:
        return find_option_by_value(options, f"{180 - int(match.group(1))}°")

    match = re.search(r"P\((-?\d+),(-?\d+)\)\s+и\s+Q\((-?\d+),(-?\d+)\)", question)
    if match:
        x1, y1, x2, y2 = [int(match.group(i)) for i in range(1, 5)]
        return find_option_by_value(options, (x1 - x2) ** 2 + (y1 - y2) ** 2)

    return None


def train_distribution(manifest_path: str | Path, split: str = "train") -> dict[str, float]:
    rows = filtered_rows(manifest_path, split=split)
    counts = Counter(row.get("answer") for row in rows if row.get("answer") in CHOICES)
    total = sum(counts.values())
    if total == 0:
        return {letter: 1.0 / len(CHOICES) for letter in CHOICES}
    return {letter: counts.get(letter, 0) / total for letter in CHOICES}


def majority_letter(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row.get("answer") for row in rows if row.get("answer") in CHOICES)
    if not counts:
        return "A"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def entropy_of_counts(counts: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    for letter in CHOICES:
        p = counts.get(letter, 0) / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy / math.log(len(CHOICES))


def evaluate_predictions(
    rows: list[dict[str, Any]],
    predictions: list[str],
    collapse_threshold: float = 0.70,
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")
    total = len(rows)
    correct = sum(int(pred == row.get("answer")) for row, pred in zip(rows, predictions))
    accuracy = correct / total if total else 0.0

    answer_counts = Counter(row.get("answer") for row in rows if row.get("answer") in CHOICES)
    pred_counts = Counter(pred for pred in predictions if pred in CHOICES)

    class_recall: dict[str, float | None] = {}
    recalls: list[float] = []
    for letter in CHOICES:
        support = answer_counts.get(letter, 0)
        if support == 0:
            class_recall[letter] = None
            continue
        hits = sum(
            int(row.get("answer") == letter and pred == letter)
            for row, pred in zip(rows, predictions)
        )
        recall = hits / support
        class_recall[letter] = recall
        recalls.append(recall)
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0

    subject_hits: dict[str, list[int]] = defaultdict(list)
    for row, pred in zip(rows, predictions):
        subject_hits[row.get("subject", "unknown")].append(int(pred == row.get("answer")))
    subject_accuracy = {
        subject: sum(values) / len(values) for subject, values in sorted(subject_hits.items())
    }
    subject_macro_accuracy = (
        sum(subject_accuracy.values()) / len(subject_accuracy) if subject_accuracy else 0.0
    )

    max_prediction_share = max(pred_counts.values(), default=0) / total if total else 0.0
    prediction_entropy = entropy_of_counts(pred_counts, total)
    collapse_penalty = (
        max(0.0, max_prediction_share - collapse_threshold) / (1.0 - collapse_threshold)
        if collapse_threshold < 1.0
        else 0.0
    )
    composite_score = (
        0.50 * balanced_accuracy
        + 0.40 * subject_macro_accuracy
        + 0.10 * prediction_entropy
        - 0.50 * collapse_penalty
    )

    return {
        "n": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "class_recall": class_recall,
        "subject_accuracy": subject_accuracy,
        "subject_macro_accuracy": subject_macro_accuracy,
        "answer_distribution": {letter: answer_counts.get(letter, 0) for letter in CHOICES},
        "prediction_distribution": {letter: pred_counts.get(letter, 0) for letter in CHOICES},
        "prediction_entropy": prediction_entropy,
        "max_prediction_share": max_prediction_share,
        "collapse_threshold": collapse_threshold,
        "collapse_flag": max_prediction_share > collapse_threshold if total else False,
        "collapse_penalty": collapse_penalty,
        "composite_score": composite_score,
    }


def prediction_rows(rows: list[dict[str, Any]], predictions: list[str], method: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, pred in zip(rows, predictions):
        out.append(
            {
                "id": row.get("id"),
                "subject": row.get("subject", "unknown"),
                "answer": row.get("answer"),
                "prediction": pred,
                "correct": pred == row.get("answer"),
                "method": method,
            }
        )
    return out


def constant_predictions(rows: list[dict[str, Any]], letter: str) -> list[str]:
    return [letter for _ in rows]


def random_stratified_predictions(
    rows: list[dict[str, Any]],
    distribution: dict[str, float],
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    letters = list(CHOICES)
    weights = [distribution.get(letter, 0.0) for letter in letters]
    if sum(weights) <= 0:
        weights = [1.0 for _ in letters]
    return rng.choices(letters, weights=weights, k=len(rows))


def text_solver_predictions(
    rows: list[dict[str, Any]],
    fallback: str,
) -> tuple[list[str], int]:
    predictions: list[str] = []
    solved = 0
    for row in rows:
        pred = solve_from_text(row)
        if pred in CHOICES:
            solved += 1
            predictions.append(pred)
        else:
            predictions.append(fallback)
    return predictions, solved


def load_prediction_file(path: str | Path, rows: list[dict[str, Any]]) -> list[str]:
    raw = read_jsonl(path)
    by_id: dict[str, str] = {}
    for item in raw:
        row_id = item.get("id")
        pred = item.get("prediction", item.get("pred"))
        if row_id and pred in CHOICES:
            by_id[str(row_id)] = pred
    missing = [str(row.get("id")) for row in rows if str(row.get("id")) not in by_id]
    if missing:
        raise ValueError(f"Prediction file misses {len(missing)} ids; first missing: {missing[:3]}")
    return [by_id[str(row.get("id"))] for row in rows]


def load_prediction_subset(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    raw = read_jsonl(path)
    by_id: dict[str, str] = {}
    for item in raw:
        row_id = item.get("id")
        pred = item.get("prediction", item.get("pred"))
        if row_id and pred in CHOICES:
            by_id[str(row_id)] = pred
    matched_rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    missing: list[str] = []
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in by_id:
            matched_rows.append(row)
            predictions.append(by_id[row_id])
        else:
            missing.append(row_id)
    return matched_rows, predictions, missing


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def update_summary(title: str, lines: list[str], out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    section = [f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} - {title}", ""]
    section.extend(lines)
    if SUMMARY_PATH.exists():
        current = SUMMARY_PATH.read_text(encoding="utf-8")
    else:
        current = (
            "# Advanced VLM Math Pipeline Summary\n\n"
            "Local-only experiment log. No GitHub push, PR, or artifact upload is performed by this harness.\n"
        )
    SUMMARY_PATH.write_text(current.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")


def command_preflight(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    info: dict[str, Any] = {
        "timestamp": now_id(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "env": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "commands": {
            "recommended_mps_prefix": "PYTORCH_ENABLE_MPS_FALLBACK=1",
            "no_git_push_policy": True,
        },
    }

    for name in [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "safetensors",
        "torchvision",
        "matplotlib",
        "lion_pytorch",
        "bitsandbytes",
    ]:
        spec = importlib.util.find_spec(name)
        item: dict[str, Any] = {"installed": spec is not None}
        if spec is not None:
            try:
                item["version"] = importlib.metadata.version(name)
            except Exception as exc:
                item["version_error"] = repr(exc)
        info[name] = item

    try:
        import torch

        info["torch"]["mps_built"] = torch.backends.mps.is_built()
        info["torch"]["mps_available"] = torch.backends.mps.is_available()
        info["torch"]["cuda_available"] = torch.cuda.is_available()
        if torch.backends.mps.is_available():
            try:
                x = torch.ones(4, device="mps", dtype=torch.bfloat16)
                info["torch"]["mps_bfloat16_smoke"] = bool((x + 1).cpu()[0].item() == 2.0)
            except Exception as exc:
                info["torch"]["mps_bfloat16_smoke_error"] = repr(exc)
    except Exception as exc:
        info["torch_probe_error"] = repr(exc)

    info["disk"] = {
        "artifacts": shutil.disk_usage(Path.cwd()).free,
        "artifacts_advanced_exists": out_dir.exists(),
    }

    out_path = out_dir / "preflight.json"
    write_json(out_path, info)
    update_summary(
        "preflight",
        [
            f"- Wrote `{out_path}`.",
            f"- Torch installed: {info.get('torch', {}).get('installed')}, MPS available: {info.get('torch', {}).get('mps_available')}.",
            "- Local-only policy is active: no push, PR, or upload.",
        ],
        out_dir,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


def command_baseline(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    eval_rows = filtered_rows(args.manifest, args.split, args.max_samples)
    train_rows = filtered_rows(args.manifest, args.train_split)
    if not eval_rows:
        raise SystemExit(f"No rows for split={args.split} in {args.manifest}")

    train_majority = majority_letter(train_rows)
    eval_majority = majority_letter(eval_rows)
    distribution = train_distribution(args.manifest, args.train_split)

    methods: list[tuple[str, list[str], dict[str, Any]]] = [
        ("train_majority", constant_predictions(eval_rows, train_majority), {"letter": train_majority}),
        (
            "eval_majority_diagnostic",
            constant_predictions(eval_rows, eval_majority),
            {"letter": eval_majority, "diagnostic_only": True},
        ),
        (
            "random_stratified",
            random_stratified_predictions(eval_rows, distribution, args.seed),
            {"seed": args.seed, "train_distribution": distribution},
        ),
    ]

    text_preds, solved = text_solver_predictions(eval_rows, fallback=train_majority)
    methods.append(("text_solver_with_majority_fallback", text_preds, {"solved": solved}))

    if args.predictions:
        trained_preds = load_prediction_file(args.predictions, eval_rows)
        methods.append(("trained_predictions_file", trained_preds, {"path": str(args.predictions)}))

    run_id = args.run_id or f"baseline-{args.split}-{now_id()}"
    run_dir = out_dir / "runs" / run_id
    results: list[dict[str, Any]] = []
    for method, preds, extra in methods:
        metrics = evaluate_predictions(eval_rows, preds, args.collapse_threshold)
        result = {
            "run_id": run_id,
            "method": method,
            "manifest": str(args.manifest),
            "split": args.split,
            "metrics": metrics,
            "extra": extra,
        }
        results.append(result)
        append_jsonl(out_dir / "metrics.jsonl", result)
        write_jsonl(run_dir / f"{method}_predictions.jsonl", prediction_rows(eval_rows, preds, method))

    write_json(run_dir / "baseline_results.json", results)
    best = max(results, key=lambda item: item["metrics"]["composite_score"])
    update_summary(
        "baseline audit",
        [
            f"- Evaluated `{args.manifest}` split `{args.split}` with {len(eval_rows)} rows.",
            f"- Train-majority letter: `{train_majority}`; eval-majority diagnostic: `{eval_majority}`.",
            f"- Best composite baseline: `{best['method']}` = {best['metrics']['composite_score']:.4f}.",
            f"- Outputs: `{run_dir}`.",
        ],
        out_dir,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


@dataclass(frozen=True)
class GeneratedExample:
    subject: str
    question: str
    options: list[str]
    answer: str
    value: str
    draw: Callable[[ImageDraw.ImageDraw], None]


ALGEBRA_Y_DOMAIN = sorted({a * x + b for a in range(1, 5) for b in range(-2, 6) for x in range(1, 6)})
BAR_VALUE_DOMAIN = [str(value) for value in range(1, 10)]
GRID_AREA_DOMAIN = sorted({width * height for width in range(2, 9) for height in range(2, 7)})
TRIANGLE_HYPOTENUSE_DOMAIN = ["5", "10", "13", "15", "17"]
ANGLE_VALUE_DOMAIN = [f"{180 - angle}°" for angle in [40, 50, 60, 70, 80, 100, 110, 120, 130]]
DET_VALUE_DOMAIN = [str(value) for value in range(-28, 29)]
DISTANCE_SQUARED_DOMAIN = sorted(
    {
        (x1 - x2) ** 2 + (y1 - y2) ** 2
        for x1 in range(0, 6)
        for y1 in range(0, 6)
        for x2 in range(0, 6)
        for y2 in range(0, 6)
        if x1 != x2 or y1 != y2
    }
)


def format_options(
    correct_value: str,
    answer_letter: str,
    distractors: list[str],
    rng: random.Random | None = None,
    domain_values: Iterable[str | int] | None = None,
) -> list[str]:
    seen = {correct_value}
    clean: list[str] = []

    if domain_values is not None:
        pool = [str(value) for value in domain_values if str(value) not in seen]
        if rng is not None:
            rng.shuffle(pool)
        for value in pool:
            clean.append(value)
            seen.add(value)
            if len(clean) >= 3:
                break

    for value in distractors:
        value = str(value)
        if value not in seen:
            clean.append(value)
            seen.add(value)
        if len(clean) >= 3:
            break

    candidate = int(re.sub(r"[^0-9-]", "", correct_value) or "0")
    delta = 1
    while len(clean) < 3:
        value = str(candidate + delta)
        if value not in seen:
            clean.append(value)
            seen.add(value)
        delta += 1
    if rng is not None:
        rng.shuffle(clean)
    values: dict[str, str] = {}
    distractor_iter = iter(clean)
    for letter in CHOICES:
        values[letter] = correct_value if letter == answer_letter else next(distractor_iter)
    return [f"{letter}) {values[letter]}" for letter in CHOICES]


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(20, 20, 20))


def generate_algebra(rng: random.Random, answer_letter: str) -> GeneratedExample:
    a = rng.randint(1, 4)
    b = rng.randint(-2, 5)
    x = rng.randint(1, 5)
    y = a * x + b
    sign = "+" if b >= 0 else ""
    question = f"На графике изображена прямая y={a}x{sign}{b}. Чему равен y при x={x}?"
    options = format_options(
        str(y),
        answer_letter,
        [str(y - 2), str(y - 1), str(y + 1), str(y + 2)],
        rng,
        ALGEBRA_Y_DOMAIN,
    )

    def draw(draw: ImageDraw.ImageDraw) -> None:
        draw.line((30, 190, 200, 190), fill=(0, 0, 0), width=2)
        draw.line((40, 205, 40, 25), fill=(0, 0, 0), width=2)
        points = []
        for px in range(0, 6):
            py = a * px + b
            sx = 40 + px * 28
            sy = 190 - py * 12
            points.append((sx, sy))
        draw.line(points, fill=(30, 90, 200), width=3)
        draw_text(draw, (70, 20), f"y={a}x{sign}{b}")
        draw_text(draw, (150, 170), f"x={x}")

    return GeneratedExample("algebra", question, options, answer_letter, str(y), draw)


def generate_bars_value(rng: random.Random, answer_letter: str) -> GeneratedExample:
    labels = ["P", "Q", "R"]
    values = [rng.randint(1, 9) for _ in labels]
    idx = rng.randrange(len(labels))
    label = labels[idx]
    value = values[idx]
    question = f"На столбчатой диаграмме показаны значения P, Q, R. Какое значение у столбца {label}?"
    options = format_options(
        str(value),
        answer_letter,
        [str(value + d) for d in [-2, -1, 1, 2, 3]],
        rng,
        BAR_VALUE_DOMAIN,
    )

    def draw(draw: ImageDraw.ImageDraw) -> None:
        draw.line((30, 190, 205, 190), fill=(0, 0, 0), width=2)
        for i, (name, val) in enumerate(zip(labels, values)):
            x0 = 55 + i * 50
            y0 = 190 - val * 15
            color = (70 + i * 40, 120, 200 - i * 30)
            draw.rectangle((x0, y0, x0 + 28, 190), fill=color, outline=(20, 20, 20))
            draw_text(draw, (x0 + 8, 194), name)
            draw_text(draw, (x0 + 8, y0 - 16), str(val))

    return GeneratedExample("plots", question, options, answer_letter, str(value), draw)


def generate_maxbar(rng: random.Random, answer_letter: str) -> GeneratedExample:
    labels = ["W", "X", "Y", "Z"]
    values = rng.sample(range(2, 10), 4)
    max_label = labels[values.index(max(values))]
    question = "На диаграмме показаны столбцы W, X, Y, Z. Какой столбец имеет максимальное значение?"
    distractors = [label for label in labels if label != max_label]
    options = format_options(max_label, answer_letter, distractors, rng, labels)

    def draw(draw: ImageDraw.ImageDraw) -> None:
        draw.line((30, 190, 205, 190), fill=(0, 0, 0), width=2)
        for i, (name, val) in enumerate(zip(labels, values)):
            x0 = 40 + i * 42
            y0 = 190 - val * 15
            draw.rectangle((x0, y0, x0 + 24, 190), fill=(200, 130, 70), outline=(20, 20, 20))
            draw_text(draw, (x0 + 7, 194), name)

    return GeneratedExample("plots", question, options, answer_letter, max_label, draw)


def generate_grid(rng: random.Random, answer_letter: str) -> GeneratedExample:
    width = rng.randint(2, 8)
    height = rng.randint(2, 6)
    area = width * height
    question = f"Прямоугольник разбит на клетки: ширина {width}, высота {height}. Чему равна площадь?"
    options = format_options(
        str(area),
        answer_letter,
        [str(area - 1), str(area + 1), str(width + height), str(area + 2)],
        rng,
        GRID_AREA_DOMAIN,
    )

    def draw(draw: ImageDraw.ImageDraw) -> None:
        cell = min(22, 150 // max(width, height))
        x0, y0 = 35, 35
        for i in range(width + 1):
            x = x0 + i * cell
            draw.line((x, y0, x, y0 + height * cell), fill=(30, 30, 30))
        for j in range(height + 1):
            y = y0 + j * cell
            draw.line((x0, y, x0 + width * cell, y), fill=(30, 30, 30))
        draw_text(draw, (x0, y0 + height * cell + 12), f"{width} x {height}")

    return GeneratedExample("geometry", question, options, answer_letter, str(area), draw)


def generate_triangle(rng: random.Random, answer_letter: str) -> GeneratedExample:
    triples = [(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (9, 12, 15)]
    a, b, c = rng.choice(triples)
    question = f"На рисунке прямоугольный треугольник с катетами {a} и {b}. Чему равна гипотенуза?"
    options = format_options(
        str(c),
        answer_letter,
        [str(c - 1), str(c + 1), str(a + b), str(max(a, b))],
        rng,
        TRIANGLE_HYPOTENUSE_DOMAIN,
    )

    def draw(draw: ImageDraw.ImageDraw) -> None:
        pts = [(45, 180), (45, 55), (180, 180)]
        draw.polygon(pts, outline=(20, 20, 20), fill=(235, 245, 255))
        draw.line((45, 180, 45, 55, 180, 180, 45, 180), fill=(20, 20, 20), width=2)
        draw_text(draw, (28, 110), str(a))
        draw_text(draw, (105, 185), str(b))
        draw_text(draw, (115, 105), "?")

    return GeneratedExample("geometry", question, options, answer_letter, str(c), draw)


def generate_angle(rng: random.Random, answer_letter: str) -> GeneratedExample:
    angle = rng.choice([40, 50, 60, 70, 80, 100, 110, 120, 130])
    other = 180 - angle
    question = f"Один из смежных углов равен {angle}°. Чему равен второй угол?"
    options = format_options(
        f"{other}°",
        answer_letter,
        [f"{angle}°", "90°", f"{other + 10}°", f"{other - 10}°"],
        rng,
        ANGLE_VALUE_DOMAIN,
    )

    def draw(draw: ImageDraw.ImageDraw) -> None:
        origin = (110, 150)
        draw.line((25, 150, 200, 150), fill=(20, 20, 20), width=2)
        rad = math.radians(180 - angle)
        end = (110 + int(80 * math.cos(rad)), 150 - int(80 * math.sin(rad)))
        draw.line((origin[0], origin[1], end[0], end[1]), fill=(200, 40, 40), width=3)
        draw_text(draw, (95, 122), f"{angle} deg")

    return GeneratedExample("geometry", question, options, answer_letter, f"{other}°", draw)


def generate_det(rng: random.Random, answer_letter: str) -> GeneratedExample:
    a, b, c, d = [rng.randint(-3, 4) for _ in range(4)]
    det = a * d - b * c
    question = "На рисунке матрица A размера 2x2. Чему равен det(A)?"
    options = format_options(
        str(det),
        answer_letter,
        [str(det - 2), str(det - 1), str(det + 1), str(det + 3)],
        rng,
        DET_VALUE_DOMAIN,
    )

    def draw(draw: ImageDraw.ImageDraw) -> None:
        draw_text(draw, (55, 45), "A =")
        draw.rectangle((95, 35, 175, 125), outline=(20, 20, 20), width=2)
        draw_text(draw, (112, 55), f"{a}   {b}")
        draw_text(draw, (112, 92), f"{c}   {d}")

    return GeneratedExample("linear_algebra", question, options, answer_letter, str(det), draw)


def generate_distance(rng: random.Random, answer_letter: str) -> GeneratedExample:
    x1, y1 = rng.randint(0, 5), rng.randint(0, 5)
    x2, y2 = rng.randint(0, 5), rng.randint(0, 5)
    while x1 == x2 and y1 == y2:
        x2, y2 = rng.randint(0, 5), rng.randint(0, 5)
    value = (x1 - x2) ** 2 + (y1 - y2) ** 2
    question = (
        f"На координатной плоскости отмечены точки P({x1},{y1}) и Q({x2},{y2}). "
        "Чему равен квадрат расстояния PQ?"
    )
    options = format_options(
        str(value),
        answer_letter,
        [str(value - 1), str(value + 1), str(value + 2), str(abs(x1 - x2) + abs(y1 - y2))],
        rng,
        DISTANCE_SQUARED_DOMAIN,
    )

    def draw(draw: ImageDraw.ImageDraw) -> None:
        x0, y0, scale = 40, 185, 24
        for tick in range(0, 6):
            x = x0 + tick * scale
            y = y0 - tick * scale
            draw.line((x, y0 - 145, x, y0), fill=(215, 215, 215), width=1)
            draw.line((x0, y, x0 + 145, y), fill=(215, 215, 215), width=1)
            draw_text(draw, (x - 3, y0 + 4), str(tick))
            draw_text(draw, (x0 - 18, y - 6), str(tick))
        draw.line((x0, y0, x0 + 150, y0), fill=(20, 20, 20), width=2)
        draw.line((x0, y0, x0, y0 - 150), fill=(20, 20, 20), width=2)
        p = (x0 + x1 * scale, y0 - y1 * scale)
        q = (x0 + x2 * scale, y0 - y2 * scale)
        draw.line((p[0], p[1], q[0], q[1]), fill=(80, 80, 180), width=2)
        draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=(220, 40, 40))
        draw.ellipse((q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4), fill=(40, 120, 40))
        draw_text(draw, (p[0] + 4, p[1] - 12), "P")
        draw_text(draw, (q[0] + 4, q[1] - 12), "Q")

    return GeneratedExample("coordinate_geometry", question, options, answer_letter, str(value), draw)


GENERATORS: list[Callable[[random.Random, str], GeneratedExample]] = [
    generate_algebra,
    generate_bars_value,
    generate_maxbar,
    generate_grid,
    generate_triangle,
    generate_angle,
    generate_det,
    generate_distance,
]


def make_image(example: GeneratedExample, path: Path, rng: random.Random) -> None:
    bg = tuple(rng.randint(238, 255) for _ in range(3))
    image = Image.new("RGB", (224, 224), bg)
    draw = ImageDraw.Draw(image)
    example.draw(draw)
    if rng.random() < 0.30:
        image = image.rotate(rng.uniform(-1.5, 1.5), resample=Image.Resampling.BICUBIC, fillcolor=bg)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def split_for_index(index: int, total: int) -> str:
    train_cut = int(total * 0.80)
    dev_cut = int(total * 0.90)
    if index < train_cut:
        return "train"
    if index < dev_cut:
        return "dev"
    return "test_public"


def split_sizes_for_total(total: int) -> dict[str, int]:
    counts: dict[str, int] = {"train": 0, "dev": 0, "test_public": 0}
    for index in range(total):
        counts[split_for_index(index, total)] += 1
    return counts


def crossed_generator_answer_schedule(size: int, seed: int) -> list[tuple[int, str]]:
    cycle = [(generator_index, letter) for generator_index in range(len(GENERATORS)) for letter in CHOICES]
    schedule: list[tuple[int, str]] = []
    while len(schedule) < size:
        remaining = size - len(schedule)
        schedule.extend(cycle[:remaining])
    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


def visual_only_question(example: GeneratedExample) -> str:
    question = example.question
    if example.subject == "algebra":
        return "На изображении показана прямая и отмечено значение x. Чему равен y?"
    if example.subject == "plots":
        return question
    if "Прямоугольник" in question:
        return "На рисунке прямоугольник разбит на клетки. Чему равна площадь?"
    if "треугольник" in question:
        return "На рисунке прямоугольный треугольник. Чему равна гипотенуза?"
    if "смеж" in question:
        return "На рисунке показаны смежные углы. Чему равен второй угол?"
    if example.subject == "coordinate_geometry":
        return "На координатной плоскости отмечены точки P и Q. Чему равен квадрат расстояния PQ?"
    return question


def generate_dataset(
    out_dir: Path,
    size: int,
    seed: int,
    overwrite: bool = False,
    visual_only: bool = False,
    legacy_schedule: bool = False,
) -> Path:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    if legacy_schedule:
        schedule = [
            (split_for_index(index, size), index % len(GENERATORS), CHOICES[index % len(CHOICES)])
            for index in range(size)
        ]
    else:
        schedule = []
        split_sizes = split_sizes_for_total(size)
        split_offsets = {"train": 11, "dev": 23, "test_public": 37}
        for split in ["train", "dev", "test_public"]:
            split_schedule = crossed_generator_answer_schedule(
                split_sizes[split],
                seed + split_offsets[split],
            )
            schedule.extend((split, generator_index, answer_letter) for generator_index, answer_letter in split_schedule)

    schedule_name = "legacy_index_cycle_v1" if legacy_schedule else "crossed_generator_answer_v2"
    option_policy = "hard_domain_v3"
    source_name = "advanced_synthetic_visual_v3" if visual_only else "advanced_synthetic_v3"
    if legacy_schedule:
        source_name = "advanced_synthetic_visual_v1_hard_options" if visual_only else "advanced_synthetic_v1_hard_options"

    for index, (split, generator_index, answer_letter) in enumerate(schedule):
        generator = GENERATORS[generator_index]
        example = generator(rng, answer_letter)
        question = visual_only_question(example) if visual_only else example.question
        row_id = f"generated_{size}_{index:06d}"
        image_name = f"{split}_{index:06d}.png"
        make_image(example, images_dir / image_name, rng)
        rows.append(
            {
                "id": row_id,
                "split": split,
                "image": f"images/{image_name}",
                "question": question,
                "options": example.options,
                "answer": example.answer,
                "subject": example.subject,
                "source": source_name,
                "metadata": {
                    "generator": generator.__name__,
                    "solver_value": example.value,
                    "answer_letter_target": answer_letter,
                    "seed": seed,
                    "visual_only": visual_only,
                    "schedule": schedule_name,
                    "option_policy": option_policy,
                },
            }
        )

    manifest = out_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    (out_dir / "README.md").write_text(
        "# advanced synthetic math VQA\n\n"
        "Generated locally by scripts/advanced_vlm_experiment.py. "
        "Do not upload or commit large generated variants unless explicitly requested.\n",
        encoding="utf-8",
    )
    return manifest


def validate_generated_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}
    generator_answer_counts: dict[str, dict[str, int]] = {}
    checked_solver_rows = 0
    skipped_solver_rows = 0
    for row in rows:
        metadata = row.get("metadata") or {}
        if "solver_value" in metadata:
            checked_solver_rows += 1
            value = str(metadata.get("solver_value", ""))
            actual = find_option_by_value(list(row.get("options", [])), value)
            if actual != row.get("answer"):
                failures.append(
                    {
                        "id": row.get("id"),
                        "expected": row.get("answer"),
                        "actual": actual,
                        "value": value,
                        "options": row.get("options"),
                    }
                )
        else:
            skipped_solver_rows += 1
        split = row.get("split", "unknown")
        split_counts.setdefault(split, {letter: 0 for letter in CHOICES})
        answer = row.get("answer")
        if answer in CHOICES:
            split_counts[split][answer] += 1
            generator = metadata.get("generator")
            if generator:
                generator_answer_counts.setdefault(str(generator), {letter: 0 for letter in CHOICES})
                generator_answer_counts[str(generator)][answer] += 1

    generator_answer_max_share = 0.0
    generator_answer_collapses: dict[str, float] = {}
    for generator, counts in generator_answer_counts.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        share = max(counts.values()) / total
        generator_answer_max_share = max(generator_answer_max_share, share)
        if share >= 0.95 and total >= len(CHOICES):
            generator_answer_collapses[generator] = share
    return {
        "n": len(rows),
        "failures": failures,
        "valid": not failures,
        "checked_solver_rows": checked_solver_rows,
        "skipped_solver_rows": skipped_solver_rows,
        "split_answer_counts": split_counts,
        "generator_answer_counts": generator_answer_counts,
        "generator_answer_max_share": generator_answer_max_share,
        "generator_answer_collapses": generator_answer_collapses,
    }


def command_generate_data(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    dataset_dir = Path(args.dataset_dir)
    manifest = generate_dataset(
        dataset_dir,
        args.size,
        args.seed,
        args.overwrite,
        args.visual_only,
        args.legacy_schedule,
    )
    rows = read_jsonl(manifest)
    validation = validate_generated_rows(rows)
    write_json(dataset_dir / "validation.json", validation)
    update_summary(
        "generated data",
        [
            f"- Generated {len(rows)} rows at `{dataset_dir}`.",
            f"- Manifest: `{manifest}`.",
            f"- Validation valid: `{validation['valid']}`.",
            f"- Visual-only questions: `{args.visual_only}`.",
            f"- Schedule: `{'legacy_index_cycle_v1' if args.legacy_schedule else 'crossed_generator_answer_v2'}`.",
            f"- Generator-answer max share: `{validation['generator_answer_max_share']:.4f}`.",
            f"- Split answer counts: `{validation['split_answer_counts']}`.",
        ],
        out_dir,
    )
    print(json.dumps({"manifest": str(manifest), "validation": validation}, ensure_ascii=False, indent=2))


def command_validate_data(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    validation = validate_generated_rows(rows)
    write_json(Path(args.out_dir) / "data_validation.json", validation)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


def copy_manifest_image(
    source_root: Path,
    image_rel: str,
    output_images_dir: Path,
    prefix: str,
    cache: dict[tuple[str, str], str],
) -> str:
    key = (str(source_root), image_rel)
    if key in cache:
        return cache[key]
    src = source_root / image_rel
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_rel.replace("/", "_"))
    out_name = f"{prefix}_{safe_name}"
    dst = output_images_dir / out_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rel = f"images/{out_name}"
    cache[key] = rel
    return rel


def counterfactual_option_rows(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows_out: list[dict[str, Any]] = []

    for row in rows:
        original_answer = str(row.get("answer", ""))
        if original_answer not in CHOICES:
            raise ValueError(f"Row {row.get('id')!r} has unsupported answer {original_answer!r}")

        values = option_values_by_label(list(row.get("options", [])))
        missing = [letter for letter in CHOICES if letter not in values]
        if missing:
            raise ValueError(f"Row {row.get('id')!r} misses option labels {missing}")

        correct_value = correct_value_for_row(row)
        new_answer = CHOICES[(CHOICES.index(original_answer) + 1) % len(CHOICES)]
        distractor_values = [values[letter] for letter in CHOICES if letter != original_answer]
        rng.shuffle(distractor_values)

        new_values: dict[str, str] = {new_answer: correct_value}
        distractor_iter = iter(distractor_values)
        for letter in CHOICES:
            if letter == new_answer:
                continue
            new_values[letter] = next(distractor_iter)

        original_id = str(row.get("id"))
        metadata = dict(row.get("metadata") or {})
        if "answer_letter_target" in metadata:
            metadata.setdefault("original_answer_letter_target", metadata["answer_letter_target"])
        metadata.update(
            {
                "ablation": "counterfactual_options",
                "original_id": original_id,
                "original_answer": original_answer,
                "counterfactual_answer": new_answer,
                "answer_letter_target": new_answer,
                "solver_value": correct_value,
                "seed": seed,
                "counterfactual_option_values": new_values,
            }
        )

        new_row = dict(row)
        new_row["id"] = f"counterfactual_{original_id}"
        new_row["options"] = [f"{letter}) {new_values[letter]}" for letter in CHOICES]
        new_row["answer"] = new_answer
        new_row["metadata"] = metadata
        rows_out.append(new_row)

    return rows_out


def command_combine_data(args: argparse.Namespace) -> None:
    out_dir = Path(args.dataset_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    cache: dict[tuple[str, str], str] = {}

    generated_manifest = Path(args.generated_manifest)
    base_manifest = Path(args.base_manifest)
    sources = [
        ("generated", generated_manifest, {"train": args.generated_train_repeat}),
        ("base", base_manifest, {"train": args.base_train_repeat, "dev": 1, "test_public": 1}),
    ]
    rows_out: list[dict[str, Any]] = []
    for prefix, manifest, repeats_by_split in sources:
        source_root = manifest.parent
        for row in read_jsonl(manifest):
            split = row.get("split")
            repeat = repeats_by_split.get(split, 0)
            for rep in range(repeat):
                new_row = dict(row)
                original_id = str(row.get("id"))
                new_row["id"] = f"{prefix}_r{rep}_{original_id}"
                new_row["image"] = copy_manifest_image(
                    source_root,
                    str(row["image"]),
                    images_dir,
                    prefix,
                    cache,
                )
                metadata = dict(row.get("metadata") or {})
                metadata.update(
                    {
                        "combined_from": prefix,
                        "source_manifest": str(manifest),
                        "original_id": original_id,
                        "repeat": rep,
                    }
                )
                new_row["metadata"] = metadata
                rows_out.append(new_row)

    manifest_out = out_dir / "manifest.jsonl"
    write_jsonl(manifest_out, rows_out)
    validation = validate_generated_rows(rows_out)
    write_json(out_dir / "validation.json", validation)
    (out_dir / "README.md").write_text(
        "# combined advanced math VQA\n\n"
        "Generated locally by scripts/advanced_vlm_experiment.py combine-data. "
        "Train combines generated data and repeated medium train rows; dev/test_public come from the base manifest.\n",
        encoding="utf-8",
    )
    update_summary(
        "combined data",
        [
            f"- Wrote combined manifest `{manifest_out}` with {len(rows_out)} rows.",
            f"- Generated train repeat: {args.generated_train_repeat}; base train repeat: {args.base_train_repeat}.",
            f"- Split answer counts: `{validation['split_answer_counts']}`.",
        ],
        Path(args.out_dir),
    )
    print(json.dumps({"manifest": str(manifest_out), "validation": validation}, ensure_ascii=False, indent=2))


def command_counterfactual_options(args: argparse.Namespace) -> None:
    source_manifest = Path(args.manifest)
    out_dir = Path(args.dataset_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    cache: dict[tuple[str, str], str] = {}

    source_rows = read_jsonl(source_manifest)
    rows_out = counterfactual_option_rows(source_rows, seed=args.seed)
    for source_row, new_row in zip(source_rows, rows_out):
        new_row["image"] = copy_manifest_image(
            source_manifest.parent,
            str(source_row["image"]),
            images_dir,
            "counterfactual",
            cache,
        )
        metadata = dict(new_row.get("metadata") or {})
        metadata["source_manifest"] = str(source_manifest)
        metadata["original_image"] = source_row.get("image")
        new_row["metadata"] = metadata

    manifest_out = out_dir / "manifest.jsonl"
    write_jsonl(manifest_out, rows_out)
    validation = validate_generated_rows(rows_out)
    write_json(out_dir / "validation.json", validation)
    (out_dir / "README.md").write_text(
        "# counterfactual-options ablation dataset\n\n"
        "Images and questions are preserved, but A/B/C/D option values are permuted "
        "and the correct letter is updated. A real visual-and-options model should "
        "remain accurate; a letter-prior model should fail.\n",
        encoding="utf-8",
    )
    update_summary(
        "counterfactual options ablation",
        [
            f"- Wrote counterfactual manifest `{manifest_out}` from `{source_manifest}`.",
            f"- Rows: {len(rows_out)}; seed: {args.seed}.",
            f"- Validation valid: `{validation['valid']}`.",
        ],
        Path(args.out_dir),
    )
    print(json.dumps({"manifest": str(manifest_out), "validation": validation}, ensure_ascii=False, indent=2))


def command_shuffle_images(args: argparse.Namespace) -> None:
    source_manifest = Path(args.manifest)
    out_dir = Path(args.dataset_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    cache: dict[tuple[str, str], str] = {}
    rng = random.Random(args.seed)
    rows = read_jsonl(source_manifest)
    rows_out: list[dict[str, Any]] = []

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", "unknown"))].append(row)

    shuffled_image_by_id: dict[str, str] = {}
    for split, split_rows in by_split.items():
        images = [str(row["image"]) for row in split_rows]
        shuffled = images[:]
        if len(shuffled) > 1:
            for _ in range(20):
                rng.shuffle(shuffled)
                if all(a != b for a, b in zip(images, shuffled)):
                    break
            else:
                shuffled = images[1:] + images[:1]
        for row, image_rel in zip(split_rows, shuffled):
            shuffled_image_by_id[str(row["id"])] = image_rel

    for row in rows:
        new_row = dict(row)
        original_id = str(row["id"])
        shuffled_rel = shuffled_image_by_id[original_id]
        new_row["id"] = f"shuffled_{original_id}"
        new_row["image"] = copy_manifest_image(
            source_manifest.parent,
            shuffled_rel,
            images_dir,
            "shuffled",
            cache,
        )
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "ablation": "image_shuffle",
                "original_id": original_id,
                "original_image": row.get("image"),
                "shuffled_image": shuffled_rel,
                "seed": args.seed,
            }
        )
        new_row["metadata"] = metadata
        rows_out.append(new_row)

    manifest_out = out_dir / "manifest.jsonl"
    write_jsonl(manifest_out, rows_out)
    validation = validate_generated_rows(rows_out)
    write_json(out_dir / "validation.json", validation)
    (out_dir / "README.md").write_text(
        "# image-shuffled ablation dataset\n\n"
        "Images are shuffled within each split while questions/options/answers remain unchanged. "
        "A real visual model should lose accuracy on visual-only data.\n",
        encoding="utf-8",
    )
    update_summary(
        "image shuffle ablation",
        [
            f"- Wrote shuffled manifest `{manifest_out}` from `{source_manifest}`.",
            f"- Rows: {len(rows_out)}; seed: {args.seed}.",
        ],
        Path(args.out_dir),
    )
    print(json.dumps({"manifest": str(manifest_out), "validation": validation}, ensure_ascii=False, indent=2))


def command_blank_images(args: argparse.Namespace) -> None:
    source_manifest = Path(args.manifest)
    out_dir = Path(args.dataset_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []

    for row in read_jsonl(source_manifest):
        new_row = dict(row)
        original_id = str(row["id"])
        image_name = f"blank_{original_id}.png"
        image_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_name)
        image = Image.new("RGB", (224, 224), tuple(args.rgb))
        image.save(images_dir / image_name)
        new_row["id"] = f"blank_{original_id}"
        new_row["image"] = f"images/{image_name}"
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "ablation": "blank_image",
                "original_id": original_id,
                "original_image": row.get("image"),
                "blank_rgb": list(args.rgb),
                "source_manifest": str(source_manifest),
            }
        )
        new_row["metadata"] = metadata
        rows_out.append(new_row)

    manifest_out = out_dir / "manifest.jsonl"
    write_jsonl(manifest_out, rows_out)
    validation = validate_generated_rows(rows_out)
    write_json(out_dir / "validation.json", validation)
    (out_dir / "README.md").write_text(
        "# blank-image ablation dataset\n\n"
        "Images are replaced with a constant blank canvas while questions/options/answers remain unchanged. "
        "Accuracy here estimates text/options/label-prior leakage.\n",
        encoding="utf-8",
    )
    update_summary(
        "blank image ablation",
        [
            f"- Wrote blank-image manifest `{manifest_out}` from `{source_manifest}`.",
            f"- Rows: {len(rows_out)}; rgb: `{list(args.rgb)}`.",
            f"- Validation valid: `{validation['valid']}`.",
        ],
        Path(args.out_dir),
    )
    print(json.dumps({"manifest": str(manifest_out), "validation": validation}, ensure_ascii=False, indent=2))


def ensure_final_suite_ablations(
    manifest: str | Path,
    run_dir: Path,
    out_dir: Path,
    seed: int,
    rgb: tuple[int, int, int],
    overwrite: bool,
) -> dict[str, Path]:
    ablation_dir = run_dir / "ablations"
    manifests = {
        "normal": Path(manifest),
        "blank": ablation_dir / "blank" / "manifest.jsonl",
        "shuffled": ablation_dir / "shuffled" / "manifest.jsonl",
        "counterfactual": ablation_dir / "counterfactual" / "manifest.jsonl",
    }

    if overwrite or not manifests["blank"].exists():
        command_blank_images(
            argparse.Namespace(
                manifest=str(manifest),
                dataset_dir=str(manifests["blank"].parent),
                rgb=rgb,
                out_dir=str(out_dir),
                overwrite=overwrite,
            )
        )
    if overwrite or not manifests["shuffled"].exists():
        command_shuffle_images(
            argparse.Namespace(
                manifest=str(manifest),
                dataset_dir=str(manifests["shuffled"].parent),
                seed=seed,
                out_dir=str(out_dir),
                overwrite=overwrite,
            )
        )
    if overwrite or not manifests["counterfactual"].exists():
        command_counterfactual_options(
            argparse.Namespace(
                manifest=str(manifest),
                dataset_dir=str(manifests["counterfactual"].parent),
                seed=seed,
                out_dir=str(out_dir),
                overwrite=overwrite,
            )
        )
    return manifests


def final_suite_verdict(
    metrics_by_variant: dict[str, dict[str, Any]],
    target_accuracy: float,
    max_blank_accuracy: float,
    max_shuffled_accuracy: float,
    min_counterfactual_accuracy: float,
    min_clean_visual_gain: float,
) -> dict[str, Any]:
    normal = metrics_by_variant["normal"]["accuracy"]
    blank = metrics_by_variant["blank"]["accuracy"]
    shuffled = metrics_by_variant["shuffled"]["accuracy"]
    counterfactual = metrics_by_variant["counterfactual"]["accuracy"]
    leakage_floor = max(blank, shuffled)
    honest_accuracy = min(normal, counterfactual)
    clean_visual_gain = honest_accuracy - leakage_floor
    counterfactual_gap = normal - counterfactual
    checks = {
        "normal_target": normal >= target_accuracy,
        "blank_near_random": blank <= max_blank_accuracy,
        "shuffled_near_random": shuffled <= max_shuffled_accuracy,
        "counterfactual_robust": counterfactual >= min_counterfactual_accuracy,
        "clean_visual_gain": clean_visual_gain >= min_clean_visual_gain,
    }
    if all(checks.values()):
        status = "pass"
        recommendation = "claim as clean visual result"
    elif honest_accuracy >= 0.50 and clean_visual_gain > 0.05 and blank <= max_blank_accuracy + 0.10:
        status = "promising"
        recommendation = "continue only with a stronger model family or larger clean curriculum"
    else:
        status = "fail"
        recommendation = "do not spend more runs on this adapter setup"
    return {
        "status": status,
        "recommendation": recommendation,
        "normal_accuracy": normal,
        "blank_accuracy": blank,
        "shuffled_accuracy": shuffled,
        "counterfactual_accuracy": counterfactual,
        "leakage_floor": leakage_floor,
        "honest_accuracy": honest_accuracy,
        "clean_visual_gain": clean_visual_gain,
        "counterfactual_gap": counterfactual_gap,
        "checks": checks,
        "thresholds": {
            "target_accuracy": target_accuracy,
            "max_blank_accuracy": max_blank_accuracy,
            "max_shuffled_accuracy": max_shuffled_accuracy,
            "min_counterfactual_accuracy": min_counterfactual_accuracy,
            "min_clean_visual_gain": min_clean_visual_gain,
        },
    }


def write_final_suite_report(run_dir: Path, suite: dict[str, Any]) -> None:
    verdict = suite["verdict"]
    lines = [
        "# Final Suite",
        "",
        "One-shot leakage-aware adapter evaluation. This is the gate for research claims.",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Recommendation: {verdict['recommendation']}",
        f"- Adapter: `{suite['adapter']}`",
        f"- Split: `{suite['split']}`; max samples: `{suite['max_samples']}`",
        "",
        "## Ablation Metrics",
        "",
        "| variant | accuracy | balanced | composite | collapse |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ["normal", "counterfactual", "blank", "shuffled"]:
        metrics = suite["variants"][name]["metrics"]
        lines.append(
            "| {name} | {acc:.4f} | {bal:.4f} | {comp:.4f} | {collapse} |".format(
                name=name,
                acc=metrics["accuracy"],
                bal=metrics["balanced_accuracy"],
                comp=metrics["composite_score"],
                collapse=metrics["collapse_flag"],
            )
        )
    lines.extend(
        [
            "",
            "## Verdict Signals",
            "",
            f"- Honest accuracy `min(normal, counterfactual)`: `{verdict['honest_accuracy']:.4f}`",
            f"- Leakage floor `max(blank, shuffled)`: `{verdict['leakage_floor']:.4f}`",
            f"- Clean visual gain: `{verdict['clean_visual_gain']:.4f}`",
            f"- Counterfactual gap `normal - counterfactual`: `{verdict['counterfactual_gap']:.4f}`",
            "",
            "## Checks",
            "",
        ]
    )
    for check, passed in verdict["checks"].items():
        lines.append(f"- `{check}`: `{passed}`")
    (run_dir / "FINAL_SUITE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_adapter_suite(
    adapter: str | Path,
    manifests: dict[str, Path],
    split: str,
    max_samples: int,
    device: str,
    dtype: str,
    num_image_tokens: int,
    collapse_threshold: float,
    out_dir: Path,
    run_dir: Path,
    suite_id: str,
) -> dict[str, dict[str, Any]]:
    module = load_mps_training_module()
    ExpConfig = module.ExpConfig
    BatchBuilder = module.BatchBuilder
    QwenVLM = module.QwenVLM
    evaluate = module.evaluate
    MathVQADataset = module.MathVQADataset

    cfg = ExpConfig(
        manifestpath=str(manifests["normal"]),
        device=device,
        dtype=dtype,
        numimagetokens=num_image_tokens,
        evalmaxsamples=max_samples,
        artifactsdir=str(run_dir),
    )
    builder = BatchBuilder(cfg)
    model = QwenVLM(cfg)
    if builder.addedtokens > 0:
        model.language.resize_token_embeddings(len(builder.tokenizer))
    model.imagetokenid = builder.imagetokenid
    model.adapter.load_state_dict(module.torch.load(adapter, map_location=device))
    model.to(device)

    results: dict[str, dict[str, Any]] = {}
    for variant, manifest in manifests.items():
        variant_dir = run_dir / variant
        dataset = MathVQADataset(str(manifest), split=split)
        effective_max = min(len(dataset), max_samples)
        model_report = evaluate(model, builder, dataset, device, effective_max)
        predictions = model_report.pop("predictions")
        write_jsonl(variant_dir / "predictions.jsonl", predictions)

        rows = filtered_rows(manifest, split, effective_max)
        pred_letters = [row["pred"] for row in predictions]
        metrics = evaluate_predictions(rows[: len(pred_letters)], pred_letters, collapse_threshold)
        metrics["model_report"] = model_report
        write_json(variant_dir / "metrics.json", metrics)
        append_jsonl(
            out_dir / "metrics.jsonl",
            {
                "run_id": f"{suite_id}-{variant}",
                "method": "final_suite_eval_adapter",
                "manifest": str(manifest),
                "split": split,
                "metrics": metrics,
                "extra": {
                    "suite_id": suite_id,
                    "variant": variant,
                    "adapter": str(adapter),
                    "device": device,
                    "dtype": dtype,
                    "num_image_tokens": num_image_tokens,
                },
            },
        )
        results[variant] = {
            "manifest": str(manifest),
            "predictions": str(variant_dir / "predictions.jsonl"),
            "metrics": metrics,
        }
    return results


def command_final_suite(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    run_id = args.run_id or f"final-suite-{now_id()}"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = args.adapter
    train_run_id = None
    if args.train_first:
        train_run_id = f"{run_id}-train"
        command_train(
            argparse.Namespace(
                manifest=args.train_manifest or args.manifest,
                device=args.device,
                epochs=args.epochs,
                eval_max_samples=args.eval_max_samples,
                cpu_timing_steps=0,
                dtype=args.dtype,
                lr=args.lr,
                num_image_tokens=args.num_image_tokens,
                optimizer="adamw",
                sam=False,
                init_adapter=args.init_adapter,
                blank_uniform_weight=args.blank_uniform_weight,
                dry_run=args.dry_run,
                out_dir=str(out_dir),
                run_id=train_run_id,
            )
        )
        adapter = str(out_dir / "runs" / train_run_id / "adapter_best.pt")

    if args.dry_run:
        plan = {
            "run_id": run_id,
            "train_first": args.train_first,
            "train_run_id": train_run_id,
            "adapter": adapter,
            "manifest": args.manifest,
            "split": args.split,
            "max_samples": args.max_samples,
            "variants": ["normal", "counterfactual", "blank", "shuffled"],
        }
        write_json(run_dir / "final_suite_plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if not adapter:
        raise SystemExit("final-suite needs --adapter or --train-first")
    if not Path(adapter).exists():
        raise SystemExit(f"Adapter not found: {adapter}")

    manifests = ensure_final_suite_ablations(
        args.manifest,
        run_dir,
        out_dir,
        args.seed,
        tuple(args.blank_rgb),
        args.overwrite_ablations,
    )
    variants = evaluate_adapter_suite(
        adapter,
        manifests,
        args.split,
        args.max_samples,
        args.device,
        args.dtype,
        args.num_image_tokens,
        args.collapse_threshold,
        out_dir,
        run_dir,
        run_id,
    )
    metrics_by_variant = {name: item["metrics"] for name, item in variants.items()}
    verdict = final_suite_verdict(
        metrics_by_variant,
        args.target_accuracy,
        args.max_blank_accuracy,
        args.max_shuffled_accuracy,
        args.min_counterfactual_accuracy,
        args.min_clean_visual_gain,
    )
    suite = {
        "run_id": run_id,
        "train_first": args.train_first,
        "train_run_id": train_run_id,
        "adapter": str(adapter),
        "manifest": str(args.manifest),
        "split": args.split,
        "max_samples": args.max_samples,
        "variants": variants,
        "verdict": verdict,
    }
    write_json(run_dir / "final_suite.json", suite)
    write_final_suite_report(run_dir, suite)
    update_summary(
        "final suite",
        [
            f"- Finished final suite `{run_id}`.",
            f"- Status: `{verdict['status']}`; clean visual gain: `{verdict['clean_visual_gain']:.4f}`.",
            f"- Outputs: `{run_dir / 'FINAL_SUITE.md'}`.",
        ],
        out_dir,
    )
    print(json.dumps(suite, ensure_ascii=False, indent=2))


def command_eval(args: argparse.Namespace) -> None:
    rows = filtered_rows(args.manifest, args.split, args.max_samples)
    if args.allow_partial:
        rows_for_eval, predictions, missing = load_prediction_subset(args.predictions, rows)
        if not rows_for_eval:
            raise SystemExit(f"No matching predictions in {args.predictions}")
        metrics = evaluate_predictions(rows_for_eval, predictions, args.collapse_threshold)
        metrics["evaluated_subset_n"] = len(rows_for_eval)
        metrics["requested_rows_total"] = len(rows)
        metrics["missing_prediction_count"] = len(missing)
    else:
        predictions = load_prediction_file(args.predictions, rows)
        rows_for_eval = rows
        metrics = evaluate_predictions(rows_for_eval, predictions, args.collapse_threshold)
    run_id = args.run_id or f"eval-{args.split}-{now_id()}"
    run_dir = Path(args.out_dir) / "runs" / run_id
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "predictions_scored.jsonl", prediction_rows(rows_for_eval, predictions, "eval_file"))
    append_jsonl(
        Path(args.out_dir) / "metrics.jsonl",
        {
            "run_id": run_id,
            "method": "eval_file",
            "manifest": str(args.manifest),
            "split": args.split,
            "metrics": metrics,
            "extra": {"predictions": str(args.predictions)},
        },
    )
    update_summary(
        "evaluation",
        [
            f"- Evaluated `{args.predictions}` on `{args.manifest}` split `{args.split}`.",
            f"- Accuracy: {metrics['accuracy']:.4f}; balanced: {metrics['balanced_accuracy']:.4f}; collapse: {metrics['collapse_flag']}.",
        ],
        Path(args.out_dir),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def load_mps_training_module() -> Any:
    module_path = REPO_ROOT / "scripts" / "train_adapter_mps.py"
    spec = importlib.util.spec_from_file_location("advanced_train_adapter_mps", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load training module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SystemExit(f"Could not import {module_path}: {exc}") from exc
    return module


def command_train(args: argparse.Namespace) -> None:
    """Run the heavy CLIP+Qwen adapter experiment locally.

    The command intentionally delegates the backbone loading to the existing MPS
    script, but stores outputs in artifacts/advanced/runs/<run_id>.
    """
    run_id = args.run_id or f"train-{now_id()}"
    run_dir = Path(args.out_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "command": "train",
        "run_id": run_id,
        "manifest": str(args.manifest),
        "epochs": args.epochs,
        "device": args.device,
        "dtype": args.dtype,
        "learning_rate": args.lr,
        "num_image_tokens": args.num_image_tokens,
        "eval_max_samples": args.eval_max_samples,
        "cpu_timing_steps": args.cpu_timing_steps,
        "optimizer": args.optimizer,
        "sam": args.sam,
        "init_adapter": args.init_adapter,
        "blank_uniform_weight": args.blank_uniform_weight,
        "note": "Delegates to scripts.train_adapter_mps for heavy CLIP+Qwen adapter training.",
    }
    write_json(run_dir / "config.json", config)

    if args.dry_run:
        update_summary(
            "train dry-run",
            [f"- Prepared local run config at `{run_dir / 'config.json'}`."],
            Path(args.out_dir),
        )
        print(json.dumps({"run_id": run_id, "dry_run": True, "run_dir": str(run_dir)}, indent=2))
        return

    module = load_mps_training_module()
    ExpConfig = module.ExpConfig
    runexperiment = module.runexperiment

    cfg = ExpConfig(
        manifestpath=str(args.manifest),
        device=args.device,
        epochs=args.epochs,
        dtype=args.dtype,
        learningrate=args.lr,
        numimagetokens=args.num_image_tokens,
        evalmaxsamples=args.eval_max_samples,
        initadapterpath=args.init_adapter,
        cputimingsteps=args.cpu_timing_steps,
        blankuniformweight=args.blank_uniform_weight,
        artifactsdir=str(run_dir),
        logpath=str(run_dir / "training_log.json"),
        adapterpath=str(run_dir / "adapter_best.pt"),
        finaladapterpath=str(run_dir / "adapter_final.pt"),
        predictionspath=str(run_dir / "dev_predictions.jsonl"),
    )
    runexperiment(cfg)

    metrics = None
    predictions_path = run_dir / "dev_predictions.jsonl"
    if predictions_path.exists():
        rows = filtered_rows(args.manifest, "dev")
        matched_rows, preds, missing = load_prediction_subset(predictions_path, rows)
        if not matched_rows:
            raise SystemExit(f"No matching predictions in {predictions_path}")
        metrics = evaluate_predictions(matched_rows, preds)
        metrics["evaluated_subset_n"] = len(matched_rows)
        metrics["dev_rows_total"] = len(rows)
        metrics["missing_prediction_count"] = len(missing)
        write_json(run_dir / "advanced_dev_metrics.json", metrics)
        append_jsonl(
            Path(args.out_dir) / "metrics.jsonl",
            {
                "run_id": run_id,
                "method": "clip_qwen_adapter",
                "manifest": str(args.manifest),
                "split": "dev",
                "metrics": metrics,
                "extra": config,
            },
        )
    update_summary(
        "train",
        [
            f"- Finished run `{run_id}` in `{run_dir}`.",
            f"- Advanced dev metrics: `{run_dir / 'advanced_dev_metrics.json'}`." if metrics else "- No predictions file found for advanced scoring.",
        ],
        Path(args.out_dir),
    )
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


def command_eval_adapter(args: argparse.Namespace) -> None:
    module = load_mps_training_module()
    ExpConfig = module.ExpConfig
    BatchBuilder = module.BatchBuilder
    QwenVLM = module.QwenVLM
    evaluate = module.evaluate
    MathVQADataset = module.MathVQADataset

    run_id = args.run_id or f"eval-adapter-{args.split}-{now_id()}"
    run_dir = Path(args.out_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExpConfig(
        manifestpath=str(args.manifest),
        device=args.device,
        dtype=args.dtype,
        numimagetokens=args.num_image_tokens,
        evalmaxsamples=args.max_samples,
        artifactsdir=str(run_dir),
    )
    builder = BatchBuilder(cfg)
    model = QwenVLM(cfg)
    if builder.addedtokens > 0:
        model.language.resize_token_embeddings(len(builder.tokenizer))
    model.imagetokenid = builder.imagetokenid
    model.adapter.load_state_dict(module.torch.load(args.adapter, map_location=args.device))
    model.to(args.device)

    dataset = MathVQADataset(str(args.manifest), split=args.split)
    model_report = evaluate(model, builder, dataset, args.device, args.max_samples)
    predictions = model_report.pop("predictions")
    write_jsonl(run_dir / "predictions.jsonl", predictions)

    rows = filtered_rows(args.manifest, args.split, args.max_samples)
    pred_letters = [row["pred"] for row in predictions]
    metrics = evaluate_predictions(rows[: len(pred_letters)], pred_letters, args.collapse_threshold)
    metrics["model_report"] = model_report
    write_json(run_dir / "metrics.json", metrics)
    append_jsonl(
        Path(args.out_dir) / "metrics.jsonl",
        {
            "run_id": run_id,
            "method": "eval_adapter",
            "manifest": str(args.manifest),
            "split": args.split,
            "metrics": metrics,
            "extra": {
                "adapter": str(args.adapter),
                "device": args.device,
                "dtype": args.dtype,
                "num_image_tokens": args.num_image_tokens,
            },
        },
    )
    update_summary(
        "adapter evaluation",
        [
            f"- Evaluated adapter `{args.adapter}` on `{args.manifest}` split `{args.split}`.",
            f"- Accuracy: {metrics['accuracy']:.4f}; balanced: {metrics['balanced_accuracy']:.4f}; collapse: {metrics['collapse_flag']}.",
            f"- Outputs: `{run_dir}`.",
        ],
        Path(args.out_dir),
    )
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


def sweep_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for lr in [3e-4, 7e-4, 1e-3]:
        for tokens in [8, 16, 32]:
            for dtype in ["bfloat16"]:
                configs.append(
                    {
                        "model_track": "clip_qwen_adapter",
                        "optimizer": "adamw",
                        "lr": lr,
                        "num_image_tokens": tokens,
                        "dtype": dtype,
                        "epochs": 2,
                        "early_stop_on_collapse": True,
                    }
                )
    configs.extend(
        [
            {
                "model_track": "qwen2_5_vl_3b_zero_shot",
                "status": "planned_optional_download",
                "eval_only": True,
            },
            {
                "model_track": "paligemma2_3b_zero_shot",
                "status": "planned_optional_download",
                "eval_only": True,
            },
            {
                "model_track": "smolvlm2_quick_sanity",
                "status": "planned_optional_download",
                "eval_only": True,
            },
        ]
    )
    return configs


def command_sweep(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    run_id = args.run_id or f"sweep-{now_id()}"
    run_dir = out_dir / "runs" / run_id
    configs = sweep_configs()
    selected = configs[: args.limit] if args.limit else configs
    for cfg in selected:
        cfg["config_hash"] = config_hash(cfg)
    write_json(run_dir / "sweep_plan.json", selected)
    update_summary(
        "sweep plan",
        [
            f"- Wrote {len(selected)} planned configs to `{run_dir / 'sweep_plan.json'}`.",
            "- Heavy training is not launched by `sweep` directly; use generated configs with `train` overnight.",
        ],
        out_dir,
    )
    print(json.dumps({"run_id": run_id, "configs": selected}, ensure_ascii=False, indent=2))


def command_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "metrics.jsonl"
    rows = read_jsonl(metrics_path) if metrics_path.exists() else []
    suite_paths = sorted(
        (out_dir / "runs").glob("*/final_suite.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows_sorted = sorted(
        rows,
        key=lambda row: row.get("metrics", {}).get("composite_score", -999),
        reverse=True,
    )
    report_lines = [
        "# Advanced VLM Local Report",
        "",
        "Local-only report. No GitHub push, PR, or upload was performed.",
        "",
        "## Top Runs",
        "",
    ]
    if rows_sorted:
        report_lines.append("| rank | run_id | method | split | accuracy | balanced | composite | collapse |")
        report_lines.append("|---:|---|---|---|---:|---:|---:|---|")
        for rank, row in enumerate(rows_sorted[: args.top_k], start=1):
            metrics = row.get("metrics", {})
            report_lines.append(
                "| {rank} | {run_id} | {method} | {split} | {acc:.4f} | {bal:.4f} | {comp:.4f} | {collapse} |".format(
                    rank=rank,
                    run_id=row.get("run_id", ""),
                    method=row.get("method", ""),
                    split=row.get("split", ""),
                    acc=metrics.get("accuracy", 0.0),
                    bal=metrics.get("balanced_accuracy", 0.0),
                    comp=metrics.get("composite_score", 0.0),
                    collapse=metrics.get("collapse_flag", False),
                )
            )
    else:
        report_lines.append("No metrics recorded yet. Run `preflight`, `baseline`, `generate-data`, or `train`.")
    if suite_paths:
        report_lines.extend(
            [
                "",
                "## Final Leakage-Aware Suites",
                "",
                "| run_id | status | normal | counterfactual | blank | shuffled | clean visual gain | recommendation |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for path in suite_paths[: args.top_k]:
            suite = json.loads(path.read_text(encoding="utf-8"))
            verdict = suite["verdict"]
            variants = suite["variants"]
            report_lines.append(
                "| {run_id} | {status} | {normal:.4f} | {counter:.4f} | {blank:.4f} | {shuffled:.4f} | {gain:.4f} | {recommendation} |".format(
                    run_id=suite["run_id"],
                    status=verdict["status"],
                    normal=variants["normal"]["metrics"]["accuracy"],
                    counter=variants["counterfactual"]["metrics"]["accuracy"],
                    blank=variants["blank"]["metrics"]["accuracy"],
                    shuffled=variants["shuffled"]["metrics"]["accuracy"],
                    gain=verdict["clean_visual_gain"],
                    recommendation=verdict["recommendation"],
                )
            )
    report_lines.extend(
        [
            "",
            "## Recommended Next Commands",
            "",
            "```bash",
            "PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/advanced_vlm_experiment.py preflight",
            "pytest -q tests_public  # toy_math_vqa is smoke-only, not a quality benchmark",
            "PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/advanced_vlm_experiment.py final-suite --manifest assets/generated_math_vqa_visual_v3_2k/manifest.jsonl --split test_public --max-samples 200 --adapter artifacts/advanced/runs/clip-qwen-v3-option-aug-init-v2-lr1e4/adapter_best.pt --device mps --run-id final-clean-v3-gate",
            "python scripts/prepare_mathvista_testmini.py --out assets/mathvista_testmini --max-samples 200",
            "python -m hw.benchmark --config configs/eval_mathvista_testmini.yaml",
            ".venv/bin/python scripts/advanced_vlm_experiment.py report",
            "```",
            "",
        ]
    )
    report_path = out_dir / "REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    update_summary(
        "report",
        [f"- Wrote `{report_path}` from {len(rows)} metric rows."],
        out_dir,
    )
    print(str(report_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Advanced local-only VLM math experiment harness")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    preflight.set_defaults(func=command_preflight)

    baseline = sub.add_parser("baseline")
    baseline.add_argument("--manifest", default=str(DEFAULT_MEDIUM_MANIFEST))
    baseline.add_argument("--split", default="dev")
    baseline.add_argument("--train-split", default="train")
    baseline.add_argument("--max-samples", type=int)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--predictions")
    baseline.add_argument("--collapse-threshold", type=float, default=0.70)
    baseline.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    baseline.add_argument("--run-id")
    baseline.set_defaults(func=command_baseline)

    gen = sub.add_parser("generate-data")
    gen.add_argument("--size", type=int, default=2000)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--dataset-dir", default="assets/generated_math_vqa_v1_2k")
    gen.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    gen.add_argument("--visual-only", action="store_true")
    gen.add_argument("--legacy-schedule", action="store_true")
    gen.add_argument("--overwrite", action="store_true")
    gen.set_defaults(func=command_generate_data)

    validate = sub.add_parser("validate-data")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    validate.set_defaults(func=command_validate_data)

    combine = sub.add_parser("combine-data")
    combine.add_argument("--generated-manifest", default="assets/generated_math_vqa_v1_2k/manifest.jsonl")
    combine.add_argument("--base-manifest", default=str(DEFAULT_MEDIUM_MANIFEST))
    combine.add_argument("--dataset-dir", default="assets/generated_math_vqa_v1_2k_plus_medium")
    combine.add_argument("--generated-train-repeat", type=int, default=1)
    combine.add_argument("--base-train-repeat", type=int, default=4)
    combine.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    combine.add_argument("--overwrite", action="store_true")
    combine.set_defaults(func=command_combine_data)

    shuffle = sub.add_parser("shuffle-images")
    shuffle.add_argument("--manifest", required=True)
    shuffle.add_argument("--dataset-dir", required=True)
    shuffle.add_argument("--seed", type=int, default=42)
    shuffle.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    shuffle.add_argument("--overwrite", action="store_true")
    shuffle.set_defaults(func=command_shuffle_images)

    counterfactual = sub.add_parser("counterfactual-options")
    counterfactual.add_argument("--manifest", required=True)
    counterfactual.add_argument("--dataset-dir", required=True)
    counterfactual.add_argument("--seed", type=int, default=42)
    counterfactual.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    counterfactual.add_argument("--overwrite", action="store_true")
    counterfactual.set_defaults(func=command_counterfactual_options)

    blank = sub.add_parser("blank-images")
    blank.add_argument("--manifest", required=True)
    blank.add_argument("--dataset-dir", required=True)
    blank.add_argument("--rgb", nargs=3, type=int, default=(248, 248, 248))
    blank.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    blank.add_argument("--overwrite", action="store_true")
    blank.set_defaults(func=command_blank_images)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--manifest", default=str(DEFAULT_MEDIUM_MANIFEST))
    eval_parser.add_argument("--split", default="dev")
    eval_parser.add_argument("--max-samples", type=int)
    eval_parser.add_argument("--predictions", required=True)
    eval_parser.add_argument("--allow-partial", action="store_true")
    eval_parser.add_argument("--collapse-threshold", type=float, default=0.70)
    eval_parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    eval_parser.add_argument("--run-id")
    eval_parser.set_defaults(func=command_eval)

    train = sub.add_parser("train")
    train.add_argument("--manifest", default=str(DEFAULT_MEDIUM_MANIFEST))
    train.add_argument("--device", default="mps", choices=["mps", "cpu"])
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--eval-max-samples", type=int, default=40)
    train.add_argument("--cpu-timing-steps", type=int, default=0)
    train.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    train.add_argument("--lr", type=float, default=7e-4)
    train.add_argument("--num-image-tokens", type=int, default=16)
    train.add_argument("--optimizer", default="adamw", choices=["adamw", "lion", "sam_adamw"])
    train.add_argument("--sam", action="store_true")
    train.add_argument("--init-adapter")
    train.add_argument("--blank-uniform-weight", type=float, default=0.0)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    train.add_argument("--run-id")
    train.set_defaults(func=command_train)

    eval_adapter = sub.add_parser("eval-adapter")
    eval_adapter.add_argument("--manifest", default=str(DEFAULT_MEDIUM_MANIFEST))
    eval_adapter.add_argument("--split", default="dev")
    eval_adapter.add_argument("--max-samples", type=int, default=40)
    eval_adapter.add_argument("--adapter", required=True)
    eval_adapter.add_argument("--device", default="mps", choices=["mps", "cpu"])
    eval_adapter.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    eval_adapter.add_argument("--num-image-tokens", type=int, default=16)
    eval_adapter.add_argument("--collapse-threshold", type=float, default=0.70)
    eval_adapter.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    eval_adapter.add_argument("--run-id")
    eval_adapter.set_defaults(func=command_eval_adapter)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--limit", type=int)
    sweep.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    sweep.add_argument("--run-id")
    sweep.set_defaults(func=command_sweep)

    final_suite = sub.add_parser("final-suite")
    final_suite.add_argument("--manifest", required=True)
    final_suite.add_argument("--split", default="test_public")
    final_suite.add_argument("--max-samples", type=int, default=200)
    final_suite.add_argument("--adapter")
    final_suite.add_argument("--train-first", action="store_true")
    final_suite.add_argument("--train-manifest")
    final_suite.add_argument("--init-adapter")
    final_suite.add_argument("--epochs", type=int, default=2)
    final_suite.add_argument("--eval-max-samples", type=int, default=80)
    final_suite.add_argument("--lr", type=float, default=1e-4)
    final_suite.add_argument("--blank-uniform-weight", type=float, default=0.0)
    final_suite.add_argument("--device", default="mps", choices=["mps", "cpu"])
    final_suite.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    final_suite.add_argument("--num-image-tokens", type=int, default=16)
    final_suite.add_argument("--seed", type=int, default=42)
    final_suite.add_argument("--blank-rgb", nargs=3, type=int, default=(248, 248, 248))
    final_suite.add_argument("--collapse-threshold", type=float, default=0.70)
    final_suite.add_argument("--target-accuracy", type=float, default=0.80)
    final_suite.add_argument("--max-blank-accuracy", type=float, default=0.35)
    final_suite.add_argument("--max-shuffled-accuracy", type=float, default=0.35)
    final_suite.add_argument("--min-counterfactual-accuracy", type=float, default=0.70)
    final_suite.add_argument("--min-clean-visual-gain", type=float, default=0.30)
    final_suite.add_argument("--overwrite-ablations", action="store_true")
    final_suite.add_argument("--dry-run", action="store_true")
    final_suite.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    final_suite.add_argument("--run-id")
    final_suite.set_defaults(func=command_final_suite)

    report = sub.add_parser("report")
    report.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    report.add_argument("--top-k", type=int, default=10)
    report.set_defaults(func=command_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
