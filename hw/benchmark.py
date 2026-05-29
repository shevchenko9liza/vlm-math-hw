from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    upper = text.strip().upper()
    for letter in choices:
        if upper == letter or upper == f"({letter})":
            return letter
    match = re.search(r"\b([" + "".join(choices) + r"])\b", upper)
    if match:
        return match.group(1)
    return None

def build_benchmark_prompt(question: str, options: list[str]) -> str:
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )

def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"overall": 0.0}
    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}
    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics

def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    from hw.dataset import MathVQADataset
    eval_cfg = config.get("eval", {})
    manifest_path = eval_cfg.get("manifest_path", "assets/toy_math_vqa/manifest.jsonl")
    split = eval_cfg.get("split", "dev")
    max_samples = eval_cfg.get("max_samples")
    if toy:
        max_samples = max_samples or 5
    dataset = MathVQADataset(manifest_path, split=split, max_samples=max_samples)
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        prompt = build_benchmark_prompt(sample.question, sample.options)
        prediction = parse_mc_answer(prompt) or "A"
        rows.append({
            "id": sample.id,
            "prediction": prediction,
            "answer": sample.answer,
            "subject": sample.subject,
        })
    output_path = eval_cfg.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return compute_accuracy(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()