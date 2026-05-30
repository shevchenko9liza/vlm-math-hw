from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUNS_DIR = Path("artifacts/real_vlm")


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


def resize_image_for_eval(image: Image.Image, max_side: int | None) -> Image.Image:
    if not max_side or max_side <= 0:
        return image
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image
    scale = max_side / longest_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def option_label(index: int) -> str:
    return chr(ord("A") + index)


def option_labels(options: list[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for index, option in enumerate(options):
        match = re.match(r"^\s*([A-Z])\)", option)
        labels.append(match.group(1).upper() if match else option_label(index))
    return tuple(labels)


def parse_choice(text: str, choices: tuple[str, ...], allow_loose: bool = True) -> str | None:
    upper = text.strip().upper()
    escaped = "".join(re.escape(choice) for choice in choices)
    for choice in choices:
        if upper == choice or upper == f"({choice})":
            return choice

    answer_pattern = rf"\b(?:FINAL ANSWER|ANSWER|ANS|OPTION|ОТВЕТ|ВАРИАНТ)\s*[:\-]?\s*([{escaped}])\b"
    match = re.search(answer_pattern, upper)
    if match:
        return match.group(1)
    if not allow_loose:
        return None

    for pattern in [rf"\(([{escaped}])\)", rf"\b([{escaped}])\b"]:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def build_prompt(question: str, options: list[str], style: str = "concise") -> str:
    options_text = "\n".join(options)
    if style == "reasoned":
        return (
            "Solve the visual math multiple-choice problem. Use the image and the text. "
            "Reason briefly, then end with a separate line exactly like `Final answer: X`, "
            "where X is one option letter.\n\n"
            f"Question: {question}\n"
            f"Options:\n{options_text}\n\n"
            "Reasoning:"
        )
    return (
        "Solve the visual math multiple-choice problem. Use the image and the text. "
        "Return exactly one option letter, with no explanation.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        "Answer:"
    )


def candidate_token_texts(label: str) -> tuple[str, ...]:
    return (label, f" {label}", f"{label})", f" {label})")


def compute_metrics(rows: list[dict[str, Any]], predictions: list[str | None]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")

    total = len(rows)
    answer_labels = sorted(
        {
            str(row.get("answer"))
            for row in rows
            if isinstance(row.get("answer"), str) and str(row.get("answer"))
        }
    )
    pred_counts = Counter(pred for pred in predictions if pred)
    answer_counts = Counter(str(row.get("answer")) for row in rows)
    correct = sum(int(pred == row.get("answer")) for row, pred in zip(rows, predictions))

    class_recall: dict[str, float | None] = {}
    recalls: list[float] = []
    for label in answer_labels:
        support = answer_counts[label]
        if support == 0:
            class_recall[label] = None
            continue
        hits = sum(int(row.get("answer") == label and pred == label) for row, pred in zip(rows, predictions))
        recall = hits / support
        class_recall[label] = recall
        recalls.append(recall)

    subject_hits: dict[str, list[int]] = defaultdict(list)
    for row, pred in zip(rows, predictions):
        subject_hits[str(row.get("subject", "unknown"))].append(int(pred == row.get("answer")))

    prediction_entropy = normalized_entropy(pred_counts, max(total, 1), answer_labels)
    max_prediction_share = max(pred_counts.values(), default=0) / total if total else 0.0

    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "class_recall": class_recall,
        "subject_accuracy": {
            subject: sum(values) / len(values) for subject, values in sorted(subject_hits.items())
        },
        "answer_distribution": {label: answer_counts.get(label, 0) for label in answer_labels},
        "prediction_distribution": {label: pred_counts.get(label, 0) for label in answer_labels},
        "invalid_prediction_count": sum(int(pred is None) for pred in predictions),
        "prediction_entropy": prediction_entropy,
        "max_prediction_share": max_prediction_share,
        "collapse_flag": max_prediction_share > 0.7 if total else False,
    }


def normalized_entropy(counts: Counter[str], total: int, labels: list[str]) -> float:
    if total <= 0 or len(labels) <= 1:
        return 0.0
    import math

    entropy = 0.0
    for label in labels:
        probability = counts.get(label, 0) / total
        if probability > 0:
            entropy -= probability * math.log(probability)
    return entropy / math.log(len(labels))


@dataclass
class LoadedVLM:
    model: Any
    processor: Any
    backend: str
    device: str


def infer_backend(model_id: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = model_id.lower()
    if "qwen2.5-vl" in lowered or "qwen2_5_vl" in lowered:
        return "qwen2_5_vl"
    if "smolvlm" in lowered:
        return "smolvlm"
    return "auto"


def torch_dtype(name: str):
    import torch

    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype {name!r}")
    return mapping[name]


def load_vlm(
    model_id: str,
    backend: str,
    device: str,
    dtype: str,
    local_files_only: bool = False,
) -> LoadedVLM:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    backend = infer_backend(model_id, backend)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    dtype_value = torch_dtype(dtype)

    if backend == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype_value,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
    elif backend == "smolvlm":
        from transformers import SmolVLMForConditionalGeneration

        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype_value,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype_value,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )

    model.eval()
    model.to(device)
    if device == "mps":
        torch.mps.empty_cache()
    return LoadedVLM(model=model, processor=processor, backend=backend, device=device)


def chat_text(processor: Any, prompt: str, backend: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        if backend == "qwen2_5_vl":
            return f"<|vision_start|><|image_pad|><|vision_end|>\n{prompt}"
        return f"<image>\n{prompt}"


def move_inputs(inputs: Any, device: str) -> Any:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generate_choice(
    loaded: LoadedVLM,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    text = chat_text(loaded.processor, prompt, loaded.backend)
    inputs = loaded.processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )
    inputs = move_inputs(inputs, loaded.device)

    with torch.inference_mode():
        generated = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, input_length:]
    decoded = loaded.processor.batch_decode(new_tokens, skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


def score_choice(
    loaded: LoadedVLM,
    image: Image.Image,
    prompt: str,
    labels: tuple[str, ...],
) -> tuple[str | None, dict[str, float]]:
    import torch

    tokenizer = getattr(loaded.processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not expose a tokenizer; option scoring is unavailable.")

    text = chat_text(loaded.processor, prompt, loaded.backend)
    inputs = loaded.processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )
    inputs = move_inputs(inputs, loaded.device)

    with torch.inference_mode():
        outputs = loaded.model(**inputs)
    logits = outputs.logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    scores: dict[str, float] = {}
    for label in labels:
        token_ids: list[int] = []
        for token_text in candidate_token_texts(label):
            ids = tokenizer.encode(token_text, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(int(ids[0]))
        token_ids = sorted(set(token_ids))
        if token_ids:
            scores[label] = max(float(log_probs[token_id].item()) for token_id in token_ids)
        else:
            scores[label] = float("-inf")

    finite_scores = {label: score for label, score in scores.items() if math.isfinite(score)}
    if not finite_scores:
        return None, scores
    prediction = max(finite_scores.items(), key=lambda item: item[1])[0]
    return prediction, scores


def evaluate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    rows = filtered_rows(manifest_path, args.split, args.max_samples)
    if not rows:
        raise SystemExit(f"No rows for split={args.split!r} in {manifest_path}")

    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUNS_DIR / args.run_id
    predictions_path = Path(args.predictions) if args.predictions else run_dir / "predictions.jsonl"
    metrics_path = Path(args.metrics) if args.metrics else run_dir / "metrics.json"

    if predictions_path.exists() and not args.overwrite:
        raise SystemExit(f"{predictions_path} already exists; pass --overwrite to replace it")
    if predictions_path.exists():
        predictions_path.unlink()

    loaded = load_vlm(
        args.model_id,
        args.backend,
        args.device,
        args.dtype,
        local_files_only=args.local_files_only,
    )
    predictions: list[str | None] = []
    start_time = time.time()

    for index, row in enumerate(rows, start=1):
        image_path = manifest_path.parent / str(row["image"])
        with Image.open(image_path) as raw_image:
            image = resize_image_for_eval(raw_image.convert("RGB"), args.max_image_side)
        options = list(row.get("options", []))
        labels = option_labels(options)
        prompt = build_prompt(str(row.get("question", "")), options, style=args.prompt_style)
        scores: dict[str, float] | None = None
        if args.inference_mode == "score":
            prediction, scores = score_choice(loaded, image, prompt, labels)
            raw_output = "option_logprobs"
        else:
            raw_output = generate_choice(loaded, image, prompt, args.max_new_tokens)
            prediction = parse_choice(
                raw_output,
                labels,
                allow_loose=args.prompt_style == "concise",
            )
        predictions.append(prediction)
        append_jsonl(
            predictions_path,
            {
                "id": row.get("id"),
                "subject": row.get("subject", "unknown"),
                "answer": row.get("answer"),
                "prediction": prediction,
                "correct": prediction == row.get("answer"),
                "raw_output": raw_output,
                "scores": scores,
                "model_id": args.model_id,
                "backend": loaded.backend,
                "inference_mode": args.inference_mode,
                "split": args.split,
            },
        )
        if args.print_every and (index == 1 or index % args.print_every == 0 or index == len(rows)):
            elapsed = time.time() - start_time
            print(
                f"[{index}/{len(rows)}] pred={prediction} answer={row.get('answer')} "
                f"elapsed={elapsed:.1f}s raw={raw_output[:80]!r}",
                flush=True,
            )

    metrics = compute_metrics(rows, predictions)
    metrics.update(
        {
            "model_id": args.model_id,
            "backend": loaded.backend,
            "inference_mode": args.inference_mode,
            "manifest": str(manifest_path),
            "split": args.split,
            "max_image_side": args.max_image_side,
            "predictions_path": str(predictions_path),
            "elapsed_sec": time.time() - start_time,
        }
    )
    write_json(metrics_path, metrics)
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False, indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation for real VLMs on manifest VQA data")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="testmini")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", choices=["auto", "qwen2_5_vl", "smolvlm"], default="auto")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--inference-mode", choices=["generate", "score"], default="generate")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt-style", choices=["concise", "reasoned"], default="concise")
    parser.add_argument("--max-image-side", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--run-id", default="real-vlm-zero-shot")
    parser.add_argument("--run-dir")
    parser.add_argument("--predictions")
    parser.add_argument("--metrics")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-every", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate_manifest(args)


if __name__ == "__main__":
    main()
