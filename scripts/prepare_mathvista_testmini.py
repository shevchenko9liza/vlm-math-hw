from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def option_label(idx: int) -> str:
    return chr(ord("A") + idx)

def normalize_choices(choices: Any) -> list[str]:
    if choices is None:
        return []
    if isinstance(choices, str):
        return [choices]
    if not isinstance(choices, list):
        return []
    normalized: list[str] = []
    for i, choice in enumerate(choices):
        label = option_label(i)
        text = str(choice)
        if text.strip().upper().startswith(f"{label})"):
            normalized.append(text)
        else:
            normalized.append(f"{label}) {text}")
    return normalized

def infer_subject(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict):
        skills = metadata.get("skills") or []
        if isinstance(skills, list) and skills:
            return str(skills[0]).replace(" ", "_").lower()
        task = metadata.get("task")
        if task:
            return str(task).replace(" ", "_").lower()
        context = metadata.get("context")
        if context:
            return str(context).replace(" ", "_").lower()
    return "mathvista"

def save_image(row: dict[str, Any], images_dir: Path, pid: str) -> str:
    image = row.get("decoded_image")
    if image is None:
        image = row.get("image")
    out_name = f"{pid}.png"
    out_path = images_dir / out_name
    if hasattr(image, "save"):
        image.convert("RGB").save(out_path)
        return f"images/{out_name}"
    raise ValueError(
        "Could not find a decoded PIL image in the MathVista row. "
        "Try loading AI4Math/MathVista through Hugging Face datasets."
    )


def convert_row(row: dict[str, Any], images_dir: Path) -> dict[str, Any]:
    pid = str(row.get("pid") or row.get("id") or len(list(images_dir.glob("*.png"))))
    question = str(row.get("question") or row.get("query") or "")
    question_type = str(row.get("question_type") or "")
    answer = str(row.get("answer") or "")
    choices = normalize_choices(row.get("choices"))
    if choices and answer and answer not in [option_label(i) for i in range(len(choices))]:
        raw_choices = [c.split(") ", 1)[-1].strip() for c in choices]
        for i, choice in enumerate(raw_choices):
            if choice.strip().lower() == answer.strip().lower():
                answer = option_label(i)
                break
    image_rel = save_image(row, images_dir, pid)
    metadata = row.get("metadata") or {}
    return {
        "id": f"mathvista_{pid}",
        "split": "testmini",
        "image": image_rel,
        "question": question,
        "options": choices,
        "answer": answer,
        "subject": infer_subject(row),
        "source": "MathVista",
        "question_type": question_type,
        "answer_type": row.get("answer_type"),
        "metadata": metadata,
    }

def is_multiple_choice_row(row: dict[str, Any]) -> bool:
    choices = normalize_choices(row.get("choices"))
    if len(choices) < 2:
        return False
    answer = str(row.get("answer") or "")
    if answer in [option_label(i) for i in range(len(choices))]:
        return True
    raw_choices = [choice.split(") ", 1)[-1].strip().lower() for choice in choices]
    return answer.strip().lower() in raw_choices

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MathVista testmini in homework manifest format")
    parser.add_argument("--out", type=Path, default=Path("assets/mathvista_testmini"))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", type=str, default="testmini")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--multiple-choice-only", action="store_true")
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install optional ML dependencies first: python -m pip install -e '.[ml]'"
        ) from exc
    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    split = load_dataset("AI4Math/MathVista", split=args.split, streaming=args.streaming)
    if not args.streaming and args.max_samples:
        split = split.select(range(min(args.max_samples, len(split))))
    written = 0
    seen = 0
    converted_rows: list[dict[str, Any]] = []
    with manifest_path.open("w", encoding="utf-8") as f:
        for raw_row in split:
            seen += 1
            row = dict(raw_row)
            if args.multiple_choice_only and not is_multiple_choice_row(row):
                continue
            converted = convert_row(row, images_dir)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            converted_rows.append(converted)
            written += 1
            if args.max_samples and written >= args.max_samples:
                break
    mc4_rows = [
        row
        for row in converted_rows
        if len(row.get("options", [])) == 4 and row.get("answer") in {"A", "B", "C", "D"}
    ]
    mc4_manifest_path = args.out / "manifest_mc4.jsonl"
    with mc4_manifest_path.open("w", encoding="utf-8") as f:
        for row in mc4_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    readme = args.out / "README.md"
    readme.write_text(
        "# MathVista testmini local cache\n\n"
        "This folder was generated by scripts/prepare_mathvista_testmini.py.\n"
        "Do not commit it to GitHub. It is ignored by .gitignore.\n",
        encoding="utf-8",
    )
    print(f"Wrote {written} examples to {manifest_path} after scanning {seen} rows")
    print(f"Wrote {len(mc4_rows)} strict A-D examples to {mc4_manifest_path}")
    print("Do not commit this folder to GitHub.")


if __name__ == "__main__":
    main()