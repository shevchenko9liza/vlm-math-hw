from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        size = self.config.image_size
        image = image.resize((size, size))
        array = torch.tensor(list(image.getdata()), dtype=torch.float32)
        array = array.view(size, size, 3).permute(2, 0, 1) / 255.0
        tiles = array.unsqueeze(0).repeat(self.config.num_tiles, 1, 1, 1)
        return tiles

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        imagetokens = IMAGE_TOKEN * self.config.num_image_tokens
        visualblock = f"{IMAGE_START_TOKEN}{imagetokens}{IMAGE_END_TOKEN}"
        optionstext = "\n".join(sample.options)
        prompt = (
            f"{visualblock}\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{optionstext}\n"
            f"Ответ:"
        )
        if include_answer:
            prompt = f"{prompt} {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        promptonly = self.build_prompt(sample, include_answer=False)
        promptids = self.tokenizer(promptonly, add_special_tokens=False)["input_ids"]
        answertext = f" {sample.answer}"
        answerids = self.tokenizer(answertext, add_special_tokens=True)["input_ids"]
        inputids = promptids + answerids
        maxlen = self.config.max_length
        inputids = inputids[:maxlen]
        labels = [self.config.ignore_index] * len(promptids) + answerids
        labels = labels[:maxlen]
        attentionmask = [1] * len(inputids)
        return {
            "input_ids": torch.tensor(inputids, dtype=torch.long),
            "attention_mask": torch.tensor(attentionmask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        padid = getattr(self.tokenizer, "pad_token_id", 0) or 0
        maxlen = max(item["input_ids"].shape[0] for item in batch)
        def pad(tensor: torch.Tensor, value: int) -> torch.Tensor:
            deficit = maxlen - tensor.shape[0]
            if deficit == 0:
                return tensor
            padding = torch.full((deficit,), value, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)
        inputids = torch.stack([pad(item["input_ids"], padid) for item in batch])
        attentionmask = torch.stack([pad(item["attention_mask"], 0) for item in batch])
        labels = torch.stack([pad(item["labels"], self.config.ignore_index) for item in batch])
        pixelvalues = torch.stack([item["pixel_values"] for item in batch])
        return {
            "input_ids": inputids,
            "attention_mask": attentionmask,
            "labels": labels,
            "pixel_values": pixelvalues,
        }
