from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens
        self.queries = nn.Parameter(torch.randn(num_image_tokens, vision_hidden_size) * 0.02)
        self.projection = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        attention = torch.softmax(
            self.queries @ vision_hidden_states.transpose(1, 2), dim=-1
        )
        pooled = attention @ vision_hidden_states
        return self.projection(pooled)


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    merged = input_embeds.clone()
    mask = input_ids == image_token_id
    for row in range(input_embeds.shape[0]):
        positions = mask[row].nonzero(as_tuple=True)[0]
        merged[row, positions] = visual_embeds[row, : positions.shape[0]].to(merged.dtype)
    return merged


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision encoder over pixel tiles and map to visual embeddings.

        pixel_values: [B, T, 3, H, W]. Tiles are flattened into the batch,
        encoded, then projected by the adapter to [B, num_image_tokens, D].
        """
        batch, tiles = pixel_values.shape[0], pixel_values.shape[1]
        flat = pixel_values.view(batch * tiles, *pixel_values.shape[2:])
        vision_hidden = self.vision_encoder(flat)
        if isinstance(vision_hidden, dict):
            vision_hidden = vision_hidden["last_hidden_state"]
        if vision_hidden.dim() == 2:
            vision_hidden = vision_hidden.unsqueeze(1)
        visual_embeds = self.adapter(vision_hidden)
        visual_embeds = visual_embeds.view(batch, tiles, *visual_embeds.shape[1:])
        return visual_embeds.reshape(batch, -1, visual_embeds.shape[-1])

    def build_inputs_embeds(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Merge text embeddings with visual embeddings at <image> positions."""
        input_ids = batch["input_ids"]
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        visual_embeds = self.encode_images(batch["pixel_values"])
        return merge_visual_embeddings(
            text_embeds, input_ids, visual_embeds, self.config.image_token_id
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss."""
        inputs_embeds = self.build_inputs_embeds(batch)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        inputs_embeds = self.build_inputs_embeds(batch)
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            **generation_kwargs,
        )