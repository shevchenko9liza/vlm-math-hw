from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    model.train()
    output = model(batch)
    loss = output["loss"] if isinstance(output, dict) else output
    if not torch.isfinite(loss):
        raise ValueError("Loss is not finite")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    trainer = config.get("trainer", {})
    max_steps = int(trainer.get("max_steps", 3))
    if fast_train:
        max_steps = min(max_steps, 3)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    class TinyDummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)
        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            pred = self.linear(batch["x"])
            loss = torch.nn.functional.mse_loss(pred, batch["y"])
            return {"loss": loss}
    model = TinyDummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(max_steps):
        batch = {"x": torch.randn(4, 4), "y": torch.randn(4, 1)}
        loss = train_one_step(model, batch, optimizer)
        print(f"step {step + 1}/{max_steps} | loss={loss:.4f}")
    save_path = trainer.get("save_path")
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, path)
        print(f"saved checkpoint to {path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)

if __name__ == "__main__":
    main()