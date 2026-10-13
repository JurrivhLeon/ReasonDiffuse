from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .model import AmbientFlowConfig, AmbientFlowModel


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_ambient(
    path: str | Path, model: AmbientFlowModel, extra: dict[str, Any] | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "ambient_clean_flow",
        "config": model.config_dict(),
        "model": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ambient(
    path: str | Path, device: torch.device
) -> tuple[AmbientFlowModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != "ambient_clean_flow":
        raise ValueError(f"Not an ambient clean-flow checkpoint: {path}")
    config_data = dict(checkpoint["config"])
    config_data.pop("effective_ambient_dim", None)
    model = AmbientFlowModel(AmbientFlowConfig(**config_data)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
