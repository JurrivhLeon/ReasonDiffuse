from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from reasoner_fm import (
    LatentFlowReasoner,
    ReasonerConfig,
    SudokuVAE,
    UnifiedLatentReasoner,
    VAEConfig,
)
from reasoner_ddim import (
    DDIMReasonerConfig,
    DDIMScheduleConfig,
    LatentDDIMReasoner,
    UnifiedDDIMLatentReasoner,
)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_vae(path: str | Path, model: SudokuVAE, extra: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.config.__dict__, "model": model.state_dict(), **extra}, path
    )


def load_vae(
    path: str | Path, device: torch.device
) -> tuple[SudokuVAE, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    model = SudokuVAE(VAEConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def save_reasoner(
    path: str | Path, model: LatentFlowReasoner, extra: dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.config.__dict__, "model": model.state_dict(), **extra}, path
    )


def load_reasoner(
    path: str | Path, device: torch.device
) -> tuple[LatentFlowReasoner, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    model = LatentFlowReasoner(ReasonerConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def save_unified(
    path: str | Path, model: UnifiedLatentReasoner, extra: dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config_dict(),
            "model": model.state_dict(),
            **extra,
        },
        path,
    )


def load_unified(
    path: str | Path, device: torch.device
) -> tuple[UnifiedLatentReasoner, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    vae = SudokuVAE(VAEConfig(**config["vae"]))
    reasoner = LatentFlowReasoner(ReasonerConfig(**config["reasoner"]))
    model = UnifiedLatentReasoner(vae, reasoner, task=config.get("task", "sudoku")).to(
        device
    )
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def save_ddim_unified(
    path: str | Path, model: UnifiedDDIMLatentReasoner, extra: dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config_dict(),
            "model": model.state_dict(),
            **extra,
        },
        path,
    )


def load_ddim_unified(
    path: str | Path, device: torch.device
) -> tuple[UnifiedDDIMLatentReasoner, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    vae = SudokuVAE(VAEConfig(**config["vae"]))
    reasoner = LatentDDIMReasoner(DDIMReasonerConfig(**config["reasoner"]))
    schedule = DDIMScheduleConfig(**config.get("schedule", {}))
    model = UnifiedDDIMLatentReasoner(
        vae, reasoner, schedule=schedule, task=config.get("task", "sudoku")
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
