#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoner_ddim import (
    DDIMReasonerConfig,
    DDIMScheduleConfig,
    LatentDDIMReasoner,
    UnifiedDDIMLatentReasoner,
)
from sudoku_diffusion.checkpoint import load_ddim_unified, save_ddim_unified
from sudoku_diffusion.data import SudokuNpyDataset
from sudoku_diffusion.metrics import exact_and_cell_accuracy, sudoku_violations
from reasoner_fm import (
    LatentFlowReasoner,
    ReasonerConfig,
    SudokuVAE,
    UnifiedLatentReasoner,
    VAEConfig,
)


def main() -> None:
    ds = SudokuNpyDataset("data/sudoku-smoke", split="train", limit=4)
    item = ds[0]
    assert item["puzzle_tokens"].shape == (81,)
    assert item["solution_classes"].min().item() >= 0
    assert item["solution_classes"].max().item() <= 8
    assert item["solution_digits"].min().item() >= 1
    assert item["solution_digits"].max().item() <= 9

    puzzle = torch.stack([ds[i]["puzzle_tokens"] for i in range(4)])
    target_cls = torch.stack([ds[i]["solution_classes"] for i in range(4)])
    target_digits = torch.stack([ds[i]["solution_digits"] for i in range(4)])

    vae = SudokuVAE(VAEConfig(d_model=32, d_z=16, layers=1, heads=4))
    out = vae(puzzle, target_cls)
    assert out["logits"].shape == (4, 81, 9)
    assert out["z"].shape == (4, 81, 16)
    assert torch.isfinite(out["loss"])

    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    out["loss"].backward()
    opt.step()

    reasoner = LatentFlowReasoner(ReasonerConfig(d_model=32, d_z=16, layers=1, heads=4))
    tau = torch.rand(4)
    flow = reasoner(torch.randn(4, 81, 16), puzzle, tau)
    assert flow["velocity"].shape == (4, 81, 16)
    assert flow["q_logits"].shape == (4,)

    unified = UnifiedLatentReasoner(vae, reasoner)
    unified.set_latent_stats(torch.zeros(16), torch.ones(16))
    logits = unified.decode_normalized(puzzle, torch.randn(4, 81, 16))
    assert logits.shape == (4, 81, 9)
    sampled, q = unified.sample(puzzle, r_steps=1, k_samples=2)
    assert sampled.shape == (4, 81)
    assert q.shape == (4,)

    ddim_reasoner = LatentDDIMReasoner(
        DDIMReasonerConfig(d_model=32, d_z=16, layers=1, heads=4)
    )
    ddim_unified = UnifiedDDIMLatentReasoner(
        vae,
        ddim_reasoner,
        schedule=DDIMScheduleConfig(train_timesteps=10),
    )
    ddim_unified.set_latent_stats(torch.zeros(16), torch.ones(16))
    t_idx = torch.randint(1, ddim_unified.train_timesteps + 1, (4,))
    z_star = torch.randn(4, 81, 16)
    z_t, _ = ddim_unified.q_sample(z_star, t_idx)
    ddim_out = ddim_unified.reasoner_step(z_t, puzzle, t_idx)
    assert ddim_out["x0_pred"].shape == (4, 81, 16)
    assert ddim_out["q_logits"].shape == (4,)
    ddim_sampled, ddim_q = ddim_unified.sample(puzzle, r_steps=2, k_samples=2)
    assert ddim_sampled.shape == (4, 81)
    assert ddim_q.shape == (4,)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ddim.pt"
        save_ddim_unified(path, ddim_unified, {"test": True})
        loaded, ckpt = load_ddim_unified(path, torch.device("cpu"))
        assert ckpt["test"] is True
        loaded_sampled, loaded_q = loaded.sample(puzzle, r_steps=1, k_samples=1)
        assert loaded_sampled.shape == (4, 81)
        assert loaded_q.shape == (4,)

    pred = target_digits.clone()
    exact, cell = exact_and_cell_accuracy(pred, target_digits)
    assert exact.item() == 1.0
    assert cell.item() == 1.0
    assert sudoku_violations(pred).min().item() >= 0
    print("smoke ok")


if __name__ == "__main__":
    main()
