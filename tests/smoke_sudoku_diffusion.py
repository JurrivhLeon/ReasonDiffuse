#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sudoku_diffusion.data import SudokuNpyDataset
from sudoku_diffusion.metrics import exact_and_cell_accuracy, sudoku_violations
from diffusion_reasoning_model import LatentFlowReasoner, ReasonerConfig, SudokuVAE, UnifiedLatentReasoner, VAEConfig


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

    pred = target_digits.clone()
    exact, cell = exact_and_cell_accuracy(pred, target_digits)
    assert exact.item() == 1.0
    assert cell.item() == 1.0
    assert sudoku_violations(pred).min().item() >= 0
    print("smoke ok")


if __name__ == "__main__":
    main()
