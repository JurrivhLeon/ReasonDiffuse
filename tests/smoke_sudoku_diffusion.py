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
from utils.sudoku.checkpoint import (
    load_ddim_unified,
    load_unified,
    save_ddim_unified,
    save_unified,
)
from utils.sudoku.data import SudokuNpyDataset
from utils.sudoku.metrics import exact_and_cell_accuracy, sudoku_violations
from reasoner_fm import (
    LatentFlowReasoner,
    MazeVAE,
    ReasonerConfig,
    SudokuVAE,
    UnifiedLatentReasoner,
    VAEConfig,
)


def exercise_maze_interface() -> None:
    bsz = 2
    seq_len = 900
    d_z = 8
    puzzle = torch.randint(0, 6, (bsz, seq_len), dtype=torch.long)
    target_cls = torch.randint(0, 5, (bsz, seq_len), dtype=torch.long)

    vae = MazeVAE(
        VAEConfig(
            d_model=16,
            d_z=d_z,
            layers=1,
            heads=4,
            seq_len=seq_len,
            input_vocab_size=6,
            output_vocab_size=5,
            prediction_offset=0,
        )
    )
    out = vae(puzzle, target_cls)
    assert out["logits"].shape == (bsz, seq_len, 5)
    assert out["z"].shape == (bsz, seq_len, d_z)
    assert torch.isfinite(out["loss"])

    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    out["loss"].backward()
    opt.step()

    reasoner = LatentFlowReasoner(
        ReasonerConfig(
            d_model=16,
            d_z=d_z,
            layers=1,
            heads=4,
            seq_len=seq_len,
            input_vocab_size=6,
        )
    )
    tau = torch.rand(bsz)
    flow = reasoner(torch.randn(bsz, seq_len, d_z), puzzle, tau)
    assert flow["velocity"].shape == (bsz, seq_len, d_z)
    assert flow["q_logits"].shape == (bsz,)

    unified = UnifiedLatentReasoner(vae, reasoner, task="maze")
    unified.set_latent_stats(torch.zeros(d_z), torch.ones(d_z))
    logits = unified.decode_normalized(puzzle, torch.randn(bsz, seq_len, d_z))
    assert logits.shape == (bsz, seq_len, 5)
    sampled, q = unified.sample(puzzle, r_steps=1, k_samples=2)
    assert sampled.shape == (bsz, seq_len)
    assert sampled.min().item() >= 0
    assert sampled.max().item() <= 4
    assert q.shape == (bsz,)

    ddim_reasoner = LatentDDIMReasoner(
        DDIMReasonerConfig(
            d_model=16,
            d_z=d_z,
            layers=1,
            heads=4,
            seq_len=seq_len,
            input_vocab_size=6,
        )
    )
    ddim_unified = UnifiedDDIMLatentReasoner(
        vae,
        ddim_reasoner,
        schedule=DDIMScheduleConfig(train_timesteps=10),
        task="maze",
    )
    ddim_unified.set_latent_stats(torch.zeros(d_z), torch.ones(d_z))
    t_idx = torch.randint(1, ddim_unified.train_timesteps + 1, (bsz,))
    z_star = torch.randn(bsz, seq_len, d_z)
    z_t, _ = ddim_unified.q_sample(z_star, t_idx)
    ddim_out = ddim_unified.reasoner_step(z_t, puzzle, t_idx)
    assert ddim_out["x0_pred"].shape == (bsz, seq_len, d_z)
    ddim_sampled, ddim_q = ddim_unified.sample(puzzle, r_steps=1, k_samples=2)
    assert ddim_sampled.shape == (bsz, seq_len)
    assert ddim_sampled.min().item() >= 0
    assert ddim_sampled.max().item() <= 4
    assert ddim_q.shape == (bsz,)

    with tempfile.TemporaryDirectory() as tmp:
        fm_path = Path(tmp) / "maze_fm.pt"
        save_unified(fm_path, unified, {"test": True})
        loaded_fm, fm_ckpt = load_unified(fm_path, torch.device("cpu"))
        assert fm_ckpt["test"] is True
        assert isinstance(loaded_fm.vae, MazeVAE)
        loaded_sampled, loaded_q = loaded_fm.sample(puzzle, r_steps=1, k_samples=1)
        assert loaded_sampled.shape == (bsz, seq_len)
        assert loaded_q.shape == (bsz,)

        ddim_path = Path(tmp) / "maze_ddim.pt"
        save_ddim_unified(ddim_path, ddim_unified, {"test": True})
        loaded_ddim, ddim_ckpt = load_ddim_unified(ddim_path, torch.device("cpu"))
        assert ddim_ckpt["test"] is True
        assert isinstance(loaded_ddim.vae, MazeVAE)
        loaded_ddim_sampled, loaded_ddim_q = loaded_ddim.sample(
            puzzle, r_steps=1, k_samples=1
        )
        assert loaded_ddim_sampled.shape == (bsz, seq_len)
        assert loaded_ddim_q.shape == (bsz,)


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
    sampled, q = unified.sample(
        puzzle, r_steps=4, k_samples=1, cycles=2, cycle_tau_start=0.25
    )
    assert sampled.shape == (4, 81)
    try:
        unified.sample(puzzle, r_steps=4, k_samples=1, cycles=2, cycle_tau_start=0.3)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected non-divisible cycle_tau_start to fail")

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

    exercise_maze_interface()

    pred = target_digits.clone()
    exact, cell = exact_and_cell_accuracy(pred, target_digits)
    assert exact.item() == 1.0
    assert cell.item() == 1.0
    assert sudoku_violations(pred).min().item() >= 0
    print("smoke ok")


if __name__ == "__main__":
    main()
