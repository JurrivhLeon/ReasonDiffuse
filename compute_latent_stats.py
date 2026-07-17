#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sudoku_diffusion.checkpoint import load_vae
from sudoku_diffusion.data import make_loader, to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sudoku-smoke")
    p.add_argument("--vae", default="checkpoints/sudoku_vae.pt")
    p.add_argument("--out", default="checkpoints/sudoku_latent_stats.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vae, _ = load_vae(args.vae, device)
    vae.eval()
    loader = make_loader(args.data_dir, "train", args.batch_size, shuffle=False, limit=args.limit)

    count = 0
    total = None
    total_sq = None
    with torch.inference_mode():
        for raw in loader:
            batch = to_device(raw, device)
            mu, _ = vae.encode(batch.puzzle_tokens, batch.solution_classes)
            flat = mu.reshape(-1, mu.shape[-1]).float()
            total = flat.sum(dim=0) if total is None else total + flat.sum(dim=0)
            total_sq = (flat * flat).sum(dim=0) if total_sq is None else total_sq + (flat * flat).sum(dim=0)
            count += flat.shape[0]

    mean = total / count
    var = (total_sq / count - mean.square()).clamp_min(1e-6)
    std = var.sqrt()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean.cpu(), "std": std.cpu(), "count": count}, args.out)
    print(f"saved {args.out} count={count} mean_abs={mean.abs().mean().item():.4f} std_mean={std.mean().item():.4f}")


if __name__ == "__main__":
    main()
