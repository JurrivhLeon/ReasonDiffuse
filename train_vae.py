#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from utils.sudoku.checkpoint import save_vae
from utils.sudoku.data import make_loader, to_device
from utils.sudoku.metrics import exact_and_cell_accuracy
from reasoner_fm import SudokuVAE, VAEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sudoku-smoke")
    p.add_argument("--out", default="checkpoints/sudoku_vae.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--steps", type=int, default=None, help="Deprecated alias for --max-steps"
    )
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-z", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps is not None:
        args.max_steps = args.steps
    device = torch.device(args.device)
    loader = make_loader(
        args.data_dir, "train", args.batch_size, shuffle=True, limit=args.train_limit
    )
    model = SudokuVAE(
        VAEConfig(
            d_model=args.d_model,
            d_z=args.d_z,
            layers=args.layers,
            heads=args.heads,
            beta=args.beta,
        )
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    it = iter(loader)

    model.train()
    progress = tqdm(range(1, args.max_steps + 1), desc="vae", total=args.max_steps)
    for step in progress:
        try:
            raw = next(it)
        except StopIteration:
            it = iter(loader)
            raw = next(it)
        batch = to_device(raw, device)
        out = model(batch.puzzle_tokens, batch.solution_classes)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_scale = (
            min(1.0, step / max(1, args.warmup_steps)) if args.warmup_steps > 0 else 1.0
        )
        for group in opt.param_groups:
            group["lr"] = args.lr * lr_scale
        opt.step()

        if step == 1 or step % max(1, args.max_steps // 10) == 0:
            pred = out["logits"].argmax(dim=-1) + 1
            exact, cell = exact_and_cell_accuracy(pred, batch.solution_digits)
            progress.set_postfix(
                loss=f"{out['loss'].item():.4f}",
                recon=f"{out['recon'].item():.4f}",
                kl=f"{out['kl'].item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"step={step} loss={out['loss'].item():.4f} recon={out['recon'].item():.4f} kl={out['kl'].item():.4f} exact={exact.item():.4f} cell={cell.item():.4f}"
            )

    save_vae(args.out, model, {"steps": args.max_steps})
    print(f"saved {Path(args.out)}")


if __name__ == "__main__":
    main()
