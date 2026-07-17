#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn.functional as F

from sudoku_diffusion.checkpoint import load_vae, save_unified
from sudoku_diffusion.data import make_loader, to_device
from sudoku_diffusion.metrics import exact_and_cell_accuracy
from diffusion_reasoning_model import LatentFlowReasoner, ReasonerConfig, UnifiedLatentReasoner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sudoku-smoke")
    p.add_argument("--vae", default="checkpoints/sudoku_vae.pt")
    p.add_argument("--stats", default="checkpoints/sudoku_latent_stats.pt")
    p.add_argument("--out", default="checkpoints/sudoku_reasoner.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=None, help="Optimizer steps. Overrides --epochs if set.")
    p.add_argument("--epochs", type=int, default=60000, help="TRM-style group epochs by default")
    p.add_argument("--epoch-unit", choices=("groups", "examples"), default="groups")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lambda-fm", type=float, default=1.0)
    p.add_argument("--lambda-answer", type=float, default=1.0)
    p.add_argument("--lambda-q", type=float, default=0.1)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _epoch_size(data_dir: str, unit: str, limit: int | None) -> int:
    if limit is not None:
        return limit
    with (Path(data_dir) / "train" / "dataset.json").open("r") as f:
        metadata = json.load(f)
    if unit == "groups":
        return int(metadata["total_groups"])
    return int(metadata["total_puzzles"] * metadata["mean_puzzle_examples"])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vae, vae_ckpt = load_vae(args.vae, device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    stats = torch.load(args.stats, map_location=device)

    cfg = ReasonerConfig(
        d_model=args.d_model,
        d_z=vae_ckpt["config"]["d_z"],
        layers=args.layers,
        heads=args.heads,
        lambda_fm=args.lambda_fm,
        lambda_answer=args.lambda_answer,
        lambda_q=args.lambda_q,
    )
    reasoner = LatentFlowReasoner(cfg).to(device)
    model = UnifiedLatentReasoner(vae, reasoner).to(device)
    model.set_latent_stats(stats["mean"], stats["std"])
    opt = torch.optim.AdamW(model.reasoner.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    loader = make_loader(args.data_dir, "train", args.batch_size, shuffle=True, limit=args.train_limit)
    it = iter(loader)
    epoch_size = _epoch_size(args.data_dir, args.epoch_unit, args.train_limit)
    total_steps = args.steps if args.steps is not None else math.ceil(args.epochs * epoch_size / args.batch_size)
    print(f"training reasoner for total_steps={total_steps} epochs={args.epochs} epoch_unit={args.epoch_unit} epoch_size={epoch_size}")

    model.train()
    model.vae.eval()
    progress = tqdm(range(1, total_steps + 1), desc="reasoner", total=total_steps)
    for step in progress:
        try:
            raw = next(it)
        except StopIteration:
            it = iter(loader)
            raw = next(it)
        batch = to_device(raw, device)
        with torch.no_grad():
            z_star = model.encode_solution_normalized(batch.puzzle_tokens, batch.solution_classes)

        eps = torch.randn_like(z_star)
        tau = torch.rand(z_star.shape[0], device=device)
        z_tau = (1.0 - tau[:, None, None]) * z_star + tau[:, None, None] * eps
        target_v = eps - z_star
        out = model.reasoner_step(z_tau, batch.puzzle_tokens, tau)
        fm_loss = F.mse_loss(out["velocity"], target_v)

        z_pred = z_tau - tau[:, None, None] * out["velocity"]
        logits = model.decode_normalized(batch.puzzle_tokens, z_pred)
        answer_loss = F.cross_entropy(logits.reshape(-1, 9), batch.solution_classes.reshape(-1))

        with torch.no_grad():
            pred_digits = logits.argmax(dim=-1) + 1
            q_target = pred_digits.eq(batch.solution_digits).all(dim=1).float()
        q_loss = F.binary_cross_entropy_with_logits(out["q_logits"], q_target)

        loss = cfg.lambda_fm * fm_loss + cfg.lambda_answer * answer_loss + cfg.lambda_q * q_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.reasoner.parameters(), 1.0)
        lr_scale = min(1.0, step / max(1, args.warmup_steps)) if args.warmup_steps > 0 else 1.0
        for group in opt.param_groups:
            group["lr"] = args.lr * lr_scale
        opt.step()

        if step == 1 or step % max(1, total_steps // 10) == 0:
            exact, cell = exact_and_cell_accuracy(pred_digits, batch.solution_digits)
            epoch_done = min(args.epochs, step * args.batch_size / max(epoch_size, 1))
            progress.set_postfix(
                epoch=f"{epoch_done:.2f}/{args.epochs}",
                loss=f"{loss.item():.4f}",
                fm=f"{fm_loss.item():.4f}",
                ce=f"{answer_loss.item():.4f}",
                q=f"{q_loss.item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"step={step} epoch={epoch_done:.2f}/{args.epochs} loss={loss.item():.4f} fm={fm_loss.item():.4f} "
                f"ce={answer_loss.item():.4f} q={q_loss.item():.4f} exact={exact.item():.4f} cell={cell.item():.4f}"
            )

    save_unified(args.out, model, {"steps": total_steps, "epochs": args.epochs, "epoch_unit": args.epoch_unit, "vae": args.vae, "stats": args.stats})
    print(f"saved {Path(args.out)}")


if __name__ == "__main__":
    main()

