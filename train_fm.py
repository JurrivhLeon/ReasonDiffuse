#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from reasoner_fm import (
    LatentFlowReasoner,
    ReasonerConfig,
    SudokuVAE,
    UnifiedLatentReasoner,
    VAEConfig,
)
from sudoku_diffusion.checkpoint import load_vae, save_json, save_unified, save_vae
from sudoku_diffusion.data import (
    iter_group_all_example_batches,
    iter_group_batches,
    iter_group_eval_batches,
    make_loader,
    split_group_ids,
    to_device,
)
from sudoku_diffusion.metrics import exact_and_cell_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory. The run is saved to <output-dir>/run_<TIMESTAMP> unless --run-dir is set.",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Exact run directory. Overrides --output-dir.",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--best-out", default=None)
    p.add_argument("--vae-out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling", choices=("group", "flat"), default="group")
    p.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Group limit for group sampling; example limit for flat sampling",
    )
    p.add_argument("--val-group-fraction", type=float, default=0.05)
    p.add_argument("--eval-every-epochs", type=float, default=500.0)
    p.add_argument("--eval-batch-size", type=int, default=768)
    p.add_argument(
        "--val-aug-count",
        type=int,
        default=0,
        help="Validation uses original plus this many augmented variants per held-out group",
    )
    p.add_argument(
        "--val-R",
        type=int,
        default=4,
        help="Rollout refinement steps for reasoner validation",
    )
    p.add_argument(
        "--val-K",
        type=int,
        default=1,
        help="Rollout samples per puzzle for reasoner validation",
    )

    p.add_argument("--vae-batch-size", type=int, default=768)
    p.add_argument("--vae-max-steps", type=int, default=50000)
    p.add_argument("--vae-lr", type=float, default=1e-4)
    p.add_argument("--vae-warmup-steps", type=int, default=2000)
    p.add_argument("--vae-beta1", type=float, default=0.9)
    p.add_argument("--vae-beta2", type=float, default=0.95)
    p.add_argument("--vae-weight-decay", type=float, default=1.0)
    p.add_argument("--vae-kl-beta", type=float, default=1e-4)
    p.add_argument("--vae-d-model", type=int, default=128)
    p.add_argument("--vae-d-z", type=int, default=64)
    p.add_argument("--vae-layers", type=int, default=4)
    p.add_argument("--vae-heads", type=int, default=4)
    p.add_argument("--vae-target-val-exact", type=float, default=None)
    p.add_argument(
        "--vae-patience-evals",
        type=int,
        default=0,
        help="0 disables patience early stopping",
    )
    p.add_argument(
        "--vae-min-aug-epochs",
        type=float,
        default=1.0,
        help="Minimum full augmented train-set passes before VAE early stopping can trigger",
    )

    p.add_argument("--stats-batch-size", type=int, default=768)

    p.add_argument("--reasoner-batch-size", type=int, default=768)
    p.add_argument("--reasoner-steps", type=int, default=None)
    p.add_argument("--reasoner-epochs", type=int, default=60000)
    p.add_argument("--reasoner-lr", type=float, default=1e-4)
    p.add_argument("--reasoner-warmup-steps", type=int, default=2000)
    p.add_argument("--reasoner-beta1", type=float, default=0.9)
    p.add_argument("--reasoner-beta2", type=float, default=0.95)
    p.add_argument("--reasoner-weight-decay", type=float, default=1.0)
    p.add_argument("--reasoner-d-model", type=int, default=256)
    p.add_argument("--reasoner-layers", type=int, default=6)
    p.add_argument("--reasoner-heads", type=int, default=8)
    p.add_argument("--lambda-fm", type=float, default=1.0)
    p.add_argument("--lambda-answer", type=float, default=1.0)
    p.add_argument("--lambda-q", type=float, default=0.1)

    p.add_argument("--rollout-batch-size", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--rollout-epochs", type=int, default=0)
    p.add_argument(
        "--rollout-R",
        type=int,
        default=4,
        help="Differentiable rollout steps used during rollout training",
    )
    p.add_argument("--rollout-lr", type=float, default=1e-4)
    p.add_argument("--rollout-warmup-steps", type=int, default=2000)
    p.add_argument("--rollout-beta1", type=float, default=0.9)
    p.add_argument("--rollout-beta2", type=float, default=0.95)
    p.add_argument("--rollout-weight-decay", type=float, default=1.0)
    p.add_argument("--lambda-rollout", type=float, default=0.1)
    p.add_argument(
        "--lambda-rollout-fm", type=float, default=None, help="Defaults to --lambda-fm"
    )
    p.add_argument(
        "--lambda-rollout-q", type=float, default=None, help="Defaults to --lambda-q"
    )
    p.add_argument(
        "--lambda-intermediate",
        type=float,
        default=0.0,
        help="Optional CE on non-final intermediate rollout decodes",
    )
    p.add_argument(
        "--intermediate-weight-gamma",
        type=float,
        default=0.0,
        help="Weights non-final intermediate CE by normalized (r/R)^gamma; 0 is uniform.",
    )
    return p.parse_args()


def configure_output_paths(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else Path("checkpoints") / "sudoku"
        )
        run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)
    if args.out is None:
        args.out = str(run_dir / "unified.pt")
    if args.best_out is None:
        args.best_out = str(run_dir / "unified.best.pt")
    if args.vae_out is None:
        args.vae_out = str(run_dir / "vae.pt")
    args.config_out = str(run_dir / "config.json")
    save_json(args.config_out, vars(args))
    print(f"run_dir={args.run_dir}")
    print(f"config_out={args.config_out}")
    print(f"out={args.out}")
    print(f"best_out={args.best_out}")
    print(f"vae_out={args.vae_out}")


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def update_config_json(args: argparse.Namespace, updates: dict) -> None:
    config_path = Path(args.config_out)
    with config_path.open("r") as f:
        config = json.load(f)
    config.update(updates)
    save_json(config_path, config)


def best_path(path: str, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    p = Path(path)
    return str(p.with_name(f"{p.stem}.best{p.suffix}"))


def set_warmup_lr(
    opt: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int
) -> None:
    scale = min(1.0, step / max(1, warmup_steps)) if warmup_steps > 0 else 1.0
    for group in opt.param_groups:
        group["lr"] = base_lr * scale


def augmented_train_examples(
    data_dir: str, train_group_count: int, train_limit: int | None
) -> int:
    if train_limit is not None:
        # In group mode, train_limit limits groups. Each Sudoku augmented puzzle has one row.
        from sudoku_diffusion.data import SudokuArrays

        arrays = SudokuArrays(data_dir, split="train")
        total = 0
        for group_id in range(min(train_group_count, arrays.total_groups)):
            total += int(
                arrays.group_indices[group_id + 1] - arrays.group_indices[group_id]
            )
        return max(total, 1)
    with (Path(data_dir) / "train" / "dataset.json").open("r") as f:
        metadata = json.load(f)
    total_groups = max(int(metadata["total_groups"]), 1)
    total_examples = int(
        round(
            float(metadata["total_puzzles"]) * float(metadata["mean_puzzle_examples"])
        )
    )
    return max(1, math.ceil(total_examples * train_group_count / total_groups))


def flat_batches(
    data_dir: str, batch_size: int, total_steps: int, train_limit: int | None
):
    loader = make_loader(data_dir, "train", batch_size, shuffle=True, limit=train_limit)
    it = iter(loader)
    for step in range(1, total_steps + 1):
        try:
            raw = next(it)
        except StopIteration:
            it = iter(loader)
            raw = next(it)
        yield step, raw


@torch.inference_mode()
def evaluate_vae(
    model: SudokuVAE, args: argparse.Namespace, device: torch.device, val_group_ids
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total = 0
    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "exact": 0.0, "cell": 0.0}
    for raw in iter_group_eval_batches(
        args.data_dir, args.eval_batch_size, val_group_ids, aug_count=args.val_aug_count
    ):
        batch = to_device(raw, device)
        out = model(batch.puzzle_tokens, batch.solution_classes)
        pred = out["logits"].argmax(dim=-1) + 1
        exact, cell = exact_and_cell_accuracy(pred, batch.solution_digits)
        bsz = batch.puzzle_tokens.shape[0]
        total += bsz
        sums["loss"] += out["loss"].item() * bsz
        sums["recon"] += out["recon"].item() * bsz
        sums["kl"] += out["kl"].item() * bsz
        sums["exact"] += exact.item() * bsz
        sums["cell"] += cell.item() * bsz
    if was_training:
        model.train()
    return {k: v / max(total, 1) for k, v in sums.items()} | {"count": float(total)}


@torch.inference_mode()
def evaluate_reasoner(
    model: UnifiedLatentReasoner,
    args: argparse.Namespace,
    device: torch.device,
    val_group_ids,
) -> dict[str, float]:
    """Validate reasoner with the same noise-to-solution rollout used at test time."""
    vae_was_training = model.vae.training
    reasoner_was_training = model.reasoner.training
    model.vae.eval()
    model.reasoner.eval()
    total = 0
    sums = {"exact": 0.0, "cell": 0.0, "q": 0.0}
    for raw in iter_group_eval_batches(
        args.data_dir, args.eval_batch_size, val_group_ids, aug_count=args.val_aug_count
    ):
        batch = to_device(raw, device)
        pred_digits, q_logits = model.sample(
            batch.puzzle_tokens, r_steps=args.val_R, k_samples=args.val_K
        )
        exact, cell = exact_and_cell_accuracy(pred_digits, batch.solution_digits)
        bsz = batch.puzzle_tokens.shape[0]
        total += bsz
        sums["exact"] += exact.item() * bsz
        sums["cell"] += cell.item() * bsz
        sums["q"] += q_logits.sigmoid().mean().item() * bsz
    if vae_was_training:
        model.vae.train()
    if reasoner_was_training:
        model.reasoner.train()
    metrics = {k: v / max(total, 1) for k, v in sums.items()}
    metrics["count"] = float(total)
    metrics["rollout_R"] = float(args.val_R)
    metrics["rollout_K"] = float(args.val_K)
    # Loss is kept for existing tie-break code; lower negative exact is better.
    metrics["loss"] = -metrics["exact"]
    return metrics


def train_vae(
    args: argparse.Namespace, device: torch.device, train_group_ids, val_group_ids
) -> SudokuVAE:
    model = SudokuVAE(
        VAEConfig(
            d_model=args.vae_d_model,
            d_z=args.vae_d_z,
            layers=args.vae_layers,
            heads=args.vae_heads,
            beta=args.vae_kl_beta,
        )
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.vae_lr,
        betas=(args.vae_beta1, args.vae_beta2),
        weight_decay=args.vae_weight_decay,
    )
    model.train()
    best_val_exact = -1.0
    best_val_recon = float("inf")
    stale_evals = 0
    best_vae_out = best_path(args.vae_out) if args.vae_out is not None else None
    train_group_count = max(len(train_group_ids), 1)
    min_aug_examples = augmented_train_examples(
        args.data_dir,
        train_group_count,
        args.train_limit if args.sampling == "group" else None,
    )
    min_early_stop_steps = math.ceil(
        args.vae_min_aug_epochs * min_aug_examples / args.vae_batch_size
    )
    epochs_needed = (
        math.ceil(args.vae_max_steps * args.vae_batch_size / train_group_count) + 1
    )
    eval_interval_steps = max(
        1, math.ceil(args.eval_every_epochs * train_group_count / args.vae_batch_size)
    )
    group_iter = iter_group_batches(
        args.data_dir,
        args.vae_batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed,
    )
    flat_iter = flat_batches(
        args.data_dir, args.vae_batch_size, args.vae_max_steps, args.train_limit
    )
    print(
        f"stage=vae max_steps={args.vae_max_steps} sampling={args.sampling} eval_every_epochs={args.eval_every_epochs} min_aug_examples={min_aug_examples} min_early_stop_steps={min_early_stop_steps}"
    )
    progress = tqdm(
        range(1, args.vae_max_steps + 1), desc="vae", total=args.vae_max_steps
    )
    for step in progress:
        if args.sampling == "group":
            epoch_done, raw = next(group_iter)
        else:
            _, raw = next(flat_iter)
            epoch_done = step * args.vae_batch_size / train_group_count
        batch = to_device(raw, device)
        out = model(batch.puzzle_tokens, batch.solution_classes)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_warmup_lr(opt, args.vae_lr, step, args.vae_warmup_steps)
        opt.step()
        if step == 1 or step % 100 == 0:
            pred = out["logits"].argmax(dim=-1) + 1
            exact, cell = exact_and_cell_accuracy(pred, batch.solution_digits)
            progress.set_postfix(
                epoch=f"{epoch_done:.1f}",
                loss=f"{out['loss'].item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"stage=vae step={step} epoch={epoch_done:.2f} loss={out['loss'].item():.4f} recon={out['recon'].item():.4f} kl={out['kl'].item():.4f} exact={exact.item():.4f} cell={cell.item():.4f}"
            )
        if val_group_ids.size and (
            step == 1 or step % eval_interval_steps == 0 or step == args.vae_max_steps
        ):
            metrics = evaluate_vae(model, args, device, val_group_ids)
            tqdm.write(
                f"stage=vae_eval step={step} epoch={epoch_done:.2f} val_loss={metrics['loss']:.4f} val_recon={metrics['recon']:.4f} val_exact={metrics['exact']:.4f} val_cell={metrics['cell']:.4f}"
            )
            is_better = (metrics["exact"] > best_val_exact) or (
                metrics["exact"] == best_val_exact and metrics["recon"] < best_val_recon
            )
            if is_better:
                best_val_exact = metrics["exact"]
                best_val_recon = metrics["recon"]
                stale_evals = 0
                if best_vae_out is not None:
                    save_vae(
                        best_vae_out,
                        model,
                        {
                            "steps": step,
                            "selection": "max_exact_then_min_recon",
                            "val_metrics": metrics,
                        },
                    )
                    tqdm.write(f"saved best VAE {best_vae_out}")
            else:
                stale_evals += 1
            early_stop_allowed = step >= min_early_stop_steps
            if not early_stop_allowed and (
                args.vae_target_val_exact is not None or args.vae_patience_evals > 0
            ):
                tqdm.write(
                    f"VAE early stop disabled until step {min_early_stop_steps} ({args.vae_min_aug_epochs:g} augmented epoch floor)"
                )
            if (
                early_stop_allowed
                and args.vae_target_val_exact is not None
                and metrics["exact"] >= args.vae_target_val_exact
            ):
                tqdm.write(
                    f"early stop VAE: val_exact {metrics['exact']:.4f} >= target {args.vae_target_val_exact:.4f}"
                )
                break
            if (
                early_stop_allowed
                and args.vae_patience_evals > 0
                and stale_evals >= args.vae_patience_evals
            ):
                tqdm.write(
                    f"early stop VAE: no validation improvement for {stale_evals} evals"
                )
                break
    return model


@torch.inference_mode()
def compute_latent_stats(
    args: argparse.Namespace, device: torch.device, vae: SudokuVAE, train_group_ids
) -> tuple[torch.Tensor, torch.Tensor, int]:
    vae.eval()
    count = 0
    total = None
    total_sq = None
    print("stage=stats")
    for raw in tqdm(
        iter_group_all_example_batches(
            args.data_dir, args.stats_batch_size, train_group_ids
        ),
        desc="stats",
    ):
        batch = to_device(raw, device)
        mu, _ = vae.encode(batch.puzzle_tokens, batch.solution_classes)
        flat = mu.reshape(-1, mu.shape[-1]).float()
        total = flat.sum(dim=0) if total is None else total + flat.sum(dim=0)
        total_sq = (
            (flat * flat).sum(dim=0)
            if total_sq is None
            else total_sq + (flat * flat).sum(dim=0)
        )
        count += flat.shape[0]
    mean = total / count
    var = (total_sq / count - mean.square()).clamp_min(1e-6)
    std = var.sqrt()
    print(
        f"stage=stats count={count} mean_abs={mean.abs().mean().item():.4f} std_mean={std.mean().item():.4f}"
    )
    return mean, std, count


def train_reasoner(
    args: argparse.Namespace,
    device: torch.device,
    model: UnifiedLatentReasoner,
    train_group_ids,
    val_group_ids,
) -> int:
    for param in model.vae.parameters():
        param.requires_grad_(False)
    model.vae.eval()
    model.reasoner.train()
    opt = torch.optim.AdamW(
        model.reasoner.parameters(),
        lr=args.reasoner_lr,
        betas=(args.reasoner_beta1, args.reasoner_beta2),
        weight_decay=args.reasoner_weight_decay,
    )
    train_group_count = max(len(train_group_ids), 1)
    total_steps = (
        args.reasoner_steps
        if args.reasoner_steps is not None
        else math.ceil(
            args.reasoner_epochs * train_group_count / args.reasoner_batch_size
        )
    )
    epochs_needed = (
        math.ceil(total_steps * args.reasoner_batch_size / train_group_count) + 1
    )
    group_iter = iter_group_batches(
        args.data_dir,
        args.reasoner_batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed + 100000,
    )
    flat_iter = flat_batches(
        args.data_dir, args.reasoner_batch_size, total_steps, args.train_limit
    )
    print(
        f"stage=reasoner total_steps={total_steps} epochs={args.reasoner_epochs} sampling={args.sampling} train_groups={train_group_count}"
    )
    cfg = model.reasoner.config
    progress = tqdm(range(1, total_steps + 1), desc="reasoner", total=total_steps)
    for step in progress:
        if args.sampling == "group":
            epoch_done, raw = next(group_iter)
        else:
            _, raw = next(flat_iter)
            epoch_done = step * args.reasoner_batch_size / train_group_count
        batch = to_device(raw, device)
        with torch.no_grad():
            z_star = model.encode_solution_normalized(
                batch.puzzle_tokens, batch.solution_classes
            )
        eps = torch.randn_like(z_star)
        tau = torch.rand(z_star.shape[0], device=device)
        z_tau = (1.0 - tau[:, None, None]) * z_star + tau[:, None, None] * eps
        target_v = eps - z_star
        out = model.reasoner_step(z_tau, batch.puzzle_tokens, tau)
        fm_loss = F.mse_loss(out["velocity"], target_v)
        z_pred = z_tau - tau[:, None, None] * out["velocity"]
        logits = model.decode_normalized(batch.puzzle_tokens, z_pred)
        answer_loss = F.cross_entropy(
            logits.reshape(-1, 9), batch.solution_classes.reshape(-1)
        )
        with torch.no_grad():
            pred_digits = logits.argmax(dim=-1) + 1
            q_target = pred_digits.eq(batch.solution_digits).all(dim=1).float()
        q_loss = F.binary_cross_entropy_with_logits(out["q_logits"], q_target)
        loss = (
            cfg.lambda_fm * fm_loss
            + cfg.lambda_answer * answer_loss
            + cfg.lambda_q * q_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.reasoner.parameters(), 1.0)
        set_warmup_lr(opt, args.reasoner_lr, step, args.reasoner_warmup_steps)
        opt.step()
        if step == 1 or step % 100 == 0:
            exact, cell = exact_and_cell_accuracy(pred_digits, batch.solution_digits)
            progress.set_postfix(
                epoch=f"{epoch_done:.1f}/{args.reasoner_epochs}",
                loss=f"{loss.item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"stage=reasoner step={step} epoch={epoch_done:.2f}/{args.reasoner_epochs} loss={loss.item():.4f} fm={fm_loss.item():.4f} ce={answer_loss.item():.4f} q={q_loss.item():.4f} exact={exact.item():.4f} cell={cell.item():.4f}"
            )
    return total_steps


def train_rollout(
    args: argparse.Namespace,
    device: torch.device,
    model: UnifiedLatentReasoner,
    train_group_ids,
    val_group_ids,
) -> int:
    total_steps = args.rollout_steps
    train_group_count = max(len(train_group_ids), 1)
    if total_steps is None:
        total_steps = math.ceil(
            args.rollout_epochs * train_group_count / args.rollout_batch_size
        )
    if total_steps <= 0:
        print("stage=rollout skipped")
        return 0

    for param in model.vae.parameters():
        param.requires_grad_(False)
    model.vae.eval()
    model.reasoner.train()

    opt = torch.optim.AdamW(
        model.reasoner.parameters(),
        lr=args.rollout_lr,
        betas=(args.rollout_beta1, args.rollout_beta2),
        weight_decay=args.rollout_weight_decay,
    )
    epochs_needed = (
        math.ceil(total_steps * args.rollout_batch_size / train_group_count) + 1
    )
    eval_interval_steps = max(
        1,
        math.ceil(args.eval_every_epochs * train_group_count / args.rollout_batch_size),
    )
    group_iter = iter_group_batches(
        args.data_dir,
        args.rollout_batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed + 200000,
    )
    flat_iter = flat_batches(
        args.data_dir, args.rollout_batch_size, total_steps, args.train_limit
    )
    lambda_fm = (
        args.lambda_fm if args.lambda_rollout_fm is None else args.lambda_rollout_fm
    )
    lambda_q = args.lambda_q if args.lambda_rollout_q is None else args.lambda_rollout_q
    best_unified_out = best_path(args.out, args.best_out)
    best_val_exact = -1.0
    best_val_loss = float("inf")
    if val_group_ids.size:
        init_metrics = evaluate_reasoner(model, args, device, val_group_ids)
        best_val_exact = init_metrics["exact"]
        best_val_loss = init_metrics["loss"]
        print(
            f"stage=rollout_eval_init R={args.val_R} K={args.val_K} val_exact={init_metrics['exact']:.4f} val_cell={init_metrics['cell']:.4f} val_q={init_metrics['q']:.4f}"
        )

    rollout_epoch_target = (
        args.rollout_epochs
        if args.rollout_epochs > 0
        else total_steps * args.rollout_batch_size / train_group_count
    )
    print(
        f"stage=rollout total_steps={total_steps} epochs={rollout_epoch_target:g} R_train={args.rollout_R} sampling={args.sampling} train_groups={train_group_count}"
    )
    progress = tqdm(range(1, total_steps + 1), desc="rollout", total=total_steps)
    for step in progress:
        if args.sampling == "group":
            epoch_done, raw = next(group_iter)
        else:
            _, raw = next(flat_iter)
            epoch_done = step * args.rollout_batch_size / train_group_count
        batch = to_device(raw, device)
        with torch.no_grad():
            z_star = model.encode_solution_normalized(
                batch.puzzle_tokens, batch.solution_classes
            )

        eps = torch.randn_like(z_star)
        tau = torch.rand(z_star.shape[0], device=device)
        z_tau = (1.0 - tau[:, None, None]) * z_star + tau[:, None, None] * eps
        target_v = eps - z_star
        tf_out = model.reasoner_step(z_tau, batch.puzzle_tokens, tau)
        fm_loss = F.mse_loss(tf_out["velocity"], target_v)

        roll_out = model.rollout(
            batch.puzzle_tokens,
            r_steps=args.rollout_R,
            return_intermediates=args.lambda_intermediate > 0.0,
        )
        logits = roll_out["logits"]
        rollout_loss = F.cross_entropy(
            logits.reshape(-1, 9), batch.solution_classes.reshape(-1)
        )
        intermediate_loss = logits.new_tensor(0.0)
        if args.lambda_intermediate > 0.0:
            intermediate_logits = roll_out["intermediate_logits"]
            if intermediate_logits:
                losses = torch.stack(
                    [
                        F.cross_entropy(
                            mid.reshape(-1, 9), batch.solution_classes.reshape(-1)
                        )
                        for mid in intermediate_logits
                    ]
                )
                r = torch.arange(
                    1, losses.numel() + 1, device=device, dtype=losses.dtype
                )
                weights = (r / max(args.rollout_R, 1)) ** args.intermediate_weight_gamma
                weights = weights / weights.sum().clamp_min(1e-12)
                intermediate_loss = (weights * losses).sum()
        with torch.no_grad():
            pred_digits = logits.argmax(dim=-1) + 1
            q_target = pred_digits.eq(batch.solution_digits).all(dim=1).float()
        q_loss = F.binary_cross_entropy_with_logits(roll_out["q_logits"], q_target)
        loss = (
            lambda_fm * fm_loss
            + args.lambda_rollout * rollout_loss
            + lambda_q * q_loss
            + args.lambda_intermediate * intermediate_loss
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.reasoner.parameters(), 1.0)
        set_warmup_lr(opt, args.rollout_lr, step, args.rollout_warmup_steps)
        opt.step()

        if step == 1 or step % 100 == 0:
            exact, cell = exact_and_cell_accuracy(pred_digits, batch.solution_digits)
            progress.set_postfix(
                epoch=f"{epoch_done:.1f}/{rollout_epoch_target:g}",
                loss=f"{loss.item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"stage=rollout step={step} epoch={epoch_done:.2f}/{rollout_epoch_target:g} loss={loss.item():.4f} fm={fm_loss.item():.4f} roll_ce={rollout_loss.item():.4f} q={q_loss.item():.4f} int={intermediate_loss.item():.4f} exact={exact.item():.4f} cell={cell.item():.4f}"
            )
        if val_group_ids.size and (
            step == 1 or step % eval_interval_steps == 0 or step == total_steps
        ):
            metrics = evaluate_reasoner(model, args, device, val_group_ids)
            tqdm.write(
                f"stage=rollout_eval step={step} epoch={epoch_done:.2f} R={args.val_R} K={args.val_K} val_exact={metrics['exact']:.4f} val_cell={metrics['cell']:.4f} val_q={metrics['q']:.4f}"
            )
            is_better = (metrics["exact"] > best_val_exact) or (
                metrics["exact"] == best_val_exact and metrics["loss"] < best_val_loss
            )
            if is_better:
                best_val_exact = metrics["exact"]
                best_val_loss = metrics["loss"]
                save_unified(
                    best_unified_out,
                    model,
                    {
                        "stage": "rollout",
                        "rollout_steps": step,
                        "rollout_epoch": epoch_done,
                        "rollout_R": args.rollout_R,
                        "selection": "max_rollout_exact",
                        "val_metrics": metrics,
                    },
                )
                tqdm.write(f"saved best unified {best_unified_out}")
    return total_steps


def main() -> None:
    args = parse_args()
    configure_output_paths(args)
    device = torch.device(args.device)
    train_group_ids, val_group_ids = split_group_ids(
        args.data_dir,
        args.val_group_fraction,
        seed=args.seed,
        limit_groups=args.train_limit if args.sampling == "group" else None,
    )
    print(
        f"groups train={train_group_ids.size} val={val_group_ids.size} sampling={args.sampling}"
    )
    vae = train_vae(args, device, train_group_ids, val_group_ids)
    best_vae_out = best_path(args.vae_out) if args.vae_out is not None else None
    if best_vae_out is not None and Path(best_vae_out).exists():
        vae, best_vae_ckpt = load_vae(best_vae_out, device)
        print(f"loaded best VAE for stats/reasoner from {best_vae_out}")
    if args.vae_out is not None:
        save_vae(args.vae_out, vae, {"steps": args.vae_max_steps})
        print(f"saved {Path(args.vae_out)}")
    mean, std, stats_count = compute_latent_stats(args, device, vae, train_group_ids)
    reasoner = LatentFlowReasoner(
        ReasonerConfig(
            d_model=args.reasoner_d_model,
            d_z=args.vae_d_z,
            layers=args.reasoner_layers,
            heads=args.reasoner_heads,
            lambda_fm=args.lambda_fm,
            lambda_answer=args.lambda_answer,
            lambda_q=args.lambda_q,
        )
    ).to(device)
    unified = UnifiedLatentReasoner(vae, reasoner).to(device)
    unified.set_latent_stats(mean, std)
    param_counts = {
        "vae": parameter_counts(vae),
        "reasoner": parameter_counts(reasoner),
        "unified": parameter_counts(unified),
    }
    update_config_json(args, {"parameter_counts": param_counts})
    print(
        "parameters "
        f"vae={param_counts['vae']['total']} "
        f"reasoner={param_counts['reasoner']['total']} "
        f"unified={param_counts['unified']['total']}"
    )
    reasoner_steps = train_reasoner(
        args, device, unified, train_group_ids, val_group_ids
    )
    rollout_requested = args.rollout_steps is not None or args.rollout_epochs > 0
    if rollout_requested:
        rollout_steps = train_rollout(
            args, device, unified, train_group_ids, val_group_ids
        )
    else:
        rollout_steps = 0
    save_unified(
        args.out,
        unified,
        {
            "vae_steps": args.vae_max_steps,
            "reasoner_steps": reasoner_steps,
            "reasoner_epochs": args.reasoner_epochs,
            "rollout_steps": rollout_steps,
            "rollout_epochs": args.rollout_epochs,
            "rollout_R": args.rollout_R,
            "sampling": args.sampling,
            "train_groups": int(train_group_ids.size),
            "val_groups": int(val_group_ids.size),
            "latent_stats_count": stats_count,
        },
    )
    print(f"saved {Path(args.out)}")


if __name__ == "__main__":
    main()
