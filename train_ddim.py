#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from reasoner_ddim import (
    DDIMReasonerConfig,
    DDIMScheduleConfig,
    LatentDDIMReasoner,
    UnifiedDDIMLatentReasoner,
)
from utils.sudoku.checkpoint import load_vae, save_ddim_unified, save_vae
from utils.task_data import (
    answer_ce,
    exact_and_cell_accuracy,
    get_task_spec,
    iter_group_batches,
    iter_group_eval_batches,
    logits_to_tokens,
    split_group_ids,
    to_device,
)
from train_fm import (
    best_path,
    compute_latent_stats,
    configure_output_paths,
    flat_batches,
    parameter_counts,
    set_warmup_lr,
    train_vae,
    update_config_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("sudoku", "maze"), default="sudoku")
    p.add_argument("--data-dir", default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--best-out", default=None)
    p.add_argument("--vae-out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling", choices=("group", "flat"), default="group")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--val-group-fraction", type=float, default=0.05)
    p.add_argument("--eval-every-epochs", type=float, default=500.0)
    p.add_argument("--eval-batch-size", type=int, default=768)
    p.add_argument("--val-aug-count", type=int, default=0)
    p.add_argument("--val-R", type=int, default=4)
    p.add_argument("--val-K", type=int, default=1)

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
    p.add_argument("--vae-patience-evals", type=int, default=0)
    p.add_argument("--vae-min-aug-epochs", type=float, default=1.0)

    p.add_argument("--stats-batch-size", type=int, default=768)

    p.add_argument("--ddim-batch-size", type=int, default=768)
    p.add_argument("--ddim-steps", type=int, default=None)
    p.add_argument("--ddim-epochs", type=int, default=60000)
    p.add_argument("--ddim-lr", type=float, default=1e-4)
    p.add_argument("--ddim-warmup-steps", type=int, default=2000)
    p.add_argument("--ddim-beta1", type=float, default=0.9)
    p.add_argument("--ddim-beta2", type=float, default=0.95)
    p.add_argument("--ddim-weight-decay", type=float, default=1.0)
    p.add_argument("--ddim-d-model", type=int, default=256)
    p.add_argument("--ddim-layers", type=int, default=6)
    p.add_argument("--ddim-heads", type=int, default=8)
    p.add_argument("--train-timesteps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--lambda-x0", type=float, default=1.0)
    p.add_argument("--lambda-answer", type=float, default=1.0)
    p.add_argument("--lambda-q", type=float, default=0.1)

    p.add_argument("--rollout-batch-size", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--rollout-epochs", type=int, default=0)
    p.add_argument("--rollout-R", type=int, default=4)
    p.add_argument("--rollout-lr", type=float, default=1e-4)
    p.add_argument("--rollout-warmup-steps", type=int, default=2000)
    p.add_argument("--rollout-beta1", type=float, default=0.9)
    p.add_argument("--rollout-beta2", type=float, default=0.95)
    p.add_argument("--rollout-weight-decay", type=float, default=1.0)
    p.add_argument("--lambda-rollout", type=float, default=0.1)
    p.add_argument("--lambda-rollout-x0", type=float, default=None)
    p.add_argument("--lambda-rollout-q", type=float, default=None)
    p.add_argument("--lambda-intermediate", type=float, default=0.0)
    p.add_argument("--intermediate-weight-gamma", type=float, default=0.0)
    return p.parse_args()


@torch.inference_mode()
def evaluate_ddim_reasoner(
    model: UnifiedDDIMLatentReasoner,
    args: argparse.Namespace,
    device: torch.device,
    val_group_ids,
) -> dict[str, float]:
    vae_was_training = model.vae.training
    reasoner_was_training = model.reasoner.training
    model.vae.eval()
    model.reasoner.eval()
    total = 0
    sums = {"exact": 0.0, "cell": 0.0, "q": 0.0}
    for raw in iter_group_eval_batches(
        args.data_dir, args.task, args.eval_batch_size, val_group_ids, aug_count=args.val_aug_count
    ):
        batch = to_device(raw, device)
        pred_tokens, q_logits = model.sample(
            batch.puzzle_tokens, r_steps=args.val_R, k_samples=args.val_K
        )
        exact, cell = exact_and_cell_accuracy(pred_tokens, batch.solution_tokens)
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
    metrics["loss"] = -metrics["exact"]
    return metrics


def ddim_pointwise_loss(
    model: UnifiedDDIMLatentReasoner, batch, spec
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad():
        z_star = model.encode_solution_normalized(
            batch.puzzle_tokens, batch.solution_classes
        )
    t_idx = torch.randint(
        1,
        model.train_timesteps + 1,
        (z_star.shape[0],),
        device=z_star.device,
    )
    z_t, _ = model.q_sample(z_star, t_idx)
    out = model.reasoner_step(z_t, batch.puzzle_tokens, t_idx)
    x0_loss = F.mse_loss(out["x0_pred"], z_star)
    logits = model.decode_normalized(batch.puzzle_tokens, out["x0_pred"])
    answer_loss = answer_ce(
        logits, batch.solution_classes, model.vae.config.output_vocab_size, model.vae.config.ignore_index
    )
    with torch.no_grad():
        pred_tokens = logits_to_tokens(logits, spec)
        q_target = pred_tokens.eq(batch.solution_tokens).all(dim=1).float()
    q_loss = F.binary_cross_entropy_with_logits(out["q_logits"], q_target)
    return (
        model.reasoner.config.lambda_x0 * x0_loss
        + model.reasoner.config.lambda_answer * answer_loss
        + model.reasoner.config.lambda_q * q_loss,
        {
            "x0": x0_loss,
            "ce": answer_loss,
            "q": q_loss,
            "logits": logits,
        },
    )


def train_ddim_reasoner(
    args: argparse.Namespace,
    device: torch.device,
    model: UnifiedDDIMLatentReasoner,
    train_group_ids,
) -> int:
    for param in model.vae.parameters():
        param.requires_grad_(False)
    model.vae.eval()
    model.reasoner.train()
    opt = torch.optim.AdamW(
        model.reasoner.parameters(),
        lr=args.ddim_lr,
        betas=(args.ddim_beta1, args.ddim_beta2),
        weight_decay=args.ddim_weight_decay,
    )
    train_group_count = max(len(train_group_ids), 1)
    total_steps = (
        args.ddim_steps
        if args.ddim_steps is not None
        else math.ceil(args.ddim_epochs * train_group_count / args.ddim_batch_size)
    )
    epochs_needed = (
        math.ceil(total_steps * args.ddim_batch_size / train_group_count) + 1
    )
    group_iter = iter_group_batches(
        args.data_dir,
        args.task,
        args.ddim_batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed + 100000,
    )
    flat_iter = flat_batches(
        args.data_dir, args.task, args.ddim_batch_size, total_steps, args.train_limit
    )
    spec = get_task_spec(args.task, args.data_dir)
    print(
        f"stage=ddim total_steps={total_steps} epochs={args.ddim_epochs} sampling={args.sampling} train_groups={train_group_count} T={model.train_timesteps}"
    )
    progress = tqdm(range(1, total_steps + 1), desc="ddim", total=total_steps)
    for step in progress:
        if args.sampling == "group":
            epoch_done, raw = next(group_iter)
        else:
            _, raw = next(flat_iter)
            epoch_done = step * args.ddim_batch_size / train_group_count
        batch = to_device(raw, device)
        loss, parts = ddim_pointwise_loss(model, batch, spec)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.reasoner.parameters(), 1.0)
        set_warmup_lr(opt, args.ddim_lr, step, args.ddim_warmup_steps)
        opt.step()
        if step == 1 or step % 100 == 0:
            pred_tokens = logits_to_tokens(parts["logits"], spec)
            exact, cell = exact_and_cell_accuracy(pred_tokens, batch.solution_tokens)
            progress.set_postfix(
                epoch=f"{epoch_done:.1f}/{args.ddim_epochs}",
                loss=f"{loss.item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"stage=ddim step={step} epoch={epoch_done:.2f}/{args.ddim_epochs} loss={loss.item():.4f} x0={parts['x0'].item():.4f} ce={parts['ce'].item():.4f} q={parts['q'].item():.4f} exact={exact.item():.4f} cell={cell.item():.4f}"
            )
    return total_steps


def train_rollout(
    args: argparse.Namespace,
    device: torch.device,
    model: UnifiedDDIMLatentReasoner,
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
        args.task,
        args.rollout_batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed + 200000,
    )
    flat_iter = flat_batches(
        args.data_dir, args.task, args.rollout_batch_size, total_steps, args.train_limit
    )
    lambda_x0 = (
        args.lambda_x0 if args.lambda_rollout_x0 is None else args.lambda_rollout_x0
    )
    lambda_q = args.lambda_q if args.lambda_rollout_q is None else args.lambda_rollout_q
    best_unified_out = best_path(args.out, args.best_out)
    best_val_exact = -1.0
    best_val_loss = float("inf")
    spec = get_task_spec(args.task, args.data_dir)
    if val_group_ids.size:
        init_metrics = evaluate_ddim_reasoner(model, args, device, val_group_ids)
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
        pointwise_loss, pointwise = ddim_pointwise_loss(model, batch, spec)

        roll_out = model.rollout(
            batch.puzzle_tokens,
            r_steps=args.rollout_R,
            return_intermediates=args.lambda_intermediate > 0.0,
        )
        logits = roll_out["logits"]
        rollout_loss = answer_ce(
            logits, batch.solution_classes, model.vae.config.output_vocab_size, model.vae.config.ignore_index
        )
        intermediate_loss = logits.new_tensor(0.0)
        if args.lambda_intermediate > 0.0:
            intermediate_logits = roll_out["intermediate_logits"]
            if intermediate_logits:
                losses = torch.stack(
                    [
                        answer_ce(
                            mid, batch.solution_classes, model.vae.config.output_vocab_size, model.vae.config.ignore_index
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
            pred_tokens = logits_to_tokens(logits, spec)
            q_target = pred_tokens.eq(batch.solution_tokens).all(dim=1).float()
        q_loss = F.binary_cross_entropy_with_logits(roll_out["q_logits"], q_target)
        loss = (
            lambda_x0 * pointwise["x0"]
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
            exact, cell = exact_and_cell_accuracy(pred_tokens, batch.solution_tokens)
            progress.set_postfix(
                epoch=f"{epoch_done:.1f}/{rollout_epoch_target:g}",
                loss=f"{loss.item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"stage=rollout step={step} epoch={epoch_done:.2f}/{rollout_epoch_target:g} loss={loss.item():.4f} x0={pointwise['x0'].item():.4f} point={pointwise_loss.item():.4f} roll_ce={rollout_loss.item():.4f} q={q_loss.item():.4f} int={intermediate_loss.item():.4f} exact={exact.item():.4f} cell={cell.item():.4f}"
            )
        if val_group_ids.size and (
            step == 1 or step % eval_interval_steps == 0 or step == total_steps
        ):
            metrics = evaluate_ddim_reasoner(model, args, device, val_group_ids)
            tqdm.write(
                f"stage=rollout_eval step={step} epoch={epoch_done:.2f} R={args.val_R} K={args.val_K} val_exact={metrics['exact']:.4f} val_cell={metrics['cell']:.4f} val_q={metrics['q']:.4f}"
            )
            is_better = (metrics["exact"] > best_val_exact) or (
                metrics["exact"] == best_val_exact and metrics["loss"] < best_val_loss
            )
            if is_better:
                best_val_exact = metrics["exact"]
                best_val_loss = metrics["loss"]
                save_ddim_unified(
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
        args.task,
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
        vae, _ = load_vae(best_vae_out, device)
        print(f"loaded best VAE for stats/reasoner from {best_vae_out}")
    if args.vae_out is not None:
        save_vae(args.vae_out, vae, {"steps": args.vae_max_steps})
        print(f"saved {Path(args.vae_out)}")
    mean, std, stats_count = compute_latent_stats(args, device, vae, train_group_ids)
    reasoner = LatentDDIMReasoner(
        DDIMReasonerConfig(
            d_model=args.ddim_d_model,
            d_z=args.vae_d_z,
            layers=args.ddim_layers,
            heads=args.ddim_heads,
            lambda_x0=args.lambda_x0,
            lambda_answer=args.lambda_answer,
            lambda_q=args.lambda_q,
            seq_len=vae.config.seq_len,
            input_vocab_size=vae.config.input_vocab_size,
        )
    ).to(device)
    schedule = DDIMScheduleConfig(
        train_timesteps=args.train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    unified = UnifiedDDIMLatentReasoner(
        vae, reasoner, schedule=schedule, task=args.task
    ).to(device)
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
    ddim_steps = train_ddim_reasoner(args, device, unified, train_group_ids)
    rollout_requested = args.rollout_steps is not None or args.rollout_epochs > 0
    rollout_steps = (
        train_rollout(args, device, unified, train_group_ids, val_group_ids)
        if rollout_requested
        else 0
    )
    save_ddim_unified(
        args.out,
        unified,
        {
            "vae_steps": args.vae_max_steps,
            "ddim_steps": ddim_steps,
            "ddim_epochs": args.ddim_epochs,
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
