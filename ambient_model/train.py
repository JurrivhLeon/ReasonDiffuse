#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from utils.task_data import (
    get_task_spec,
    iter_group_batches,
    iter_group_eval_batches,
    make_loader,
    split_group_ids,
    to_device,
)

from .checkpoint import save_ambient, save_json
from .layers import token_cross_entropy
from .model import AmbientFlowConfig, AmbientFlowModel
from .runtime import ModuleEMA, exact_and_cell_accuracy, make_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ambient categorical flow model"
    )
    parser.add_argument("--task", choices=("sudoku", "maze"), default="sudoku")
    parser.add_argument("--data-dir", default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--embedding-type", choices=("fixed", "learned"), default=None)
    parser.add_argument("--ambient-dim", type=int, default=64)
    parser.add_argument("--cosine-temperature", type=float, default=0.1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--best-out", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument(
        "--ddp-batch-size-mode", choices=("global", "per-rank"), default="global"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("group", "flat"), default="group")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-group-fraction", type=float, default=0.05)
    parser.add_argument("--eval-every-epochs", type=float, default=500.0)
    parser.add_argument("--eval-batch-size", type=int, default=768)
    parser.add_argument("--val-aug-count", type=int, default=0)
    parser.add_argument("--val-R", type=int, default=8)
    parser.add_argument("--val-K", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-clean", type=float, default=1.0)
    parser.add_argument("--lambda-q", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--stablemax", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--ema-decay", type=float, default=0.0)

    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--rollout-epochs", type=int, default=0)
    parser.add_argument("--rollout-R", type=int, default=8)
    parser.add_argument("--rollout-lr", type=float, default=1e-4)
    parser.add_argument("--rollout-warmup-steps", type=int, default=500)
    parser.add_argument("--rollout-beta1", type=float, default=0.9)
    parser.add_argument("--rollout-beta2", type=float, default=0.95)
    parser.add_argument("--rollout-weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-rollout", type=float, default=1.0)
    parser.add_argument(
        "--lambda-rollout-clean",
        "--lambda-rollout-fm",
        dest="lambda_rollout_clean",
        type=float,
        default=None,
    )
    parser.add_argument("--lambda-rollout-q", type=float, default=None)
    parser.add_argument("--lambda-intermediate", type=float, default=1.0)
    parser.add_argument("--intermediate-weight-gamma", type=float, default=1.0)

    args = parser.parse_args()

    if args.embedding_type is None:
        args.embedding_type = "fixed" if args.task == "sudoku" else "learned"
    if args.ambient_dim < 1:
        parser.error("--ambient-dim must be positive")
    if args.cosine_temperature <= 0.0:
        parser.error("--cosine-temperature must be positive")
    if args.val_R < 1 or args.val_K < 1:
        parser.error("validation rollout counts must be positive")
    if args.rollout_R < 1:
        parser.error("--rollout-R must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema-decay must be in [0, 1)")
    return args


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_rank_zero() -> bool:
    return rank() == 0


def rank_print(*args, **kwargs) -> None:
    if is_rank_zero():
        print(*args, **kwargs)


def barrier() -> None:
    if is_dist():
        dist.barrier()


def init_distributed(args: argparse.Namespace) -> None:
    if not args.ddp:
        return
    if not torch.cuda.is_available():
        raise ValueError("--ddp requires CUDA")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", rank() % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> AmbientFlowModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def maybe_ddp(model: AmbientFlowModel, args: argparse.Namespace) -> torch.nn.Module:
    if not args.ddp:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()],
        output_device=torch.cuda.current_device(),
        find_unused_parameters=args.ddp_find_unused_parameters,
        broadcast_buffers=False,
    )


def local_batch_size(args: argparse.Namespace, configured: int) -> int:
    if not args.ddp or args.ddp_batch_size_mode == "per-rank":
        return configured
    return max(1, math.ceil(configured / world_size()))


def configure_outputs(args: argparse.Namespace) -> None:
    payload = None
    if is_rank_zero():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.run_dir is None:
            base = (
                Path(args.output_dir)
                if args.output_dir is not None
                else Path("checkpoints") / "ambient_model" / args.task
            )
            args.run_dir = str(base / f"run_{timestamp}")
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        args.out = args.out or str(run_dir / "ambient_flow.pt")
        args.best_out = args.best_out or str(run_dir / "ambient_flow.best.pt")
        args.config_out = str(run_dir / "config.json")
        payload = {
            "run_dir": args.run_dir,
            "out": args.out,
            "best_out": args.best_out,
            "config_out": args.config_out,
        }
    if is_dist():
        objects = [payload]
        dist.broadcast_object_list(objects, src=0)
        payload = objects[0]
    for key, value in payload.items():
        setattr(args, key, value)
    if is_rank_zero():
        save_json(args.config_out, vars(args))
    rank_print(f"run_dir={args.run_dir}")


def flat_batches(args: argparse.Namespace, batch_size: int, total_steps: int):
    loader = make_loader(
        args.data_dir,
        args.task,
        "train",
        batch_size,
        shuffle=True,
        limit=args.train_limit,
    )
    iterator = iter(loader)
    for step in range(1, total_steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        yield float(step), raw


def make_batches(
    args: argparse.Namespace,
    batch_size: int,
    total_steps: int,
    train_group_ids,
    seed_offset: int,
):
    if args.sampling == "flat":
        return flat_batches(args, batch_size, total_steps)
    group_count = max(len(train_group_ids), 1)
    epochs_needed = math.ceil(total_steps * batch_size / group_count) + 1
    return iter_group_batches(
        args.data_dir,
        args.task,
        batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed + seed_offset + rank(),
    )


def set_warmup_lr(
    optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int
) -> None:
    scale = min(1.0, step / max(warmup_steps, 1)) if warmup_steps > 0 else 1.0
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def clean_loss(
    logits: torch.Tensor,
    solution_classes: torch.Tensor,
    model: AmbientFlowModel,
    args: argparse.Namespace,
) -> torch.Tensor:
    return token_cross_entropy(
        logits,
        solution_classes,
        model.config.output_vocab_size,
        model.config.ignore_index,
        args.label_smoothing,
        args.stablemax,
    )


def predictions(logits: torch.Tensor, model: AmbientFlowModel) -> torch.Tensor:
    return logits.argmax(dim=-1) + model.config.prediction_offset


@torch.inference_mode()
def evaluate(
    model: AmbientFlowModel,
    args: argparse.Namespace,
    device: torch.device,
    val_group_ids,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total = 0
    sums = {"exact": 0.0, "cell": 0.0, "q": 0.0}
    for raw in iter_group_eval_batches(
        args.data_dir,
        args.task,
        args.eval_batch_size,
        val_group_ids,
        aug_count=args.val_aug_count,
    ):
        batch = to_device(raw, device)
        predicted, q_logits = model.sample(
            batch.puzzle_tokens,
            r_steps=args.val_R,
            k_samples=args.val_K,
        )
        exact, cell = exact_and_cell_accuracy(predicted, batch.solution_tokens)
        batch_size = predicted.shape[0]
        total += batch_size
        sums["exact"] += exact.item() * batch_size
        sums["cell"] += cell.item() * batch_size
        sums["q"] += q_logits.sigmoid().mean().item() * batch_size
    model.train(was_training)
    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    metrics["count"] = float(total)
    return metrics


def validate_and_save(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    val_group_ids,
    ema: ModuleEMA | None,
    best_exact: float,
    stage: str,
    step: int,
) -> float:
    if is_rank_zero():
        raw_model = unwrap(model)
        context = (
            ema.average_parameters(raw_model) if ema is not None else nullcontext()
        )
        with context:
            metrics = evaluate(raw_model, args, device, val_group_ids)
            if metrics["exact"] > best_exact:
                best_exact = metrics["exact"]
                save_ambient(
                    args.best_out,
                    raw_model,
                    {
                        "stage": stage,
                        "step": step,
                        "selection": "max_exact",
                        "val_metrics": metrics,
                        "ema_decay": 0.0 if ema is None else ema.decay,
                        "weights": "raw" if ema is None else "ema",
                    },
                )
            tqdm.write(
                f"stage={stage}_eval step={step} val_exact={metrics['exact']:.4f} "
                f"val_cell={metrics['cell']:.4f} val_q={metrics['q']:.4f}"
            )
    if is_dist():
        value = torch.tensor(best_exact, device=device)
        dist.broadcast(value, src=0)
        best_exact = float(value.item())
    barrier()
    return best_exact


def compute_steps(
    explicit: int | None, epochs: int, configured_batch: int, group_count: int
) -> int:
    return (
        explicit
        if explicit is not None
        else math.ceil(epochs * group_count / configured_batch)
    )


def optimize_step(
    output: dict[str, torch.Tensor],
    batch,
    model: AmbientFlowModel,
    args: argparse.Namespace,
    lambda_clean: float,
    lambda_q: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    categorical_loss = clean_loss(output["logits"], batch.solution_classes, model, args)
    with torch.no_grad():
        predicted = predictions(output["logits"], model)
        q_target = predicted.eq(batch.solution_tokens).all(dim=1).float()
    q_loss = F.binary_cross_entropy_with_logits(output["q_logits"], q_target)
    total_loss = lambda_clean * categorical_loss + lambda_q * q_loss
    return total_loss, categorical_loss, q_loss, predicted


def train_model(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    train_group_ids,
    val_group_ids,
    ema: ModuleEMA | None,
) -> tuple[int, float]:
    raw_model = unwrap(model)
    group_count = max(len(train_group_ids), 1)
    total_steps = compute_steps(args.steps, args.epochs, args.batch_size, group_count)
    batch_size = local_batch_size(args, args.batch_size)
    batches = make_batches(args, batch_size, total_steps, train_group_ids, 100000)
    eval_interval = max(
        1, math.ceil(args.eval_every_epochs * group_count / args.batch_size)
    )
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    best_exact = -1.0
    model.train()
    progress = tqdm(range(1, total_steps + 1), desc="train", disable=not is_rank_zero())
    for step in progress:
        epoch_done, raw = next(batches)
        batch = to_device(raw, device)
        tau = torch.rand(batch.puzzle_tokens.shape[0], device=device)
        noise = torch.randn(
            batch.puzzle_tokens.shape[0],
            raw_model.seq_len,
            raw_model.ambient_dim,
            device=device,
        )
        output = model(
            batch.puzzle_tokens,
            batch.solution_classes,
            mode="teacher",
            tau=tau,
            noise=noise,
        )
        loss, categorical_loss, q_loss, predicted = optimize_step(
            output,
            batch,
            raw_model,
            args,
            args.lambda_clean,
            args.lambda_q,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        set_warmup_lr(optimizer, args.lr, step, args.warmup_steps)
        optimizer.step()
        if ema is not None:
            ema.update(raw_model)
        if is_rank_zero() and (step == 1 or step % 100 == 0):
            exact, cell = exact_and_cell_accuracy(predicted, batch.solution_tokens)
            progress.set_postfix(loss=f"{loss.item():.4f}", exact=f"{exact.item():.4f}")
            tqdm.write(
                f"stage=train step={step} epoch={epoch_done:.2f} "
                f"loss={loss.item():.4f} ce={categorical_loss.item():.4f} "
                f"q={q_loss.item():.4f} exact={exact.item():.4f} "
                f"cell={cell.item():.4f}"
            )
        if val_group_ids.size and (
            step == 1 or step % eval_interval == 0 or step == total_steps
        ):
            best_exact = validate_and_save(
                model, args, device, val_group_ids, ema, best_exact, "train", step
            )
            model.train()
    return total_steps, best_exact


def train_rollout(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    train_group_ids,
    val_group_ids,
    ema: ModuleEMA | None,
    best_exact: float,
) -> tuple[int, float]:
    raw_model = unwrap(model)
    group_count = max(len(train_group_ids), 1)
    total_steps = compute_steps(
        args.rollout_steps,
        args.rollout_epochs,
        args.rollout_batch_size,
        group_count,
    )
    if total_steps <= 0:
        rank_print("stage=rollout skipped")
        return 0, best_exact

    batch_size = local_batch_size(args, args.rollout_batch_size)
    batches = make_batches(args, batch_size, total_steps, train_group_ids, 200000)
    eval_interval = max(
        1,
        math.ceil(args.eval_every_epochs * group_count / args.rollout_batch_size),
    )
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.rollout_lr,
        betas=(args.rollout_beta1, args.rollout_beta2),
        weight_decay=args.rollout_weight_decay,
    )
    lambda_clean = (
        args.lambda_clean
        if args.lambda_rollout_clean is None
        else args.lambda_rollout_clean
    )
    lambda_q = args.lambda_q if args.lambda_rollout_q is None else args.lambda_rollout_q
    model.train()
    progress = tqdm(
        range(1, total_steps + 1), desc="rollout", disable=not is_rank_zero()
    )
    for step in progress:
        epoch_done, raw = next(batches)
        batch = to_device(raw, device)

        tau = torch.rand(batch.puzzle_tokens.shape[0], device=device)
        noise = torch.randn(
            batch.puzzle_tokens.shape[0],
            raw_model.seq_len,
            raw_model.ambient_dim,
            device=device,
        )
        teacher_output = model(
            batch.puzzle_tokens,
            batch.solution_classes,
            mode="teacher",
            tau=tau,
            noise=noise,
        )
        teacher_clean_loss = clean_loss(
            teacher_output["logits"], batch.solution_classes, raw_model, args
        )

        rollout_output = model(
            batch.puzzle_tokens,
            mode="rollout",
            r_steps=args.rollout_R,
            return_intermediates=args.lambda_intermediate > 0.0,
        )
        rollout_loss = clean_loss(
            rollout_output["logits"], batch.solution_classes, raw_model, args
        )
        intermediate_loss = rollout_loss.new_zeros(())
        if args.lambda_intermediate > 0.0:
            intermediate_logits = rollout_output["intermediate_logits"]
            if intermediate_logits:
                losses = torch.stack(
                    [
                        clean_loss(logits, batch.solution_classes, raw_model, args)
                        for logits in intermediate_logits
                    ]
                )
                indices = torch.arange(
                    1, losses.numel() + 1, device=device, dtype=losses.dtype
                )
                weights = (
                    indices / max(args.rollout_R, 1)
                ) ** args.intermediate_weight_gamma
                weights = weights / weights.sum().clamp_min(1e-12)
                intermediate_loss = (weights * losses).sum()

        with torch.no_grad():
            predicted = predictions(rollout_output["logits"], raw_model)
            q_target = predicted.eq(batch.solution_tokens).all(dim=1).float()
        q_loss = F.binary_cross_entropy_with_logits(
            rollout_output["q_logits"], q_target
        )
        loss = (
            lambda_clean * teacher_clean_loss
            + args.lambda_rollout * rollout_loss
            + lambda_q * q_loss
            + args.lambda_intermediate * intermediate_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        set_warmup_lr(optimizer, args.rollout_lr, step, args.rollout_warmup_steps)
        optimizer.step()
        if ema is not None:
            ema.update(raw_model)

        if is_rank_zero() and (step == 1 or step % 100 == 0):
            exact, cell = exact_and_cell_accuracy(predicted, batch.solution_tokens)
            progress.set_postfix(loss=f"{loss.item():.4f}", exact=f"{exact.item():.4f}")
            tqdm.write(
                f"stage=rollout step={step} epoch={epoch_done:.2f} "
                f"loss={loss.item():.4f} clean={teacher_clean_loss.item():.4f} "
                f"roll_ce={rollout_loss.item():.4f} q={q_loss.item():.4f} "
                f"int={intermediate_loss.item():.4f} exact={exact.item():.4f} "
                f"cell={cell.item():.4f}"
            )
        if val_group_ids.size and (
            step == 1 or step % eval_interval == 0 or step == total_steps
        ):
            best_exact = validate_and_save(
                model,
                args,
                device,
                val_group_ids,
                ema,
                best_exact,
                "rollout",
                step,
            )
            model.train()
    return total_steps, best_exact


def main() -> None:
    args = parse_args()
    init_distributed(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed + rank())
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank())
    configure_outputs(args)
    train_group_ids, val_group_ids = split_group_ids(
        args.data_dir,
        args.task,
        args.val_group_fraction,
        seed=args.seed,
        limit_groups=args.train_limit if args.sampling == "group" else None,
    )
    spec = get_task_spec(args.task, args.data_dir)
    config = AmbientFlowConfig(
        task=args.task,
        embedding_type=args.embedding_type,
        ambient_dim=args.ambient_dim,
        cosine_temperature=args.cosine_temperature,
        stablemax=args.stablemax,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        seq_len=spec.seq_len,
        input_vocab_size=spec.input_vocab_size,
        output_vocab_size=spec.output_vocab_size,
        prediction_offset=spec.prediction_offset,
        ignore_index=spec.ignore_index,
        lambda_clean=args.lambda_clean,
        lambda_q=args.lambda_q,
    )
    raw_model = AmbientFlowModel(config).to(device)
    model = maybe_ddp(raw_model, args)
    ema = make_ema(raw_model, args.ema_decay)
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in raw_model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in raw_model.parameters()
            if parameter.requires_grad
        ),
    }
    if is_rank_zero():
        with Path(args.config_out).open("r") as file:
            run_config = json.load(file)
        run_config.update(
            {
                "model_config": raw_model.config_dict(),
                "parameter_counts": parameter_counts,
                "train_groups": int(train_group_ids.size),
                "val_groups": int(val_group_ids.size),
            }
        )
        save_json(args.config_out, run_config)
    rank_print(
        f"embedding_type={args.embedding_type} ambient_dim={raw_model.ambient_dim} "
        f"parameters={parameter_counts['total']}"
    )
    train_steps, best_exact = train_model(
        model, args, device, train_group_ids, val_group_ids, ema
    )
    rollout_steps, _ = train_rollout(
        model,
        args,
        device,
        train_group_ids,
        val_group_ids,
        ema,
        best_exact,
    )
    if is_rank_zero():
        context = (
            ema.average_parameters(raw_model) if ema is not None else nullcontext()
        )
        with context:
            save_ambient(
                args.out,
                raw_model,
                {
                    "train_steps": train_steps,
                    "rollout_steps": rollout_steps,
                    "train_groups": int(train_group_ids.size),
                    "val_groups": int(val_group_ids.size),
                    "ema_decay": 0.0 if ema is None else ema.decay,
                    "weights": "raw" if ema is None else "ema",
                },
            )
    barrier()
    rank_print(f"saved={args.out}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
