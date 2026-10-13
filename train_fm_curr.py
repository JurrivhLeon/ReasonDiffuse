#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from reasoner_fm import (
    LatentFlowReasoner,
    ReasonerConfig,
    TokenGridVAE,
    UnifiedLatentReasoner,
)
from utils.ema import ModuleEMA, make_ema
from utils.sudoku.checkpoint import load_vae, save_json, save_unified, save_vae
from utils.task_data import (
    TaskArrays,
    answer_ce,
    exact_and_cell_accuracy,
    get_task_spec,
    iter_group_all_example_batches,
    iter_group_batches,
    iter_group_eval_batches,
    logits_to_tokens,
    make_loader,
    split_group_ids,
    to_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("sudoku", "maze"), default="sudoku")
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
    p.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel. Launch with torchrun; one process per GPU.",
    )
    p.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        help="Pass find_unused_parameters=True to DistributedDataParallel.",
    )
    p.add_argument(
        "--ddp-batch-size-mode",
        choices=("global", "per-rank"),
        default="global",
        help="For DDP, treat configured batch sizes as global totals split across ranks, or as per-rank sizes.",
    )
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
    p.add_argument("--vae-dropout", type=float, default=0.0)
    p.add_argument("--vae-label-smoothing", type=float, default=0.0)
    p.add_argument(
        "--stablemax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use StableMax cross entropy for VAE and reasoner answer losses",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Reasoner EMA decay for FM/rollout stages; 0 disables EMA, typical values are 0.999 or 0.9999",
    )
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
    p.add_argument("--reasoner-dropout", type=float, default=0.0)
    p.add_argument("--lambda-fm", type=float, default=1.0)
    p.add_argument("--lambda-answer", type=float, default=1.0)
    p.add_argument("--answer-label-smoothing", type=float, default=0.0)
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
    p.add_argument("--lambda-rollout", type=float, default=0.5)
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
    p.add_argument(
        "--rollout-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start rollout training from a scheduled teacher-forced state and use grid FM loss.",
    )
    p.add_argument(
        "--rollout-curriculum-tau-start",
        type=float,
        default=0.9,
        help="Curriculum start in data-time convention: 1 is solution/easy, 0 is pure noise/hard.",
    )
    p.add_argument(
        "--rollout-curriculum-tau-end",
        type=float,
        default=0.0,
        help="Curriculum value reached halfway through rollout training and held afterward; data-time convention: 1 is solution/easy, 0 is pure noise/hard.",
    )
    args = p.parse_args()
    if not 0.0 <= args.ema_decay < 1.0:
        p.error("--ema-decay must be in [0, 1); use 0 to disable EMA")
    for name in ("rollout_curriculum_tau_start", "rollout_curriculum_tau_end"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            p.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_rank_zero() -> bool:
    return get_rank() == 0


def rank_print(*args, **kwargs) -> None:
    if is_rank_zero():
        print(*args, **kwargs)


def ddp_barrier() -> None:
    if is_dist():
        dist.barrier()


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    if not args.ddp:
        return 0, 1, 0
    if not torch.cuda.is_available():
        raise ValueError("--ddp requires CUDA")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def configure_output_paths(args: argparse.Namespace) -> None:
    if is_rank_zero():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.run_dir is not None:
            run_dir = Path(args.run_dir)
        else:
            output_dir = (
                Path(args.output_dir)
                if args.output_dir is not None
                else Path("checkpoints") / args.task
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
        payload = {
            "run_dir": args.run_dir,
            "out": args.out,
            "best_out": args.best_out,
            "vae_out": args.vae_out,
            "config_out": args.config_out,
        }
    else:
        payload = None
    if is_dist():
        obj = [payload]
        dist.broadcast_object_list(obj, src=0)
        payload = obj[0]
        args.run_dir = payload["run_dir"]
        args.out = payload["out"]
        args.best_out = payload["best_out"]
        args.vae_out = payload["vae_out"]
        args.config_out = payload["config_out"]
    rank_print(f"run_dir={args.run_dir}")
    rank_print(f"config_out={args.config_out}")
    rank_print(f"out={args.out}")
    rank_print(f"best_out={args.best_out}")
    rank_print(f"vae_out={args.vae_out}")


def scalar_mean(value: torch.Tensor) -> torch.Tensor:
    return value.mean() if value.ndim > 0 else value


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


def resolve_device(args: argparse.Namespace) -> torch.device:
    return torch.device(args.device)


def maybe_ddp(module: torch.nn.Module, args: argparse.Namespace, name: str):
    if not getattr(args, "ddp", False):
        return module
    rank_print(f"ddp wrapping {name} world_size={get_world_size()}")
    return DistributedDataParallel(
        module,
        device_ids=[torch.cuda.current_device()],
        output_device=torch.cuda.current_device(),
        find_unused_parameters=args.ddp_find_unused_parameters,
        broadcast_buffers=False,
    )


def unwrap_parallel(module: torch.nn.Module):
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def local_batch_size(args: argparse.Namespace, batch_size: int) -> int:
    if (
        not getattr(args, "ddp", False)
        or getattr(args, "ddp_batch_size_mode", "global") == "per-rank"
    ):
        return batch_size
    return max(1, math.ceil(batch_size / get_world_size()))


@contextmanager
def unwrap_unified_for_save(model):
    old_vae = model.vae
    old_reasoner = model.reasoner
    model.vae = unwrap_parallel(old_vae)
    model.reasoner = unwrap_parallel(old_reasoner)
    try:
        yield model
    finally:
        model.vae = old_vae
        model.reasoner = old_reasoner


def _ema_target_module(model: torch.nn.Module, ema_module: torch.nn.Module | None):
    if ema_module is not None:
        return unwrap_parallel(ema_module)
    return (
        unwrap_parallel(model.reasoner)
        if hasattr(model, "reasoner")
        else unwrap_parallel(model)
    )


def evaluate_with_optional_ema(
    evaluate_fn,
    model,
    args,
    device,
    val_group_ids,
    ema: ModuleEMA | None,
    ema_module: torch.nn.Module | None = None,
) -> dict[str, float]:
    if ema is None:
        with unwrap_unified_for_save(model) as eval_model:
            return evaluate_fn(eval_model, args, device, val_group_ids)
    with ema.average_parameters(_ema_target_module(model, ema_module)):
        with unwrap_unified_for_save(model) as eval_model:
            return evaluate_fn(eval_model, args, device, val_group_ids)


def save_with_optional_ema(
    save_fn,
    path,
    model,
    extra: dict,
    ema: ModuleEMA | None,
    ema_module: torch.nn.Module | None = None,
    weights_key: str = "reasoner_weights",
) -> None:
    if not is_rank_zero():
        return
    extra = dict(extra)
    if ema is None:
        extra.update({"ema_decay": 0.0, weights_key: "raw"})
        with unwrap_unified_for_save(model) as save_model:
            save_fn(path, save_model, extra)
        return
    extra.update({"ema_decay": ema.decay, weights_key: "ema"})
    with ema.average_parameters(_ema_target_module(model, ema_module)):
        with unwrap_unified_for_save(model) as save_model:
            save_fn(path, save_model, extra)


def set_warmup_lr(
    opt: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int
) -> None:
    scale = min(1.0, step / max(1, warmup_steps)) if warmup_steps > 0 else 1.0
    for group in opt.param_groups:
        group["lr"] = base_lr * scale


def augmented_train_examples(
    data_dir: str, task: str, train_group_count: int, train_limit: int | None
) -> int:
    if train_limit is not None:
        arrays = TaskArrays(data_dir, task=task, split="train")
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
    data_dir: str, task: str, batch_size: int, total_steps: int, train_limit: int | None
):
    loader = make_loader(
        data_dir, task, "train", batch_size, shuffle=True, limit=train_limit
    )
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
    model: TokenGridVAE, args: argparse.Namespace, device: torch.device, val_group_ids
) -> dict[str, float]:
    spec = get_task_spec(args.task, args.data_dir)
    was_training = model.training
    model.eval()
    total = 0
    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "exact": 0.0, "cell": 0.0}
    for raw in iter_group_eval_batches(
        args.data_dir,
        args.task,
        args.eval_batch_size,
        val_group_ids,
        aug_count=args.val_aug_count,
    ):
        batch = to_device(raw, device)
        out = model(batch.puzzle_tokens, batch.solution_classes)
        pred = logits_to_tokens(out["logits"], spec)
        exact, cell = exact_and_cell_accuracy(pred, batch.solution_tokens)
        bsz = batch.puzzle_tokens.shape[0]
        total += bsz
        sums["loss"] += scalar_mean(out["loss"]).item() * bsz
        sums["recon"] += scalar_mean(out["recon"]).item() * bsz
        sums["kl"] += scalar_mean(out["kl"]).item() * bsz
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
        args.data_dir,
        args.task,
        args.eval_batch_size,
        val_group_ids,
        aug_count=args.val_aug_count,
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
    # Loss is kept for existing tie-break code; lower negative exact is better.
    metrics["loss"] = -metrics["exact"]
    return metrics


def train_vae(
    args: argparse.Namespace, device: torch.device, train_group_ids, val_group_ids
) -> TokenGridVAE:
    spec = get_task_spec(args.task, args.data_dir)
    model = spec.make_vae(
        spec.make_vae_config(
            d_model=args.vae_d_model,
            d_z=args.vae_d_z,
            layers=args.vae_layers,
            heads=args.vae_heads,
            beta=args.vae_kl_beta,
            dropout=args.vae_dropout,
            label_smoothing=args.vae_label_smoothing,
            stablemax=args.stablemax,
        )
    ).to(device)
    model = maybe_ddp(model, args, "vae")
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
    vae_batch_size = local_batch_size(args, args.vae_batch_size)
    min_aug_examples = augmented_train_examples(
        args.data_dir,
        args.task,
        train_group_count,
        args.train_limit if args.sampling == "group" else None,
    )
    min_early_stop_steps = math.ceil(
        args.vae_min_aug_epochs * min_aug_examples / args.vae_batch_size
    )
    epochs_needed = (
        math.ceil(args.vae_max_steps * vae_batch_size / train_group_count) + 1
    )
    eval_interval_steps = max(
        1, math.ceil(args.eval_every_epochs * train_group_count / args.vae_batch_size)
    )
    group_iter = iter_group_batches(
        args.data_dir,
        args.task,
        vae_batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed + get_rank(),
    )
    flat_iter = flat_batches(
        args.data_dir,
        args.task,
        vae_batch_size,
        args.vae_max_steps,
        args.train_limit,
    )
    rank_print(
        f"stage=vae max_steps={args.vae_max_steps} sampling={args.sampling} batch_size={args.vae_batch_size} local_batch_size={vae_batch_size} eval_every_epochs={args.eval_every_epochs} min_aug_examples={min_aug_examples} min_early_stop_steps={min_early_stop_steps}"
    )
    progress = tqdm(
        range(1, args.vae_max_steps + 1),
        desc="vae",
        total=args.vae_max_steps,
        disable=not is_rank_zero(),
    )
    for step in progress:
        if args.sampling == "group":
            epoch_done, raw = next(group_iter)
        else:
            _, raw = next(flat_iter)
            epoch_done = step * args.vae_batch_size / train_group_count
        batch = to_device(raw, device)
        out = model(batch.puzzle_tokens, batch.solution_classes)
        loss = scalar_mean(out["loss"])
        recon = scalar_mean(out["recon"])
        kl = scalar_mean(out["kl"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_warmup_lr(opt, args.vae_lr, step, args.vae_warmup_steps)
        opt.step()
        if is_rank_zero() and (step == 1 or step % 100 == 0):
            pred = logits_to_tokens(out["logits"], spec)
            exact, cell = exact_and_cell_accuracy(pred, batch.solution_tokens)
            progress.set_postfix(
                epoch=f"{epoch_done:.1f}",
                loss=f"{loss.item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"stage=vae step={step} epoch={epoch_done:.2f} loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f} exact={exact.item():.4f} cell={cell.item():.4f}"
            )
        stop_training = False
        if val_group_ids.size and (
            step == 1 or step % eval_interval_steps == 0 or step == args.vae_max_steps
        ):
            if is_rank_zero():
                eval_model = unwrap_parallel(model)
                metrics = evaluate_vae(eval_model, args, device, val_group_ids)
                tqdm.write(
                    f"stage=vae_eval step={step} epoch={epoch_done:.2f} val_loss={metrics['loss']:.4f} val_recon={metrics['recon']:.4f} val_exact={metrics['exact']:.4f} val_cell={metrics['cell']:.4f}"
                )
                is_better = (metrics["exact"] > best_val_exact) or (
                    metrics["exact"] == best_val_exact
                    and metrics["recon"] < best_val_recon
                )
                if is_better:
                    best_val_exact = metrics["exact"]
                    best_val_recon = metrics["recon"]
                    stale_evals = 0
                    if best_vae_out is not None:
                        save_vae(
                            best_vae_out,
                            eval_model,
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
                    stop_training = True
                if (
                    early_stop_allowed
                    and args.vae_patience_evals > 0
                    and stale_evals >= args.vae_patience_evals
                ):
                    tqdm.write(
                        f"early stop VAE: no validation improvement for {stale_evals} evals"
                    )
                    stop_training = True
            if is_dist():
                flag = torch.tensor(int(stop_training), device=device)
                dist.broadcast(flag, src=0)
                stop_training = bool(flag.item())
            if stop_training:
                break
    return unwrap_parallel(model)


@torch.inference_mode()
def compute_latent_stats(
    args: argparse.Namespace, device: torch.device, vae: TokenGridVAE, train_group_ids
) -> tuple[torch.Tensor, torch.Tensor, int]:
    vae.eval()
    count = 0
    total = None
    total_sq = None
    rank_print("stage=stats")
    for raw in tqdm(
        iter_group_all_example_batches(
            args.data_dir, args.task, args.stats_batch_size, train_group_ids
        ),
        desc="stats",
        disable=not is_rank_zero(),
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
    rank_print(
        f"stage=stats count={count} mean_abs={mean.abs().mean().item():.4f} std_mean={std.mean().item():.4f}"
    )
    return mean, std, count


def train_reasoner(
    args: argparse.Namespace,
    device: torch.device,
    model: UnifiedLatentReasoner,
    train_group_ids,
    val_group_ids,
    ema: ModuleEMA | None,
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
    reasoner_batch_size = local_batch_size(args, args.reasoner_batch_size)
    total_steps = (
        args.reasoner_steps
        if args.reasoner_steps is not None
        else math.ceil(
            args.reasoner_epochs * train_group_count / args.reasoner_batch_size
        )
    )
    epochs_needed = math.ceil(total_steps * reasoner_batch_size / train_group_count) + 1
    group_iter = iter_group_batches(
        args.data_dir,
        args.task,
        reasoner_batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed + 100000 + get_rank(),
    )
    flat_iter = flat_batches(
        args.data_dir,
        args.task,
        reasoner_batch_size,
        total_steps,
        args.train_limit,
    )
    rank_print(
        f"stage=reasoner total_steps={total_steps} epochs={args.reasoner_epochs} sampling={args.sampling} batch_size={args.reasoner_batch_size} local_batch_size={reasoner_batch_size} train_groups={train_group_count}"
    )
    cfg = unwrap_parallel(model.reasoner).config
    progress = tqdm(
        range(1, total_steps + 1),
        desc="reasoner",
        total=total_steps,
        disable=not is_rank_zero(),
    )
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
        answer_loss = answer_ce(
            logits,
            batch.solution_classes,
            unwrap_parallel(model.vae).config.output_vocab_size,
            unwrap_parallel(model.vae).config.ignore_index,
            args.answer_label_smoothing,
            args.stablemax,
        )
        with torch.no_grad():
            pred_tokens = logits_to_tokens(
                logits, get_task_spec(args.task, args.data_dir)
            )
            q_target = pred_tokens.eq(batch.solution_tokens).all(dim=1).float()
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
        if ema is not None:
            ema.update(unwrap_parallel(model.reasoner))
        if is_rank_zero() and (step == 1 or step % 100 == 0):
            exact, cell = exact_and_cell_accuracy(pred_tokens, batch.solution_tokens)
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


def rollout_curriculum_data_tau(
    args: argparse.Namespace, step: int, total_steps: int
) -> float:
    if total_steps <= 1:
        return float(args.rollout_curriculum_tau_end)
    start = float(args.rollout_curriculum_tau_start)
    end = float(args.rollout_curriculum_tau_end)
    progress = (step - 1) / max(total_steps - 1, 1)
    if progress >= 0.5:
        return end
    half_progress = progress / 0.5
    return start + half_progress * (end - start)


def fm_grid_loss(
    model: UnifiedLatentReasoner,
    batch,
    z_star: torch.Tensor,
    eps: torch.Tensor,
    r_steps: int,
) -> torch.Tensor:
    target_v = eps - z_star
    losses = []
    for tau_value in torch.linspace(1.0, 0.0, r_steps + 1, device=z_star.device):
        tau = torch.full((z_star.shape[0],), tau_value.item(), device=z_star.device)
        z_tau = (1.0 - tau[:, None, None]) * z_star + tau[:, None, None] * eps
        out = model.reasoner_step(z_tau, batch.puzzle_tokens, tau)
        losses.append(F.mse_loss(out["velocity"], target_v))
    return torch.stack(losses).mean()


def curriculum_rollout_start(
    args: argparse.Namespace,
    step: int,
    total_steps: int,
    r_steps: int,
    z_star: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, float, int, float]:
    data_tau = rollout_curriculum_data_tau(args, step, total_steps)
    code_tau = 1.0 - data_tau
    active_steps = max(1, min(r_steps, int(math.ceil(code_tau * r_steps - 1e-12))))
    code_tau_start = active_steps / max(r_steps, 1)
    data_tau_start = 1.0 - code_tau_start
    z_start = (1.0 - code_tau_start) * z_star + code_tau_start * eps
    return z_start, code_tau_start, active_steps, data_tau_start


def intermediate_weights_for_rollout(
    args: argparse.Namespace,
    num_losses: int,
    r_steps: int,
    active_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_losses <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if args.rollout_curriculum:
        start_abs = r_steps - active_steps
        positions = torch.arange(
            start_abs + 1,
            start_abs + num_losses + 1,
            device=device,
            dtype=dtype,
        )
    else:
        positions = torch.arange(1, num_losses + 1, device=device, dtype=dtype)
    weights = (positions / max(r_steps, 1)) ** args.intermediate_weight_gamma
    return weights / weights.sum().clamp_min(1e-12)


def train_rollout(
    args: argparse.Namespace,
    device: torch.device,
    model: UnifiedLatentReasoner,
    train_group_ids,
    val_group_ids,
    ema: ModuleEMA | None,
) -> int:
    total_steps = args.rollout_steps
    train_group_count = max(len(train_group_ids), 1)
    if total_steps is None:
        total_steps = math.ceil(
            args.rollout_epochs * train_group_count / args.rollout_batch_size
        )
    if total_steps <= 0:
        rank_print("stage=rollout skipped")
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
    rollout_batch_size = local_batch_size(args, args.rollout_batch_size)
    epochs_needed = math.ceil(total_steps * rollout_batch_size / train_group_count) + 1
    eval_interval_steps = max(
        1,
        math.ceil(args.eval_every_epochs * train_group_count / args.rollout_batch_size),
    )
    group_iter = iter_group_batches(
        args.data_dir,
        args.task,
        rollout_batch_size,
        train_group_ids,
        epochs_needed,
        seed=args.seed + 200000 + get_rank(),
    )
    flat_iter = flat_batches(
        args.data_dir, args.task, rollout_batch_size, total_steps, args.train_limit
    )
    lambda_fm = (
        args.lambda_fm if args.lambda_rollout_fm is None else args.lambda_rollout_fm
    )
    lambda_q = args.lambda_q if args.lambda_rollout_q is None else args.lambda_rollout_q
    best_unified_out = best_path(args.out, args.best_out)
    best_val_exact = -1.0
    best_val_loss = float("inf")
    if val_group_ids.size:
        if is_rank_zero():
            init_metrics = evaluate_with_optional_ema(
                evaluate_reasoner, model, args, device, val_group_ids, ema
            )
            best_val_exact = init_metrics["exact"]
            best_val_loss = init_metrics["loss"]
            tqdm.write(
                f"stage=rollout_eval_init R={args.val_R} K={args.val_K} val_exact={init_metrics['exact']:.4f} val_cell={init_metrics['cell']:.4f} val_q={init_metrics['q']:.4f}"
            )
        ddp_barrier()

    rollout_epoch_target = (
        args.rollout_epochs
        if args.rollout_epochs > 0
        else total_steps * args.rollout_batch_size / train_group_count
    )
    rank_print(
        f"stage=rollout total_steps={total_steps} epochs={rollout_epoch_target:g} R_train={args.rollout_R} sampling={args.sampling} batch_size={args.rollout_batch_size} local_batch_size={rollout_batch_size} train_groups={train_group_count} curriculum={args.rollout_curriculum} curriculum_tau={args.rollout_curriculum_tau_start:g}->{args.rollout_curriculum_tau_end:g}"
    )
    progress = tqdm(
        range(1, total_steps + 1),
        desc="rollout",
        total=total_steps,
        disable=not is_rank_zero(),
    )
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
        if args.rollout_curriculum:
            fm_loss = fm_grid_loss(model, batch, z_star, eps, args.rollout_R)
            z_start, tau_start, active_steps, data_tau_start = curriculum_rollout_start(
                args, step, total_steps, args.rollout_R, z_star, eps
            )
            roll_out = model.rollout(
                batch.puzzle_tokens,
                r_steps=active_steps,
                noise=z_start,
                return_intermediates=args.lambda_intermediate > 0.0,
                tau_start=tau_start,
            )
        else:
            tau = torch.rand(z_star.shape[0], device=device)
            z_tau = (1.0 - tau[:, None, None]) * z_star + tau[:, None, None] * eps
            target_v = eps - z_star
            tf_out = model.reasoner_step(z_tau, batch.puzzle_tokens, tau)
            fm_loss = F.mse_loss(tf_out["velocity"], target_v)
            active_steps = args.rollout_R
            data_tau_start = 0.0
            roll_out = model.rollout(
                batch.puzzle_tokens,
                r_steps=args.rollout_R,
                return_intermediates=args.lambda_intermediate > 0.0,
            )
        logits = roll_out["logits"]
        rollout_loss = answer_ce(
            logits,
            batch.solution_classes,
            unwrap_parallel(model.vae).config.output_vocab_size,
            unwrap_parallel(model.vae).config.ignore_index,
            args.answer_label_smoothing,
            args.stablemax,
        )
        intermediate_loss = logits.new_tensor(0.0)
        if args.lambda_intermediate > 0.0:
            intermediate_logits = roll_out["intermediate_logits"]
            if intermediate_logits:
                losses = torch.stack(
                    [
                        answer_ce(
                            mid,
                            batch.solution_classes,
                            unwrap_parallel(model.vae).config.output_vocab_size,
                            unwrap_parallel(model.vae).config.ignore_index,
                            args.answer_label_smoothing,
                            args.stablemax,
                        )
                        for mid in intermediate_logits
                    ]
                )
                weights = intermediate_weights_for_rollout(
                    args,
                    losses.numel(),
                    args.rollout_R,
                    active_steps,
                    device,
                    losses.dtype,
                )
                intermediate_loss = (weights * losses).sum()
        with torch.no_grad():
            pred_tokens = logits_to_tokens(
                logits, get_task_spec(args.task, args.data_dir)
            )
            q_target = pred_tokens.eq(batch.solution_tokens).all(dim=1).float()
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
        if ema is not None:
            ema.update(unwrap_parallel(model.reasoner))

        if is_rank_zero() and (step == 1 or step % 100 == 0):
            exact, cell = exact_and_cell_accuracy(pred_tokens, batch.solution_tokens)
            progress.set_postfix(
                epoch=f"{epoch_done:.1f}/{rollout_epoch_target:g}",
                loss=f"{loss.item():.4f}",
                exact=f"{exact.item():.4f}",
                cell=f"{cell.item():.4f}",
            )
            tqdm.write(
                f"stage=rollout step={step} epoch={epoch_done:.2f}/{rollout_epoch_target:g} loss={loss.item():.4f} fm={fm_loss.item():.4f} roll_ce={rollout_loss.item():.4f} q={q_loss.item():.4f} int={intermediate_loss.item():.4f} curr_t={data_tau_start:.4f} active_R={active_steps} exact={exact.item():.4f} cell={cell.item():.4f}"
            )
        if val_group_ids.size and (
            step == 1 or step % eval_interval_steps == 0 or step == total_steps
        ):
            if is_rank_zero():
                metrics = evaluate_with_optional_ema(
                    evaluate_reasoner, model, args, device, val_group_ids, ema
                )
                tqdm.write(
                    f"stage=rollout_eval step={step} epoch={epoch_done:.2f} R={args.val_R} K={args.val_K} val_exact={metrics['exact']:.4f} val_cell={metrics['cell']:.4f} val_q={metrics['q']:.4f}"
                )
                is_better = (metrics["exact"] > best_val_exact) or (
                    metrics["exact"] == best_val_exact
                    and metrics["loss"] < best_val_loss
                )
                if is_better:
                    best_val_exact = metrics["exact"]
                    best_val_loss = metrics["loss"]
                    save_with_optional_ema(
                        save_unified,
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
                        ema,
                    )
                    tqdm.write(f"saved best unified {best_unified_out}")
            ddp_barrier()
    return total_steps


def main() -> None:
    args = parse_args()
    init_distributed(args)
    device = resolve_device(args)
    configure_output_paths(args)
    train_group_ids, val_group_ids = split_group_ids(
        args.data_dir,
        args.task,
        args.val_group_fraction,
        seed=args.seed,
        limit_groups=args.train_limit if args.sampling == "group" else None,
    )
    rank_print(
        f"groups train={train_group_ids.size} val={val_group_ids.size} sampling={args.sampling} ddp={args.ddp} world_size={get_world_size()}"
    )
    vae = train_vae(args, device, train_group_ids, val_group_ids)
    ddp_barrier()
    best_vae_out = best_path(args.vae_out) if args.vae_out is not None else None
    if best_vae_out is not None and Path(best_vae_out).exists():
        vae, best_vae_ckpt = load_vae(best_vae_out, device)
        rank_print(f"loaded best VAE for stats/reasoner from {best_vae_out}")
    if args.vae_out is not None and is_rank_zero():
        save_vae(args.vae_out, vae, {"steps": args.vae_max_steps})
        rank_print(f"saved {Path(args.vae_out)}")
    ddp_barrier()
    mean, std, stats_count = compute_latent_stats(args, device, vae, train_group_ids)
    reasoner = LatentFlowReasoner(
        ReasonerConfig(
            d_model=args.reasoner_d_model,
            d_z=args.vae_d_z,
            layers=args.reasoner_layers,
            heads=args.reasoner_heads,
            dropout=args.reasoner_dropout,
            lambda_fm=args.lambda_fm,
            lambda_answer=args.lambda_answer,
            lambda_q=args.lambda_q,
            seq_len=vae.config.seq_len,
            input_vocab_size=vae.config.input_vocab_size,
        )
    ).to(device)
    unified = UnifiedLatentReasoner(vae, reasoner, task=args.task).to(device)
    unified.reasoner = maybe_ddp(unified.reasoner, args, "reasoner")
    unified.set_latent_stats(mean, std)
    ema = make_ema(unwrap_parallel(unified.reasoner), args.ema_decay)
    if ema is not None:
        rank_print(f"reasoner_ema decay={ema.decay}")
    param_counts = {
        "vae": parameter_counts(vae),
        "reasoner": parameter_counts(reasoner),
        "unified": parameter_counts(unified),
    }
    if is_rank_zero():
        update_config_json(args, {"parameter_counts": param_counts})
    rank_print(
        "parameters "
        f"vae={param_counts['vae']['total']} "
        f"reasoner={param_counts['reasoner']['total']} "
        f"unified={param_counts['unified']['total']}"
    )
    reasoner_steps = train_reasoner(
        args, device, unified, train_group_ids, val_group_ids, ema
    )
    rollout_requested = args.rollout_steps is not None or args.rollout_epochs > 0
    if rollout_requested:
        rollout_steps = train_rollout(
            args, device, unified, train_group_ids, val_group_ids, ema
        )
    else:
        rollout_steps = 0
    save_with_optional_ema(
        save_unified,
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
            "ddp": bool(args.ddp),
            "world_size": get_world_size(),
            "ddp_batch_size_mode": args.ddp_batch_size_mode,
            "rollout_curriculum": bool(args.rollout_curriculum),
            "rollout_curriculum_tau_start": args.rollout_curriculum_tau_start,
            "rollout_curriculum_tau_end": args.rollout_curriculum_tau_end,
        },
        ema,
    )
    ddp_barrier()
    rank_print(f"saved {Path(args.out)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
