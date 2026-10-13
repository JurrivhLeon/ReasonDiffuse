#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from utils.task_data import make_loader, to_device

from .checkpoint import load_ambient
from .runtime import exact_and_cell_accuracy, sudoku_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ambient flow reasoner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="data/sudoku-smoke")
    parser.add_argument("--task", choices=("sudoku", "maze"), default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--cycle-tau-start", type=float, default=0.25)
    parser.add_argument("--cycle-noise", type=float, default=0.25)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_ambient(args.model, device)
    model.eval()
    task = model.config.task
    if args.task is not None and args.task != task:
        raise ValueError(
            f"Checkpoint task is {task!r}, but --task requested {args.task!r}"
        )
    loader = make_loader(
        args.data_dir,
        task,
        args.split,
        args.batch_size,
        shuffle=False,
        limit=args.limit,
    )
    total = 0
    exact_sum = 0.0
    cell_sum = 0.0
    q_sum = 0.0
    violation_sum = 0.0
    progress = tqdm(loader, desc="ambient-eval", unit="batch")
    for raw in progress:
        batch = to_device(raw, device)
        predicted, q_logits = model.sample(
            batch.puzzle_tokens,
            r_steps=args.R,
            k_samples=args.K,
            cycles=args.cycles,
            cycle_tau_start=args.cycle_tau_start,
            cycle_noise=args.cycle_noise,
        )
        exact, cell = exact_and_cell_accuracy(predicted, batch.solution_tokens)
        batch_size = predicted.shape[0]
        total += batch_size
        exact_sum += exact.item() * batch_size
        cell_sum += cell.item() * batch_size
        q_sum += q_logits.sigmoid().mean().item() * batch_size
        if task == "sudoku":
            violation_sum += (
                sudoku_violations(predicted).float().mean().item() * batch_size
            )
        progress.set_postfix(
            exact=f"{exact_sum / total:.4f}",
            cell=f"{cell_sum / total:.4f}",
            q=f"{q_sum / total:.4f}",
        )
    metrics = {
        "n": total,
        "exact": exact_sum / max(total, 1),
        "cell": cell_sum / max(total, 1),
        "q_mean": q_sum / max(total, 1),
        "violations": violation_sum / max(total, 1) if task == "sudoku" else None,
    }
    print(
        f"n={total} exact={metrics['exact']:.4f} cell={metrics['cell']:.4f} "
        f"violations={metrics['violations'] if metrics['violations'] is not None else 'n/a'} "
        f"q_mean={metrics['q_mean']:.4f} R={args.R} K={args.K} "
        f"cycles={args.cycles}"
    )
    output = args.output
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(Path(args.model).resolve().parent / f"eval_{timestamp}.json")
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": vars(args),
        "model_config": model.config_dict(),
        "checkpoint": {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model", "config"}
        },
        "metrics": metrics,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    print(f"saved_eval={output_path}")


if __name__ == "__main__":
    main()
