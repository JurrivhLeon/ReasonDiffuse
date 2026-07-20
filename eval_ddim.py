#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from utils.sudoku.checkpoint import load_ddim_unified
from utils.sudoku.data import make_loader, to_device
from utils.sudoku.metrics import exact_and_cell_accuracy, sudoku_violations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sudoku-smoke")
    p.add_argument("--model", default="checkpoints/sudoku/ddim/unified.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output",
        default=None,
        help="JSON file for evaluation parameters and metrics. Defaults to <model-dir>/eval_ddim_<TIMESTAMP>.json.",
    )
    return p.parse_args()


@torch.inference_mode()
def sample_batch(
    model,
    puzzle_tokens: torch.Tensor,
    r_steps: int,
    k_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.sample(puzzle_tokens, r_steps=r_steps, k_samples=k_samples)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_ddim_unified(args.model, device)
    model.eval()
    loader = make_loader(
        args.data_dir, args.split, args.batch_size, shuffle=False, limit=args.limit
    )

    n = 0
    exact_sum = 0.0
    cell_sum = 0.0
    violation_sum = 0.0
    q_sum = 0.0
    total = len(loader.dataset) if hasattr(loader, "dataset") else args.limit
    progress = tqdm(loader, desc="eval_ddim", total=len(loader), unit="batch")
    for raw in progress:
        batch = to_device(raw, device)
        pred, q = sample_batch(model, batch.puzzle_tokens, args.R, args.K)
        exact, cell = exact_and_cell_accuracy(pred, batch.solution_digits)
        violations = sudoku_violations(pred)
        bsz = pred.shape[0]
        n += bsz
        exact_sum += exact.item() * bsz
        cell_sum += cell.item() * bsz
        violation_sum += violations.float().mean().item() * bsz
        q_sum += q.sigmoid().mean().item() * bsz
        progress.set_postfix(
            n=f"{n}/{total}",
            exact=f"{exact_sum/n:.4f}",
            cell=f"{cell_sum/n:.4f}",
            violations=f"{violation_sum/n:.2f}",
            q=f"{q_sum/n:.4f}",
        )

    metrics = {
        "n": n,
        "exact": exact_sum / max(n, 1),
        "cell": cell_sum / max(n, 1),
        "violations": violation_sum / max(n, 1),
        "q_mean": q_sum / max(n, 1),
    }
    print(
        f"n={n} exact={metrics['exact']:.4f} cell={metrics['cell']:.4f} "
        f"violations={metrics['violations']:.4f} q_mean={metrics['q_mean']:.4f} "
        f"R={args.R} K={args.K}"
    )

    output = args.output
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(Path(args.model).resolve().parent / f"eval_ddim_{timestamp}.json")
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": vars(args),
        "metrics": metrics,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"saved_eval={output_path}")


if __name__ == "__main__":
    main()
