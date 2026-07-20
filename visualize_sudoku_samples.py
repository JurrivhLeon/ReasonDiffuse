#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path

import torch

from utils.sudoku.checkpoint import load_unified
from utils.sudoku.data import make_loader, to_device
from utils.sudoku.metrics import sudoku_violations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out-html", default=None)
    p.add_argument("--out-txt", default=None)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--cycle-tau-start", type=float, default=0.25)
    p.add_argument("--cycle-noise", type=float, default=0.25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def board_rows(values: torch.Tensor) -> list[list[int]]:
    return values.view(9, 9).detach().cpu().tolist()


def board_text(values: torch.Tensor) -> str:
    rows = board_rows(values)
    lines = []
    for r, row in enumerate(rows):
        if r in (3, 6):
            lines.append("------+-------+------")
        chunks = []
        for c in range(0, 9, 3):
            chunks.append(" ".join("." if x == 0 else str(x) for x in row[c : c + 3]))
        lines.append(" | ".join(chunks))
    return "\n".join(lines)


def render_board_html(
    title: str,
    values: torch.Tensor,
    target: torch.Tensor | None = None,
    givens: torch.Tensor | None = None,
) -> str:
    rows = board_rows(values)
    target_rows = board_rows(target) if target is not None else None
    given_rows = board_rows(givens) if givens is not None else None
    out = [
        f"<div class='board-block'><div class='board-title'>{html.escape(title)}</div><table class='sudoku'>"
    ]
    for r, row in enumerate(rows):
        out.append("<tr>")
        for c, value in enumerate(row):
            classes = []
            if r in (2, 5):
                classes.append("box-bottom")
            if c in (2, 5):
                classes.append("box-right")
            if given_rows is not None and given_rows[r][c] != 0:
                classes.append("given")
            if target_rows is not None and value != target_rows[r][c]:
                classes.append("wrong")
            display = "." if value == 0 else str(value)
            out.append(f"<td class='{' '.join(classes)}'>{display}</td>")
        out.append("</tr>")
    out.append("</table></div>")
    return "\n".join(out)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_unified(args.model, device)
    model.eval()

    loader = make_loader(
        args.data_dir,
        args.split,
        args.batch_size,
        shuffle=False,
        limit=max(args.num_samples, args.batch_size),
    )
    raw = next(iter(loader))
    batch = to_device(raw, device)
    pred, q = model.sample(
        batch.puzzle_tokens,
        r_steps=args.R,
        k_samples=args.K,
        cycles=args.cycles,
        cycle_tau_start=args.cycle_tau_start,
        cycle_noise=args.cycle_noise,
    )
    violations = sudoku_violations(pred)

    n = min(args.num_samples, pred.shape[0])
    out_stem = Path(args.model).with_suffix("")
    out_html = (
        Path(args.out_html)
        if args.out_html is not None
        else out_stem.parent / f"{out_stem.name}_samples.html"
    )
    out_txt = (
        Path(args.out_txt)
        if args.out_txt is not None
        else out_stem.parent / f"{out_stem.name}_samples.txt"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    txt_parts = []
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><style>",
        "body{font-family:system-ui,Arial,sans-serif;margin:24px;color:#111}",
        ".sample{margin:0 0 32px 0;padding-bottom:24px;border-bottom:1px solid #ddd}",
        ".meta{font-size:14px;margin:0 0 12px 0}",
        ".boards{display:flex;gap:24px;flex-wrap:wrap}",
        ".board-title{font-weight:600;margin-bottom:6px}",
        "table.sudoku{border-collapse:collapse;border:2px solid #111}",
        "table.sudoku td{width:28px;height:28px;text-align:center;border:1px solid #aaa;font:16px ui-monospace,monospace}",
        "td.box-right{border-right:2px solid #111!important}",
        "td.box-bottom{border-bottom:2px solid #111!important}",
        "td.given{background:#eef5ff;font-weight:700}",
        "td.wrong{background:#ffd9d9;color:#a00000;font-weight:700}",
        "</style></head><body>",
        f"<h1>Sudoku Samples</h1><p>model={html.escape(args.model)} R={args.R} K={args.K}</p>",
    ]

    for i in range(n):
        puzzle = batch.puzzle_values[i].detach().cpu()
        target = batch.solution_digits[i].detach().cpu()
        prediction = pred[i].detach().cpu()
        correct = bool(prediction.eq(target).all().item())
        cell_acc = float(prediction.eq(target).float().mean().item())
        q_score = float(q[i].sigmoid().item())
        vio = int(violations[i].item())

        txt_parts.append(
            f"sample {i} correct={correct} cell_acc={cell_acc:.4f} violations={vio} q={q_score:.4f}\n"
            f"puzzle:\n{board_text(puzzle)}\n\n"
            f"target:\n{board_text(target)}\n\n"
            f"prediction:\n{board_text(prediction)}\n"
        )
        html_parts.append("<div class='sample'>")
        html_parts.append(
            f"<div class='meta'>sample {i} correct={correct} cell_acc={cell_acc:.4f} violations={vio} q={q_score:.4f}</div>"
        )
        html_parts.append("<div class='boards'>")
        html_parts.append(render_board_html("Puzzle", puzzle))
        html_parts.append(render_board_html("Target", target, givens=puzzle))
        html_parts.append(
            render_board_html("Prediction", prediction, target=target, givens=puzzle)
        )
        html_parts.append("</div></div>")

    html_parts.append("</body></html>")
    out_txt.write_text("\n\n".join(txt_parts))
    out_html.write_text("\n".join(html_parts))
    print(f"wrote {out_txt}")
    print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
