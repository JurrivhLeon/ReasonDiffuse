from __future__ import annotations

import torch


def exact_and_cell_accuracy(pred_digits: torch.Tensor, target_digits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    correct = pred_digits.eq(target_digits)
    cell = correct.float().mean()
    exact = correct.all(dim=1).float().mean()
    return exact, cell


def sudoku_violations(pred_digits: torch.Tensor) -> torch.Tensor:
    """Count duplicate/missing digit violations across 27 Sudoku units."""
    boards = pred_digits.view(-1, 9, 9)
    all_units = []
    all_units.extend([boards[:, i, :] for i in range(9)])
    all_units.extend([boards[:, :, i] for i in range(9)])
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            all_units.append(boards[:, r : r + 3, c : c + 3].reshape(-1, 9))

    violations = torch.zeros(boards.shape[0], device=pred_digits.device)
    for unit in all_units:
        counts = torch.stack([(unit == digit).sum(dim=1) for digit in range(1, 10)], dim=1)
        violations = violations + (counts - 1).abs().sum(dim=1)
    return violations

