from __future__ import annotations

from contextlib import contextmanager

import torch


class ModuleEMA:
    def __init__(self, module: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    parameter.detach(), alpha=1.0 - self.decay
                )

    @contextmanager
    def average_parameters(self, module: torch.nn.Module):
        self.backup = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if name in self.shadow
        }
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if name in self.shadow:
                    parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in module.named_parameters():
                    if name in self.backup:
                        parameter.copy_(self.backup[name])
            self.backup = {}


def make_ema(module: torch.nn.Module, decay: float) -> ModuleEMA | None:
    return None if decay <= 0.0 else ModuleEMA(module, decay)


def exact_and_cell_accuracy(
    predicted: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    correct = predicted.eq(target)
    return correct.all(dim=1).float().mean(), correct.float().mean()


def sudoku_violations(grids: torch.Tensor) -> torch.Tensor:
    grids = grids.reshape(-1, 9, 9)
    expected = torch.arange(1, 10, device=grids.device)

    def unit_errors(group: torch.Tensor) -> torch.Tensor:
        counts = group.unsqueeze(-1).eq(expected).sum(dim=-2)
        return (counts - 1).abs().sum(dim=-1)

    row_errors = unit_errors(grids).sum(dim=-1)
    column_errors = unit_errors(grids.transpose(1, 2)).sum(dim=-1)
    boxes = grids.reshape(-1, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    box_errors = unit_errors(boxes).sum(dim=-1)
    return row_errors + column_errors + box_errors
