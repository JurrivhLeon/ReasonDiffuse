from __future__ import annotations

from contextlib import contextmanager

import torch


class ModuleEMA:
    def __init__(self, module: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def store(self, module: torch.nn.Module) -> None:
        self.backup = {
            name: param.detach().clone()
            for name, param in module.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if name in self.shadow:
                param.copy_(
                    self.shadow[name].to(device=param.device, dtype=param.dtype)
                )

    @torch.no_grad()
    def restore(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if name in self.backup:
                param.copy_(
                    self.backup[name].to(device=param.device, dtype=param.dtype)
                )
        self.backup = {}

    @contextmanager
    def average_parameters(self, module: torch.nn.Module):
        self.store(module)
        self.copy_to(module)
        try:
            yield
        finally:
            self.restore(module)


def make_ema(module: torch.nn.Module, decay: float) -> ModuleEMA | None:
    if decay <= 0.0:
        return None
    return ModuleEMA(module, decay)
