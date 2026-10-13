from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _log_stablemax(
    logits: torch.Tensor, dim: int = -1, epsilon: float = 1e-30
) -> torch.Tensor:
    logits = logits.to(torch.float64).nan_to_num(posinf=1e30, neginf=-1e30)
    weights = torch.empty_like(logits)
    negative = logits < 0
    weights[negative] = 1.0 / (1.0 - logits[negative] + epsilon)
    weights[~negative] = logits[~negative] + 1.0
    log_weights = weights.clamp_min(epsilon).log()
    return log_weights - weights.sum(dim=dim, keepdim=True).clamp_min(epsilon).log()


def categorical_probabilities(
    logits: torch.Tensor, stablemax: bool = False
) -> torch.Tensor:
    if stablemax:
        return _log_stablemax(logits).exp().to(dtype=logits.dtype)
    return logits.softmax(dim=-1)


def token_cross_entropy(
    logits: torch.Tensor,
    target_classes: torch.Tensor,
    output_vocab_size: int,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    stablemax: bool = True,
) -> torch.Tensor:
    if stablemax and label_smoothing != 0.0:
        raise ValueError("stablemax and label_smoothing should not be combined")
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be in [0, 1]")
    if label_smoothing == 0.0 and not stablemax:
        return F.cross_entropy(
            logits.reshape(-1, output_vocab_size),
            target_classes.reshape(-1),
            ignore_index=ignore_index,
        )
    flat_logits = logits.reshape(-1, output_vocab_size)
    flat_targets = target_classes.reshape(-1)
    valid = flat_targets != ignore_index
    if not valid.any():
        return flat_logits.sum() * 0.0
    flat_logits = flat_logits[valid]
    flat_targets = flat_targets[valid]
    log_probs = (
        _log_stablemax(flat_logits) if stablemax else F.log_softmax(flat_logits, dim=-1)
    )
    target_log_probs = log_probs.gather(1, flat_targets[:, None]).squeeze(1)
    if label_smoothing == 0.0:
        return -target_log_probs.mean()
    if output_vocab_size < 2:
        raise ValueError("label_smoothing requires at least two classes")
    other_log_probs = log_probs.sum(dim=-1) - target_log_probs
    return (
        -(1.0 - label_smoothing) * target_log_probs
        - label_smoothing * other_log_probs / (output_vocab_size - 1)
    ).mean()


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(embedding, (0, 1)) if dim % 2 else embedding


class RotarySelfAttention(nn.Module):
    def __init__(
        self, d_model: int, heads: int, dropout: float = 0.0, seq_len: int = 81
    ):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        inverse_frequency = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        positions = torch.arange(seq_len).float()
        frequencies = torch.einsum("i,j->ij", positions, inverse_frequency)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        even_dim = (self.head_dim // 2) * 2
        rotary, remainder = x[..., :even_dim], x[..., even_dim:]
        pairs = rotary.reshape(*rotary.shape[:-1], -1, 2)
        cos = self.cos[: x.shape[2], : pairs.shape[-2]][None, None, :, :]
        sin = self.sin[: x.shape[2], : pairs.shape[-2]][None, None, :, :]
        first, second = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack(
            [first * cos - second * sin, first * sin + second * cos], dim=-1
        ).flatten(-2)
        return torch.cat([rotated, remainder], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = (
            self.qkv(x)
            .view(batch, seq_len, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attended = F.scaled_dot_product_attention(
            self._rope(query),
            self._rope(key),
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(attended.transpose(1, 2).reshape(batch, seq_len, d_model))


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, heads: int, dropout: float = 0.0, seq_len: int = 81
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = RotarySelfAttention(d_model, heads, dropout, seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class TransformerStack(nn.Module):
    def __init__(
        self,
        layers: int,
        d_model: int,
        heads: int,
        dropout: float = 0.0,
        seq_len: int = 81,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, heads, dropout, seq_len) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
