from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    d_model: int = 128
    d_z: int = 64
    layers: int = 4
    heads: int = 4
    dropout: float = 0.0
    beta: float = 1e-3
    seq_len: int = 81
    input_vocab_size: int = 10
    output_vocab_size: int = 9
    ignore_index: int = -100
    prediction_offset: int = 1


@dataclass
class ReasonerConfig:
    d_model: int = 256
    d_z: int = 64
    layers: int = 6
    heads: int = 8
    dropout: float = 0.0
    seq_len: int = 81
    input_vocab_size: int = 10
    lambda_fm: float = 1.0
    lambda_answer: float = 1.0
    lambda_q: float = 0.1


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


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
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        positions = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        d_even = (self.head_dim // 2) * 2
        x_rope, x_pass = x[..., :d_even], x[..., d_even:]
        x_pair = x_rope.reshape(*x_rope.shape[:-1], -1, 2)
        cos = self.cos[: x.shape[2], : x_pair.shape[-2]][None, None, :, :]
        sin = self.sin[: x.shape[2], : x_pair.shape[-2]][None, None, :, :]
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        rotated = torch.stack(
            [x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1
        ).flatten(-2)
        return torch.cat([rotated, x_pass], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = (
            self.qkv(x)
            .view(bsz, seq_len, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._rope(q)
        k = self._rope(k)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(bsz, seq_len, d_model)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, heads: int, dropout: float = 0.0, seq_len: int = 81
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RotarySelfAttention(d_model, heads, dropout, seq_len=seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


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
            [
                TransformerBlock(d_model, heads, dropout, seq_len=seq_len)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TokenGridVAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.puzzle_emb = nn.Embedding(config.input_vocab_size, config.d_model)
        self.solution_emb = nn.Embedding(config.output_vocab_size, config.d_model)
        self.encoder = TransformerStack(
            config.layers,
            config.d_model,
            config.heads,
            config.dropout,
            seq_len=config.seq_len,
        )
        self.to_mu_logvar = nn.Linear(config.d_model, 2 * config.d_z)
        self.z_proj = nn.Linear(config.d_z, config.d_model)
        self.decoder = TransformerStack(
            config.layers,
            config.d_model,
            config.heads,
            config.dropout,
            seq_len=config.seq_len,
        )
        self.out = nn.Linear(config.d_model, config.output_vocab_size)

    def encode(
        self, puzzle_tokens: torch.Tensor, solution_classes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if puzzle_tokens.shape[1] != self.config.seq_len:
            raise ValueError(
                f"Expected puzzle seq_len={self.config.seq_len}, got {puzzle_tokens.shape[1]}"
            )
        solution_emb_classes = solution_classes.masked_fill(
            solution_classes == self.config.ignore_index, 0
        )
        h = self.puzzle_emb(puzzle_tokens) + self.solution_emb(solution_emb_classes)
        h = self.encoder(h)
        mu, logvar = self.to_mu_logvar(h).chunk(2, dim=-1)
        return mu, logvar.clamp(min=-10.0, max=10.0)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def decode(self, puzzle_tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.puzzle_emb(puzzle_tokens) + self.z_proj(z)
        h = self.decoder(h)
        return self.out(h)

    def forward(
        self, puzzle_tokens: torch.Tensor, solution_classes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(puzzle_tokens, solution_classes)
        z = self.sample(mu, logvar)
        logits = self.decode(puzzle_tokens, z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        recon = F.cross_entropy(
            logits.reshape(-1, self.config.output_vocab_size),
            solution_classes.reshape(-1),
            ignore_index=self.config.ignore_index,
        )
        return {
            "logits": logits,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "kl": kl,
            "recon": recon,
            "loss": recon + self.config.beta * kl,
        }


class SudokuVAE(TokenGridVAE):
    def __init__(self, config: VAEConfig):
        super().__init__(config)


class MazeVAE(TokenGridVAE):
    def __init__(self, config: VAEConfig | None = None):
        if config is None:
            config = VAEConfig()
        defaults = VAEConfig()
        values = asdict(config)
        if config.seq_len == defaults.seq_len:
            values["seq_len"] = 900
        if config.input_vocab_size == defaults.input_vocab_size:
            values["input_vocab_size"] = 6
        if config.output_vocab_size == defaults.output_vocab_size:
            values["output_vocab_size"] = 5
        if config.prediction_offset == defaults.prediction_offset:
            values["prediction_offset"] = 0
        super().__init__(VAEConfig(**values))


class LatentFlowReasoner(nn.Module):
    def __init__(self, config: ReasonerConfig):
        super().__init__()
        self.config = config
        self.puzzle_emb = nn.Embedding(config.input_vocab_size, config.d_model)
        self.z_proj = nn.Linear(config.d_z, config.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.backbone = TransformerStack(
            config.layers,
            config.d_model,
            config.heads,
            config.dropout,
            seq_len=config.seq_len,
        )
        self.velocity = nn.Linear(config.d_model, config.d_z)
        self.q_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self, z_tau: torch.Tensor, puzzle_tokens: torch.Tensor, tau: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        t = self.time_mlp(timestep_embedding(tau, self.config.d_model))[:, None, :]
        h = self.z_proj(z_tau) + self.puzzle_emb(puzzle_tokens) + t
        h = self.backbone(h)
        return {
            "velocity": self.velocity(h),
            "q_logits": self.q_head(h.mean(dim=1)).squeeze(-1),
            "hidden": h,
        }


@dataclass
class UnifiedConfig:
    task: str = "sudoku"
    vae: VAEConfig = field(default_factory=VAEConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)


class UnifiedLatentReasoner(nn.Module):
    """Unified VAE + latent flow reasoner checkpoint module.

    The first concrete task is Sudoku. Later benchmarks can add task-specific
    VAE/codecs while keeping the same normalized latent-flow interface.
    """

    def __init__(
        self, vae: TokenGridVAE, reasoner: LatentFlowReasoner, task: str = "sudoku"
    ):
        super().__init__()
        if vae.config.d_z != reasoner.config.d_z:
            raise ValueError(
                f"VAE d_z {vae.config.d_z} must match reasoner d_z {reasoner.config.d_z}"
            )
        if vae.config.seq_len != reasoner.config.seq_len:
            raise ValueError(
                f"VAE seq_len {vae.config.seq_len} must match reasoner seq_len {reasoner.config.seq_len}"
            )
        if vae.config.input_vocab_size != reasoner.config.input_vocab_size:
            raise ValueError(
                f"VAE input_vocab_size {vae.config.input_vocab_size} must match reasoner input_vocab_size {reasoner.config.input_vocab_size}"
            )
        self.task = task
        self.vae = vae
        self.reasoner = reasoner
        self.register_buffer(
            "latent_mean", torch.zeros(vae.config.d_z), persistent=True
        )
        self.register_buffer("latent_std", torch.ones(vae.config.d_z), persistent=True)

    @property
    def d_z(self) -> int:
        return self.vae.config.d_z

    @property
    def seq_len(self) -> int:
        return self.vae.config.seq_len

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = (
            mean.detach()
            .to(device=self.latent_mean.device, dtype=self.latent_mean.dtype)
            .view(-1)
        )
        std = (
            std.detach()
            .to(device=self.latent_std.device, dtype=self.latent_std.dtype)
            .view(-1)
            .clamp_min(1e-6)
        )
        if mean.numel() != self.d_z or std.numel() != self.d_z:
            raise ValueError(f"Expected latent stats with {self.d_z} values")
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

    def stats_view(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latent_mean[None, None, :], self.latent_std[
            None, None, :
        ].clamp_min(1e-6)

    def encode_solution_normalized(
        self, puzzle_tokens: torch.Tensor, solution_classes: torch.Tensor
    ) -> torch.Tensor:
        mu, _ = self.vae.encode(puzzle_tokens, solution_classes)
        mean, std = self.stats_view()
        return (mu - mean) / std

    def decode_normalized(
        self, puzzle_tokens: torch.Tensor, z_norm: torch.Tensor
    ) -> torch.Tensor:
        mean, std = self.stats_view()
        return self.vae.decode(puzzle_tokens, z_norm * std + mean)

    def reasoner_step(
        self, z_norm: torch.Tensor, puzzle_tokens: torch.Tensor, tau: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.reasoner(z_norm, puzzle_tokens, tau)

    def rollout(
        self,
        puzzle_tokens: torch.Tensor,
        r_steps: int = 8,
        noise: torch.Tensor | None = None,
        return_intermediates: bool = False,
        tau_start: float = 1.0,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        z = (
            noise
            if noise is not None
            else torch.randn(
                puzzle_tokens.shape[0],
                self.seq_len,
                self.d_z,
                device=puzzle_tokens.device,
            )
        )
        tau_start = float(tau_start)
        step_size = tau_start / max(r_steps, 1)
        intermediate_logits = []
        for i in range(r_steps):
            tau = torch.full(
                (z.shape[0],),
                tau_start - i * step_size,
                device=z.device,
                dtype=z.dtype,
            )
            out = self.reasoner_step(z, puzzle_tokens, tau)
            z = z - step_size * out["velocity"]
            if return_intermediates and i < r_steps - 1:
                intermediate_logits.append(self.decode_normalized(puzzle_tokens, z))
        logits = self.decode_normalized(puzzle_tokens, z)
        q_logits = self.reasoner_step(
            z, puzzle_tokens, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        )["q_logits"]
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "z": z,
            "logits": logits,
            "q_logits": q_logits,
        }
        if return_intermediates:
            result["intermediate_logits"] = intermediate_logits
        return result

    @torch.inference_mode()
    def sample(
        self,
        puzzle_tokens: torch.Tensor,
        r_steps: int = 8,
        k_samples: int = 1,
        cycles: int = 1,
        cycle_tau_start: float = 0.25,
        cycle_noise: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = puzzle_tokens.shape[0]
        puzzle_rep = puzzle_tokens.repeat_interleave(k_samples, dim=0)
        z = torch.randn(
            bsz * k_samples, self.seq_len, self.d_z, device=puzzle_tokens.device
        )
        out = self.rollout(puzzle_rep, r_steps=r_steps, noise=z, tau_start=1.0)
        z = out["z"]
        for _ in range(max(int(cycles), 1) - 1):
            eps = torch.randn_like(z)
            alpha = float(cycle_noise)
            z = (1.0 - alpha) * z + alpha * eps
            out = self.rollout(
                puzzle_rep, r_steps=r_steps, noise=z, tau_start=cycle_tau_start
            )
            z = out["z"]
        logits = out["logits"]
        pred = logits.argmax(dim=-1) + self.vae.config.prediction_offset
        q = out["q_logits"]
        pred = pred.view(bsz, k_samples, self.seq_len)
        q = q.view(bsz, k_samples)
        best = q.argmax(dim=1)
        final = pred[torch.arange(bsz, device=q.device), best]
        return final, q.max(dim=1).values

    def config_dict(self) -> dict:
        return {
            "task": self.task,
            "vae_type": self.vae.__class__.__name__,
            "vae": asdict(self.vae.config),
            "reasoner": asdict(self.reasoner.config),
        }


def vae_config_dict(config: VAEConfig) -> dict:
    return asdict(config)


def reasoner_config_dict(config: ReasonerConfig) -> dict:
    return asdict(config)
