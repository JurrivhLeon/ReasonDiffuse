from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from reasoner_fm import SudokuVAE, TransformerStack, timestep_embedding


@dataclass
class DDIMScheduleConfig:
    train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass
class DDIMReasonerConfig:
    d_model: int = 256
    d_z: int = 64
    layers: int = 6
    heads: int = 8
    dropout: float = 0.0
    lambda_x0: float = 1.0
    lambda_answer: float = 1.0
    lambda_q: float = 0.1


class LatentDDIMReasoner(nn.Module):
    def __init__(self, config: DDIMReasonerConfig):
        super().__init__()
        self.config = config
        self.puzzle_emb = nn.Embedding(10, config.d_model)
        self.z_proj = nn.Linear(config.d_z, config.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.backbone = TransformerStack(
            config.layers, config.d_model, config.heads, config.dropout
        )
        self.x0 = nn.Linear(config.d_model, config.d_z)
        self.q_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self, z_t: torch.Tensor, puzzle_tokens: torch.Tensor, t_norm: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        t = self.time_mlp(timestep_embedding(t_norm, self.config.d_model))[:, None, :]
        h = self.z_proj(z_t) + self.puzzle_emb(puzzle_tokens) + t
        h = self.backbone(h)
        return {
            "x0_pred": self.x0(h),
            "q_logits": self.q_head(h.mean(dim=1)).squeeze(-1),
            "hidden": h,
        }


class UnifiedDDIMLatentReasoner(nn.Module):
    def __init__(
        self,
        vae: SudokuVAE,
        reasoner: LatentDDIMReasoner,
        schedule: DDIMScheduleConfig | None = None,
        task: str = "sudoku",
    ):
        super().__init__()
        if vae.config.d_z != reasoner.config.d_z:
            raise ValueError(
                f"VAE d_z {vae.config.d_z} must match reasoner d_z {reasoner.config.d_z}"
            )
        self.task = task
        self.vae = vae
        self.reasoner = reasoner
        self.schedule_config = schedule or DDIMScheduleConfig()
        self.register_buffer(
            "latent_mean", torch.zeros(vae.config.d_z), persistent=True
        )
        self.register_buffer("latent_std", torch.ones(vae.config.d_z), persistent=True)
        self._register_schedule(self.schedule_config)

    @property
    def d_z(self) -> int:
        return self.vae.config.d_z

    @property
    def train_timesteps(self) -> int:
        return int(self.schedule_config.train_timesteps)

    def _register_schedule(self, config: DDIMScheduleConfig) -> None:
        if config.train_timesteps < 1:
            raise ValueError("train_timesteps must be positive")
        betas = torch.linspace(
            config.beta_start, config.beta_end, config.train_timesteps
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cat([torch.ones(1), torch.cumprod(alphas, dim=0)], dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)

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

    def t_norm(self, t_idx: torch.Tensor) -> torch.Tensor:
        return t_idx.to(dtype=self.alpha_bars.dtype) / float(self.train_timesteps)

    def gather_alpha_bar(self, t_idx: torch.Tensor) -> torch.Tensor:
        return self.alpha_bars[t_idx.long()].to(dtype=self.alpha_bars.dtype)

    def q_sample(
        self,
        z_star: torch.Tensor,
        t_idx: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(z_star)
        alpha_bar = self.gather_alpha_bar(t_idx).to(
            device=z_star.device, dtype=z_star.dtype
        )[:, None, None]
        z_t = alpha_bar.sqrt() * z_star + (1.0 - alpha_bar).sqrt() * eps
        return z_t, eps

    def reasoner_step(
        self, z_norm: torch.Tensor, puzzle_tokens: torch.Tensor, t_idx: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.reasoner(
            z_norm, puzzle_tokens, self.t_norm(t_idx).to(z_norm.device)
        )

    def ddim_step(
        self,
        z_t: torch.Tensor,
        puzzle_tokens: torch.Tensor,
        t_idx: torch.Tensor,
        prev_t_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.reasoner_step(z_t, puzzle_tokens, t_idx)
        x0_pred = out["x0_pred"]
        alpha_bar = self.gather_alpha_bar(t_idx).to(device=z_t.device, dtype=z_t.dtype)[
            :, None, None
        ]
        prev_alpha_bar = self.gather_alpha_bar(prev_t_idx).to(
            device=z_t.device, dtype=z_t.dtype
        )[:, None, None]
        eps_pred = (z_t - alpha_bar.sqrt() * x0_pred) / (
            1.0 - alpha_bar
        ).sqrt().clamp_min(1e-6)
        z_prev = (
            prev_alpha_bar.sqrt() * x0_pred + (1.0 - prev_alpha_bar).sqrt() * eps_pred
        )
        return z_prev, x0_pred, out["q_logits"]

    def ddim_timesteps(self, r_steps: int, device: torch.device) -> torch.Tensor:
        if r_steps < 1:
            raise ValueError("r_steps must be positive")
        steps = torch.linspace(
            self.train_timesteps, 0, r_steps + 1, device=device
        ).round()
        steps = steps.long()
        steps[0] = self.train_timesteps
        steps[-1] = 0
        for i in range(1, steps.numel()):
            if steps[i] >= steps[i - 1]:
                steps[i] = max(int(steps[i - 1].item()) - 1, 0)
        return steps.clamp(min=0, max=self.train_timesteps)

    def rollout(
        self,
        puzzle_tokens: torch.Tensor,
        r_steps: int = 8,
        noise: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        z = (
            noise
            if noise is not None
            else torch.randn(
                puzzle_tokens.shape[0], 81, self.d_z, device=puzzle_tokens.device
            )
        )
        timesteps = self.ddim_timesteps(r_steps, z.device)
        intermediate_logits = []
        x0_pred = z
        q_logits = z.new_zeros(z.shape[0])
        for i in range(r_steps):
            t_idx = torch.full((z.shape[0],), int(timesteps[i].item()), device=z.device)
            prev_t_idx = torch.full(
                (z.shape[0],), int(timesteps[i + 1].item()), device=z.device
            )
            z, x0_pred, q_logits = self.ddim_step(z, puzzle_tokens, t_idx, prev_t_idx)
            if return_intermediates and i < r_steps - 1:
                intermediate_logits.append(
                    self.decode_normalized(puzzle_tokens, x0_pred)
                )
        final_t = torch.zeros(z.shape[0], device=z.device, dtype=torch.long)
        final_out = self.reasoner_step(z, puzzle_tokens, final_t)
        logits = self.decode_normalized(puzzle_tokens, x0_pred)
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "z": z,
            "x0_pred": x0_pred,
            "logits": logits,
            "q_logits": final_out["q_logits"] if r_steps > 0 else q_logits,
        }
        if return_intermediates:
            result["intermediate_logits"] = intermediate_logits
        return result

    @torch.inference_mode()
    def sample(
        self, puzzle_tokens: torch.Tensor, r_steps: int = 8, k_samples: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = puzzle_tokens.shape[0]
        puzzle_rep = puzzle_tokens.repeat_interleave(k_samples, dim=0)
        z = torch.randn(bsz * k_samples, 81, self.d_z, device=puzzle_tokens.device)
        out = self.rollout(puzzle_rep, r_steps=r_steps, noise=z)
        logits = out["logits"]
        pred = logits.argmax(dim=-1) + 1
        q = out["q_logits"]
        pred = pred.view(bsz, k_samples, 81)
        q = q.view(bsz, k_samples)
        best = q.argmax(dim=1)
        final = pred[torch.arange(bsz, device=q.device), best]
        return final, q.max(dim=1).values

    def config_dict(self) -> dict:
        return {
            "task": self.task,
            "vae": asdict(self.vae.config),
            "reasoner": asdict(self.reasoner.config),
            "schedule": asdict(self.schedule_config),
        }


def ddim_reasoner_config_dict(config: DDIMReasonerConfig) -> dict:
    return asdict(config)


def ddim_schedule_config_dict(config: DDIMScheduleConfig) -> dict:
    return asdict(config)
