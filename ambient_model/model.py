from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import TransformerStack, categorical_probabilities, timestep_embedding


@dataclass
class AmbientFlowConfig:
    task: str = "sudoku"
    embedding_type: str = "fixed"
    ambient_dim: int = 64
    cosine_temperature: float = 0.1
    stablemax: bool = False
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.0
    seq_len: int = 81
    input_vocab_size: int = 10
    output_vocab_size: int = 9
    prediction_offset: int = 1
    ignore_index: int = -100
    lambda_clean: float = 1.0
    lambda_q: float = 0.1

    def __post_init__(self) -> None:
        if self.task not in {"sudoku", "maze"}:
            raise ValueError(f"Unsupported task: {self.task}")
        if self.embedding_type not in {"fixed", "learned"}:
            raise ValueError("embedding_type must be 'fixed' or 'learned'")
        if self.ambient_dim < 1:
            raise ValueError("ambient_dim must be positive")
        if self.cosine_temperature <= 0.0:
            raise ValueError("cosine_temperature must be positive")

    @property
    def effective_ambient_dim(self) -> int:
        return (
            self.output_vocab_size
            if self.embedding_type == "fixed"
            else self.ambient_dim
        )


class AmbientTokenCodec(nn.Module):
    def __init__(self, config: AmbientFlowConfig):
        super().__init__()
        self.config = config
        if config.embedding_type == "fixed":
            self.register_buffer(
                "fixed_codebook", torch.eye(config.output_vocab_size), persistent=True
            )
            self.register_parameter("learned_codebook", None)
        else:
            codebook = F.normalize(
                torch.randn(config.output_vocab_size, config.ambient_dim), dim=-1
            )
            self.learned_codebook = nn.Parameter(codebook)
            self.register_buffer("fixed_codebook", None, persistent=False)

    @property
    def ambient_dim(self) -> int:
        return self.config.effective_ambient_dim

    def codebook(self) -> torch.Tensor:
        if self.config.embedding_type == "fixed":
            return self.fixed_codebook
        return F.normalize(self.learned_codebook, dim=-1)

    def encode(
        self, solution_classes: torch.Tensor, detach_for_path: bool = False
    ) -> torch.Tensor:
        safe_classes = solution_classes.masked_fill(
            solution_classes == self.config.ignore_index, 0
        )
        codebook = self.codebook()
        if detach_for_path and self.config.embedding_type == "learned":
            codebook = codebook.detach()
        target = F.embedding(safe_classes, codebook)
        valid = solution_classes != self.config.ignore_index
        return target * valid.unsqueeze(-1).to(target.dtype)

    def logits(self, clean_state: torch.Tensor) -> torch.Tensor:
        if clean_state.shape[-1] != self.ambient_dim:
            raise ValueError(
                f"Expected ambient dimension {self.ambient_dim}, got {clean_state.shape[-1]}"
            )
        if self.config.embedding_type == "fixed":
            return clean_state
        normalized = F.normalize(clean_state, dim=-1)
        return torch.einsum("bld,vd->blv", normalized, self.codebook()) / float(
            self.config.cosine_temperature
        )

    def endpoint(self, probabilities: torch.Tensor) -> torch.Tensor:
        return torch.einsum("blv,vd->bld", probabilities, self.codebook())


class AmbientFlowModel(nn.Module):
    def __init__(self, config: AmbientFlowConfig):
        super().__init__()
        self.config = config
        self.codec = AmbientTokenCodec(config)
        self.puzzle_emb = nn.Embedding(config.input_vocab_size, config.d_model)
        self.state_proj = nn.Linear(config.effective_ambient_dim, config.d_model)
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
        self.clean_head = nn.Linear(config.d_model, config.effective_ambient_dim)
        self.q_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )

    @property
    def ambient_dim(self) -> int:
        return self.config.effective_ambient_dim

    @property
    def seq_len(self) -> int:
        return self.config.seq_len

    def encode_target(
        self, solution_classes: torch.Tensor, detach_for_path: bool = False
    ) -> torch.Tensor:
        return self.codec.encode(solution_classes, detach_for_path=detach_for_path)

    def reasoner_step(
        self,
        state: torch.Tensor,
        puzzle_tokens: torch.Tensor,
        tau: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state.shape[1:] != (self.seq_len, self.ambient_dim):
            raise ValueError(
                f"Expected state [B, {self.seq_len}, {self.ambient_dim}], got {tuple(state.shape)}"
            )
        time = self.time_mlp(timestep_embedding(tau, self.config.d_model))[:, None, :]
        hidden = self.state_proj(state) + self.puzzle_emb(puzzle_tokens) + time
        hidden = self.backbone(hidden)
        clean_state = self.clean_head(hidden)
        logits = self.codec.logits(clean_state)
        probabilities = categorical_probabilities(
            logits, stablemax=self.config.stablemax
        )
        return {
            "clean_state": clean_state,
            "logits": logits,
            "probabilities": probabilities,
            "clean_endpoint": self.codec.endpoint(probabilities),
            "q_logits": self.q_head(hidden.mean(dim=1)).squeeze(-1),
            "hidden": hidden,
        }

    @staticmethod
    def velocity_from_clean(
        state: torch.Tensor,
        clean_endpoint: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        # tau decreases from one (noise) to zero (clean) during denoising.
        return (clean_endpoint - state) / tau.clamp_min(1e-6)[:, None, None]

    @classmethod
    def euler_step(
        cls,
        state: torch.Tensor,
        clean_endpoint: torch.Tensor,
        tau: torch.Tensor,
        tau_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        velocity = cls.velocity_from_clean(state, clean_endpoint, tau)
        step_size = (tau - tau_next)[:, None, None]
        return state + step_size * velocity, velocity

    def teacher_forced(
        self,
        puzzle_tokens: torch.Tensor,
        solution_classes: torch.Tensor,
        tau: torch.Tensor,
        noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target = self.encode_target(solution_classes, detach_for_path=True)
        tau_view = tau[:, None, None]
        state_tau = (1.0 - tau_view) * target + tau_view * noise
        result = self.reasoner_step(state_tau, puzzle_tokens, tau)
        result["state_tau"] = state_tau
        return result

    def rollout(
        self,
        puzzle_tokens: torch.Tensor,
        r_steps: int = 8,
        noise: torch.Tensor | None = None,
        return_intermediates: bool = False,
        tau_start: float = 1.0,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if r_steps < 1:
            raise ValueError("r_steps must be positive")
        if not 0.0 < tau_start <= 1.0:
            raise ValueError("tau_start must be in (0, 1]")
        state = (
            noise
            if noise is not None
            else torch.randn(
                puzzle_tokens.shape[0],
                self.seq_len,
                self.ambient_dim,
                device=puzzle_tokens.device,
            )
        )
        intermediate_logits: list[torch.Tensor] = []
        output = None
        for index in range(r_steps):
            tau_value = tau_start * (1.0 - index / r_steps)
            tau_next_value = tau_start * (1.0 - (index + 1) / r_steps)
            tau = torch.full(
                (state.shape[0],), tau_value, device=state.device, dtype=state.dtype
            )
            tau_next = torch.full_like(tau, tau_next_value)
            output = self.reasoner_step(state, puzzle_tokens, tau)
            state, velocity = self.euler_step(
                state, output["clean_endpoint"], tau, tau_next
            )
            if return_intermediates and index < r_steps - 1:
                intermediate_logits.append(output["logits"])
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "state": state,
            "logits": output["logits"],
            "probabilities": output["probabilities"],
            "velocity": velocity,
            "q_logits": output["q_logits"],
        }
        if return_intermediates:
            result["intermediate_logits"] = intermediate_logits
        return result

    def forward(
        self,
        puzzle_tokens: torch.Tensor,
        solution_classes: torch.Tensor | None = None,
        *,
        mode: str = "step",
        state: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        r_steps: int = 8,
        return_intermediates: bool = False,
    ) -> dict:
        if mode == "step":
            if state is None or tau is None:
                raise ValueError("step mode requires state and tau")
            return self.reasoner_step(state, puzzle_tokens, tau)
        if mode == "teacher":
            if solution_classes is None or tau is None or noise is None:
                raise ValueError(
                    "teacher mode requires solution_classes, tau, and noise"
                )
            return self.teacher_forced(puzzle_tokens, solution_classes, tau, noise)
        if mode == "rollout":
            return self.rollout(
                puzzle_tokens,
                r_steps=r_steps,
                noise=noise,
                return_intermediates=return_intermediates,
            )
        raise ValueError(f"Unsupported forward mode: {mode}")

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
        if k_samples < 1:
            raise ValueError("k_samples must be positive")
        if cycles < 1:
            raise ValueError("cycles must be positive")
        if not 0.0 <= cycle_noise <= 1.0:
            raise ValueError("cycle_noise must be in [0, 1]")
        batch_size = puzzle_tokens.shape[0]
        puzzle_repeated = puzzle_tokens.repeat_interleave(k_samples, dim=0)
        state = torch.randn(
            batch_size * k_samples,
            self.seq_len,
            self.ambient_dim,
            device=puzzle_tokens.device,
        )
        output = self.rollout(puzzle_repeated, r_steps=r_steps, noise=state)
        state = output["state"]
        for _ in range(cycles - 1):
            state = (1.0 - cycle_noise) * state + cycle_noise * torch.randn_like(state)
            output = self.rollout(
                puzzle_repeated,
                r_steps=r_steps,
                noise=state,
                tau_start=cycle_tau_start,
            )
            state = output["state"]
        predicted = output["logits"].argmax(dim=-1) + self.config.prediction_offset
        q_logits = output["q_logits"]
        predicted = predicted.view(batch_size, k_samples, self.seq_len)
        q_logits = q_logits.view(batch_size, k_samples)
        best = q_logits.argmax(dim=1)
        rows = torch.arange(batch_size, device=q_logits.device)
        return predicted[rows, best], q_logits[rows, best]

    def config_dict(self) -> dict:
        config = asdict(self.config)
        config["effective_ambient_dim"] = self.ambient_dim
        return config
