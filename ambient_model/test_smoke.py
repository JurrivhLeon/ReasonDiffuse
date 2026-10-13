#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import load_ambient, save_ambient
from .layers import token_cross_entropy
from .model import AmbientFlowConfig, AmbientFlowModel


def make_model(task: str, embedding_type: str) -> AmbientFlowModel:
    sudoku = task == "sudoku"
    return AmbientFlowModel(
        AmbientFlowConfig(
            task=task,
            embedding_type=embedding_type,
            ambient_dim=7,
            d_model=16,
            layers=1,
            heads=4,
            seq_len=81 if sudoku else 16,
            input_vocab_size=10 if sudoku else 6,
            output_vocab_size=9 if sudoku else 5,
            prediction_offset=1 if sudoku else 0,
        )
    )


def exercise_configuration(task: str, embedding_type: str) -> None:
    torch.manual_seed(7)
    model = make_model(task, embedding_type)
    batch_size = 2
    puzzle = torch.randint(
        0, model.config.input_vocab_size, (batch_size, model.seq_len)
    )
    classes = torch.randint(
        0, model.config.output_vocab_size, (batch_size, model.seq_len)
    )
    target = model.encode_target(classes)
    assert target.shape == (batch_size, model.seq_len, model.ambient_dim)
    target_logits = model.codec.logits(target)
    assert torch.equal(target_logits.argmax(dim=-1), classes)
    if embedding_type == "fixed":
        assert model.ambient_dim == model.config.output_vocab_size
        assert torch.equal(model.codec.codebook(), torch.eye(model.ambient_dim))
    else:
        norms = model.codec.codebook().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    tau = torch.rand(batch_size)
    noise = torch.randn(batch_size, model.seq_len, model.ambient_dim)
    output = model(
        puzzle,
        classes,
        mode="teacher",
        tau=tau,
        noise=noise,
    )
    assert "velocity" not in output
    assert output["logits"].shape == (
        batch_size,
        model.seq_len,
        model.config.output_vocab_size,
    )
    assert output["clean_endpoint"].shape == noise.shape
    loss = token_cross_entropy(
        output["logits"], classes, model.config.output_vocab_size
    ) + F.binary_cross_entropy_with_logits(output["q_logits"], torch.zeros(batch_size))
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    rollout = model(
        puzzle,
        mode="rollout",
        r_steps=2,
        return_intermediates=True,
    )
    assert rollout["logits"].shape[-1] == model.config.output_vocab_size
    assert len(rollout["intermediate_logits"]) == 1
    model.zero_grad(set_to_none=True)
    rollout_loss = token_cross_entropy(
        rollout["logits"], classes, model.config.output_vocab_size
    ) + token_cross_entropy(
        rollout["intermediate_logits"][0],
        classes,
        model.config.output_vocab_size,
    )
    rollout_loss.backward()
    assert model.state_proj.weight.grad is not None
    assert torch.count_nonzero(model.state_proj.weight.grad) > 0
    predicted, q_logits = model.sample(
        puzzle,
        r_steps=2,
        k_samples=2,
        cycles=2,
        cycle_tau_start=0.25,
    )
    assert predicted.shape == (batch_size, model.seq_len)
    assert q_logits.shape == (batch_size,)


def exercise_euler_step() -> None:
    state = torch.randn(2, 3, 4)
    endpoint = torch.randn_like(state)
    tau = torch.ones(2)
    tau_next = torch.zeros(2)
    updated, velocity = AmbientFlowModel.euler_step(state, endpoint, tau, tau_next)
    assert torch.allclose(updated, endpoint)
    assert torch.equal(velocity, endpoint - state)

    tau = torch.full((2,), 0.5)
    tau_next = torch.full((2,), 0.25)
    updated, velocity = AmbientFlowModel.euler_step(state, endpoint, tau, tau_next)
    assert torch.allclose(velocity, 2.0 * (endpoint - state))
    assert torch.allclose(updated, 0.5 * state + 0.5 * endpoint)


def exercise_learned_gradient_separation() -> None:
    model = make_model("maze", "learned")
    puzzle = torch.randint(0, 6, (2, model.seq_len))
    classes = torch.randint(0, 5, (2, model.seq_len))
    tau = torch.rand(2)
    noise = torch.randn(2, model.seq_len, model.ambient_dim)
    output = model(
        puzzle,
        classes,
        mode="teacher",
        tau=tau,
        noise=noise,
    )
    assert not output["state_tau"].requires_grad
    model.zero_grad(set_to_none=True)
    loss = token_cross_entropy(output["logits"], classes, 5)
    loss.backward()
    gradient = model.codec.learned_codebook.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def exercise_q_selection() -> None:
    model = make_model("maze", "fixed")
    puzzle = torch.zeros(2, model.seq_len, dtype=torch.long)
    original_rollout = model.rollout

    def fake_rollout(self, puzzle_tokens, r_steps=8, noise=None, **kwargs):
        count = puzzle_tokens.shape[0]
        logits = torch.zeros(count, self.seq_len, self.config.output_vocab_size)
        q_logits = torch.empty(count)
        for row in range(count):
            sample_index = row % 2
            logits[row, :, sample_index] = 1.0
            q_logits[row] = float(sample_index)
        return {"state": noise, "logits": logits, "q_logits": q_logits}

    model.rollout = types.MethodType(fake_rollout, model)
    predicted, selected_q = model.sample(puzzle, r_steps=1, k_samples=2)
    model.rollout = original_rollout
    assert torch.equal(predicted, torch.ones_like(predicted))
    assert torch.equal(selected_q, torch.ones_like(selected_q))


def exercise_checkpoint() -> None:
    model = make_model("sudoku", "learned")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ambient_flow.pt"
        save_ambient(path, model, {"smoke": True})
        loaded, checkpoint = load_ambient(path, torch.device("cpu"))
        assert checkpoint["model_type"] == "ambient_clean_flow"
        assert checkpoint["smoke"] is True
        assert loaded.config.embedding_type == "learned"
        assert loaded.ambient_dim == 7
        for expected, actual in zip(model.parameters(), loaded.parameters()):
            assert torch.equal(expected, actual)


def main() -> None:
    for task in ("sudoku", "maze"):
        for embedding_type in ("fixed", "learned"):
            exercise_configuration(task, embedding_type)
    exercise_euler_step()
    exercise_learned_gradient_separation()
    exercise_q_selection()
    exercise_checkpoint()
    print("ambient clean-flow smoke ok")


if __name__ == "__main__":
    main()
