from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from reasoner_fm import MazeVAE, SudokuVAE, VAEConfig


@dataclass(frozen=True)
class TaskSpec:
    name: str
    seq_len: int
    input_vocab_size: int
    output_vocab_size: int
    prediction_offset: int
    ignore_index: int = -100

    def make_vae_config(
        self,
        d_model: int,
        d_z: int,
        layers: int,
        heads: int,
        beta: float,
        dropout: float = 0.0,
    ) -> VAEConfig:
        return VAEConfig(
            d_model=d_model,
            d_z=d_z,
            layers=layers,
            heads=heads,
            dropout=dropout,
            beta=beta,
            seq_len=self.seq_len,
            input_vocab_size=self.input_vocab_size,
            output_vocab_size=self.output_vocab_size,
            ignore_index=self.ignore_index,
            prediction_offset=self.prediction_offset,
        )

    def make_vae(self, config: VAEConfig):
        if self.name == "sudoku":
            return SudokuVAE(config)
        if self.name == "maze":
            return MazeVAE(config)
        raise ValueError(f"Unsupported task: {self.name}")


@dataclass(frozen=True)
class TaskBatch:
    puzzle_tokens: torch.Tensor
    puzzle_values: torch.Tensor
    solution_classes: torch.Tensor
    solution_tokens: torch.Tensor

    @property
    def solution_digits(self) -> torch.Tensor:
        return self.solution_tokens


def _metadata(root: str | Path, split: str = "train", set_name: str = "all") -> dict:
    with (Path(root) / split / "dataset.json").open("r") as f:
        return json.load(f)


def get_task_spec(task: str, data_dir: str | Path) -> TaskSpec:
    task = task.lower()
    metadata = _metadata(data_dir, "train")
    seq_len = int(metadata["seq_len"])
    vocab_size = int(metadata["vocab_size"])
    if task == "sudoku":
        if seq_len != 81:
            raise ValueError(f"Sudoku expects seq_len=81, got {seq_len}")
        return TaskSpec(
            name="sudoku",
            seq_len=81,
            input_vocab_size=10,
            output_vocab_size=9,
            prediction_offset=1,
        )
    if task == "maze":
        if seq_len != 900:
            raise ValueError(f"Maze-Hard expects seq_len=900, got {seq_len}")
        if vocab_size != 6:
            raise ValueError(f"Maze-Hard expects vocab_size=6, got {vocab_size}")
        return TaskSpec(
            name="maze",
            seq_len=seq_len,
            input_vocab_size=vocab_size,
            output_vocab_size=vocab_size - 1,
            prediction_offset=0,
        )
    raise ValueError(f"Unsupported task: {task}")


class TaskArrays:
    def __init__(self, root: str | Path, task: str, split: str = "train", set_name: str = "all"):
        self.root = Path(root)
        self.task = task.lower()
        self.spec = get_task_spec(self.task, self.root)
        split_dir = self.root / split
        self.metadata = _metadata(self.root, split=split, set_name=set_name)
        self.inputs = np.load(split_dir / f"{set_name}__inputs.npy", mmap_mode="r")
        self.labels = np.load(split_dir / f"{set_name}__labels.npy", mmap_mode="r")
        self.puzzle_indices = np.load(split_dir / f"{set_name}__puzzle_indices.npy")
        self.group_indices = np.load(split_dir / f"{set_name}__group_indices.npy")
        if self.inputs.shape[1] != self.spec.seq_len or self.labels.shape[1] != self.spec.seq_len:
            raise ValueError(
                f"Expected {self.task} seq_len={self.spec.seq_len}, got {self.inputs.shape}, {self.labels.shape}"
            )

    @property
    def total_groups(self) -> int:
        return int(self.group_indices.shape[0] - 1)


class TaskNpyDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        task: str,
        split: str = "train",
        set_name: str = "all",
        limit: int | None = None,
    ):
        self.arrays = TaskArrays(root, task=task, split=split, set_name=set_name)
        self.limit = min(limit, len(self.arrays.inputs)) if limit is not None else len(self.arrays.inputs)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _convert_rows(
            self.arrays.task,
            self.arrays.inputs,
            self.arrays.labels,
            np.asarray([idx], dtype=np.int64),
        )


def _convert_rows(task: str, inputs: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, torch.Tensor]:
    puzzle_raw = torch.from_numpy(np.asarray(inputs[indices], dtype=np.int64).copy())
    label_raw = torch.from_numpy(np.asarray(labels[indices], dtype=np.int64).copy())
    if task == "sudoku":
        puzzle_tokens = (puzzle_raw - 1).clamp(min=0, max=9).long()
        solution_tokens = (label_raw - 1).clamp(min=1, max=9).long()
        solution_classes = solution_tokens - 1
    elif task == "maze":
        puzzle_tokens = puzzle_raw.clamp(min=0, max=5).long()
        solution_tokens = (label_raw - 1).clamp(min=0, max=4).long()
        solution_classes = solution_tokens.clone()
        solution_classes[label_raw == 0] = -100
    else:
        raise ValueError(f"Unsupported task: {task}")
    return {
        "puzzle_tokens": puzzle_tokens,
        "puzzle_values": puzzle_tokens.clone(),
        "solution_classes": solution_classes,
        "solution_tokens": solution_tokens,
    }


def _collate_dataset_items(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat([item[key] for item in items], dim=0)
        for key in ("puzzle_tokens", "puzzle_values", "solution_classes", "solution_tokens")
    }


def split_group_ids(
    data_dir: str | Path,
    task: str,
    val_fraction: float,
    seed: int = 0,
    limit_groups: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = TaskArrays(data_dir, task=task, split="train")
    group_ids = np.arange(arrays.total_groups, dtype=np.int64)
    if limit_groups is not None:
        group_ids = group_ids[: min(limit_groups, group_ids.size)]
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(group_ids)
    val_count = int(round(shuffled.size * val_fraction))
    if val_fraction > 0 and val_count == 0 and shuffled.size > 1:
        val_count = 1
    val_ids = np.sort(shuffled[:val_count])
    train_ids = np.sort(shuffled[val_count:])
    if train_ids.size == 0:
        raise ValueError("Validation split leaves no training groups")
    return train_ids, val_ids


def iter_group_batches(
    data_dir: str | Path,
    task: str,
    batch_size: int,
    group_ids: np.ndarray,
    epochs: int,
    seed: int = 0,
    split: str = "train",
) -> Iterator[tuple[float, dict[str, torch.Tensor]]]:
    arrays = TaskArrays(data_dir, task=task, split=split)
    pending: list[int] = []
    groups_seen = 0
    total_groups = max(int(group_ids.size), 1)
    for epoch in range(epochs):
        rng = np.random.Generator(np.random.Philox(seed=seed + epoch + 1))
        for group_id in rng.permutation(group_ids):
            puzzle_start = int(arrays.group_indices[group_id])
            puzzle_end = int(arrays.group_indices[group_id + 1])
            puzzle_id = int(rng.integers(puzzle_start, puzzle_end))
            row_start = int(arrays.puzzle_indices[puzzle_id])
            row_end = int(arrays.puzzle_indices[puzzle_id + 1])
            row_idx = int(rng.integers(row_start, row_end))
            pending.append(row_idx)
            groups_seen += 1
            if len(pending) == batch_size:
                epoch_done = groups_seen / total_groups
                yield epoch_done, _convert_rows(
                    arrays.task, arrays.inputs, arrays.labels, np.asarray(pending, dtype=np.int64)
                )
                pending = []


def iter_group_eval_batches(
    data_dir: str | Path,
    task: str,
    batch_size: int,
    group_ids: np.ndarray,
    split: str = "train",
    aug_count: int = 0,
) -> Iterator[dict[str, torch.Tensor]]:
    arrays = TaskArrays(data_dir, task=task, split=split)
    indices: list[int] = []
    for group_id in group_ids:
        start_puzzle = int(arrays.group_indices[group_id])
        end_puzzle = int(arrays.group_indices[group_id + 1])
        num_puzzles = max(0, end_puzzle - start_puzzle)
        take = min(num_puzzles, 1 + max(0, aug_count))
        for offset in range(take):
            puzzle_id = start_puzzle + offset
            indices.append(int(arrays.puzzle_indices[puzzle_id]))
            if len(indices) == batch_size:
                yield _convert_rows(arrays.task, arrays.inputs, arrays.labels, np.asarray(indices, dtype=np.int64))
                indices = []
    if indices:
        yield _convert_rows(arrays.task, arrays.inputs, arrays.labels, np.asarray(indices, dtype=np.int64))


def iter_group_all_example_batches(
    data_dir: str | Path,
    task: str,
    batch_size: int,
    group_ids: np.ndarray,
    split: str = "train",
) -> Iterator[dict[str, torch.Tensor]]:
    arrays = TaskArrays(data_dir, task=task, split=split)
    indices: list[int] = []
    for group_id in group_ids:
        for puzzle_id in range(int(arrays.group_indices[group_id]), int(arrays.group_indices[group_id + 1])):
            start = int(arrays.puzzle_indices[puzzle_id])
            end = int(arrays.puzzle_indices[puzzle_id + 1])
            for row_idx in range(start, end):
                indices.append(row_idx)
                if len(indices) == batch_size:
                    yield _convert_rows(arrays.task, arrays.inputs, arrays.labels, np.asarray(indices, dtype=np.int64))
                    indices = []
    if indices:
        yield _convert_rows(arrays.task, arrays.inputs, arrays.labels, np.asarray(indices, dtype=np.int64))


def make_loader(
    data_dir: str | Path,
    task: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    limit: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    ds = TaskNpyDataset(data_dir, task=task, split=split, limit=limit)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_dataset_items,
    )


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> TaskBatch:
    return TaskBatch(
        puzzle_tokens=batch["puzzle_tokens"].to(device, non_blocking=True),
        puzzle_values=batch["puzzle_values"].to(device, non_blocking=True),
        solution_classes=batch["solution_classes"].to(device, non_blocking=True),
        solution_tokens=batch["solution_tokens"].to(device, non_blocking=True),
    )


def exact_and_cell_accuracy(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    correct = pred_tokens.eq(target_tokens)
    return correct.all(dim=1).float().mean(), correct.float().mean()


def answer_ce(logits: torch.Tensor, target_classes: torch.Tensor, output_vocab_size: int, ignore_index: int = -100) -> torch.Tensor:
    import torch.nn.functional as F

    return F.cross_entropy(
        logits.reshape(-1, output_vocab_size),
        target_classes.reshape(-1),
        ignore_index=ignore_index,
    )


def logits_to_tokens(logits: torch.Tensor, spec: TaskSpec) -> torch.Tensor:
    return logits.argmax(dim=-1) + spec.prediction_offset
