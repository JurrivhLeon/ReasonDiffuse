from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SudokuBatch:
    puzzle_tokens: torch.Tensor
    puzzle_values: torch.Tensor
    solution_classes: torch.Tensor
    solution_digits: torch.Tensor


class SudokuNpyDataset(Dataset):
    """TRM-format Sudoku dataset with explicit token conversion.

    TRM stores PAD=0, Sudoku blank=1, and digits 1..9 as tokens 2..10.
    This dataset returns puzzle tokens for embeddings and solution classes
    in 0..8 for cross entropy.
    """

    def __init__(self, root: str | Path, split: str = "train", set_name: str = "all", limit: int | None = None):
        self.root = Path(root)
        self.split = split
        self.set_name = set_name
        split_dir = self.root / split
        with (split_dir / "dataset.json").open("r") as f:
            self.metadata = json.load(f)
        self.inputs = np.load(split_dir / f"{set_name}__inputs.npy", mmap_mode="r")
        self.labels = np.load(split_dir / f"{set_name}__labels.npy", mmap_mode="r")
        if self.inputs.shape[1] != 81 or self.labels.shape[1] != 81:
            raise ValueError(f"Expected Sudoku seq_len=81, got {self.inputs.shape}, {self.labels.shape}")
        self.limit = min(limit, len(self.inputs)) if limit is not None else len(self.inputs)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        puzzle_trm = torch.from_numpy(np.asarray(self.inputs[idx], dtype=np.int64).copy())
        label_trm = torch.from_numpy(np.asarray(self.labels[idx], dtype=np.int64).copy())

        # Embedding tokens: blank=0, givens 1..9.
        puzzle_tokens = (puzzle_trm - 1).clamp(min=0, max=9).long()
        puzzle_values = puzzle_tokens.clone()

        # Labels are complete Sudoku digits in TRM tokens 2..10.
        solution_digits = (label_trm - 1).clamp(min=1, max=9).long()
        solution_classes = solution_digits - 1

        return {
            "puzzle_tokens": puzzle_tokens,
            "puzzle_values": puzzle_values,
            "solution_classes": solution_classes,
            "solution_digits": solution_digits,
        }


class SudokuArrays:
    def __init__(self, root: str | Path, split: str = "train", set_name: str = "all"):
        self.root = Path(root)
        split_dir = self.root / split
        with (split_dir / "dataset.json").open("r") as f:
            self.metadata = json.load(f)
        self.inputs = np.load(split_dir / f"{set_name}__inputs.npy", mmap_mode="r")
        self.labels = np.load(split_dir / f"{set_name}__labels.npy", mmap_mode="r")
        self.puzzle_indices = np.load(split_dir / f"{set_name}__puzzle_indices.npy")
        self.group_indices = np.load(split_dir / f"{set_name}__group_indices.npy")
        if self.inputs.shape[1] != 81 or self.labels.shape[1] != 81:
            raise ValueError(f"Expected Sudoku seq_len=81, got {self.inputs.shape}, {self.labels.shape}")

    @property
    def total_groups(self) -> int:
        return int(self.group_indices.shape[0] - 1)


def split_group_ids(
    data_dir: str | Path,
    val_fraction: float,
    seed: int = 0,
    limit_groups: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = SudokuArrays(data_dir, split="train")
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


def _convert_rows(inputs: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, torch.Tensor]:
    puzzle_trm = torch.from_numpy(np.asarray(inputs[indices], dtype=np.int64).copy())
    label_trm = torch.from_numpy(np.asarray(labels[indices], dtype=np.int64).copy())
    puzzle_tokens = (puzzle_trm - 1).clamp(min=0, max=9).long()
    solution_digits = (label_trm - 1).clamp(min=1, max=9).long()
    return {
        "puzzle_tokens": puzzle_tokens,
        "puzzle_values": puzzle_tokens.clone(),
        "solution_classes": solution_digits - 1,
        "solution_digits": solution_digits,
    }


def iter_group_batches(
    data_dir: str | Path,
    batch_size: int,
    group_ids: np.ndarray,
    epochs: int,
    seed: int = 0,
    split: str = "train",
) -> Iterator[tuple[float, dict[str, torch.Tensor]]]:
    """Yield TRM-style batches.

    Each epoch shuffles base puzzle groups and samples one augmented puzzle from
    each group. Batches are packed across epoch boundaries, matching TRM's
    effective behavior when it trains multiple epochs per iterator pass.
    """

    arrays = SudokuArrays(data_dir, split=split)
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
                yield epoch_done, _convert_rows(arrays.inputs, arrays.labels, np.asarray(pending, dtype=np.int64))
                pending = []


def iter_group_eval_batches(
    data_dir: str | Path,
    batch_size: int,
    group_ids: np.ndarray,
    split: str = "train",
    aug_count: int = 0,
) -> Iterator[dict[str, torch.Tensor]]:
    arrays = SudokuArrays(data_dir, split=split)
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
                yield _convert_rows(arrays.inputs, arrays.labels, np.asarray(indices, dtype=np.int64))
                indices = []
    if indices:
        yield _convert_rows(arrays.inputs, arrays.labels, np.asarray(indices, dtype=np.int64))



def iter_group_all_example_batches(
    data_dir: str | Path,
    batch_size: int,
    group_ids: np.ndarray,
    split: str = "train",
) -> Iterator[dict[str, torch.Tensor]]:
    arrays = SudokuArrays(data_dir, split=split)
    indices: list[int] = []
    for group_id in group_ids:
        for puzzle_id in range(int(arrays.group_indices[group_id]), int(arrays.group_indices[group_id + 1])):
            start = int(arrays.puzzle_indices[puzzle_id])
            end = int(arrays.puzzle_indices[puzzle_id + 1])
            for row_idx in range(start, end):
                indices.append(row_idx)
                if len(indices) == batch_size:
                    yield _convert_rows(arrays.inputs, arrays.labels, np.asarray(indices, dtype=np.int64))
                    indices = []
    if indices:
        yield _convert_rows(arrays.inputs, arrays.labels, np.asarray(indices, dtype=np.int64))


def make_loader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    limit: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    ds = SudokuNpyDataset(data_dir, split=split, limit=limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> SudokuBatch:
    return SudokuBatch(
        puzzle_tokens=batch["puzzle_tokens"].to(device, non_blocking=True),
        puzzle_values=batch["puzzle_values"].to(device, non_blocking=True),
        solution_classes=batch["solution_classes"].to(device, non_blocking=True),
        solution_digits=batch["solution_digits"].to(device, non_blocking=True),
    )

