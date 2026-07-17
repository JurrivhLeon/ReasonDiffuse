#!/usr/bin/env python3
"""Validate TRM-format puzzle datasets.

The dataset builders write directories with train/test splits, dataset.json
metadata, and one or more named subsets such as all__inputs.npy. This script
checks that those files are internally consistent and loadable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = (
    "inputs",
    "labels",
    "puzzle_identifiers",
    "puzzle_indices",
    "group_indices",
)


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _check_range(name: str, values: np.ndarray, vocab_size: int) -> None:
    if values.size == 0:
        raise AssertionError(f"{name} is empty")
    min_value = int(values.min())
    max_value = int(values.max())
    if min_value < 0 or max_value >= vocab_size:
        raise AssertionError(
            f"{name} token range [{min_value}, {max_value}] exceeds [0, {vocab_size})"
        )


def check_split(dataset_dir: Path, split: str) -> None:
    split_dir = dataset_dir / split
    metadata_path = split_dir / "dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = _load_json(metadata_path)
    seq_len = int(metadata["seq_len"])
    vocab_size = int(metadata["vocab_size"])
    sets = metadata["sets"]

    for set_name in sets:
        arrays = {}
        for field in REQUIRED_FIELDS:
            path = split_dir / f"{set_name}__{field}.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            arrays[field] = np.load(path, mmap_mode="r")

        inputs = arrays["inputs"]
        labels = arrays["labels"]
        puzzle_identifiers = arrays["puzzle_identifiers"]
        puzzle_indices = arrays["puzzle_indices"]
        group_indices = arrays["group_indices"]

        if inputs.ndim != 2 or labels.ndim != 2:
            raise AssertionError(f"{dataset_dir}/{split}/{set_name}: inputs/labels must be rank 2")
        if inputs.shape != labels.shape:
            raise AssertionError(
                f"{dataset_dir}/{split}/{set_name}: inputs {inputs.shape} != labels {labels.shape}"
            )
        if inputs.shape[1] != seq_len:
            raise AssertionError(
                f"{dataset_dir}/{split}/{set_name}: seq_len {inputs.shape[1]} != metadata {seq_len}"
            )
        if puzzle_indices.ndim != 1 or group_indices.ndim != 1 or puzzle_identifiers.ndim != 1:
            raise AssertionError(f"{dataset_dir}/{split}/{set_name}: index arrays must be rank 1")
        if int(puzzle_indices[0]) != 0 or int(group_indices[0]) != 0:
            raise AssertionError(f"{dataset_dir}/{split}/{set_name}: indices must start at 0")
        if int(puzzle_indices[-1]) != inputs.shape[0]:
            raise AssertionError(
                f"{dataset_dir}/{split}/{set_name}: puzzle_indices[-1] {int(puzzle_indices[-1])} "
                f"!= num examples {inputs.shape[0]}"
            )
        if int(group_indices[-1]) != puzzle_identifiers.shape[0]:
            raise AssertionError(
                f"{dataset_dir}/{split}/{set_name}: group_indices[-1] {int(group_indices[-1])} "
                f"!= num puzzles {puzzle_identifiers.shape[0]}"
            )
        if len(puzzle_indices) != puzzle_identifiers.shape[0] + 1:
            raise AssertionError(
                f"{dataset_dir}/{split}/{set_name}: len(puzzle_indices) must be num puzzles + 1"
            )

        _check_range("inputs", inputs, vocab_size)
        _check_range("labels", labels, vocab_size)

        print(
            f"ok {dataset_dir}/{split}/{set_name}: "
            f"examples={inputs.shape[0]} seq_len={seq_len} vocab={vocab_size} "
            f"puzzles={puzzle_identifiers.shape[0]} groups={len(group_indices) - 1}"
        )


def check_dataset(dataset_dir: Path) -> None:
    found_split = False
    for split in ("train", "test"):
        if (dataset_dir / split).exists():
            found_split = True
            check_split(dataset_dir, split)
    if not found_split:
        raise FileNotFoundError(f"No train/ or test/ split found in {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    for dataset_dir in args.dataset_dirs:
        check_dataset(dataset_dir)


if __name__ == "__main__":
    main()
