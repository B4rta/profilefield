"""Deterministic spatial splits and explicit leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialSplit:
    train: IntArray
    validation: IntArray
    calibration: IntArray
    test: IntArray
    block_ids: IntArray
    sample_ids: tuple[str, ...]
    strategy: str
    seed: int

    def to_manifest(self) -> dict[str, Any]:
        def records(indices: IntArray) -> list[dict[str, Any]]:
            return [
                {
                    "index": int(i),
                    "sample_id": self.sample_ids[i],
                    "block_id": int(self.block_ids[i]),
                }
                for i in indices
            ]

        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "train": records(self.train),
            "validation": records(self.validation),
            "calibration": records(self.calibration),
            "test": records(self.test),
        }


def assign_grid_blocks(coordinates: FloatArray, n_blocks_x: int, n_blocks_y: int) -> IntArray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape [n, 2]")
    if n_blocks_x < 2 or n_blocks_y < 2:
        raise ValueError("At least two blocks are required per axis")
    minimum = coordinates.min(axis=0)
    span = np.maximum(coordinates.max(axis=0) - minimum, 1e-12)
    scaled = np.clip((coordinates - minimum) / span, 0.0, 1.0 - np.finfo(float).eps)
    bx = np.floor(scaled[:, 0] * n_blocks_x).astype(np.int64)
    by = np.floor(scaled[:, 1] * n_blocks_y).astype(np.int64)
    return bx + n_blocks_x * by


def blocked_split(
    coordinates: FloatArray,
    sample_ids: list[str] | tuple[str, ...],
    seed: int,
    n_blocks_x: int,
    n_blocks_y: int,
    train_fraction: float,
    validation_fraction: float,
    calibration_fraction: float = 0.0,
) -> SpatialSplit:
    """Assign entire geographic blocks to partitions before any fitting."""
    n = len(coordinates)
    if len(sample_ids) != n or len(set(sample_ids)) != n:
        raise ValueError("Sample IDs must be unique and align with coordinates")
    fractions = (train_fraction, validation_fraction, calibration_fraction)
    if any(value < 0 for value in fractions) or sum(fractions) >= 1.0:
        raise ValueError("Fractions must be non-negative and leave a non-empty test fraction")
    block_ids = assign_grid_blocks(coordinates, n_blocks_x, n_blocks_y)
    blocks = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(blocks)
    n_blocks = len(ordered)
    n_train = max(1, round(n_blocks * train_fraction))
    n_validation = max(1, round(n_blocks * validation_fraction))
    n_calibration = round(n_blocks * calibration_fraction)
    if n_train + n_validation + n_calibration >= n_blocks:
        overflow = n_train + n_validation + n_calibration - (n_blocks - 1)
        n_train = max(1, n_train - overflow)
    train_blocks = set(ordered[:n_train].tolist())
    validation_blocks = set(ordered[n_train : n_train + n_validation].tolist())
    cal_start = n_train + n_validation
    calibration_blocks = set(ordered[cal_start : cal_start + n_calibration].tolist())
    test_blocks = set(ordered[cal_start + n_calibration :].tolist())

    def indices(selected: set[int]) -> IntArray:
        return np.flatnonzero(np.isin(block_ids, list(selected))).astype(np.int64)

    result = SpatialSplit(
        train=indices(train_blocks),
        validation=indices(validation_blocks),
        calibration=indices(calibration_blocks),
        test=indices(test_blocks),
        block_ids=block_ids,
        sample_ids=tuple(str(value) for value in sample_ids),
        strategy="geographic_blocks",
        seed=seed,
    )
    validate_split(result, require_block_isolation=True)
    return result


def split_from_labels(
    labels: list[str] | NDArray[np.str_],
    sample_ids: list[str] | tuple[str, ...],
    block_ids: IntArray,
) -> SpatialSplit:
    normalized = np.char.lower(np.asarray(labels, dtype=str))
    aliases = {"val": "validation", "valid": "validation", "cal": "calibration"}
    normalized = np.asarray([aliases.get(item, item) for item in normalized], dtype=str)

    def indices(label: str) -> IntArray:
        return np.flatnonzero(normalized == label).astype(np.int64)

    result = SpatialSplit(
        train=indices("train"),
        validation=indices("validation"),
        calibration=indices("calibration"),
        test=indices("test"),
        block_ids=np.asarray(block_ids, dtype=np.int64),
        sample_ids=tuple(str(value) for value in sample_ids),
        strategy="official",
        seed=0,
    )
    validate_split(result, require_block_isolation=False)
    return result


def validate_split(split: SpatialSplit, require_block_isolation: bool) -> None:
    partitions = {
        "train": split.train,
        "validation": split.validation,
        "calibration": split.calibration,
        "test": split.test,
    }
    non_empty = {name: values for name, values in partitions.items() if len(values)}
    for name, values in non_empty.items():
        if len(np.unique(values)) != len(values):
            raise AssertionError(f"Duplicate indices inside {name}")
    names = list(non_empty)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = np.intersect1d(non_empty[left], non_empty[right])
            if overlap.size:
                raise AssertionError(f"Sample overlap between {left} and {right}: {overlap[:5]}")
            left_ids = {split.sample_ids[index] for index in non_empty[left]}
            right_ids = {split.sample_ids[index] for index in non_empty[right]}
            if left_ids & right_ids:
                raise AssertionError(f"Sample ID overlap between {left} and {right}")
            if require_block_isolation:
                left_blocks = set(split.block_ids[non_empty[left]].tolist())
                right_blocks = set(split.block_ids[non_empty[right]].tolist())
                if left_blocks & right_blocks:
                    raise AssertionError(f"Block leakage between {left} and {right}")
    if len(split.train) == 0 or len(split.test) == 0:
        raise AssertionError("Train and test partitions must be non-empty")
