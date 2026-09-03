from __future__ import annotations

import numpy as np
import pytest

from profilefield.data.splits import SpatialSplit, blocked_split, validate_split


def test_blocked_split_is_deterministic_and_disjoint() -> None:
    axis = np.linspace(0.0, 1.0, 8)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])
    ids = [f"s-{index}" for index in range(len(coordinates))]
    left = blocked_split(coordinates, ids, 17, 4, 4, 0.5, 0.25, 0.0625)
    right = blocked_split(coordinates, ids, 17, 4, 4, 0.5, 0.25, 0.0625)
    assert np.array_equal(left.train, right.train)
    assert np.array_equal(left.test, right.test)
    validate_split(left, require_block_isolation=True)
    assert not set(left.block_ids[left.train]) & set(left.block_ids[left.test])


def test_overlap_is_rejected() -> None:
    split = SpatialSplit(
        train=np.asarray([0, 1], dtype=np.int64),
        validation=np.asarray([2], dtype=np.int64),
        calibration=np.asarray([], dtype=np.int64),
        test=np.asarray([1, 3], dtype=np.int64),
        block_ids=np.arange(4),
        sample_ids=("a", "b", "c", "d"),
        strategy="bad",
        seed=0,
    )
    with pytest.raises(AssertionError, match="overlap"):
        validate_split(split, require_block_isolation=True)


def test_block_leakage_is_rejected() -> None:
    split = SpatialSplit(
        train=np.asarray([0, 1], dtype=np.int64),
        validation=np.asarray([2], dtype=np.int64),
        calibration=np.asarray([], dtype=np.int64),
        test=np.asarray([3], dtype=np.int64),
        block_ids=np.asarray([0, 0, 1, 0], dtype=np.int64),
        sample_ids=("a", "b", "c", "d"),
        strategy="bad",
        seed=0,
    )
    with pytest.raises(AssertionError, match="Block leakage"):
        validate_split(split, require_block_isolation=True)
