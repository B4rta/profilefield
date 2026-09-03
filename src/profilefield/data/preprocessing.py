"""Preprocessors that record exactly which samples were used for fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class TrainOnlyStandardizer:
    mean_: FloatArray | None = None
    scale_: FloatArray | None = None
    fit_sample_ids_: tuple[str, ...] = ()

    def fit(
        self, values: FloatArray, sample_ids: list[str] | tuple[str, ...]
    ) -> TrainOnlyStandardizer:
        if len(values) != len(sample_ids):
            raise ValueError("Values and sample IDs must have equal length")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Training sample IDs must be unique")
        self.mean_ = np.asarray(values, dtype=np.float64).mean(axis=0)
        scale = np.asarray(values, dtype=np.float64).std(axis=0)
        self.scale_ = np.where(scale < 1e-12, 1.0, scale)
        self.fit_sample_ids_ = tuple(str(item) for item in sample_ids)
        return self

    def transform(self, values: FloatArray) -> FloatArray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer has not been fit")
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, values: FloatArray) -> FloatArray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer has not been fit")
        return np.asarray(values, dtype=np.float64) * self.scale_ + self.mean_

    def inverse_variance(self, variance: FloatArray) -> FloatArray:
        if self.scale_ is None:
            raise RuntimeError("Standardizer has not been fit")
        return np.asarray(variance, dtype=np.float64) * np.square(self.scale_)

    def assert_fit_on(self, allowed_sample_ids: set[str]) -> None:
        unexpected = set(self.fit_sample_ids_) - allowed_sample_ids
        if unexpected:
            raise AssertionError(f"Preprocessor saw non-training IDs: {sorted(unexpected)[:5]}")
