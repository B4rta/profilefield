"""Post-hoc variance scaling fitted only on an isolated calibration partition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class VarianceScaleCalibrator:
    scale_: FloatArray | None = None
    calibration_sample_ids_: tuple[str, ...] = ()

    def fit(
        self,
        truth: FloatArray,
        mean: FloatArray,
        variance: FloatArray,
        sample_ids: list[str] | tuple[str, ...],
    ) -> VarianceScaleCalibrator:
        if len(truth) != len(sample_ids):
            raise ValueError("Calibration values and IDs must align")
        standardized_squared_error = (truth - mean) ** 2 / np.maximum(variance, 1e-10)
        self.scale_ = np.maximum(np.mean(standardized_squared_error, axis=0), 1e-6)
        self.calibration_sample_ids_ = tuple(str(value) for value in sample_ids)
        return self

    def transform(self, variance: FloatArray) -> FloatArray:
        if self.scale_ is None:
            raise RuntimeError("Calibrator has not been fit")
        return np.asarray(variance, dtype=np.float64) * self.scale_

    def assert_isolated_from(self, test_sample_ids: set[str]) -> None:
        overlap = set(self.calibration_sample_ids_) & test_sample_ids
        if overlap:
            raise AssertionError(f"Calibration/test overlap: {sorted(overlap)[:5]}")


@dataclass
class JointVarianceScaleCalibrator:
    """One temperature for a joint profile posterior.

    A shared scale preserves the relative uncertainty structure across RH
    ordinates. It is preferable to output-wise scaling when posterior draws must
    subsequently satisfy an ordered-profile constraint.
    """

    scale_: float | None = None
    calibration_sample_ids_: tuple[str, ...] = ()

    def fit(
        self,
        truth: FloatArray,
        mean: FloatArray,
        variance: FloatArray,
        sample_ids: list[str] | tuple[str, ...],
    ) -> JointVarianceScaleCalibrator:
        if len(truth) != len(sample_ids):
            raise ValueError("Calibration values and IDs must align")
        squared_error = np.asarray(truth - mean, dtype=np.float64) ** 2
        safe_variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-10)
        # Joint moment matching is stable when constrained decoders have nearly
        # zero variance at a few low RH ordinates; element-wise ratios would let
        # those coordinates dominate the temperature by many orders of magnitude.
        self.scale_ = max(float(np.sum(squared_error) / np.sum(safe_variance)), 1e-6)
        self.calibration_sample_ids_ = tuple(str(value) for value in sample_ids)
        return self

    def transform(self, variance: FloatArray) -> FloatArray:
        if self.scale_ is None:
            raise RuntimeError("Calibrator has not been fit")
        return np.asarray(variance, dtype=np.float64) * self.scale_

    def assert_isolated_from(self, test_sample_ids: set[str]) -> None:
        overlap = set(self.calibration_sample_ids_) & test_sample_ids
        if overlap:
            raise AssertionError(f"Calibration/test overlap: {sorted(overlap)[:5]}")
