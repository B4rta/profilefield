"""Shared prediction contracts and training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from profilefield.data.preprocessing import TrainOnlyStandardizer

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialPrediction:
    mean: FloatArray
    variance: FloatArray
    samples: FloatArray
    full_covariance: FloatArray | None
    output_covariance: FloatArray


@dataclass
class ModelTransforms:
    x: TrainOnlyStandardizer
    y: TrainOnlyStandardizer

    @classmethod
    def fit(
        cls,
        coordinates: FloatArray,
        targets: FloatArray,
        sample_ids: list[str] | tuple[str, ...],
    ) -> ModelTransforms:
        return cls(
            x=TrainOnlyStandardizer().fit(coordinates, sample_ids),
            y=TrainOnlyStandardizer().fit(targets, sample_ids),
        )


def as_tensor(values: FloatArray, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(np.asarray(values), device=device, dtype=dtype)


def unscale_samples(samples: FloatArray, transform: TrainOnlyStandardizer) -> FloatArray:
    shape = samples.shape
    flat = samples.reshape(-1, shape[-1])
    return transform.inverse_transform(flat).reshape(shape)


def unscale_full_covariance(covariance: FloatArray, scales: FloatArray, n: int) -> FloatArray:
    """Unscale covariance ordered as [sample0-task0, sample0-task1, ...]."""
    repeated = np.tile(scales, n)
    return covariance * np.outer(repeated, repeated)
