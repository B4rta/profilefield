"""Train-only functional basis for spatial residual-profile modelling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def project_monotone_profiles(values: FloatArray) -> FloatArray:
    """Project profile vectors onto the non-negative monotone cone."""

    profiles = np.asarray(values, dtype=np.float64)
    return np.maximum.accumulate(np.maximum(profiles, 0.0), axis=-1)


class FunctionalResidualBasis:
    """Fixed local-linear functional basis with train-only residual centring.

    PCA is intentionally not used here: its scores are decorrelated on the
    training set by construction and would remove the zero-lag cross-output
    structure that a coregionalized spatial process is meant to learn. Instead,
    smooth residual profiles are represented by coefficients of overlapping
    piecewise-linear hat functions. Their correlations remain scientifically
    meaningful and can differ with spatial lag.
    """

    def __init__(self, n_components: int, ridge: float = 1e-6) -> None:
        if n_components < 2:
            raise ValueError("n_components must be at least two")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        self.n_components = int(n_components)
        self.ridge = float(ridge)
        self.basis_: FloatArray | None = None
        self.encoding_operator_: FloatArray | None = None
        self.residual_mean_: FloatArray | None = None
        self.score_variance_ratio_: FloatArray | None = None
        self.reconstruction_variance_explained_: float | None = None
        self.reconstruction_residual_covariance_: FloatArray | None = None
        self.fit_sample_ids_: tuple[str, ...] = ()

    def fit(
        self,
        truth: FloatArray,
        context_mean: FloatArray,
        sample_ids: list[str] | tuple[str, ...],
    ) -> FunctionalResidualBasis:
        truth_array, mean_array = self._validate_pair(truth, context_mean)
        if len(sample_ids) != len(truth_array):
            raise ValueError("sample_ids must align with the training profiles")
        component_count = min(self.n_components, truth_array.shape[1])
        self.basis_ = _linear_hat_basis(truth_array.shape[1], component_count)
        gram = self.basis_.T @ self.basis_ + self.ridge * np.eye(component_count)
        self.encoding_operator_ = np.linalg.solve(gram, self.basis_.T)
        residual = truth_array - mean_array
        self.residual_mean_ = np.mean(residual, axis=0)
        scores = (residual - self.residual_mean_) @ self.encoding_operator_.T
        score_variance = np.var(scores, axis=0, ddof=1)
        self.score_variance_ratio_ = score_variance / max(float(score_variance.sum()), 1e-12)
        reconstruction = scores @ self.basis_.T + self.residual_mean_
        reconstruction_error = residual - reconstruction
        covariance = np.atleast_2d(np.cov(reconstruction_error, rowvar=False))
        self.reconstruction_residual_covariance_ = covariance + np.eye(covariance.shape[0]) * 1e-6
        denominator = max(float(np.sum((residual - self.residual_mean_) ** 2)), 1e-12)
        self.reconstruction_variance_explained_ = float(
            1.0 - np.sum((residual - reconstruction) ** 2) / denominator
        )
        self.fit_sample_ids_ = tuple(str(value) for value in sample_ids)
        return self

    def encode(self, truth: FloatArray, context_mean: FloatArray) -> FloatArray:
        _, encoding, residual_mean = self._require_fitted()
        truth_array, mean_array = self._validate_pair(truth, context_mean)
        return np.asarray((truth_array - mean_array - residual_mean) @ encoding.T, dtype=np.float64)

    def decode(
        self,
        scores: FloatArray,
        context_mean: FloatArray,
        *,
        enforce_monotonicity: bool = True,
    ) -> FloatArray:
        basis, _, residual_mean = self._require_fitted()
        score_array = np.asarray(scores, dtype=np.float64)
        mean_array = np.asarray(context_mean, dtype=np.float64)
        if score_array.shape[-1] != basis.shape[1]:
            raise ValueError("scores have the wrong latent dimension")
        if mean_array.shape[-1] != basis.shape[0]:
            raise ValueError("context_mean has the wrong profile dimension")
        decoded = score_array @ basis.T + residual_mean + mean_array
        return project_monotone_profiles(decoded) if enforce_monotonicity else decoded

    def sample_reconstruction_nugget(
        self,
        sample_count: int,
        location_count: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Draw the train-estimated profile-scale residual left outside the basis."""

        if self.reconstruction_residual_covariance_ is None:
            raise RuntimeError("Functional residual basis has not been fit")
        cholesky = np.linalg.cholesky(self.reconstruction_residual_covariance_)
        noise = rng.standard_normal((sample_count, location_count, len(cholesky)))
        return np.asarray(noise @ cholesky.T, dtype=np.float64)

    def assert_fit_on(self, expected_ids: set[str]) -> None:
        if set(self.fit_sample_ids_) != set(expected_ids):
            raise AssertionError("Functional residual basis was not fit on the training set only")

    def _require_fitted(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        if self.basis_ is None or self.encoding_operator_ is None or self.residual_mean_ is None:
            raise RuntimeError("Functional residual basis has not been fit")
        return self.basis_, self.encoding_operator_, self.residual_mean_

    @staticmethod
    def _validate_pair(
        truth: FloatArray, context_mean: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        truth_array = np.asarray(truth, dtype=np.float64)
        mean_array = np.asarray(context_mean, dtype=np.float64)
        if truth_array.ndim != 2 or truth_array.shape != mean_array.shape:
            raise ValueError("truth and context_mean must be equal [sample, output] arrays")
        return truth_array, mean_array


def _linear_hat_basis(output_count: int, component_count: int) -> FloatArray:
    positions = np.linspace(0.0, 1.0, output_count)
    knots = np.linspace(0.0, 1.0, component_count)
    spacing = float(knots[1] - knots[0])
    basis = np.maximum(1.0 - np.abs(positions[:, None] - knots[None, :]) / spacing, 0.0)
    basis /= np.maximum(np.sum(basis, axis=1, keepdims=True), 1e-12)
    return basis.astype(np.float64)


def select_residual_mean_weight(
    context_mean: FloatArray,
    residual_correction: FloatArray,
    truth: FloatArray,
    candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> float:
    """Select spatial residual-mean shrinkage on validation observations only."""

    context = np.asarray(context_mean, dtype=np.float64)
    correction = np.asarray(residual_correction, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    if context.shape != correction.shape or context.shape != observed.shape:
        raise ValueError("context_mean, residual_correction, and truth must align")
    valid = tuple(float(value) for value in candidates)
    if not valid or any(value < 0.0 or value > 1.0 for value in valid):
        raise ValueError("candidates must be non-empty values in [0, 1]")
    losses = []
    for weight in valid:
        prediction = project_monotone_profiles(context + weight * correction)
        losses.append(float(np.mean((prediction - observed) ** 2)))
    return valid[int(np.argmin(losses))]
