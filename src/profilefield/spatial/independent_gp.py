"""Independent sparse variational GPs with a fair multi-output interface."""

from __future__ import annotations

from typing import Any

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from numpy.typing import NDArray
from torch import Tensor, nn

from profilefield.reproducibility import set_seed
from profilefield.spatial.common import (
    FloatArray,
    ModelTransforms,
    SpatialPrediction,
    as_tensor,
    unscale_full_covariance,
    unscale_samples,
)
from profilefield.spatial.kernels import matern_kernel, select_inducing_points


class _IndependentApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: Tensor, output_dim: int, nu: float) -> None:
        batch_shape = torch.Size([output_dim])
        expanded = inducing_points.unsqueeze(0).expand(output_dim, -1, -1).contiguous()
        distribution = gpytorch.variational.CholeskyVariationalDistribution(
            expanded.size(-2), batch_shape=batch_shape
        )
        strategy = gpytorch.variational.VariationalStrategy(
            self, expanded, distribution, learn_inducing_locations=True
        )
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = matern_kernel(inducing_points.size(-1), nu, batch_shape)

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class IndependentSVGP(nn.Module):
    """One variational GP per output with no cross-output covariance."""

    def __init__(
        self,
        output_dim: int,
        inducing_count: int,
        nu: float = 1.5,
        learning_rate: float = 0.03,
        training_steps: int = 500,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.inducing_count = inducing_count
        self.nu = nu
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.gp: _IndependentApproximateGP | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.transforms: ModelTransforms | None = None
        self.history: list[float] = []

    def fit(
        self,
        coordinates: FloatArray,
        targets: FloatArray,
        sample_ids: list[str] | tuple[str, ...],
    ) -> IndependentSVGP:
        set_seed(self.seed)
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise ValueError("Targets must have shape [n, output_dim]")
        self.transforms = ModelTransforms.fit(coordinates, values, sample_ids)
        x_scaled = self.transforms.x.transform(coordinates)
        y_scaled = self.transforms.y.transform(values)
        inducing = select_inducing_points(x_scaled, self.inducing_count)
        inducing_tensor = as_tensor(inducing, self.device, self.dtype)
        self.gp = _IndependentApproximateGP(inducing_tensor, self.output_dim, self.nu).to(
            self.device, self.dtype
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([self.output_dim])
        ).to(self.device, self.dtype)
        x_tensor = as_tensor(x_scaled, self.device, self.dtype)
        y_tensor = as_tensor(y_scaled.T, self.device, self.dtype)
        self.gp.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(
            list(self.gp.parameters()) + list(self.likelihood.parameters()), lr=self.learning_rate
        )
        objective = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp, num_data=len(x_tensor))
        self.history = []
        for _ in range(self.training_steps):
            optimizer.zero_grad(set_to_none=True)
            output = self.gp(x_tensor)
            loss = -objective(output, y_tensor).sum()
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach().cpu()))
        return self

    def predict(
        self,
        coordinates: FloatArray,
        posterior_samples: int = 128,
        full_covariance: bool = True,
    ) -> SpatialPrediction:
        gp, likelihood, transforms = self._require_fitted()
        x_tensor = as_tensor(transforms.x.transform(coordinates), self.device, self.dtype)
        gp.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive = likelihood(gp(x_tensor))
            mean_scaled = predictive.mean.transpose(0, 1).cpu().numpy()
            variance_scaled = predictive.variance.transpose(0, 1).cpu().numpy()
            draws_scaled = predictive.rsample(torch.Size([posterior_samples]))
            draws_scaled = draws_scaled.permute(0, 2, 1).cpu().numpy()
            batch_covariance = predictive.covariance_matrix.cpu().numpy()
        n = len(x_tensor)
        joint: FloatArray | None = None
        if full_covariance:
            joint = np.zeros((n * self.output_dim, n * self.output_dim), dtype=np.float64)
            for task in range(self.output_dim):
                positions = np.arange(task, n * self.output_dim, self.output_dim)
                joint[np.ix_(positions, positions)] = batch_covariance[task]
            joint = unscale_full_covariance(joint, transforms.y.scale_, n)  # type: ignore[arg-type]
        outputscale = gp.covar_module.outputscale.detach().cpu().numpy()
        output_covariance = np.diag(outputscale * np.square(transforms.y.scale_))  # type: ignore[arg-type]
        return SpatialPrediction(
            mean=transforms.y.inverse_transform(mean_scaled),
            variance=transforms.y.inverse_variance(variance_scaled),
            samples=unscale_samples(draws_scaled, transforms.y),
            full_covariance=joint,
            output_covariance=output_covariance,
        )

    def checkpoint(self) -> dict[str, Any]:
        gp, likelihood, transforms = self._require_fitted()
        return {
            "kind": "independent_svgp",
            "settings": {
                "output_dim": self.output_dim,
                "inducing_count": self.inducing_count,
                "nu": self.nu,
                "seed": self.seed,
            },
            "gp": gp.state_dict(),
            "likelihood": likelihood.state_dict(),
            "x_mean": transforms.x.mean_,
            "x_scale": transforms.x.scale_,
            "y_mean": transforms.y.mean_,
            "y_scale": transforms.y.scale_,
            "history": self.history,
        }

    def _require_fitted(
        self,
    ) -> tuple[_IndependentApproximateGP, gpytorch.likelihoods.GaussianLikelihood, ModelTransforms]:
        if self.gp is None or self.likelihood is None or self.transforms is None:
            raise RuntimeError("Model has not been fit")
        return self.gp, self.likelihood, self.transforms


Array = NDArray[np.float64]
