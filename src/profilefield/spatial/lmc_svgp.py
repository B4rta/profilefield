"""Linear model of coregionalization sparse variational multi-output GP."""

from __future__ import annotations

from typing import Any

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
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


class _LMCApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(
        self, inducing_points: Tensor, output_dim: int, latent_count: int, nu: float
    ) -> None:
        batch_shape = torch.Size([latent_count])
        expanded = inducing_points.unsqueeze(0).expand(latent_count, -1, -1).contiguous()
        distribution = gpytorch.variational.CholeskyVariationalDistribution(
            expanded.size(-2), batch_shape=batch_shape
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self, expanded, distribution, learn_inducing_locations=True
        )
        strategy = gpytorch.variational.LMCVariationalStrategy(
            base_strategy,
            num_tasks=output_dim,
            num_latents=latent_count,
            latent_dim=-1,
        )
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = matern_kernel(inducing_points.size(-1), nu, batch_shape)

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class LMCSVGP(nn.Module):
    """Sparse multi-output GP with learned LMC mixing coefficients."""

    def __init__(
        self,
        output_dim: int,
        latent_count: int,
        inducing_count: int,
        nu: float = 1.5,
        learning_rate: float = 0.03,
        training_steps: int = 500,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if latent_count < 1:
            raise ValueError("latent_count must be positive")
        self.output_dim = output_dim
        self.latent_count = latent_count
        self.inducing_count = inducing_count
        self.nu = nu
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.gp: _LMCApproximateGP | None = None
        self.likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood | None = None
        self.transforms: ModelTransforms | None = None
        self.history: list[float] = []

    def fit(
        self,
        coordinates: FloatArray,
        targets: FloatArray,
        sample_ids: list[str] | tuple[str, ...],
    ) -> LMCSVGP:
        set_seed(self.seed)
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise ValueError("Targets must have shape [n, output_dim]")
        self.transforms = ModelTransforms.fit(coordinates, values, sample_ids)
        x_scaled = self.transforms.x.transform(coordinates)
        y_scaled = self.transforms.y.transform(values)
        inducing = select_inducing_points(x_scaled, self.inducing_count)
        inducing_tensor = as_tensor(inducing, self.device, self.dtype)
        self.gp = _LMCApproximateGP(
            inducing_tensor, self.output_dim, self.latent_count, self.nu
        ).to(self.device, self.dtype)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.output_dim, rank=0
        ).to(self.device, self.dtype)
        x_tensor = as_tensor(x_scaled, self.device, self.dtype)
        y_tensor = as_tensor(y_scaled, self.device, self.dtype)
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
            loss = -objective(output, y_tensor)
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
        if posterior_samples < 1:
            raise ValueError("posterior_samples must be positive")
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive = likelihood(gp(x_tensor))
            mean_scaled = predictive.mean.cpu().numpy()
            variance_scaled = predictive.variance.cpu().numpy()
            # A low-rank root of the full multi-output covariance can truncate
            # observation noise. Sample exact spatial latent fields before LMC
            # mixing, then add independent task/global noise and LMC jitter.
            strategy = gp.variational_strategy
            latent = strategy.base_variational_strategy(x_tensor, diag=False)
            with gpytorch.settings.max_cholesky_size(len(x_tensor) + 1):
                latent_draws = latent.rsample(torch.Size([posterior_samples]))
            mixed = latent_draws.transpose(-1, -2) @ strategy.lmc_coefficients
            if likelihood.rank != 0:
                raise RuntimeError("Exact LMC sampler currently requires diagonal task noise")
            noise_variance = torch.full(
                (self.output_dim,), float(strategy.jitter_val), dtype=self.dtype, device=self.device
            )
            if likelihood.has_task_noise:
                noise_variance = noise_variance + likelihood.task_noises
            if likelihood.has_global_noise:
                noise_variance = noise_variance + likelihood.noise
            mixed = mixed + torch.randn_like(mixed) * noise_variance.sqrt()
            draws_scaled = mixed.cpu().numpy()
            covariance_scaled = (
                predictive.covariance_matrix.cpu().numpy() if full_covariance else None
            )
        n = len(x_tensor)
        scales = transforms.y.scale_
        if scales is None:
            raise RuntimeError("Target transform lost its fitted scale")
        joint = (
            unscale_full_covariance(covariance_scaled, scales, n)
            if covariance_scaled is not None
            else None
        )
        coefficients = gp.variational_strategy.lmc_coefficients.detach().cpu().numpy()
        outputscale = gp.covar_module.outputscale.detach().cpu().numpy()
        covariance_standardized = coefficients.T @ np.diag(outputscale) @ coefficients
        output_covariance = covariance_standardized * np.outer(scales, scales)
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
            "kind": "lmc_svgp",
            "settings": {
                "output_dim": self.output_dim,
                "latent_count": self.latent_count,
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
    ) -> tuple[
        _LMCApproximateGP,
        gpytorch.likelihoods.MultitaskGaussianLikelihood,
        ModelTransforms,
    ]:
        if self.gp is None or self.likelihood is None or self.transforms is None:
            raise RuntimeError("Model has not been fit")
        return self.gp, self.likelihood, self.transforms
