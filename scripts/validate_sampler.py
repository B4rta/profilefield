"""Reproduce the joint-sampler covariance diagnostic with auditable outputs."""

from __future__ import annotations

import time
from pathlib import Path

import gpytorch
import numpy as np
import torch

from profilefield.artifacts import RunArtifacts
from profilefield.spatial.common import as_tensor, unscale_full_covariance, unscale_samples
from profilefield.spatial.lmc_svgp import LMCSVGP


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(4)
    seed = 1907
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(32, 2))
    common = np.sin(x[:, 0] * 4) + np.cos(x[:, 1] * 2)
    y = np.column_stack([common, 0.8 * common + 0.1 * x[:, 0]])
    model = LMCSVGP(2, 2, 12, training_steps=5, seed=seed).fit(x, y, list(map(str, range(32))))
    run = RunArtifacts.create(Path("outputs/validation"), "joint_sampler_v2", seed)
    run.initialize({"experiment": {"name": "joint_sampler_v2", "seed": seed},
                    "diagnostic": {"locations": [64, 500], "posterior_draws": 2048}},
                   {"source": "deterministic_synthetic_sampler_diagnostic"},
                   {"training_count": 32, "query_role": "numerical_covariance_validation"},
                   Path.cwd(), seed)
    rows = []
    for count in (64, 500):
        query = rng.uniform(size=(count, 2))
        torch.manual_seed(seed + count)
        fixed = model.predict(query, posterior_samples=2048, full_covariance=False)
        gp, likelihood, transforms = model._require_fitted()
        query_tensor = as_tensor(transforms.x.transform(query), model.device, model.dtype)
        torch.manual_seed(seed + count)
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(800), gpytorch.settings.max_root_decomposition_size(100):
            predictive = likelihood(gp(query_tensor))
            legacy_samples = predictive.rsample(torch.Size([2048])).cpu().numpy()
            legacy_variance = predictive.variance.cpu().numpy()
            expected_covariance = unscale_full_covariance(
                predictive.covariance_matrix.cpu().numpy(), transforms.y.scale_, count
            )
        legacy_samples = unscale_samples(legacy_samples, transforms.y)
        legacy_variance = transforms.y.inverse_variance(legacy_variance)
        for name, variance, samples in (
            ("legacy_output_root", legacy_variance, legacy_samples),
            ("exact_latent_factor", fixed.variance, fixed.samples),
        ):
            sampled_variance = samples.var(axis=0)
            sampled_covariance = np.cov(samples.reshape(len(samples), -1), rowvar=False)
            diagonal = np.sqrt(np.diag(expected_covariance))
            normalized_error = (sampled_covariance - expected_covariance) / np.outer(diagonal, diagonal)
            off_diagonal = ~np.eye(len(diagonal), dtype=bool)
            rows.append({"model": name, "query_locations": count,
                         "sample_variance_to_model_variance": float(sampled_variance.sum() / variance.sum()),
                         "median_coordinate_variance_ratio": float(np.median(sampled_variance / variance)),
                         "normalized_offdiagonal_covariance_rmse": float(np.sqrt(np.mean(normalized_error[off_diagonal] ** 2))),
                         "draws": len(samples)})
    run.write_table("sampler_diagnostics", rows)
    run.finalize(time.perf_counter() - started)
    print(run.path)


if __name__ == "__main__":
    main()
