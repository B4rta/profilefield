"""Restore trusted, integrity-checked spatial states without optimization."""

from __future__ import annotations

from typing import Any

import gpytorch
import numpy as np
import torch

from profilefield.data.preprocessing import TrainOnlyStandardizer
from profilefield.spatial.common import ModelTransforms
from profilefield.spatial.independent_gp import IndependentSVGP, _IndependentApproximateGP
from profilefield.spatial.lmc_svgp import LMCSVGP, _LMCApproximateGP


def restore_spatial(state: dict[str, Any]) -> IndependentSVGP | LMCSVGP:
    """Reconstruct a CPU float64 model from a locally trusted checkpoint dict.

    This function does not deserialize files or fit any parameters. Callers must
    verify provenance before using pickle-based torch/joblib deserialization.
    """
    settings = state["settings"]
    model: IndependentSVGP | LMCSVGP
    if state["kind"] == "independent_svgp":
        model = IndependentSVGP(**settings)
        inducing = state["gp"]["variational_strategy.inducing_points"][0]
        model.gp = _IndependentApproximateGP(inducing, model.output_dim, model.nu).double()
        model.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([model.output_dim])
        ).double()
    elif state["kind"] == "lmc_svgp":
        model = LMCSVGP(**settings)
        inducing = state["gp"]["variational_strategy.base_variational_strategy.inducing_points"][0]
        model.gp = _LMCApproximateGP(
            inducing, model.output_dim, model.latent_count, model.nu
        ).double()
        model.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=model.output_dim, rank=0
        ).double()
    else:
        raise ValueError("Unsupported spatial checkpoint kind")
    model.gp.load_state_dict(state["gp"], strict=True)
    model.likelihood.load_state_dict(state["likelihood"], strict=True)
    model.transforms = ModelTransforms(*[
        TrainOnlyStandardizer(
            mean_=np.asarray(state[f"{axis}_mean"], dtype=np.float64),
            scale_=np.asarray(state[f"{axis}_scale"], dtype=np.float64),
        ) for axis in ("x", "y")
    ])
    model.history = list(state["history"])
    return model
