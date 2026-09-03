"""Biomazon/GEDI ordered-RH common experiment framework."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from profilefield.artifacts import RunArtifacts
from profilefield.calibration.scaling import JointVarianceScaleCalibrator
from profilefield.config import Config
from profilefield.data.biomazon import load_biomazon
from profilefield.data.preprocessing import TrainOnlyStandardizer
from profilefield.evaluation.bootstrap import block_bootstrap_interval
from profilefield.evaluation.figures import calibration_figure, covariance_figure, variogram_figure
from profilefield.evaluation.joint import (
    derived_profile_rows,
    posterior_profile_covariance_rows,
    profile_variogram_score,
)
from profilefield.evaluation.metrics import (
    calibration_rows,
    energy_score,
    gaussian_crps,
    gaussian_nll,
    latent_metric_rows,
)
from profilefield.evaluation.profiles import per_block_profile_rows, profile_metric_rows
from profilefield.evaluation.spatial import (
    cross_variogram_error,
    empirical_cross_variograms,
    posterior_cross_variogram_error,
)
from profilefield.profiles.functional_residual import (
    FunctionalResidualBasis,
    project_monotone_profiles,
    select_residual_mean_weight,
)
from profilefield.profiles.models import (
    ContextualProfileVAE,
    DeterministicContextProfile,
    train_contextual_vae,
    train_deterministic_profile,
)
from profilefield.reproducibility import set_seed
from profilefield.spatial.independent_gp import IndependentSVGP
from profilefield.spatial.lmc_svgp import LMCSVGP

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ProfileDistribution:
    mean: FloatArray
    variance: FloatArray
    samples: FloatArray


@dataclass
class ProfilePredictor:
    name: str
    predict: Any


def run_biomazon(config: Config, project_root: str | Path = ".") -> list[Path]:
    root = Path(project_root).resolve()
    return [_run_seed(config, int(seed), root) for seed in config["experiment"].get("seeds", [0])]


def _run_seed(config: Config, seed: int, root: Path) -> Path:
    started = time.perf_counter()
    set_seed(seed)
    dataset = load_biomazon(config, root, seed)
    artifacts = RunArtifacts.create(
        root / config["experiment"]["output_root"], config["experiment"]["name"], seed
    )
    artifacts.initialize(config, dataset.manifest, dataset.split.to_manifest(), root, seed)
    split = dataset.split
    train_ids = [dataset.sample_ids[index] for index in split.train]
    feature_scaler = TrainOnlyStandardizer().fit(dataset.features[split.train], train_ids)
    feature_scaler.assert_fit_on(set(train_ids))
    features = feature_scaler.transform(dataset.features)
    x_train = features[split.train]
    y_train = dataset.profiles[split.train]
    model_config = config["model"]
    sample_count = int(config["evaluation"]["posterior_samples"])
    rng = np.random.default_rng(seed)
    predictors: list[ProfilePredictor] = []
    runtime_rows: list[dict[str, Any]] = []

    linear_started = time.perf_counter()
    ridge = Ridge(alpha=float(model_config.get("ridge_alpha", 1.0))).fit(x_train, y_train)
    ridge_train_prediction = ridge.predict(x_train)
    ridge_covariance = _residual_covariance(y_train, ridge_train_prediction)
    predictors.append(
        ProfilePredictor(
            "Per-RH ridge",
            lambda indices: _regression_distribution(
                ridge.predict(features[indices]), ridge_covariance, sample_count, rng, False
            ),
        )
    )
    joblib.dump(ridge, artifacts.path / "models/per_rh_ridge.joblib")
    runtime_rows.append(_runtime_row("Per-RH ridge", time.perf_counter() - linear_started))

    tree_started = time.perf_counter()
    forest = RandomForestRegressor(
        n_estimators=int(model_config.get("tree_estimators", 200)),
        min_samples_leaf=int(model_config.get("tree_min_samples_leaf", 3)),
        random_state=seed,
        n_jobs=-1,
    ).fit(x_train, y_train)
    forest_train_prediction = forest.predict(x_train)
    tree_covariance = _residual_covariance(y_train, forest_train_prediction)
    predictors.append(
        ProfilePredictor(
            "Per-RH random forest",
            lambda indices: _regression_distribution(
                forest.predict(features[indices]), tree_covariance, sample_count, rng, False
            ),
        )
    )
    joblib.dump(forest, artifacts.path / "models/per_rh_random_forest.joblib")
    runtime_rows.append(_runtime_row("Per-RH random forest", time.perf_counter() - tree_started))

    dtype = torch.float32
    context_train = torch.as_tensor(x_train, dtype=dtype)
    profile_train = torch.as_tensor(y_train, dtype=dtype)
    context_all = torch.as_tensor(features, dtype=dtype)
    latent_dim = int(model_config["latent_dim"])
    hidden_dim = int(model_config["hidden_dim"])
    neural_steps = int(model_config["training_steps"])
    learning_rate = float(model_config["learning_rate"])

    deterministic_started = time.perf_counter()
    deterministic = DeterministicContextProfile(
        x_train.shape[1], latent_dim, y_train.shape[1], hidden_dim
    )
    deterministic_result = train_deterministic_profile(
        deterministic, context_train, profile_train, neural_steps, learning_rate
    )
    deterministic_covariance = _residual_covariance(
        y_train, _torch_predict(deterministic, context_train)
    )
    predictors.append(
        ProfilePredictor(
            "Deterministic shared decoder",
            lambda indices: _regression_distribution(
                _torch_predict(deterministic, context_all[indices]),
                deterministic_covariance,
                sample_count,
                rng,
                True,
            ),
        )
    )
    torch.save(
        deterministic.state_dict(), artifacts.path / "models/deterministic_shared_decoder.pt"
    )
    _save_history(
        artifacts.path / "diagnostics/training_deterministic.csv",
        deterministic_result.history,
    )
    runtime_rows.append(
        _runtime_row("Deterministic shared decoder", time.perf_counter() - deterministic_started)
    )

    vae_started = time.perf_counter()
    vae = ContextualProfileVAE(x_train.shape[1], latent_dim, y_train.shape[1], hidden_dim)
    vae_result = train_contextual_vae(
        vae,
        context_train,
        profile_train,
        neural_steps,
        learning_rate,
        float(model_config.get("kl_weight", 0.05)),
    )
    predictors.append(
        ProfilePredictor(
            "Conditional VAE",
            lambda indices: _cvae_distribution(vae, context_all[indices], sample_count),
        )
    )
    torch.save(vae.state_dict(), artifacts.path / "models/contextual_vae.pt")
    _save_history(artifacts.path / "diagnostics/training_cvae.csv", vae_result.history)
    runtime_rows.append(_runtime_row("Conditional VAE", time.perf_counter() - vae_started))

    with torch.no_grad():
        train_residual = (
            vae.encode_residual(context_train, profile_train).cpu().numpy().astype(np.float64)
        )
    inducing_count = int(model_config["inducing_points"])
    kernel_nu = float(model_config["kernel_nu"])
    gp_learning_rate = float(model_config.get("gp_learning_rate", learning_rate))
    gp_training_steps = int(model_config.get("gp_training_steps", neural_steps))
    independent_started = time.perf_counter()
    independent = IndependentSVGP(
        output_dim=latent_dim,
        inducing_count=inducing_count,
        nu=kernel_nu,
        learning_rate=gp_learning_rate,
        training_steps=gp_training_steps,
        seed=seed,
        dtype=torch.float64,
    ).fit(dataset.coordinates[split.train], train_residual, train_ids)
    runtime_rows.append(
        _runtime_row("GeoVAE independent-SVGP", time.perf_counter() - independent_started)
    )
    lmc_started = time.perf_counter()
    lmc = LMCSVGP(
        output_dim=latent_dim,
        latent_count=int(model_config["lmc_latents"]),
        inducing_count=inducing_count,
        nu=kernel_nu,
        learning_rate=gp_learning_rate,
        training_steps=gp_training_steps,
        seed=seed,
        dtype=torch.float64,
    ).fit(dataset.coordinates[split.train], train_residual, train_ids)
    runtime_rows.append(_runtime_row("GeoVAE-Field LMC-SVGP", time.perf_counter() - lmc_started))
    torch.save(independent.checkpoint(), artifacts.path / "models/independent_svgp.pt")
    torch.save(lmc.checkpoint(), artifacts.path / "models/lmc_svgp.pt")

    def spatial_profile_distribution(
        indices: NDArray[np.int64], model: IndependentSVGP | LMCSVGP
    ) -> ProfileDistribution:
        latent_prediction = model.predict(
            dataset.coordinates[indices], posterior_samples=sample_count, full_covariance=False
        )
        with torch.no_grad():
            context_mean = vae.context_encoder(context_all[indices]).cpu().numpy()
            mean = (
                vae.decoder(torch.as_tensor(context_mean + latent_prediction.mean, dtype=dtype))
                .cpu()
                .numpy()
            )
            latent_samples = latent_prediction.samples + context_mean[None, :, :]
            decoded = (
                vae.decoder(torch.as_tensor(latent_samples.reshape(-1, latent_dim), dtype=dtype))
                .cpu()
                .numpy()
            )
        samples = decoded.reshape(sample_count, len(indices), -1).astype(np.float64)
        return ProfileDistribution(
            mean=mean.astype(np.float64),
            variance=np.var(samples, axis=0) + 1e-8,
            samples=samples,
        )

    predictors.extend(
        [
            ProfilePredictor(
                "GeoVAE independent-SVGP",
                lambda indices: spatial_profile_distribution(indices, independent),
            ),
            ProfilePredictor(
                "GeoVAE-Field LMC-SVGP",
                lambda indices: spatial_profile_distribution(indices, lmc),
            ),
        ]
    )

    functional_started = time.perf_counter()
    functional_basis = FunctionalResidualBasis(latent_dim).fit(
        y_train, forest_train_prediction, train_ids
    )
    functional_basis.assert_fit_on(set(train_ids))
    functional_train_scores = functional_basis.encode(y_train, forest_train_prediction)
    functional_dim = functional_train_scores.shape[1]
    functional_independent = IndependentSVGP(
        output_dim=functional_dim,
        inducing_count=inducing_count,
        nu=kernel_nu,
        learning_rate=gp_learning_rate,
        training_steps=gp_training_steps,
        seed=seed,
        dtype=torch.float64,
    ).fit(dataset.coordinates[split.train], functional_train_scores, train_ids)
    runtime_rows.append(
        _runtime_row("RF-FRC independent-SVGP", time.perf_counter() - functional_started)
    )
    functional_lmc_started = time.perf_counter()
    functional_lmc = LMCSVGP(
        output_dim=functional_dim,
        latent_count=min(int(model_config["lmc_latents"]), functional_dim),
        inducing_count=inducing_count,
        nu=kernel_nu,
        learning_rate=gp_learning_rate,
        training_steps=gp_training_steps,
        seed=seed,
        dtype=torch.float64,
    ).fit(dataset.coordinates[split.train], functional_train_scores, train_ids)
    runtime_rows.append(
        _runtime_row("ProfileField RF-FRC LMC-SVGP", time.perf_counter() - functional_lmc_started)
    )
    joblib.dump(functional_basis, artifacts.path / "models/functional_residual_basis.joblib")
    torch.save(
        functional_independent.checkpoint(),
        artifacts.path / "models/functional_independent_svgp.pt",
    )
    torch.save(functional_lmc.checkpoint(), artifacts.path / "models/functional_lmc_svgp.pt")

    residual_mean_weights: dict[str, float] = {}
    spatial_dependence_weights: dict[str, float] = {}
    spatial_score_scales: dict[str, FloatArray] = {}
    validation_weight_diagnostics: dict[str, list[dict[str, float]]] = {}
    validation_indices = split.validation
    validation_selection_indices = validation_indices
    validation_location_limit = int(
        config["evaluation"].get("validation_objective_max_locations", 512)
    )
    if len(validation_indices) > validation_location_limit:
        selection_rng = np.random.default_rng(seed + 6001)
        validation_selection_indices = np.sort(
            selection_rng.choice(validation_indices, size=validation_location_limit, replace=False)
        ).astype(np.int64)
    functional_score_covariance = _residual_covariance(
        functional_train_scores, np.zeros_like(functional_train_scores)
    )
    score_cholesky = np.linalg.cholesky(functional_score_covariance)
    dependence_weight_grid = tuple(
        float(value)
        for value in model_config.get("spatial_dependence_weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    )
    if not dependence_weight_grid or any(
        value < 0.0 or value > 1.0 for value in dependence_weight_grid
    ):
        raise ValueError("spatial_dependence_weight_grid must contain values in [0, 1]")
    for name, model in (
        ("RF-FRC independent-SVGP", functional_independent),
        ("ProfileField RF-FRC LMC-SVGP", functional_lmc),
    ):
        if len(validation_selection_indices):
            validation_context = forest.predict(features[validation_selection_indices]).astype(
                np.float64
            )
            validation_draw_count = min(
                sample_count, int(config["evaluation"].get("validation_posterior_samples", 64))
            )
            validation_scores = model.predict(
                dataset.coordinates[validation_selection_indices],
                posterior_samples=validation_draw_count,
                full_covariance=False,
            )
            validation_full_mean = functional_basis.decode(
                validation_scores.mean, validation_context, enforce_monotonicity=False
            )
            residual_mean_weights[name] = select_residual_mean_weight(
                validation_context,
                validation_full_mean - validation_context,
                dataset.profiles[validation_selection_indices],
                tuple(
                    float(value)
                    for value in model_config.get(
                        "residual_mean_weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0]
                    )
                ),
            )
            validation_rng = np.random.default_rng(seed + 8001)
            iid_score_draws = (
                validation_rng.standard_normal(
                    (validation_draw_count, len(validation_selection_indices), functional_dim)
                )
                @ score_cholesky.T
            )
            reconstruction_nugget = functional_basis.sample_reconstruction_nugget(
                validation_draw_count, len(validation_selection_indices), validation_rng
            )
            centered_gp_scores = validation_scores.samples - validation_scores.mean[None, :, :]
            target_score_scale = np.sqrt(np.maximum(np.diag(functional_score_covariance), 1e-10))
            predictive_score_scale = np.sqrt(
                np.maximum(np.mean(centered_gp_scores**2, axis=(0, 1)), 1e-10)
            )
            spatial_score_scales[name] = target_score_scale / predictive_score_scale
            centered_gp_scores = centered_gp_scores * spatial_score_scales[name][None, None, :]
            shrunk_mean = validation_context + residual_mean_weights[name] * (
                validation_full_mean - validation_context
            )
            validation_selected_rh = np.unique(
                np.linspace(
                    0,
                    dataset.profiles.shape[1] - 1,
                    min(5, dataset.profiles.shape[1]),
                )
                .round()
                .astype(int)
            )
            energy_scale = max(
                float(
                    np.mean(np.linalg.norm(dataset.profiles[validation_selection_indices], axis=1))
                ),
                1e-8,
            )
            candidates: list[dict[str, float]] = []
            for dependence_weight in dependence_weight_grid:
                combined_scores = (
                    np.sqrt(dependence_weight) * centered_gp_scores
                    + np.sqrt(1.0 - dependence_weight) * iid_score_draws
                )
                score_profile_deviation = combined_scores @ functional_basis.basis_.T  # type: ignore[union-attr]
                candidate_samples = project_monotone_profiles(
                    shrunk_mean[None, :, :] + score_profile_deviation + reconstruction_nugget
                )
                spatial_errors = posterior_cross_variogram_error(
                    dataset.coordinates[validation_selection_indices],
                    dataset.profiles[validation_selection_indices][:, validation_selected_rh],
                    candidate_samples[:, :, validation_selected_rh],
                    int(config["evaluation"].get("variogram_bins", 10)),
                    seed=seed,
                    max_pairs=int(config["evaluation"].get("validation_variogram_pairs", 20_000)),
                    max_draws=min(16, validation_draw_count),
                )
                candidate_energy = energy_score(
                    dataset.profiles[validation_selection_indices], candidate_samples
                )
                candidate_profile_variogram = profile_variogram_score(
                    dataset.profiles[validation_selection_indices],
                    candidate_samples,
                    validation_selected_rh,
                )
                objective = (
                    spatial_errors["posterior_variogram_mismatch"]
                    + spatial_errors["posterior_cross_variogram_mismatch"]
                    + candidate_profile_variogram
                    + candidate_energy / energy_scale
                )
                candidates.append(
                    {
                        "weight": dependence_weight,
                        "objective": objective,
                        "energy_score": candidate_energy,
                        "profile_variogram_score": candidate_profile_variogram,
                        **spatial_errors,
                    }
                )
            validation_weight_diagnostics[name] = candidates
            spatial_dependence_weights[name] = min(candidates, key=lambda row: row["objective"])[
                "weight"
            ]
        else:
            residual_mean_weights[name] = 1.0
            spatial_dependence_weights[name] = 1.0
            spatial_score_scales[name] = np.ones(functional_dim, dtype=np.float64)
            validation_weight_diagnostics[name] = []

    functional_independent_rng = np.random.default_rng(seed + 7001)
    functional_lmc_rng = np.random.default_rng(seed + 7001)

    def functional_spatial_distribution(
        indices: NDArray[np.int64],
        model: IndependentSVGP | LMCSVGP,
        model_name: str,
        nugget_rng: np.random.Generator,
    ) -> ProfileDistribution:
        context_mean = forest.predict(features[indices]).astype(np.float64)
        score_prediction = model.predict(
            dataset.coordinates[indices], posterior_samples=sample_count, full_covariance=False
        )
        full_mean = functional_basis.decode(
            score_prediction.mean, context_mean, enforce_monotonicity=False
        )
        mean_weight = residual_mean_weights[model_name]
        dependence_weight = spatial_dependence_weights[model_name]
        shrunk_mean = context_mean + mean_weight * (full_mean - context_mean)
        centered_gp_scores = (
            score_prediction.samples - score_prediction.mean[None, :, :]
        ) * spatial_score_scales[model_name][None, None, :]
        iid_score_draws = (
            nugget_rng.standard_normal((sample_count, len(indices), functional_dim))
            @ score_cholesky.T
        )
        combined_scores = (
            np.sqrt(dependence_weight) * centered_gp_scores
            + np.sqrt(1.0 - dependence_weight) * iid_score_draws
        )
        if functional_basis.basis_ is None:
            raise RuntimeError("Functional basis unexpectedly lost its fitted matrix")
        centered_spatial_draws = combined_scores @ functional_basis.basis_.T
        nugget = functional_basis.sample_reconstruction_nugget(
            sample_count, len(indices), nugget_rng
        )
        samples = project_monotone_profiles(
            shrunk_mean[None, :, :] + centered_spatial_draws + nugget
        )
        mean = project_monotone_profiles(shrunk_mean)
        return ProfileDistribution(
            mean=mean.astype(np.float64),
            variance=np.var(samples, axis=0) + 1e-8,
            samples=samples,
        )

    predictors.extend(
        [
            ProfilePredictor(
                "RF-FRC independent-SVGP",
                lambda indices: functional_spatial_distribution(
                    indices,
                    functional_independent,
                    "RF-FRC independent-SVGP",
                    functional_independent_rng,
                ),
            ),
            ProfilePredictor(
                "ProfileField RF-FRC LMC-SVGP",
                lambda indices: functional_spatial_distribution(
                    indices,
                    functional_lmc,
                    "ProfileField RF-FRC LMC-SVGP",
                    functional_lmc_rng,
                ),
            ),
        ]
    )

    profile_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    test_distributions: dict[str, ProfileDistribution] = {}
    selected_rh = np.unique(
        np.linspace(0, dataset.profiles.shape[1] - 1, min(5, dataset.profiles.shape[1]))
        .round()
        .astype(int)
    )
    selected_joint_rh = np.unique(
        np.linspace(0, dataset.profiles.shape[1] - 1, min(11, dataset.profiles.shape[1]))
        .round()
        .astype(int)
    )
    truth_variograms = empirical_cross_variograms(
        dataset.coordinates[split.test],
        dataset.profiles[split.test][:, selected_rh],
        int(config["evaluation"].get("variogram_bins", 10)),
    )
    estimated_variograms: dict[str, list[dict[str, Any]]] = {}
    derived_thresholds: dict[str, float] = {}
    calibration_scales: dict[str, float] = {}
    for predictor in predictors:
        calibration_distribution = (
            predictor.predict(split.calibration) if len(split.calibration) else None
        )
        distribution = predictor.predict(split.test)
        if calibration_distribution is not None:
            calibrator = JointVarianceScaleCalibrator().fit(
                dataset.profiles[split.calibration],
                calibration_distribution.mean,
                calibration_distribution.variance,
                [dataset.sample_ids[index] for index in split.calibration],
            )
            calibrator.assert_isolated_from({dataset.sample_ids[index] for index in split.test})
            scale = calibrator.scale_
            if scale is None:
                raise AssertionError("Calibration scale was not fit")
            calibration_scales[predictor.name] = scale
            calibrated_samples = distribution.mean[None, :, :] + (
                distribution.samples - distribution.mean[None, :, :]
            ) * np.sqrt(scale)
            # Calibration can otherwise destroy ordered-profile physics because
            # each RH ordinate has a separate scale. Re-project every draw and
            # recompute its actual post-projection variance.
            samples = project_monotone_profiles(calibrated_samples)
            mean = np.mean(samples, axis=0)
            distribution = ProfileDistribution(mean, np.var(samples, axis=0) + 1e-8, samples)
        test_distributions[predictor.name] = distribution
        profile_rows.extend(
            profile_metric_rows(
                predictor.name,
                dataset.profiles[split.test],
                distribution.mean,
                dataset.rh_percentiles,
            )
        )
        profile_rows.extend(
            [
                {
                    "model": predictor.name,
                    "scope": "all",
                    "metric": "profile_nll",
                    "value": gaussian_nll(
                        dataset.profiles[split.test], distribution.mean, distribution.variance
                    ),
                },
                {
                    "model": predictor.name,
                    "scope": "all",
                    "metric": "profile_crps",
                    "value": float(
                        gaussian_crps(
                            dataset.profiles[split.test], distribution.mean, distribution.variance
                        ).mean()
                    ),
                },
                {
                    "model": predictor.name,
                    "scope": "all",
                    "metric": "profile_energy_score",
                    "value": energy_score(dataset.profiles[split.test], distribution.samples),
                },
                {
                    "model": predictor.name,
                    "scope": "selected_rh_joint",
                    "metric": "profile_variogram_score",
                    "value": profile_variogram_score(
                        dataset.profiles[split.test],
                        distribution.samples,
                        selected_joint_rh,
                    ),
                },
                {
                    "model": predictor.name,
                    "scope": "all_posterior_draws",
                    "metric": "posterior_monotonicity_violation_percentage",
                    "value": float(
                        100.0
                        * np.count_nonzero(np.diff(distribution.samples, axis=-1) < -1e-7)
                        / max(np.diff(distribution.samples, axis=-1).size, 1)
                    ),
                },
                {
                    "model": predictor.name,
                    "scope": "all_posterior_draws",
                    "metric": "posterior_negative_value_percentage",
                    "value": float(100.0 * np.mean(distribution.samples < -1e-7)),
                },
            ]
        )
        profile_rows.extend(
            posterior_profile_covariance_rows(
                predictor.name,
                dataset.profiles[split.test],
                distribution.samples,
                selected_joint_rh,
            )
        )
        derived_rows, current_thresholds = derived_profile_rows(
            predictor.name,
            y_train,
            dataset.profiles[split.test],
            distribution.samples,
            dataset.rh_percentiles,
        )
        profile_rows.extend(derived_rows)
        if derived_thresholds and current_thresholds != derived_thresholds:
            raise AssertionError("Train-derived event thresholds changed between models")
        derived_thresholds = current_thresholds
        block_rows.extend(
            per_block_profile_rows(
                predictor.name,
                dataset.profiles[split.test],
                distribution.mean,
                split.block_ids[split.test],
            )
        )
        per_sample_mse = np.mean((distribution.mean - dataset.profiles[split.test]) ** 2, axis=1)
        estimate, lower, upper = block_bootstrap_interval(
            per_sample_mse,
            split.block_ids[split.test],
            statistic=lambda value: float(np.sqrt(np.mean(value))),
            replicates=int(config["evaluation"].get("bootstrap_replicates", 2000)),
            seed=seed,
        )
        profile_rows.extend(
            {
                "model": predictor.name,
                "scope": "all",
                "metric": metric,
                "value": value,
            }
            for metric, value in (
                ("profile_rmse_block_estimate", estimate),
                ("profile_rmse_block_ci_lower", lower),
                ("profile_rmse_block_ci_upper", upper),
            )
        )
        coverage.extend(
            calibration_rows(
                predictor.name,
                dataset.profiles[split.test],
                distribution.mean,
                distribution.variance,
            )
        )
        variograms = empirical_cross_variograms(
            dataset.coordinates[split.test],
            distribution.mean[:, selected_rh],
            int(config["evaluation"].get("variogram_bins", 10)),
        )
        estimated_variograms[predictor.name] = variograms
        mismatch = cross_variogram_error(truth_variograms, variograms)
        mismatch.update(
            posterior_cross_variogram_error(
                dataset.coordinates[split.test],
                dataset.profiles[split.test][:, selected_rh],
                distribution.samples[:, :, selected_rh],
                int(config["evaluation"].get("variogram_bins", 10)),
                seed=seed,
                max_pairs=int(config["evaluation"].get("posterior_variogram_pairs", 50_000)),
                max_draws=int(config["evaluation"].get("posterior_variogram_draws", 16)),
            )
        )
        profile_rows.extend(
            {"model": predictor.name, "scope": "all", "metric": key, "value": value}
            for key, value in mismatch.items()
        )
    profile_rows.extend(runtime_rows)

    with torch.no_grad():
        test_residual = (
            vae.encode_residual(
                context_all[split.test],
                torch.as_tensor(dataset.profiles[split.test], dtype=dtype),
            )
            .cpu()
            .numpy()
        )
    true_residual_covariance = np.cov(train_residual, rowvar=False)
    latent_rows: list[dict[str, Any]] = []
    covariance_estimates: dict[str, FloatArray] = {}
    for name, model in (
        ("GeoVAE independent-SVGP", independent),
        ("GeoVAE-Field LMC-SVGP", lmc),
    ):
        prediction = model.predict(
            dataset.coordinates[split.test], posterior_samples=sample_count, full_covariance=False
        )
        covariance_estimates[name] = prediction.output_covariance
        latent_rows.extend(
            latent_metric_rows(
                name,
                test_residual,
                prediction.mean,
                prediction.variance,
                prediction.samples,
                prediction.output_covariance,
                true_residual_covariance,
            )
        )

    forest_test_prediction = forest.predict(features[split.test])
    functional_test_scores = functional_basis.encode(
        dataset.profiles[split.test], forest_test_prediction
    )
    functional_true_covariance = np.cov(functional_train_scores, rowvar=False)
    functional_covariance_estimates: dict[str, FloatArray] = {}
    for name, model in (
        ("RF-FRC independent-SVGP", functional_independent),
        ("ProfileField RF-FRC LMC-SVGP", functional_lmc),
    ):
        prediction = model.predict(
            dataset.coordinates[split.test], posterior_samples=sample_count, full_covariance=False
        )
        functional_covariance_estimates[name] = prediction.output_covariance
        latent_rows.extend(
            latent_metric_rows(
                name,
                functional_test_scores,
                prediction.mean,
                prediction.variance,
                prediction.samples,
                prediction.output_covariance,
                functional_true_covariance,
            )
        )

    artifacts.write_table("metrics_profile", profile_rows)
    artifacts.write_table("metrics_blocks", block_rows)
    artifacts.write_table("metrics_latent", latent_rows)
    artifacts.write_table("calibration", coverage)
    artifacts.write_json(
        "diagnostics/preprocessing.json",
        {
            "feature_mean": feature_scaler.mean_,
            "feature_scale": feature_scaler.scale_,
            "fit_sample_ids": feature_scaler.fit_sample_ids_,
            "normalization_fit_on_training_only": True,
        },
    )
    artifacts.write_json(
        "diagnostics/functional_residual.json",
        {
            "fit_sample_ids": functional_basis.fit_sample_ids_,
            "fit_on_training_only": True,
            "context_model": "Per-RH random forest",
            "components": functional_dim,
            "basis_kind": "fixed_local_linear_hat_functions",
            "component_score_variance_ratio": functional_basis.score_variance_ratio_,
            "reconstruction_variance_explained": (
                functional_basis.reconstruction_variance_explained_
            ),
            "reconstruction_nugget_trace": float(
                np.trace(functional_basis.reconstruction_residual_covariance_)
                if functional_basis.reconstruction_residual_covariance_ is not None
                else float("nan")
            ),
            "residual_mean_weights_selected_on_validation": residual_mean_weights,
            "residual_mean_weight_grid": model_config.get(
                "residual_mean_weight_grid", [0.0, 0.25, 0.5, 0.75, 1.0]
            ),
            "residual_mean_weight_selection_used_test": False,
            "spatial_dependence_weights_selected_on_validation": spatial_dependence_weights,
            "spatial_score_marginal_scales_fit_on_validation": spatial_score_scales,
            "spatial_score_scale_target": "train empirical functional-score standard deviation",
            "spatial_score_scale_used_test": False,
            "spatial_dependence_weight_grid": dependence_weight_grid,
            "spatial_dependence_validation_objective": (
                "posterior variogram mismatch + posterior cross-variogram mismatch + "
                "profile variogram score + normalized energy score"
            ),
            "spatial_dependence_validation_diagnostics": validation_weight_diagnostics,
            "spatial_dependence_weight_selection_used_test": False,
            "validation_selection_full_count": len(validation_indices),
            "validation_selection_objective_count": len(validation_selection_indices),
            "validation_selection_sample_ids": [
                dataset.sample_ids[index] for index in validation_selection_indices
            ],
            "event_thresholds_fit_on_training_only": derived_thresholds,
            "joint_calibration_scales": calibration_scales,
            "joint_calibration_fit_on_isolated_partition": len(split.calibration) > 0,
        },
    )
    artifacts.write_json(
        "diagnostics/fairness.json",
        {
            "same_contextual_vae_for_spatial_ablations": True,
            "same_residual_latent_observations": True,
            "same_random_forest_context_for_functional_ablations": True,
            "same_train_only_functional_basis_for_functional_ablations": True,
            "same_functional_residual_observations": True,
            "same_inducing_initialization_rule": "deterministic_farthest_point",
            "test_used_for_selection": False,
            "calibration_isolated": len(split.calibration) > 0,
            "spatial_split_strategy": split.strategy,
        },
    )
    pd.DataFrame(truth_variograms).assign(model="Truth").to_csv(
        artifacts.path / "diagnostics/profile_variograms_truth.csv", index=False
    )
    pd.concat(
        [pd.DataFrame(rows).assign(model=name) for name, rows in estimated_variograms.items()],
        ignore_index=True,
    ).to_csv(artifacts.path / "diagnostics/profile_variograms_models.csv", index=False)
    covariance_figure(
        true_residual_covariance,
        covariance_estimates,
        artifacts.path / "figures",
        stem="geovae_latent_correlation",
    )
    covariance_figure(
        functional_true_covariance,
        functional_covariance_estimates,
        artifacts.path / "figures",
        stem="functional_residual_correlation",
    )
    calibration_figure(coverage, artifacts.path / "figures")
    variogram_figure(truth_variograms, estimated_variograms, artifacts.path / "figures")
    artifacts.finalize(
        time.perf_counter() - started,
        {
            "model_suite": [predictor.name for predictor in predictors],
            "train_count": len(split.train),
            "validation_count": len(split.validation),
            "calibration_count": len(split.calibration),
            "test_count": len(split.test),
        },
    )
    return artifacts.path


def _residual_covariance(truth: FloatArray, prediction: FloatArray) -> FloatArray:
    residual = np.asarray(truth) - np.asarray(prediction)
    covariance = np.atleast_2d(np.cov(residual, rowvar=False))
    return covariance + np.eye(covariance.shape[0]) * 1e-6


def _regression_distribution(
    mean: FloatArray,
    residual_covariance: FloatArray,
    sample_count: int,
    rng: np.random.Generator,
    enforce_monotonicity: bool,
) -> ProfileDistribution:
    mean = np.asarray(mean, dtype=np.float64)
    cholesky = np.linalg.cholesky(residual_covariance)
    noise = rng.standard_normal((sample_count, len(mean), mean.shape[1])) @ cholesky.T
    samples = mean[None, :, :] + noise
    output_mean = mean
    if enforce_monotonicity:
        samples = np.maximum.accumulate(np.maximum(samples, 0.0), axis=-1)
        output_mean = np.maximum.accumulate(np.maximum(mean, 0.0), axis=-1)
    return ProfileDistribution(
        mean=output_mean,
        variance=np.var(samples, axis=0) + 1e-8,
        samples=samples,
    )


def _cvae_distribution(
    model: ContextualProfileVAE, context: torch.Tensor, sample_count: int
) -> ProfileDistribution:
    model.eval()
    with torch.no_grad():
        context_mean = model.context_encoder(context)
        latent = context_mean.unsqueeze(0) + torch.randn(
            sample_count,
            len(context),
            context_mean.shape[1],
            dtype=context.dtype,
            device=context.device,
        )
        samples = model.decoder(latent.reshape(-1, context_mean.shape[1])).reshape(
            sample_count, len(context), -1
        )
        mean = model.predict_context_only(context)
    values = samples.cpu().numpy().astype(np.float64)
    return ProfileDistribution(
        mean=mean.cpu().numpy().astype(np.float64),
        variance=np.var(values, axis=0) + 1e-8,
        samples=values,
    )


def _torch_predict(model: torch.nn.Module, values: torch.Tensor) -> FloatArray:
    model.eval()
    with torch.no_grad():
        return model(values).cpu().numpy().astype(np.float64)


def _save_history(path: Path, history: list[float]) -> None:
    pd.DataFrame({"step": np.arange(len(history)), "loss": history}).to_csv(path, index=False)


def _runtime_row(model: str, seconds: float) -> dict[str, Any]:
    return {"model": model, "scope": "all", "metric": "fit_seconds", "value": seconds}
