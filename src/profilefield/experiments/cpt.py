"""Immutable baseline audit and operational CPT LMC continuity experiment."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from profilefield.artifacts import RunArtifacts
from profilefield.baselines.geovae_sgs import published_metric_rows, verify_baseline
from profilefield.config import Config
from profilefield.data.cpt import CPTAdapter
from profilefield.evaluation.bootstrap import block_bootstrap_interval
from profilefield.evaluation.metrics import (
    calibration_rows,
    energy_score,
    gaussian_crps,
    latent_metric_rows,
)
from profilefield.evaluation.profiles import per_block_profile_rows, profile_metric_rows
from profilefield.evaluation.spatial import (
    cross_variogram_error,
    empirical_cross_variograms,
)
from profilefield.reproducibility import set_seed, sha256_file
from profilefield.spatial.common import SpatialPrediction
from profilefield.spatial.independent_gp import IndependentSVGP
from profilefield.spatial.lmc_svgp import LMCSVGP


def run_cpt_audit(config: Config, project_root: str | Path = ".") -> Path:
    started = time.perf_counter()
    root = Path(project_root).resolve()
    seed = int(config["experiment"]["seed"])
    baseline = verify_baseline(root, config["baseline"]["manifest"])
    repository = root / config["baseline"]["repository"]
    pairs_path = repository / config["baseline"]["published_pairs"]
    rounds_path = repository / "validations/reviewer_analysis/geovae_gp_reviewer_rounds.csv"
    calibration_path = repository / "validations/reviewer_analysis/kc_calibration_splits.json"
    rounds = pd.read_csv(rounds_path)
    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration_manifest = json.load(handle)
    outer_ids = {
        item.strip() for value in rounds["holdout_ids"].astype(str) for item in value.split(",")
    }
    calibration_ids = set(calibration_manifest["selected_ids"])
    if outer_ids & calibration_ids:
        raise AssertionError("Published calibration and outer test IDs overlap")
    data_manifest = {
        "source": "GeoVAE-SGS Sample_data",
        "baseline": baseline,
        "pairs_sha256": sha256_file(pairs_path),
        "rounds_sha256": sha256_file(rounds_path),
        "n_profiles": 59,
        "n_layers": 5,
        "policy": "read_only",
    }
    split_manifest = {
        "strategy": "published_outer_rounds",
        "rounds": rounds[["round", "holdout_ids"]].to_dict(orient="records"),
        "outer_unique_test_ids": sorted(outer_ids),
        "calibration_ids": sorted(calibration_ids),
        "calibration_isolated": True,
    }
    artifacts = RunArtifacts.create(
        root / config["experiment"]["output_root"],
        f"{config['experiment']['name']}_audit",
        seed,
    )
    artifacts.initialize(config, data_manifest, split_manifest, root, seed)
    profile_rows, coverage_rows = published_metric_rows(pairs_path)
    artifacts.write_table("metrics_profile", profile_rows)
    round_rows: list[dict[str, Any]] = []
    pairs = pd.read_csv(pairs_path)
    for (method, round_id), group in pairs.groupby(["method", "round"], sort=True):
        truth, prediction = group["ln_qc_obs"], group["ln_qc_pred"]
        valid = truth.notna() & prediction.notna()
        error = prediction[valid].to_numpy(float) - truth[valid].to_numpy(float)
        for metric, value in (
            ("mae", np.mean(np.abs(error))),
            ("rmse", np.sqrt(np.mean(error**2))),
            ("bias", np.mean(error)),
        ):
            round_rows.append(
                {
                    "model": method,
                    "block_id": int(str(round_id)),
                    "metric": metric,
                    "value": float(value),
                    "n_samples": len(error),
                }
            )
    artifacts.write_table("metrics_blocks", round_rows)
    latent_summary_path = (
        repository / "validations/reviewer_analysis/latent_correlation_summary.csv"
    )
    latent_summary = pd.read_csv(latent_summary_path)
    latent_rows = [
        {
            "model": "Published VAE encoder diagnostic",
            "scope": f"layer_{int(float(str(row.layer_id)))}",
            "metric": column,
            "value": float(getattr(row, column)),
        }
        for row in latent_summary.itertuples(index=False)
        for column in (
            "mean_abs_offdiag_r",
            "median_abs_offdiag_r",
            "max_abs_offdiag_r",
        )
    ]
    artifacts.write_table("metrics_latent", latent_rows)
    artifacts.write_table("calibration", coverage_rows)
    shutil.copy2(latent_summary_path, artifacts.path / "diagnostics" / latent_summary_path.name)
    matrix_path = (
        repository / "validations/reviewer_analysis/latent_correlation_matrix_by_layer.csv"
    )
    shutil.copy2(matrix_path, artifacts.path / "diagnostics" / matrix_path.name)
    source_figure = (
        repository / "validations/reviewer_analysis/figures/latent_correlation_heatmaps.png"
    )
    if source_figure.exists():
        shutil.copy2(source_figure, artifacts.path / "figures" / source_figure.name)
    artifacts.finalize(
        time.perf_counter() - started,
        {
            "baseline_verified": True,
            "calibration_isolated": True,
            "note": "Metrics were recomputed from immutable published pair artifacts.",
        },
    )
    return artifacts.path


def run_cpt_comparison(config: Config, project_root: str | Path = ".") -> list[Path]:
    """Retrain the published VAE fold-wise and replace only its spatial latent model."""
    root = Path(project_root).resolve()
    verify_baseline(root, config["baseline"]["manifest"])
    seeds = config["experiment"].get("seeds", [config["experiment"].get("seed", 2026)])
    return [_run_cpt_seed(config, int(seed), root) for seed in seeds]


def _run_cpt_seed(config: Config, seed: int, root: Path) -> Path:
    started = time.perf_counter()
    set_seed(seed)
    baseline_config = config.get("baseline_model", {})
    profile_points = int(baseline_config.get("profile_points", 174))
    latent_dim = int(baseline_config.get("latent_dim", 9))
    adapter = CPTAdapter(root, profile_points=profile_points)
    adapter.core.SEED = seed
    adapter.core.GP_NOISE_LEVEL = float(
        baseline_config.get("gp_noise_level", 0.0004413460154094174)
    )
    adapter.core.SPATIAL_MAX_RESTARTS = int(baseline_config.get("spatial_max_restarts", 8))
    adapter.core.SPATIAL_CV_FOLDS = int(baseline_config.get("spatial_cv_folds", 4))
    adapter.core.GP_N_RESTARTS = int(baseline_config.get("gp_n_restarts", 8))
    adapter.core.SPATIAL_KERNELS = tuple(
        baseline_config.get("spatial_kernels", ("matern05", "matern15", "matern25", "rbf"))
    )
    rounds = adapter.published_rounds()
    fold_limit = int(config["experiment"].get("fold_limit", len(rounds)))
    rounds = rounds[:fold_limit]
    split_manifest = {
        "strategy": "published_GeoVAE-SGS_outer_rounds",
        "rounds": [{"round": number, "test_ids": list(ids)} for number, ids in rounds],
        "test_used_for_selection": False,
    }
    data_manifest = {
        "source": "GeoVAE-SGS Sample_data",
        "baseline": verify_baseline(root, config["baseline"]["manifest"]),
        "preprocessing": "GeoVAE-SGS layer segmentation, resampling, train-only NormalScore",
        "profile_points": profile_points,
    }
    artifacts = RunArtifacts.create(
        root / config["experiment"]["output_root"], config["experiment"]["name"], seed
    )
    artifacts.initialize(config, data_manifest, split_manifest, root, seed)
    profile_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    variogram_rows: list[dict[str, Any]] = []
    model_config = config["model"]
    posterior_samples = int(config["evaluation"]["posterior_samples"])
    for round_id, holdout_ids in rounds:
        fold_started = time.perf_counter()
        folds = adapter.prepare_fold(holdout_ids)
        vaes = adapter.train_baseline_vaes(
            folds,
            latent_dim=latent_dim,
            hidden=int(baseline_config.get("hidden", 293)),
            epochs=int(baseline_config.get("epochs", 300)),
            kl_weight=float(baseline_config.get("kl_weight", 0.062352834818264306)),
        )
        for layer, fold in folds.items():
            if layer not in vaes:
                continue
            vae = vaes[layer]
            train_latent = adapter.encode(vae, fold.train_profiles_ns)
            test_latent = adapter.encode(vae, fold.test_profiles_ns)
            empirical_covariance = np.cov(train_latent, rowvar=False)
            independent = _fit_published_independent(
                adapter,
                train_latent,
                fold.train_coordinates,
                fold.test_coordinates,
                posterior_samples,
                seed + round_id * 100 + layer,
                artifacts.path
                / "models"
                / f"round{round_id:02d}_layer{layer}_published_independent.joblib",
            )
            matched_independent = IndependentSVGP(
                output_dim=latent_dim,
                inducing_count=int(model_config["inducing_points"]),
                nu=float(model_config["kernel_nu"]),
                learning_rate=float(model_config["learning_rate"]),
                training_steps=int(model_config["training_steps"]),
                seed=seed + round_id * 100 + layer,
            ).fit(fold.train_coordinates, train_latent, fold.train_ids)
            lmc = LMCSVGP(
                output_dim=latent_dim,
                latent_count=int(model_config["lmc_latents"]),
                inducing_count=int(model_config["inducing_points"]),
                nu=float(model_config["kernel_nu"]),
                learning_rate=float(model_config["learning_rate"]),
                training_steps=int(model_config["training_steps"]),
                seed=seed + round_id * 100 + layer,
            ).fit(fold.train_coordinates, train_latent, fold.train_ids)
            predictions = {
                "Published independent GP": independent,
                "Matched independent-SVGP": matched_independent.predict(
                    fold.test_coordinates,
                    posterior_samples=posterior_samples,
                    full_covariance=True,
                ),
                "LMC-SVGP": lmc.predict(
                    fold.test_coordinates,
                    posterior_samples=posterior_samples,
                    full_covariance=True,
                ),
            }
            model_path = artifacts.path / "models" / f"round{round_id:02d}_layer{layer}_lmc.pt"
            torch.save(lmc.checkpoint(), model_path)
            torch.save(
                matched_independent.checkpoint(),
                artifacts.path
                / "models"
                / f"round{round_id:02d}_layer{layer}_matched_independent_svgp.pt",
            )
            torch.save(
                {
                    "state_dict": vae.state_dict(),
                    "profile_points": profile_points,
                    "latent_dim": latent_dim,
                },
                artifacts.path / "models" / f"round{round_id:02d}_layer{layer}_vae.pt",
            )
            for name, prediction in predictions.items():
                predicted_profile = adapter.decode(vae, prediction.mean, fold.normal_score)
                model_label = f"{name}"
                for row in profile_metric_rows(
                    model_label,
                    fold.test_profiles_raw,
                    predicted_profile,
                    ordered=False,
                ):
                    row.update({"round": round_id, "layer": layer})
                    profile_rows.append(row)
                ids_to_block = {
                    value: index for index, value in enumerate(sorted(set(fold.test_ids)))
                }
                blocks = np.asarray(
                    [ids_to_block[value] for value in fold.test_ids], dtype=np.int64
                )
                for row in per_block_profile_rows(
                    model_label, fold.test_profiles_raw, predicted_profile, blocks
                ):
                    row.update({"round": round_id, "layer": layer})
                    block_rows.append(row)
                for row in latent_metric_rows(
                    model_label,
                    test_latent,
                    prediction.mean,
                    prediction.variance,
                    prediction.samples,
                    prediction.output_covariance,
                    empirical_covariance,
                ):
                    row.update({"round": round_id, "layer": layer})
                    latent_rows.append(row)
                for row in calibration_rows(
                    model_label, test_latent, prediction.mean, prediction.variance
                ):
                    row.update({"round": round_id, "layer": layer})
                    coverage_rows.append(row)
                truth_variogram = empirical_cross_variograms(
                    fold.test_coordinates, test_latent, n_bins=1
                )
                estimate_variogram = empirical_cross_variograms(
                    fold.test_coordinates, prediction.mean, n_bins=1
                )
                mismatch = cross_variogram_error(truth_variogram, estimate_variogram)
                latent_rows.extend(
                    {
                        "model": model_label,
                        "scope": "all",
                        "metric": metric,
                        "value": value,
                        "round": round_id,
                        "layer": layer,
                    }
                    for metric, value in mismatch.items()
                )
                variogram_rows.extend(
                    {
                        **row,
                        "model": model_label,
                        "kind": "estimate",
                        "round": round_id,
                        "layer": layer,
                    }
                    for row in estimate_variogram
                )
                variogram_rows.extend(
                    {
                        **row,
                        "model": model_label,
                        "kind": "truth",
                        "round": round_id,
                        "layer": layer,
                    }
                    for row in truth_variogram
                )
                per_profile_mse = np.mean((predicted_profile - fold.test_profiles_raw) ** 2, axis=1)
                estimate, lower, upper = block_bootstrap_interval(
                    per_profile_mse,
                    blocks,
                    statistic=lambda value: float(np.sqrt(np.mean(value))),
                    replicates=int(config["evaluation"].get("bootstrap_replicates", 2000)),
                    seed=seed + round_id * 100 + layer,
                )
                profile_rows.extend(
                    {
                        "model": model_label,
                        "scope": "all",
                        "metric": metric,
                        "value": value,
                        "round": round_id,
                        "layer": layer,
                    }
                    for metric, value in (
                        ("profile_rmse_block_estimate", estimate),
                        ("profile_rmse_block_ci_lower", lower),
                        ("profile_rmse_block_ci_upper", upper),
                    )
                )
                decoded_samples = np.stack(
                    [adapter.decode(vae, draw, fold.normal_score) for draw in prediction.samples]
                )
                profile_rows.extend(
                    [
                        {
                            "model": model_label,
                            "scope": "all",
                            "metric": "profile_crps",
                            "value": float(
                                gaussian_crps(
                                    fold.test_profiles_raw,
                                    decoded_samples.mean(axis=0),
                                    decoded_samples.var(axis=0) + 1e-8,
                                ).mean()
                            ),
                            "round": round_id,
                            "layer": layer,
                        },
                        {
                            "model": model_label,
                            "scope": "all",
                            "metric": "profile_energy_score",
                            "value": energy_score(fold.test_profiles_raw, decoded_samples),
                            "round": round_id,
                            "layer": layer,
                        },
                    ]
                )
        latent_rows.append(
            {
                "model": "both",
                "scope": "round",
                "metric": "wall_time_seconds",
                "value": time.perf_counter() - fold_started,
                "round": round_id,
                "layer": "all",
            }
        )
    artifacts.write_table("metrics_profile", profile_rows)
    artifacts.write_table("metrics_blocks", block_rows)
    artifacts.write_table("metrics_latent", latent_rows)
    artifacts.write_table("calibration", coverage_rows)
    pd.DataFrame(variogram_rows).to_csv(
        artifacts.path / "diagnostics/cross_variograms.csv", index=False
    )
    artifacts.write_json(
        "diagnostics/fairness.json",
        {
            "same_foldwise_VAE": True,
            "same_train_latent_observations": True,
            "matched_svgp_ablation": True,
            "matched_svgp_kernel_family": f"Matern-{model_config['kernel_nu']}",
            "matched_svgp_inducing_points": int(model_config["inducing_points"]),
            "matched_svgp_training_steps": int(model_config["training_steps"]),
            "published_outer_holdouts": True,
            "normal_score_fit_on_training_only": True,
            "test_used_for_selection": False,
            "primary_difference_under_test": (
                "matched independent-SVGP versus LMC-SVGP coregionalization"
            ),
            "continuity_reference": "published independent exact GP",
        },
    )
    artifacts.finalize(time.perf_counter() - started, {"completed_rounds": len(rounds)})
    return artifacts.path


def _fit_published_independent(
    adapter: CPTAdapter,
    train_latent: np.ndarray,
    train_coordinates: np.ndarray,
    test_coordinates: np.ndarray,
    posterior_samples: int,
    seed: int,
    checkpoint_path: Path,
) -> SpatialPrediction:
    set_seed(seed)
    fitted = adapter.core.fit_gp_models(
        {1: (train_latent, train_coordinates)}, n_restarts=adapter.core.GP_N_RESTARTS
    )[1]
    rotated = adapter.core._rotate_2d(test_coordinates, fitted["angle"])
    means = []
    variances = []
    draws = []
    for output, gp in enumerate(fitted["gps"]):
        mean, std = gp.predict(rotated, return_std=True)
        means.append(mean)
        variances.append(std**2)
        draws.append(gp.sample_y(rotated, n_samples=posterior_samples, random_state=seed + output))
    mean_array = np.column_stack(means)
    variance_array = np.column_stack(variances)
    sample_array = np.stack(draws, axis=-1).transpose(1, 0, 2)
    n, output_dim = mean_array.shape
    full = np.zeros((n * output_dim, n * output_dim), dtype=np.float64)
    for output, gp in enumerate(fitted["gps"]):
        covariance = gp.predict(rotated, return_cov=True)[1]
        positions = np.arange(output, n * output_dim, output_dim)
        full[np.ix_(positions, positions)] = covariance
    output_covariance = np.diag(np.var(train_latent, axis=0, ddof=1))
    # Keep the exact sklearn objects as continuity evidence rather than translating parameters.
    joblib.dump(fitted, checkpoint_path)
    return SpatialPrediction(
        mean=mean_array,
        variance=variance_array,
        samples=sample_array,
        full_covariance=full,
        output_covariance=output_covariance,
    )
