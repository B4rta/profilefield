"""End-to-end fair synthetic independent-SVGP versus LMC-SVGP experiment."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

from profilefield.artifacts import RunArtifacts
from profilefield.config import Config
from profilefield.data.splits import blocked_split
from profilefield.data.synthetic import generate_synthetic
from profilefield.evaluation.bootstrap import block_bootstrap_interval
from profilefield.evaluation.figures import (
    calibration_figure,
    covariance_figure,
    field_figure,
    variogram_figure,
)
from profilefield.evaluation.metrics import calibration_rows, latent_metric_rows
from profilefield.evaluation.profiles import per_block_profile_rows, profile_metric_rows
from profilefield.evaluation.spatial import (
    cross_variogram_error,
    empirical_cross_variograms,
    local_gradient_distribution,
)
from profilefield.reproducibility import set_seed
from profilefield.spatial.common import SpatialPrediction
from profilefield.spatial.independent_gp import IndependentSVGP
from profilefield.spatial.lmc_svgp import LMCSVGP


def run_synthetic(config: Config, project_root: str | Path = ".") -> list[Path]:
    """Run every configured seed and return completed artifact directories."""
    paths: list[Path] = []
    for seed_value in config["experiment"]["seeds"]:
        paths.append(_run_seed(config, int(seed_value), Path(project_root).resolve()))
    return paths


def _run_seed(config: Config, seed: int, project_root: Path) -> Path:
    started = time.perf_counter()
    set_seed(seed)
    dataset = generate_synthetic(config["data"], seed)
    split_config = config["split"]
    split = blocked_split(
        dataset.coordinates,
        dataset.sample_ids,
        seed=seed,
        n_blocks_x=int(split_config["n_blocks_x"]),
        n_blocks_y=int(split_config["n_blocks_y"]),
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
    )
    output_root = project_root / config["experiment"]["output_root"]
    artifacts = RunArtifacts.create(output_root, config["experiment"]["name"], seed)
    artifacts.initialize(config, dataset.manifest, split.to_manifest(), project_root, seed)
    model_config = config["model"]
    common = {
        "output_dim": dataset.latent_true.shape[1],
        "inducing_count": int(model_config["inducing_points"]),
        "nu": float(model_config["kernel_nu"]),
        "learning_rate": float(model_config["learning_rate"]),
        "training_steps": int(model_config["training_steps"]),
        "seed": seed,
    }
    models: dict[str, IndependentSVGP | LMCSVGP] = {
        "Independent-SVGP": IndependentSVGP(**common),
        "LMC-SVGP": LMCSVGP(latent_count=int(model_config["lmc_latents"]), **common),
    }
    train = split.train
    test = split.test
    train_ids = [dataset.sample_ids[index] for index in train]
    posterior_samples = int(config["evaluation"]["posterior_samples"])
    process = psutil.Process()
    predictions: dict[str, SpatialPrediction] = {}
    runtime_rows: list[dict[str, Any]] = []
    for name, model in models.items():
        before_rss = process.memory_info().rss
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tracemalloc.start()
        model_started = time.perf_counter()
        model.fit(dataset.coordinates[train], dataset.latent_observed[train], train_ids)
        fit_seconds = time.perf_counter() - model_started
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        after_rss = process.memory_info().rss
        prediction = model.predict(
            dataset.coordinates[test], posterior_samples=posterior_samples, full_covariance=True
        )
        predictions[name] = prediction
        torch.save(model.checkpoint(), artifacts.path / "models" / f"{_slug(name)}.pt")
        runtime_rows.extend(
            [
                {"model": name, "scope": "all", "metric": "fit_seconds", "value": fit_seconds},
                {
                    "model": name,
                    "scope": "all",
                    "metric": "peak_python_bytes",
                    "value": int(peak_python),
                },
                {
                    "model": name,
                    "scope": "all",
                    "metric": "rss_change_bytes",
                    "value": int(after_rss - before_rss),
                },
                {
                    "model": name,
                    "scope": "all",
                    "metric": "peak_gpu_bytes",
                    "value": int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else 0,
                },
            ]
        )
        pd.DataFrame(
            {"step": np.arange(len(model.history)), "negative_elbo": model.history}
        ).to_csv(artifacts.path / "diagnostics" / f"training_{_slug(name)}.csv", index=False)

    # The equality is explicit and recorded: neither branch receives different observations.
    for model in models.values():
        if model.transforms is None:
            raise AssertionError("Model did not retain preprocessing provenance")
        model.transforms.x.assert_fit_on(set(train_ids))
        model.transforms.y.assert_fit_on(set(train_ids))
        if set(model.transforms.y.fit_sample_ids_) != set(train_ids):
            raise AssertionError(
                "GP ablations did not use exactly the frozen training observations"
            )

    true_latent = dataset.latent_true[test]
    true_profiles = dataset.profiles_true[test]
    latent_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    all_calibration_rows: list[dict[str, Any]] = []
    truth_variograms = empirical_cross_variograms(
        dataset.coordinates[test], true_latent, int(config["evaluation"]["variogram_bins"])
    )
    estimated_variograms: dict[str, list[dict[str, Any]]] = {}
    covariance_estimates: dict[str, np.ndarray] = {}
    bootstrap_summary: list[dict[str, Any]] = []
    for name, prediction in predictions.items():
        covariance_estimates[name] = prediction.output_covariance
        latent_rows.extend(
            latent_metric_rows(
                name,
                true_latent,
                prediction.mean,
                prediction.variance,
                prediction.samples,
                prediction.output_covariance,
                dataset.coregionalization,
            )
        )
        predicted_profiles = dataset.decode(prediction.mean)
        profile_rows.extend(profile_metric_rows(name, true_profiles, predicted_profiles))
        block_rows.extend(
            per_block_profile_rows(name, true_profiles, predicted_profiles, split.block_ids[test])
        )
        all_calibration_rows.extend(
            calibration_rows(name, true_latent, prediction.mean, prediction.variance)
        )
        estimated = empirical_cross_variograms(
            dataset.coordinates[test],
            prediction.mean,
            int(config["evaluation"]["variogram_bins"]),
        )
        estimated_variograms[name] = estimated
        mismatch = cross_variogram_error(truth_variograms, estimated)
        latent_rows.extend(
            {"model": name, "scope": "all", "metric": metric, "value": value}
            for metric, value in mismatch.items()
        )
        gradient_truth = local_gradient_distribution(dataset.coordinates[test], true_latent)
        gradient_estimate = local_gradient_distribution(dataset.coordinates[test], prediction.mean)
        latent_rows.append(
            {
                "model": name,
                "scope": "all",
                "metric": "local_gradient_mean_error",
                "value": float(abs(gradient_estimate.mean() - gradient_truth.mean())),
            }
        )
        errors = np.mean((predicted_profiles - true_profiles) ** 2, axis=1)
        estimate, lower, upper = block_bootstrap_interval(
            errors,
            split.block_ids[test],
            statistic=lambda value: float(np.sqrt(np.mean(value))),
            replicates=int(config["evaluation"]["bootstrap_replicates"]),
            seed=seed,
        )
        bootstrap_summary.append(
            {
                "model": name,
                "metric": "profile_rmse",
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "unit": "geographic_block",
            }
        )

    latent_rows.extend(runtime_rows)
    artifacts.write_table("metrics_profile", profile_rows)
    artifacts.write_table("metrics_blocks", block_rows)
    artifacts.write_table("metrics_latent", latent_rows)
    artifacts.write_table("calibration", all_calibration_rows)
    pd.DataFrame(truth_variograms).assign(model="Truth").to_csv(
        artifacts.path / "diagnostics" / "cross_variograms_truth.csv", index=False
    )
    combined_variograms: list[dict[str, Any]] = []
    for name, rows in estimated_variograms.items():
        combined_variograms.extend({**row, "model": name} for row in rows)
    pd.DataFrame(combined_variograms).to_csv(
        artifacts.path / "diagnostics" / "cross_variograms_models.csv", index=False
    )
    artifacts.write_json(
        "diagnostics/covariance_matrices.json",
        {
            "truth": dataset.coregionalization,
            **{name: value for name, value in covariance_estimates.items()},
        },
    )
    artifacts.write_json("diagnostics/block_bootstrap.json", bootstrap_summary)
    artifacts.write_json(
        "diagnostics/fairness.json",
        {
            "same_training_sample_ids": True,
            "training_sample_ids": train_ids,
            "same_inducing_initialization_rule": "deterministic_farthest_point",
            "same_kernel_family": f"Matern-{model_config['kernel_nu']}",
            "same_training_steps": int(model_config["training_steps"]),
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
        },
    )
    covariance_figure(dataset.coregionalization, covariance_estimates, artifacts.path / "figures")
    calibration_figure(all_calibration_rows, artifacts.path / "figures")
    variogram_figure(truth_variograms, estimated_variograms, artifacts.path / "figures")
    field_figure(
        dataset.coordinates[test],
        true_latent,
        {name: prediction.mean for name, prediction in predictions.items()},
        predictions["LMC-SVGP"].samples[0],
        artifacts.path / "figures",
    )
    artifacts.finalize(
        time.perf_counter() - started,
        {
            "models": list(models),
            "train_count": len(train),
            "validation_count": len(split.validation),
            "test_count": len(test),
            "scientific_note": "Fixed hyperparameters; validation and test were not used for selection.",
        },
    )
    return artifacts.path


def _slug(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")
