"""Frozen-source external validation: no target-domain fitting or selection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from profilefield.artifacts import RunArtifacts, validate_run_directory
from profilefield.config import load_config
from profilefield.context.crossfit import spatial_crossfit_predictions
from profilefield.context.embeddings import select_embedding_columns
from profilefield.data.biomazon import load_biomazon
from profilefield.data.preprocessing import TrainOnlyStandardizer
from profilefield.evaluation.ensemble import empirical_crps, ensemble_calibration_rows
from profilefield.evaluation.joint import profile_variogram_score
from profilefield.evaluation.metrics import energy_score
from profilefield.evaluation.profiles import profile_metric_rows
from profilefield.experiments.biomazon import (
    _cvae_distribution,
    _regression_distribution,
    _residual_covariance,
    _torch_predict,
)
from profilefield.profiles.functional_residual import project_monotone_profiles
from profilefield.profiles.models import ContextualProfileVAE, DeterministicContextProfile
from profilefield.reproducibility import set_seed, sha256_file
from profilefield.spatial.restore import restore_spatial

RF = "Per-RH random forest"
IND = "RF-FRC independent-SVGP"
LMC = "ProfileField RF-FRC LMC-SVGP"


def select_external_cases(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    """Selection is identity-only, invariant to input row order and responses."""
    if maximum < 1 or frame.sample_id.duplicated().any():
        raise ValueError("Positive maximum and unique external sample IDs required")
    ordered = frame.sort_values("sample_id").reset_index(drop=True)
    if len(ordered) > maximum:
        indices = np.sort(np.random.default_rng(seed).choice(len(ordered), maximum, replace=False))
        ordered = ordered.iloc[indices]
    return ordered.reset_index(drop=True)


def minimum_distance_km(source: np.ndarray, target: np.ndarray) -> float:
    """Minimum great-circle distance between lon/lat sets; bounded memory."""
    source_rad = np.deg2rad(source)
    minimum = float("inf")
    for start in range(0, len(target), 128):
        points = np.deg2rad(target[start:start + 128])
        delta = points[:, None, :] - source_rad[None, :, :]
        hav = np.sin(delta[:, :, 1] / 2) ** 2 + (
            np.cos(points[:, None, 1]) * np.cos(source_rad[None, :, 1])
            * np.sin(delta[:, :, 0] / 2) ** 2
        )
        minimum = min(minimum, float((12742.0 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))).min()))
    return minimum


def run_external(protocol_path: str | Path, root: str | Path = ".") -> list[Path]:
    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    validate_run_directory(protocol_path)
    protocol = json.loads((protocol_path / "protocol.json").read_text())
    config = load_config(protocol_path / "config.yaml")
    if sha256_file(root / config["external"]["region_spec"]) != protocol["region_sha256"]:
        raise ValueError("Region definition changed after protocol freeze")
    frame = pd.read_parquet(root / config["external"]["table"])
    frame = select_external_cases(frame, config["external"]["maximum_cases"], config["external"]["selection_seed"])
    if len(frame) < config["external"]["minimum_cases"]:
        raise ValueError("Insufficient matched external observations; do not choose by model scores")
    return [_run_source(source, frame, config, protocol_path, root) for source in protocol["source_runs"]]


def _run_source(source: dict[str, Any], frame: pd.DataFrame, config: dict[str, Any],
                protocol_path: Path, root: Path) -> Path:
    started = time.perf_counter()
    run, seed = Path(source["path"]), int(source["seed"])
    validate_run_directory(run)
    if sha256_file(run / "artifact_integrity.json") != source["seal_sha256"]:
        raise ValueError("Frozen source state was modified")
    set_seed(seed)
    source_config = load_config(run / "config.yaml")
    dataset = load_biomazon(source_config, root, seed)
    original_manifest = json.loads((run / "data_manifest.json").read_text())
    if dataset.manifest["table_sha256"] != original_manifest["table_sha256"]:
        raise ValueError("Source data changed")
    columns = tuple(select_embedding_columns(frame, source_config["data"]["feature_prefixes"]))
    if columns != dataset.feature_columns:
        raise ValueError("External predictor schema differs from source")
    if set(frame.sample_id) & set(dataset.sample_ids):
        raise ValueError("Source and external sample IDs overlap")
    for identity in ("shot_number", "agbd_sample_id"):
        if identity in frame and identity in dataset.frame:
            if set(frame[identity].astype(str)) & set(dataset.frame[identity].astype(str)):
                raise ValueError(f"Source and external {identity} overlap")
    coordinates = frame[[source_config["data"]["x_column"], source_config["data"]["y_column"]]].to_numpy(float)
    truth = frame[list(dataset.rh_columns)].to_numpy(float)
    raw_features = frame[list(columns)].to_numpy(float)
    if not all(np.isfinite(array).all() for array in (coordinates, truth, raw_features)):
        raise ValueError("Nonfinite external input")
    if np.any(np.diff(truth, axis=1) < -1e-6):
        raise ValueError("Unordered external RH profiles")
    distance = minimum_distance_km(dataset.coordinates, coordinates)
    if distance < config["external"]["minimum_source_distance_km"]:
        raise ValueError("External geography is not sufficiently separated")
    preprocessing = json.loads((run / "diagnostics/preprocessing.json").read_text())
    scaler = TrainOnlyStandardizer(np.asarray(preprocessing["feature_mean"]), np.asarray(preprocessing["feature_scale"]), tuple(preprocessing["fit_sample_ids"]))
    x, train = scaler.transform(raw_features), dataset.split.train
    x_train, y_train = scaler.transform(dataset.features[train]), dataset.profiles[train]
    train_ids = [dataset.sample_ids[i] for i in train]
    if set(train_ids) != set(scaler.fit_sample_ids_):
        raise ValueError("Source preprocessing identities differ")
    forest = joblib.load(run / "models/per_rh_random_forest.joblib")
    # Legacy states did not store residual covariance. Replay only the already
    # prescribed SOURCE folds to reconstruct it; production models stay frozen.
    crossfit, _ = spatial_crossfit_predictions(
        forest, x_train, y_train, dataset.split.block_ids[train], train_ids,
        folds=source_config["model"]["residual_crossfit_folds"],
    )
    old_residual = json.loads((run / "diagnostics/residual_crossfit.json").read_text())
    np.testing.assert_allclose(np.mean((y_train-crossfit)**2), old_residual["training_residual_mse"], rtol=1e-10)
    basis = joblib.load(run / "models/functional_residual_basis.joblib")
    np.testing.assert_allclose((y_train-crossfit).mean(0), basis.residual_mean_, rtol=1e-9, atol=1e-9)
    score_values = basis.encode(y_train, crossfit)
    score_cholesky = np.linalg.cholesky(_residual_covariance(score_values, np.zeros_like(score_values)))
    functional = json.loads((run / "diagnostics/functional_residual.json").read_text())
    model_config = source_config["model"]
    count = source_config["evaluation"]["posterior_samples"]
    lower_bound = model_config["profile_lower_bound"]
    context = torch.as_tensor(x, dtype=torch.float32)
    train_context = torch.as_tensor(x_train, dtype=torch.float32)
    dimensions = (x.shape[1], model_config["latent_dim"], truth.shape[1], model_config["hidden_dim"])
    deterministic = DeterministicContextProfile(*dimensions)
    deterministic.load_state_dict(torch.load(run / "models/deterministic_shared_decoder.pt", map_location="cpu", weights_only=True))
    vae = ContextualProfileVAE(*dimensions)
    vae.load_state_dict(torch.load(run / "models/contextual_vae.pt", map_location="cpu", weights_only=True))
    vae.eval()
    ridge = joblib.load(run / "models/per_rh_ridge.joblib")
    gp_files = {"GeoVAE independent-SVGP": "independent_svgp.pt", "GeoVAE-Field LMC-SVGP": "lmc_svgp.pt",
                IND: "functional_independent_svgp.pt", LMC: "functional_lmc_svgp.pt"}
    models = {name: restore_spatial(torch.load(run / "models" / filename, map_location="cpu", weights_only=False))
              for name, filename in gp_files.items()}
    artifact = RunArtifacts.create(root / config["experiment"]["output_root"], config["experiment"]["name"], seed)
    blocks = pd.factorize(frame.official_spatial_block.astype(str), sort=True)[0]
    artifact.initialize(config, {"external_table_sha256": sha256_file(root / config["external"]["table"]),
                                "source_run": source, "protocol_seal_sha256": sha256_file(protocol_path / "artifact_integrity.json"),
                                "feature_columns": columns},
                        {"train": [], "validation": [], "calibration": [], "test": [
                            {"sample_id": sid, "block_id": int(block)} for sid, block in zip(frame.sample_id, blocks, strict=True)]}, root, seed)
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    names = ["Per-RH ridge", RF, "Deterministic shared decoder", "Conditional VAE", *gp_files]
    for index, name in enumerate(names):
        # Stable model-local streams do not depend on other models' draw counts.
        set_seed(seed + 10000 + index)
        rng = np.random.default_rng(seed + 10000 + index)
        if name in ("Per-RH ridge", RF, "Deterministic shared decoder"):
            if name == RF:
                mean, covariance = forest.predict(x), _residual_covariance(y_train, crossfit)
            elif name == "Per-RH ridge":
                mean, covariance = ridge.predict(x), _residual_covariance(y_train, ridge.predict(x_train))
            else:
                mean = _torch_predict(deterministic, context)
                covariance = _residual_covariance(y_train, _torch_predict(deterministic, train_context))
            draws = _regression_distribution(mean, covariance, count, rng, name == "Deterministic shared decoder").samples
        elif name == "Conditional VAE":
            draws = _cvae_distribution(vae, context, count).samples
        else:
            prediction = models[name].predict(coordinates, count, full_covariance=False)
            if name in (IND, LMC):
                mean_context = forest.predict(x)
                full_mean = basis.decode(prediction.mean, mean_context, enforce_monotonicity=False)
                mean_weight = functional["residual_mean_weights_selected_on_validation"][name]
                weight = functional["spatial_dependence_weights_selected_on_validation"][name]
                scales = np.asarray(functional["spatial_score_marginal_scales_fit_on_validation"][name])
                centered = (prediction.samples-prediction.mean[None]) * scales[None, None]
                iid = rng.standard_normal(prediction.samples.shape) @ score_cholesky.T
                noise = basis.sample_reconstruction_nugget(count, len(frame), rng)
                draws = mean_context[None] + mean_weight*(full_mean-mean_context)[None] + (
                    np.sqrt(weight)*centered + np.sqrt(1-weight)*iid) @ basis.basis_.T + noise
            else:
                with torch.no_grad():
                    latent = prediction.samples + vae.context_encoder(context).numpy()[None]
                    draws = vae.decoder(torch.as_tensor(latent.reshape(-1, model_config["latent_dim"]), dtype=torch.float32)).numpy().reshape(count, len(frame), -1)
        draws = project_monotone_profiles(draws, lower_bound=lower_bound)
        mean = draws.mean(axis=0)
        scale = functional["joint_calibration_scales"][name]
        draws = project_monotone_profiles(mean[None] + (draws-mean[None])*np.sqrt(scale), lower_bound=lower_bound)
        mean = draws.mean(axis=0)
        crps = empirical_crps(truth, draws)
        mse = np.mean((mean-truth)**2, axis=1)
        rows.extend(profile_metric_rows(name, truth, mean, dataset.rh_percentiles))
        rows.extend({"model": name, "scope": "all", "metric": metric, "value": value} for metric, value in (
            ("profile_empirical_crps", float(crps.mean())), ("profile_energy_score", energy_score(truth, draws)),
            ("profile_variogram_score", profile_variogram_score(truth, draws, np.unique(np.linspace(0, truth.shape[1]-1, min(11, truth.shape[1])).round().astype(int)))),
        ))
        coverage.extend(ensemble_calibration_rows(name, truth, draws))
        cases.extend({"model": name, "sample_id": sid, "block_id": int(blocks[i]), "seed": seed,
                      "mean_squared_error": float(mse[i]), "empirical_crps": float(crps[i].mean())}
                     for i, sid in enumerate(frame.sample_id))
    artifact.write_table("metrics_profile", rows)
    artifact.write_table("metrics_cases", cases)
    artifact.write_table("calibration_empirical", coverage)
    artifact.write_json("diagnostics/frozen.json", {"source_run": source, "target_fit_count": 0,
                        "source_fold_replay_only": True, "source_residual_mse_verified": True,
                        "minimum_source_distance_km": distance, "calibration_scales": functional["joint_calibration_scales"],
                        "mean_weights": functional["residual_mean_weights_selected_on_validation"],
                        "spatial_weights": functional["spatial_dependence_weights_selected_on_validation"]})
    artifact.finalize(time.perf_counter()-started, {"evaluation_version": "external_frozen_1", "test_count": len(frame),
                                                   "external_test_only": True, "model_suite": names})
    print(f"External seed {seed} complete: {artifact.path}", flush=True)
    return artifact.path
