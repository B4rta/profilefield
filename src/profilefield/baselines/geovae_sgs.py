"""Read-only bridge to the exact, commit-pinned GeoVAE-SGS implementation."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from profilefield.reproducibility import git_sha, sha256_file, sha256_tree


def verify_baseline(project_root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = root / manifest_path
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    repository = root / manifest["submodule_path"]
    required = [
        repository / "src/sgs_vae_gp/sgs_core.py",
        repository / "Sample_data/meta_Projekt.csv",
        repository / "validations/reviewer_analysis/combined_prediction_pairs.csv",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            "GeoVAE-SGS submodule is incomplete. Run `git submodule update --init --recursive`. "
            f"Missing: {missing}"
        )
    observed = {
        "commit": git_sha(repository),
        "sgs_core_sha256": sha256_file(required[0]),
        "sample_data_tree_sha256": sha256_tree(repository / "Sample_data"),
    }
    for key, value in observed.items():
        if value != manifest[key]:
            raise RuntimeError(f"Immutable baseline mismatch for {key}: {value} != {manifest[key]}")
    return {**manifest, "verified": True}


def import_baseline_core(project_root: str | Path) -> Any:
    source = Path(project_root).resolve() / "third_party/geovae-sgs/src"
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return importlib.import_module("sgs_vae_gp.sgs_core")


def published_metric_rows(
    pairs_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.read_csv(pairs_path)
    profile_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for method, group in frame.groupby("method", sort=True):
        # The combined reviewer artifact normalizes both methods into these columns;
        # the legacy method-specific columns are intentionally sparse.
        truth = group["ln_qc_obs"].to_numpy(float)
        prediction = group["ln_qc_pred"].to_numpy(float)
        standard_deviation = group["pred_std"].to_numpy(float)
        valid = np.isfinite(truth) & np.isfinite(prediction)
        truth = truth[valid]
        prediction = prediction[valid]
        standard_deviation = standard_deviation[valid]
        error = prediction - truth
        values = {
            "n": float(len(error)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias": float(np.mean(error)),
        }
        profile_rows.extend(
            {"model": method, "scope": "ln_qc", "metric": metric, "value": value}
            for metric, value in values.items()
        )
        for sigma, nominal in ((1.0, 0.682689492), (2.0, 0.954499736)):
            uncertainty_valid = np.isfinite(standard_deviation)
            covered = (
                np.abs(error[uncertainty_valid]) <= sigma * standard_deviation[uncertainty_valid]
            )
            empirical = float(np.mean(covered)) if len(covered) else float("nan")
            calibration_rows.append(
                {
                    "model": method,
                    "output": "ln_qc",
                    "nominal_coverage": nominal,
                    "empirical_coverage": empirical,
                    "coverage_error": abs(empirical - nominal),
                    "mean_interval_width": (
                        float(2.0 * sigma * np.mean(standard_deviation[uncertainty_valid]))
                        if uncertainty_valid.any()
                        else float("nan")
                    ),
                }
            )
    return profile_rows, calibration_rows
