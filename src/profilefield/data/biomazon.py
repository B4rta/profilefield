"""Local-only Biomazon/GEDI table adapter with checksums and spatial splits."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from profilefield.config import Config
from profilefield.context.embeddings import select_embedding_columns
from profilefield.data.splits import (
    SpatialSplit,
    assign_grid_blocks,
    blocked_split,
    split_from_labels,
    validate_split,
)
from profilefield.reproducibility import sha256_file, sha256_json

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BiomazonDataset:
    table_path: Path
    frame: pd.DataFrame
    sample_ids: tuple[str, ...]
    coordinates: FloatArray
    features: FloatArray
    profiles: FloatArray
    feature_columns: tuple[str, ...]
    rh_columns: tuple[str, ...]
    rh_percentiles: FloatArray
    split: SpatialSplit
    manifest: dict[str, Any]


def biomazon_root(data_config: dict[str, Any], project_root: str | Path) -> Path:
    if data_config.get("root"):
        return Path(data_config["root"]).expanduser().resolve()
    environment_name = str(data_config.get("root_env", "BIOMAZON_ROOT"))
    environment_value = os.environ.get(environment_name)
    if environment_value:
        return Path(environment_value).expanduser().resolve()
    default = Path(project_root).resolve() / "data/biomazon"
    raise FileNotFoundError(
        "Biomazon is not downloaded automatically. Download DOI "
        f"{data_config.get('expected_source_doi', '10.26165/JUELICH-DATA/2YP2PZ')}, then set "
        f"{environment_name} to its local directory or add data.root to the config. "
        f"Suggested local path: {default}"
    )


def load_biomazon(config: Config, project_root: str | Path = ".", seed: int = 0) -> BiomazonDataset:
    data_config = config["data"]
    root = biomazon_root(data_config, project_root)
    table_path = root / data_config["table"]
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing Biomazon feature table: {table_path}. Create/export a CSV or Parquet table "
            "containing IDs, projected x/y coordinates, official split labels when available, "
            "precomputed EO features, and ordered GEDI RH columns. No raster download is attempted."
        )
    if table_path.suffix.lower() == ".csv":
        frame = pd.read_csv(table_path)
    elif table_path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(table_path)
    else:
        raise ValueError("Biomazon table must be CSV or Parquet")
    id_column = data_config["sample_id_column"]
    x_column = data_config["x_column"]
    y_column = data_config["y_column"]
    required = [id_column, x_column, y_column]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required Biomazon columns: {missing}")
    sample_ids = tuple(frame[id_column].astype(str))
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Biomazon sample IDs must be unique")
    coordinates = frame[[x_column, y_column]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Biomazon coordinates contain missing/non-finite values")
    feature_columns = tuple(select_embedding_columns(frame, list(data_config["feature_prefixes"])))
    rh_columns = tuple(_ordered_rh_columns(frame, str(data_config.get("rh_prefix", "RH"))))
    features = frame[list(feature_columns)].to_numpy(dtype=np.float64)
    profiles = frame[list(rh_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(profiles).all():
        raise ValueError("Biomazon feature/RH table contains missing or non-finite values")
    violations = int(np.count_nonzero(np.diff(profiles, axis=1) < -1e-6))
    if violations:
        raise ValueError(f"Input GEDI RH targets contain {violations} ordering violations")
    rh_percentiles = np.asarray([_rh_number(column) for column in rh_columns], dtype=np.float64)
    split = _build_split(frame, coordinates, sample_ids, config["split"], data_config, seed)
    preprocessing = config.get("preprocessing", {})
    source_manifest_name = data_config.get("source_manifest")
    source_manifest_path = root / source_manifest_name if source_manifest_name else None
    if source_manifest_path is not None and not source_manifest_path.is_file():
        raise FileNotFoundError(f"Missing external source manifest: {source_manifest_path}")
    parameters = {
        "source_doi": data_config.get("expected_source_doi"),
        "table": str(table_path),
        "table_sha256": sha256_file(table_path),
        "source_manifest": str(source_manifest_path) if source_manifest_path else None,
        "source_manifest_sha256": (
            sha256_file(source_manifest_path) if source_manifest_path is not None else None
        ),
        "rows": len(frame),
        "feature_columns": list(feature_columns),
        "rh_columns": list(rh_columns),
        "preprocessing": preprocessing,
    }
    manifest = {**parameters, "manifest_hash": sha256_json(parameters)}
    return BiomazonDataset(
        table_path=table_path,
        frame=frame,
        sample_ids=sample_ids,
        coordinates=coordinates,
        features=features,
        profiles=profiles,
        feature_columns=feature_columns,
        rh_columns=rh_columns,
        rh_percentiles=rh_percentiles,
        split=split,
        manifest=manifest,
    )


def validate_biomazon_from_config(config: Config, project_root: str | Path = ".") -> str:
    seed = int(config["experiment"].get("seeds", [0])[0])
    dataset = load_biomazon(config, project_root, seed)
    destination = dataset.table_path.parent / "data_manifest.json"
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dataset.manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return (
        f"Validated {len(dataset.sample_ids)} samples, {len(dataset.feature_columns)} features, "
        f"{len(dataset.rh_columns)} ordered RH outputs. Manifest: {destination}"
    )


def _ordered_rh_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}[_-]?(\d+(?:\.\d+)?)$", re.IGNORECASE)
    matched = [
        (column, float(match.group(1)))
        for column in frame.columns
        if (match := pattern.match(column))
    ]
    if len(matched) < 2:
        raise ValueError(f"Expected at least two ordered RH columns matching {prefix}<percentile>")
    matched.sort(key=lambda pair: pair[1])
    return [column for column, _ in matched]


def _rh_number(column: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)$", column)
    if match is None:
        raise ValueError(f"Cannot parse RH percentile from {column}")
    return float(match.group(1))


def _build_split(
    frame: pd.DataFrame,
    coordinates: FloatArray,
    sample_ids: tuple[str, ...],
    split_config: dict[str, Any],
    data_config: dict[str, Any],
    seed: int,
) -> SpatialSplit:
    strategy = str(split_config.get("strategy", "official"))
    n_blocks_x = int(split_config.get("n_blocks_x", 8))
    n_blocks_y = int(split_config.get("n_blocks_y", 8))
    block_column = data_config.get("block_column")
    if block_column and block_column in frame:
        block_ids = pd.factorize(frame[block_column].astype(str), sort=True)[0].astype(np.int64)
    else:
        block_ids = assign_grid_blocks(coordinates, n_blocks_x, n_blocks_y)
    split_column = data_config.get("split_column")
    if strategy == "official":
        if not split_column or split_column not in frame:
            raise ValueError(
                f"Official split requested but column {split_column!r} is missing. Use strategy: geographic_blocks only if official labels are unavailable."
            )
        split = split_from_labels(frame[split_column].astype(str).tolist(), sample_ids, block_ids)
        calibration_fraction = float(split_config.get("calibration_fraction", 0.0))
        if len(split.calibration) == 0 and calibration_fraction > 0:
            split = _carve_training_calibration(split, calibration_fraction, seed)
        validate_split(split, require_block_isolation=True)
        return split
    if strategy == "geographic_blocks":
        return blocked_split(
            coordinates,
            sample_ids,
            seed,
            n_blocks_x,
            n_blocks_y,
            float(split_config["train_fraction"]),
            float(split_config["validation_fraction"]),
            float(split_config.get("calibration_fraction", 0.0)),
        )
    if strategy == "leave_region_out":
        region_column = data_config.get("region_column")
        if not region_column or region_column not in frame:
            raise ValueError("leave_region_out requires data.region_column")
        regions = sorted(frame[region_column].astype(str).unique())
        if len(regions) < 3:
            raise ValueError("leave_region_out requires at least three regions")
        test_region = str(split_config.get("test_region", regions[-1]))
        validation_region = str(split_config.get("validation_region", regions[-2]))
        labels = np.full(len(frame), "train", dtype=object)
        region_values = frame[region_column].astype(str).to_numpy()
        labels[region_values == validation_region] = "validation"
        labels[region_values == test_region] = "test"
        split = split_from_labels(labels.tolist(), sample_ids, block_ids)
        calibration_fraction = float(split_config.get("calibration_fraction", 0.0))
        if calibration_fraction > 0:
            split = _carve_training_calibration(split, calibration_fraction, seed)
        validate_split(split, require_block_isolation=True)
        return split
    raise ValueError(f"Unknown split strategy: {strategy}")


def _carve_training_calibration(
    split: SpatialSplit, calibration_fraction: float, seed: int
) -> SpatialSplit:
    train_blocks = np.unique(split.block_ids[split.train])
    rng = np.random.default_rng(seed)
    count = max(1, round(len(train_blocks) * calibration_fraction))
    calibration_blocks = set(rng.permutation(train_blocks)[:count].tolist())
    calibration = split.train[np.isin(split.block_ids[split.train], list(calibration_blocks))]
    train = split.train[~np.isin(split.block_ids[split.train], list(calibration_blocks))]
    result = SpatialSplit(
        train=train,
        validation=split.validation,
        calibration=calibration,
        test=split.test,
        block_ids=split.block_ids,
        sample_ids=split.sample_ids,
        strategy=f"{split.strategy}_with_train_block_calibration",
        seed=seed,
    )
    validate_split(result, require_block_isolation=True)
    return result
