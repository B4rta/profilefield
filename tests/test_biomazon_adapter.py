from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from profilefield.data.biomazon import load_biomazon
from profilefield.experiments.biomazon import run_biomazon


def test_biomazon_adapter_official_split_and_manifest(tmp_path: Path) -> None:
    rows = []
    split_by_block = {
        "a": "train",
        "b": "train",
        "c": "train",
        "d": "train",
        "e": "validation",
        "f": "validation",
        "g": "test",
        "h": "test",
    }
    for block_index, (block, split) in enumerate(split_by_block.items()):
        for replicate in range(2):
            base = float(block_index + replicate / 10)
            rows.append(
                {
                    "sample_id": f"{block}-{replicate}",
                    "x": block_index,
                    "y": replicate,
                    "spatial_block": block,
                    "split": split,
                    "s1_a": base,
                    "s2_b": base + 1,
                    "RH0": 0.0,
                    "RH50": base + 3,
                    "RH100": base + 7,
                }
            )
    table = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    config = {
        "experiment": {"seeds": [3]},
        "data": {
            "root": str(tmp_path),
            "table": table.name,
            "sample_id_column": "sample_id",
            "x_column": "x",
            "y_column": "y",
            "block_column": "spatial_block",
            "split_column": "split",
            "feature_prefixes": ["s1_", "s2_"],
            "rh_prefix": "RH",
            "expected_source_doi": "10.26165/JUELICH-DATA/2YP2PZ",
        },
        "split": {"strategy": "official", "calibration_fraction": 0.25},
        "preprocessing": {"feature_scaling": "standard"},
    }
    dataset = load_biomazon(config, tmp_path, 3)
    assert dataset.features.shape == (16, 2)
    assert dataset.profiles.shape == (16, 3)
    assert np.array_equal(dataset.rh_percentiles, [0, 50, 100])
    assert len(dataset.split.calibration) > 0
    assert not set(dataset.split.block_ids[dataset.split.train]) & set(
        dataset.split.block_ids[dataset.split.test]
    )
    assert len(dataset.manifest["table_sha256"]) == 64


def test_leave_region_out_keeps_calibration_and_test_blocks_disjoint(tmp_path: Path) -> None:
    rows = []
    for region_index, region in enumerate(("train_a", "train_b", "validation", "test")):
        for block_index in range(4):
            for replicate in range(2):
                value = float(region_index + block_index / 10 + replicate / 100)
                rows.append(
                    {
                        "sample_id": f"{region}-{block_index}-{replicate}",
                        "longitude": value,
                        "latitude": value + 0.5,
                        "region": region,
                        "spatial_block": f"{region}-{block_index}",
                        "ctx_time": value,
                        "RH0": 0.0,
                        "RH50": value + 2.0,
                        "RH100": value + 5.0,
                    }
                )
    table = tmp_path / "gedi.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    config = {
        "experiment": {"seeds": [11]},
        "data": {
            "root": str(tmp_path),
            "table": table.name,
            "sample_id_column": "sample_id",
            "x_column": "longitude",
            "y_column": "latitude",
            "region_column": "region",
            "block_column": "spatial_block",
            "feature_prefixes": ["ctx_"],
            "rh_prefix": "RH",
        },
        "split": {
            "strategy": "leave_region_out",
            "validation_region": "validation",
            "test_region": "test",
            "calibration_fraction": 0.25,
        },
    }
    dataset = load_biomazon(config, tmp_path, 11)
    split = dataset.split
    assert len(split.calibration) > 0
    groups = [
        set(split.block_ids[index])
        for index in (split.train, split.validation, split.calibration, split.test)
    ]
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            assert not left & right


def test_tiny_biomazon_model_suite_runs_end_to_end(tmp_path: Path) -> None:
    rows = []
    assignments = [
        ("a", "train"),
        ("b", "train"),
        ("c", "train"),
        ("d", "train"),
        ("e", "validation"),
        ("f", "validation"),
        ("g", "test"),
        ("h", "test"),
    ]
    for block_index, (block, split) in enumerate(assignments):
        for replicate in range(2):
            feature = block_index / 8 + replicate / 20
            rows.append(
                {
                    "sample_id": f"{block}-{replicate}",
                    "x": float(block_index),
                    "y": float(replicate),
                    "spatial_block": block,
                    "split": split,
                    "s1_a": feature,
                    "s2_b": feature**2,
                    "RH0": 0.05,
                    "RH50": 1.0 + feature,
                    "RH100": 2.0 + 2 * feature,
                }
            )
    table = tmp_path / "tiny.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    short_output = Path.home() / ".codex/test-output" / uuid.uuid4().hex[:8]
    config = {
        "experiment": {
            "name": "b",
            "seeds": [9],
            "output_root": str(short_output),
        },
        "data": {
            "root": str(tmp_path),
            "table": table.name,
            "sample_id_column": "sample_id",
            "x_column": "x",
            "y_column": "y",
            "block_column": "spatial_block",
            "split_column": "split",
            "feature_prefixes": ["s1_", "s2_"],
            "rh_prefix": "RH",
            "expected_source_doi": "10.26165/JUELICH-DATA/2YP2PZ",
        },
        "split": {"strategy": "official", "calibration_fraction": 0.25},
        "preprocessing": {"feature_scaling": "standard"},
        "model": {
            "latent_dim": 2,
            "hidden_dim": 8,
            "training_steps": 2,
            "learning_rate": 0.01,
            "gp_training_steps": 2,
            "gp_learning_rate": 0.02,
            "kl_weight": 0.02,
            "inducing_points": 4,
            "kernel_nu": 1.5,
            "lmc_latents": 2,
            "tree_estimators": 3,
            "tree_min_samples_leaf": 1,
        },
        "evaluation": {"posterior_samples": 4, "variogram_bins": 2},
    }
    try:
        [run_path] = run_biomazon(config, tmp_path)
        assert (run_path / "metrics_profile.csv").exists()
        assert (run_path / "models/lmc_svgp.pt").exists()
        metadata = pd.read_json(run_path / "run_metadata.json", typ="series")
        assert metadata["status"] == "complete"
    finally:
        if short_output.exists():
            shutil.rmtree(short_output)
