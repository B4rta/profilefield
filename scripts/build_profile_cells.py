"""Aggregate footprint RH tables to non-overlapping landscape-support cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from profilefield.reproducibility import sha256_file, sha256_json


def build_profile_cells(
    source: Path,
    destination: Path,
    *,
    cell_km: float,
    minimum_shots: int,
    x_column: str = "longitude",
    y_column: str = "latitude",
) -> dict[str, Any]:
    """Create mean profiles/features without allowing a cell to cross a supplied split."""

    if cell_km <= 0.0 or minimum_shots < 1:
        raise ValueError("cell_km and minimum_shots must be positive")
    frame = pd.read_parquet(source)
    degrees = cell_km / 111.0
    frame = frame.assign(
        _cell_x=np.floor(frame[x_column].to_numpy(float) / degrees).astype(np.int64),
        _cell_y=np.floor(frame[y_column].to_numpy(float) / degrees).astype(np.int64),
    )
    keys = ["_cell_x", "_cell_y"]
    for column in ("region", "agbd_split"):
        if column in frame:
            keys.append(column)
    grouped = frame.groupby(keys, sort=True, observed=True)
    counts = grouped.size().rename("shot_count")
    retained = counts[counts >= minimum_shots]
    if retained.empty:
        raise ValueError("No cells satisfy minimum_shots")
    selected = frame.set_index(keys).loc[retained.index].reset_index()
    numeric_columns = [
        column
        for column in selected.select_dtypes(include=[np.number]).columns
        if column not in {"_cell_x", "_cell_y"}
        and not column.endswith("_id")
        and column not in {"official_spatial_block", "spatial_block"}
    ]
    output = (
        selected.groupby(keys, sort=True, observed=True)[numeric_columns]
        .mean()
        .join(retained)
        .reset_index()
    )
    output.insert(
        0,
        "sample_id",
        [
            f"cell{round(cell_km * 1000)}:" + ":".join(str(value) for value in row)
            for row in output[keys].itertuples(index=False, name=None)
        ],
    )
    output["cell_id"] = [
        f"{int(x)}:{int(y)}" for x, y in output[["_cell_x", "_cell_y"]].itertuples(index=False)
    ]
    if "official_spatial_block" in selected:
        block = (
            selected.groupby(keys, sort=True, observed=True)["official_spatial_block"]
            .agg(lambda values: values.mode().iloc[0])
            .reset_index(name="official_spatial_block")
        )
        output = output.merge(block, on=keys, validate="one_to_one")
    output = output.drop(columns=["_cell_x", "_cell_y"])
    rh_columns = [
        column for column in output.columns if isinstance(column, str) and column.startswith("rh_")
    ]
    if rh_columns and np.count_nonzero(np.diff(output[rh_columns].to_numpy(float), axis=1) < -1e-7):
        raise AssertionError("Averaging unexpectedly broke RH monotonicity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    parameters = {
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "destination_path": str(destination.resolve()),
        "destination_sha256": sha256_file(destination),
        "cell_km": cell_km,
        "minimum_shots": minimum_shots,
        "source_rows": len(frame),
        "retained_cells": len(output),
        "retained_shots": int(output["shot_count"].sum()),
        "aggregation": "arithmetic mean within non-overlapping lon/lat cells and source split",
        "cross_split_cells": 0,
    }
    manifest = {**parameters, "manifest_hash": sha256_json(parameters)}
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cell-km", type=float, default=0.5)
    parser.add_argument("--minimum-shots", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            build_profile_cells(
                args.source,
                args.destination,
                cell_km=args.cell_km,
                minimum_shots=args.minimum_shots,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
