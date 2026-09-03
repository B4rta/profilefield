"""Join local AGBD_raw EO patches to gediDB RH profiles by precise footprint location."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

CHANNEL_NAMES = (
    "s2_b01",
    "s2_b02",
    "s2_b03",
    "s2_b04",
    "s2_b05",
    "s2_b06",
    "s2_b07",
    "s2_b08",
    "s2_b8a",
    "s2_b09",
    "s2_b11",
    "s2_b12",
    "s2_num_days",
    "s2_doy_cos",
    "s2_doy_sin",
    "lat_cos",
    "lat_sin",
    "lon_cos",
    "lon_sin",
    "gedi_num_days",
    "gedi_doy_cos",
    "gedi_doy_sin",
    "alos_hh",
    "alos_hv",
    "canopy_height",
    "canopy_height_std",
    "land_cover",
    "land_cover_probability",
    "slope",
    "aspect_cos",
    "aspect_sin",
    "dem",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def intersects(row: Any, bounds: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bounds
    return bool(
        row.longitude_max >= min_lon
        and row.longitude_min <= max_lon
        and row.latitude_max >= min_lat
        and row.latitude_min <= max_lat
    )


def projected_coordinates(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    reference_latitude = float(np.mean(latitude))
    return np.column_stack(
        [
            longitude * (111_320.0 * np.cos(np.deg2rad(reference_latitude))),
            latitude * 110_540.0,
        ]
    )


def valid_central_pixels(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values).all(axis=1)
    valid &= (values[:, :12] != 0.0).all(axis=1)
    return valid


def extract_candidates(
    root: Path,
    catalog: pd.DataFrame,
    bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    selected_shards = catalog[[intersects(row, bounds) for row in catalog.itertuples(index=False)]]
    rows: list[pd.DataFrame] = []
    min_lon, min_lat, max_lon, max_lat = bounds
    for shard_number, shard in enumerate(selected_shards.itertuples(index=False), start=1):
        path = root / str(shard.relative_path)
        parquet = pq.ParquetFile(path)
        row_offset = 0
        for row_group in range(parquet.metadata.num_row_groups):
            metadata_table = parquet.read_row_group(
                row_group,
                columns=["label", "metadata.lat", "metadata.lon", "metadata.rh98"],
            )
            metadata = metadata_table.column("metadata").combine_chunks()
            latitude = metadata.field("lat").to_numpy(zero_copy_only=False)
            longitude = metadata.field("lon").to_numpy(zero_copy_only=False)
            keep = (
                (longitude >= min_lon)
                & (longitude <= max_lon)
                & (latitude >= min_lat)
                & (latitude <= max_lat)
            )
            local_indices = np.flatnonzero(keep)
            if len(local_indices):
                input_table = parquet.read_row_group(row_group, columns=["input"])
                nested = input_table.column("input").combine_chunks().take(pa.array(local_indices))
                flat = pc.list_flatten(pc.list_flatten(pc.list_flatten(nested))).to_numpy(
                    zero_copy_only=False
                )
                patches = flat.reshape(len(local_indices), len(CHANNEL_NAMES), 25, 25)
                central = patches[:, :, 12, 12].astype(np.float64)
                neighbourhood = patches[:, :, 11:14, 11:14].astype(np.float64)
                valid = valid_central_pixels(central)
                if valid.any():
                    indices = local_indices[valid]
                    records: dict[str, Any] = {
                        "agbd_split": shard.split,
                        "agbd_sample_id": [
                            f"{shard.relative_path}:{row_offset + int(index)}" for index in indices
                        ],
                        "agbd_shard": shard.relative_path,
                        "agbd_row": row_offset + indices,
                        "agbd_label": metadata_table.column("label").to_numpy(zero_copy_only=False)[
                            indices
                        ],
                        "agbd_rh98": metadata.field("rh98").to_numpy(zero_copy_only=False)[indices],
                        "agbd_longitude": longitude[indices],
                        "agbd_latitude": latitude[indices],
                    }
                    for channel, name in enumerate(CHANNEL_NAMES):
                        records[f"eo_center_{name}"] = central[valid, channel]
                        records[f"eo_mean3_{name}"] = neighbourhood[valid, channel].mean(
                            axis=(1, 2)
                        )
                    rows.append(pd.DataFrame(records))
            row_offset += metadata_table.num_rows
        print(
            f"Scanned {shard_number}/{len(selected_shards)} intersecting shards: {path.name}",
            flush=True,
        )
    if not rows:
        raise RuntimeError("No valid AGBD_raw patches were found inside the requested bounds")
    return pd.concat(rows, ignore_index=True)


def join_profiles(
    candidates: pd.DataFrame,
    profiles: pd.DataFrame,
    maximum_distance_metres: float,
) -> pd.DataFrame:
    all_longitudes = np.concatenate(
        [candidates["agbd_longitude"].to_numpy(), profiles["longitude"].to_numpy()]
    )
    all_latitudes = np.concatenate(
        [candidates["agbd_latitude"].to_numpy(), profiles["latitude"].to_numpy()]
    )
    coordinates = projected_coordinates(all_longitudes, all_latitudes)
    candidate_coordinates = coordinates[: len(candidates)]
    profile_coordinates = coordinates[len(candidates) :]
    distance, candidate_index = cKDTree(candidate_coordinates).query(profile_coordinates, k=1)
    accepted = distance <= maximum_distance_metres
    if not accepted.any():
        raise RuntimeError(
            f"No AGBD_raw/GEDI footprint matches within {maximum_distance_metres:g} m"
        )
    left = profiles.loc[accepted].reset_index(drop=True)
    right = candidates.iloc[candidate_index[accepted]].reset_index(drop=True)
    duplicated = right["agbd_sample_id"].duplicated(keep=False)
    if duplicated.any():
        unique = ~right["agbd_sample_id"].duplicated(keep="first")
        left = left.loc[unique].reset_index(drop=True)
        right = right.loc[unique].reset_index(drop=True)
        distance = distance[accepted][unique]
    else:
        distance = distance[accepted]
    left["match_distance_m"] = distance
    joined = pd.concat([left, right], axis=1)
    exact_observation = (np.abs(joined["agbd"] - joined["agbd_label"]) <= 0.01) & (
        np.abs(joined["rh_98"] - joined["agbd_rh98"]) <= 0.05
    )
    joined = joined.loc[exact_observation].reset_index(drop=True)
    if joined.empty:
        raise RuntimeError(
            "Spatial candidates were found, but none matched both AGBD and RH98 observations"
        )
    return joined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agbd-root", type=Path, default=Path("data/agbd_raw"))
    parser.add_argument(
        "--gedi-profiles", type=Path, default=Path("data/gedidb_amazon/french_guiana.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/gedidb_amazon/french_guiana_eo_rh.parquet")
    )
    parser.add_argument("--padding-degrees", type=float, default=0.002)
    parser.add_argument("--max-distance-metres", type=float, default=30.0)
    args = parser.parse_args()
    root = args.agbd_root.resolve()
    catalog_path = root / "shard_catalog.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Run scripts/catalog_agbd_raw.py first: {catalog_path}")
    profiles_path = args.gedi_profiles.resolve()
    profiles = pd.read_parquet(profiles_path)
    padding = args.padding_degrees
    bounds = (
        float(profiles["longitude"].min() - padding),
        float(profiles["latitude"].min() - padding),
        float(profiles["longitude"].max() + padding),
        float(profiles["latitude"].max() + padding),
    )
    catalog = pd.read_parquet(catalog_path)
    candidates = extract_candidates(root, catalog, bounds)
    joined = join_profiles(candidates, profiles, args.max_distance_metres)
    joined["official_spatial_block"] = (
        joined["agbd_split"].astype(str) + ":" + joined["spatial_block"].astype(str)
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output, index=False, compression="zstd")
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "agbd_revision": json.loads((root / "download_complete.json").read_text(encoding="utf-8"))[
            "revision"
        ],
        "agbd_catalog_sha256": sha256_file(catalog_path),
        "gedidb_profiles": str(profiles_path),
        "gedidb_profiles_sha256": sha256_file(profiles_path),
        "bounds": bounds,
        "maximum_distance_metres": args.max_distance_metres,
        "candidate_rows": len(candidates),
        "matched_rows": len(joined),
        "match_distance_median": float(joined["match_distance_m"].median()),
        "match_distance_max": float(joined["match_distance_m"].max()),
        "agbd_absolute_difference_max": float(np.abs(joined["agbd"] - joined["agbd_label"]).max()),
        "rh98_absolute_difference_max": float(np.abs(joined["rh_98"] - joined["agbd_rh98"]).max()),
        "official_split_counts": joined["agbd_split"].value_counts().sort_index().to_dict(),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
