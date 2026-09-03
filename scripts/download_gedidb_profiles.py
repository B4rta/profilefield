"""Query quality-filtered Amazon GEDI profiles from the public GFZ gediDB."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gedidb
import geopandas as gpd
import numpy as np
import pandas as pd
from gedidb.core.gediprovider import GEDIProvider
from gedidb.providers.tiledb_provider import TileDBProvider
from shapely.geometry import box

BUCKET = "dog-ext.gedidb.gedi-l2-l4-v002.0"
ENDPOINT = "https://s3.gfz-potsdam.de"
START_TIME = "2019-04-17"
END_TIME = "2023-03-16"
VARIABLES = (
    "agbd",
    "agbd_se",
    "rh",
    "cover_z",
    "pai_z",
    "pavd_z",
    "cover",
    "pai",
    "fhd_normal",
    "sensitivity",
    "beam_type",
    "l2a_quality_flag",
    "l2b_quality_flag",
    "l4_quality_flag",
)
QUALITY_FILTERS = {
    "sensitivity": ">= 0.95 and <= 1.0",
    "beam_type": "== 'full'",
    "l2a_quality_flag": "== 1",
    "l2b_quality_flag": "== 1",
    "l4_quality_flag": "== 1",
}


@dataclass(frozen=True)
class Region:
    name: str
    longitude: float
    latitude: float


REGIONS = (
    Region("tapajos", -55.025, -3.025),
    Region("manaus", -60.075, -2.675),
    Region("acre", -67.875, -9.925),
    Region("french_guiana", -53.075, 4.925),
    Region("agbd_train", -53.875, 2.675),
    Region("agbd_validation", -53.625, 5.575),
    Region("agbd_test", -52.575, 5.025),
)


class LowMemoryGEDIProvider(GEDIProvider):
    """GEDIProvider with bounded TileDB buffers for commodity workstations."""

    def __init__(self) -> None:
        TileDBProvider.__init__(
            self,
            storage_type="s3",
            s3_bucket=BUCKET,
            local_path=None,
            url=ENDPOINT,
            region="eu-central-1",
            credentials=None,
            s3_config_overrides={
                "py.init_buffer_bytes": str(32 * 1024**2),
                "sm.memory_budget": str(2 * 1024**3),
                "sm.memory_budget_var": str(1024**3),
                "sm.mem.total_budget": str(4 * 1024**3),
                "sm.tile_cache_size": str(512 * 1024**2),
                "vfs.s3.max_parallel_ops": "8",
                "sm.compute_concurrency_level": "8",
                "sm.io_concurrency_level": "8",
                "sm.num_reader_threads": "8",
                "sm.num_tiledb_threads": "8",
            },
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def profile_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(f"{prefix}_")]
    return sorted(columns, key=lambda value: int(value.rsplit("_", maxsplit=1)[1]))


def expand_profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand gediDB list-valued profile columns into stable scalar columns."""
    result = frame
    for name, labels in (
        ("rh", list(range(101))),
        ("cover_z", list(range(0, 150, 5))),
        ("pai_z", list(range(0, 150, 5))),
        ("pavd_z", list(range(0, 150, 5))),
    ):
        if name not in result.columns:
            continue
        values = np.asarray(result.pop(name).tolist(), dtype=np.float64)
        if values.shape[1] != len(labels):
            raise ValueError(f"Profile {name} has length {values.shape[1]}, expected {len(labels)}")
        expanded = pd.DataFrame(
            values,
            columns=[f"{name}_{label}" for label in labels],
            index=result.index,
        )
        result = pd.concat([result, expanded], axis=1)
    return result


def validate_frame(frame: pd.DataFrame, region: Region) -> dict[str, Any]:
    required = {"latitude", "longitude", "agbd", "agbd_se", "sensitivity"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"gediDB response for {region.name} is missing {missing}")
    rh_columns = profile_columns(frame, "rh")
    if len(rh_columns) != 101:
        raise ValueError(f"Expected 101 RH columns, observed {len(rh_columns)}")
    numeric = frame[["latitude", "longitude", "agbd", "agbd_se", *rh_columns]].to_numpy(
        dtype=np.float64
    )
    finite_rows = np.isfinite(numeric).all(axis=1)
    monotonic_rows = (np.diff(numeric[:, 4:], axis=1) >= -1e-5).all(axis=1)
    valid = finite_rows & monotonic_rows & (numeric[:, 2] >= 0.0) & (numeric[:, 2] <= 500.0)
    return {
        "rows": len(frame),
        "valid_rows": int(valid.sum()),
        "invalid_nonfinite": int((~finite_rows).sum()),
        "invalid_rh_order": int((finite_rows & ~monotonic_rows).sum()),
        "invalid_agbd_range": int(
            (finite_rows & ((numeric[:, 2] < 0.0) | (numeric[:, 2] > 500.0))).sum()
        ),
        "valid_mask": valid,
        "rh_columns": rh_columns,
    }


def deterministic_subsample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(frame), size=maximum, replace=False))
    return frame.iloc[selected]


def query_region(
    provider: GEDIProvider,
    region: Region,
    side_degrees: float,
    maximum_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    half = side_degrees / 2.0
    bounds = (
        region.longitude - half,
        region.latitude - half,
        region.longitude + half,
        region.latitude + half,
    )
    geometry = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326")
    response = provider.get_data(
        variables=list(VARIABLES),
        query_type="bounding_box",
        geometry=geometry,
        start_time=START_TIME,
        end_time=END_TIME,
        return_type="dataframe",
        **QUALITY_FILTERS,
    )
    if not isinstance(response, pd.DataFrame) or response.empty:
        raise RuntimeError(f"gediDB returned no records for {region.name}")
    frame = expand_profile_columns(response.reset_index())
    validation = validate_frame(frame, region)
    valid_mask = np.asarray(validation.pop("valid_mask"), dtype=bool)
    rh_columns = list(validation.pop("rh_columns"))
    frame = frame.loc[valid_mask].copy()
    frame = deterministic_subsample(frame, maximum_rows, seed)
    frame.insert(0, "region", region.name)
    frame.insert(1, "sample_id", [f"{region.name}:{value}" for value in frame["shot_number"]])
    frame.insert(
        2,
        "spatial_block",
        [
            f"{region.name}:{int(np.floor(lon * 200))}:{int(np.floor(lat * 200))}"
            for lon, lat in zip(frame["longitude"], frame["latitude"], strict=True)
        ],
    )
    acquisition_time = pd.to_datetime(frame["time"], utc=True)
    day_angle = 2.0 * np.pi * (acquisition_time.dt.dayofyear.to_numpy() - 1.0) / 365.25
    frame["ctx_time_sin"] = np.sin(day_angle)
    frame["ctx_time_cos"] = np.cos(day_angle)
    frame["ctx_intercept"] = 1.0
    ordered = [
        "region",
        "sample_id",
        "spatial_block",
        "shot_number",
        "longitude",
        "latitude",
        "time",
        "ctx_time_sin",
        "ctx_time_cos",
        "ctx_intercept",
        "agbd",
        "agbd_se",
        "cover",
        "pai",
        "fhd_normal",
        "sensitivity",
        "beam_type",
        "l2a_quality_flag",
        "l2b_quality_flag",
        "l4_quality_flag",
        *rh_columns,
        *profile_columns(frame, "cover_z"),
        *profile_columns(frame, "pai_z"),
        *profile_columns(frame, "pavd_z"),
    ]
    ordered = list(dict.fromkeys(column for column in ordered if column in frame.columns))
    output = frame[ordered].reset_index(drop=True)
    validation["retained_rows"] = len(output)
    validation["bounds"] = list(bounds)
    return output, validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/gedidb_amazon"))
    parser.add_argument("--side-degrees", type=float, default=0.10)
    parser.add_argument("--max-rows-per-region", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--regions", nargs="*", choices=[region.name for region in REGIONS])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.side_degrees <= 0 or args.retries < 1:
        raise SystemExit("--side-degrees and --retries must be positive")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected_names = set(args.regions or [region.name for region in REGIONS])
    regions = [region for region in REGIONS if region.name in selected_names]
    provider = LowMemoryGEDIProvider()
    region_manifests: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    try:
        for index, region in enumerate(regions):
            path = destination / f"{region.name}.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
                validation = validate_frame(frame, region)
                validation.pop("valid_mask")
                validation.pop("rh_columns")
                validation["retained_rows"] = len(frame)
                validation["reused"] = True
            else:
                last_error: Exception | None = None
                for attempt in range(1, args.retries + 1):
                    try:
                        frame, validation = query_region(
                            provider,
                            region,
                            args.side_degrees,
                            args.max_rows_per_region,
                            args.seed + index,
                        )
                        frame.to_parquet(path, index=False, compression="zstd")
                        break
                    except Exception as error:
                        last_error = error
                        if attempt == args.retries:
                            raise
                        time.sleep(2**attempt)
                if last_error is not None and not path.exists():
                    raise last_error
                validation["reused"] = False
            frames.append(frame)
            region_manifests.append(
                {
                    "name": region.name,
                    "file": path.name,
                    "sha256": sha256_file(path),
                    **validation,
                }
            )
            print(f"{region.name}: {len(frame)} retained shots -> {path}", flush=True)
    finally:
        provider.close()
    combined = pd.concat(frames, ignore_index=True)
    combined_path = destination / "gedi_profiles.parquet"
    combined.to_parquet(combined_path, index=False, compression="zstd")
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "GFZ gediDB public TileDB",
        "source_url": "https://s3.gfz-potsdam.de",
        "bucket": BUCKET,
        "gedidb_version": gedidb.__version__,
        "time_range": [START_TIME, END_TIME],
        "variables": list(VARIABLES),
        "quality_filters": QUALITY_FILTERS,
        "side_degrees": args.side_degrees,
        "max_rows_per_region": args.max_rows_per_region,
        "seed": args.seed,
        "regions": region_manifests,
        "combined_file": combined_path.name,
        "combined_rows": len(combined),
        "combined_sha256": sha256_file(combined_path),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Complete: {len(combined)} profiles -> {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
