"""Build a compact, reproducible shard catalogue for the local AGBD_raw release."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def summarize_shard(path: Path, root: Path) -> dict[str, Any]:
    table = pq.read_table(
        path,
        columns=["label", "metadata.lat", "metadata.lon", "metadata.region_cla", "metadata.rh98"],
    )
    label = table.column("label").combine_chunks().to_numpy(zero_copy_only=False)
    latitude = table.column("lat").combine_chunks().to_numpy(zero_copy_only=False)
    longitude = table.column("lon").combine_chunks().to_numpy(zero_copy_only=False)
    region = table.column("region_cla").combine_chunks().to_numpy(zero_copy_only=False)
    rh98 = table.column("rh98").combine_chunks().to_numpy(zero_copy_only=False)
    finite = np.isfinite(label) & np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(rh98)
    if not finite.any():
        raise ValueError(f"No finite metadata rows in {path}")
    label = label[finite]
    latitude = latitude[finite]
    longitude = longitude[finite]
    rh98 = rh98[finite]
    region = region[finite]
    split = path.name.split("-", maxsplit=1)[0]
    return {
        "split": split,
        "relative_path": path.relative_to(root).as_posix(),
        "rows": table.num_rows,
        "finite_rows": int(finite.sum()),
        "bytes": path.stat().st_size,
        "latitude_min": float(latitude.min()),
        "latitude_max": float(latitude.max()),
        "latitude_mean": float(latitude.mean()),
        "longitude_min": float(longitude.min()),
        "longitude_max": float(longitude.max()),
        "longitude_mean": float(longitude.mean()),
        "agbd_mean": float(label.mean()),
        "agbd_std": float(label.std()),
        "agbd_q05": float(np.quantile(label, 0.05)),
        "agbd_q50": float(np.quantile(label, 0.50)),
        "agbd_q95": float(np.quantile(label, 0.95)),
        "rh98_mean": float(rh98.mean()),
        "rh98_std": float(rh98.std()),
        "region_counts_json": json.dumps(
            {str(key): value for key, value in sorted(Counter(map(int, region)).items())},
            sort_keys=True,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/agbd_raw"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    completion_path = root / "download_complete.json"
    if not completion_path.exists():
        raise FileNotFoundError(f"Missing completed-download manifest: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("problems"):
        raise RuntimeError("AGBD_raw download manifest reports integrity problems")
    paths = sorted((root / "data").glob("*.parquet"))
    if len(paths) != int(completion["expected_files"]):
        raise RuntimeError(
            f"Expected {completion['expected_files']} shards, discovered {len(paths)}"
        )
    rows = []
    for index, path in enumerate(paths, start=1):
        rows.append(summarize_shard(path, root))
        if index % 100 == 0:
            print(f"Catalogued {index}/{len(paths)} shards", flush=True)
    frame = pd.DataFrame(rows).sort_values(["split", "relative_path"]).reset_index(drop=True)
    output = (args.output or root / "shard_catalog.parquet").resolve()
    frame.to_parquet(output, index=False, compression="zstd")
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": completion["revision"],
        "shards": len(frame),
        "rows": int(frame["rows"].sum()),
        "bytes": int(frame["bytes"].sum()),
        "splits": {
            split: {
                "shards": len(group),
                "rows": int(group["rows"].sum()),
                "bytes": int(group["bytes"].sum()),
            }
            for split, group in frame.groupby("split", sort=True)
        },
        "catalog": output.name,
    }
    (output.parent / "shard_catalog_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
