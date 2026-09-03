"""Self-contained and auditable run outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from profilefield.config import Config, save_config
from profilefield.reproducibility import environment_snapshot, sha256_json

REQUIRED_TABLES = ("metrics_profile", "metrics_blocks", "metrics_latent", "calibration")
REQUIRED_FILES = (
    "config.yaml",
    "environment.json",
    "data_manifest.json",
    "split_manifest.json",
    "run_metadata.json",
)


@dataclass
class RunArtifacts:
    """Writer for the mandatory experiment output contract."""

    path: Path

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        experiment_name: str,
        seed: int,
        run_id: str | None = None,
    ) -> RunArtifacts:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        identifier = run_id or f"{timestamp}_seed{seed}"
        path = Path(output_root) / experiment_name / identifier
        for child in (path, path / "figures", path / "models", path / "diagnostics"):
            child.mkdir(parents=True, exist_ok=False)
        return cls(path.resolve())

    def initialize(
        self,
        config: Config,
        data_manifest: dict[str, Any],
        split_manifest: dict[str, Any],
        project_root: str | Path,
        seed: int,
    ) -> None:
        save_config(config, self.path / "config.yaml")
        self.write_json("environment.json", environment_snapshot(project_root))
        self.write_json("data_manifest.json", data_manifest)
        self.write_json("split_manifest.json", split_manifest)
        self.write_json(
            "run_metadata.json",
            {
                "status": "running",
                "seed": seed,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "data_manifest_hash": sha256_json(data_manifest),
                "split_manifest_hash": sha256_json(split_manifest),
            },
        )

    def write_json(self, name: str, value: Any) -> None:
        destination = self.path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")

    def write_table(self, name: str, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        frame.to_csv(self.path / f"{name}.csv", index=False, lineterminator="\n")

    def finalize(self, wall_time_seconds: float, extra: dict[str, Any] | None = None) -> None:
        metadata_path = self.path / "run_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "wall_time_seconds": wall_time_seconds,
            }
        )
        if extra:
            metadata.update(extra)
        self.write_json("run_metadata.json", metadata)
        for name in REQUIRED_TABLES:
            path = self.path / f"{name}.csv"
            if not path.exists():
                self.write_table(name, [])


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def validate_run_directory(path: str | Path) -> dict[str, Any]:
    """Validate the artifact contract and provenance hashes of a completed run."""
    run = Path(path).resolve()
    missing = [name for name in REQUIRED_FILES if not (run / name).is_file()]
    missing.extend(name for name in REQUIRED_TABLES if not (run / f"{name}.csv").is_file())
    missing.extend(
        name for name in ("figures", "models", "diagnostics") if not (run / name).is_dir()
    )
    if missing:
        raise FileNotFoundError(f"Incomplete run directory {run}; missing {missing}")
    with (run / "run_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (run / "data_manifest.json").open("r", encoding="utf-8") as handle:
        data_manifest = json.load(handle)
    with (run / "split_manifest.json").open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {metadata.get('status')}")
    if metadata.get("data_manifest_hash") != sha256_json(data_manifest):
        raise RuntimeError("data_manifest.json hash does not match run_metadata.json")
    if metadata.get("split_manifest_hash") != sha256_json(split_manifest):
        raise RuntimeError("split_manifest.json hash does not match run_metadata.json")
    return {
        "path": str(run),
        "status": "valid",
        "seed": metadata.get("seed"),
        "wall_time_seconds": metadata.get("wall_time_seconds"),
    }
