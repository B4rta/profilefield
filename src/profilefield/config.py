"""Configuration loading with deterministic serialization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def load_config(path: str | Path) -> Config:
    """Load a YAML mapping and retain its source path for provenance."""
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {source}")
    config = deepcopy(value)
    config["_source"] = str(source)
    return config


def save_config(config: Config, path: str | Path) -> None:
    """Write a stable YAML representation without the private source marker."""
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=True)
