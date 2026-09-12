"""Seal the external-validation protocol before querying new GEDI RH outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from profilefield.artifacts import RunArtifacts, validate_run_directory
from profilefield.config import load_config
from profilefield.reproducibility import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gedidb_v2/external_validation.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    runs = []
    for meta_path in sorted(Path(config["source"]["run_root"]).glob("*/run_metadata.json")):
        meta = json.loads(meta_path.read_text())
        if meta["status"] == "complete":
            validate_run_directory(meta_path.parent)
            runs.append({"path": str(meta_path.parent.resolve()), "seed": meta["seed"],
                         "seal_sha256": sha256_file(meta_path.parent / "artifact_integrity.json")})
    if sorted(run["seed"] for run in runs) != sorted(config["experiment"]["seeds"]):
        raise ValueError("Expected one frozen source run per configured seed")
    region = Path(config["external"]["region_spec"])
    protocol = RunArtifacts.create("outputs/protocols", "external", 2026)
    protocol.initialize(config, {"source_runs": runs}, {"external": "test_only"}, Path.cwd(), 2026)
    protocol.write_json("protocol.json", {
        "source_runs": runs, "region": json.loads(region.read_text()),
        "region_sha256": sha256_file(region), "config_sha256": sha256_file(Path(args.config)),
        "selection_uses": "AGBD coordinate counts only; no GEDI RH or predictive scores",
        "scope": "frozen French Guiana model transfer; no target-domain fitting or calibration",
        "source_code_commit": "25f4900e56e37061a183d87b6e97d16dee8e3e0d",
    })
    protocol.finalize(0.0)
    print(protocol.path, flush=True)


if __name__ == "__main__":
    main()
