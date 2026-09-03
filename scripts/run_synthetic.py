"""Run the synthetic ProfileField experiment from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from profilefield.config import load_config
from profilefield.experiments.synthetic import run_synthetic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic/smoke.yaml"))
    args = parser.parse_args()
    for path in run_synthetic(load_config(args.config), Path.cwd()):
        print(path)


if __name__ == "__main__":
    main()
