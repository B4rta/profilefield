"""Run the full fold-wise CPT continuity comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from profilefield.config import load_config
from profilefield.experiments.cpt import run_cpt_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cpt/continuity.yaml"))
    args = parser.parse_args()
    for path in run_cpt_comparison(load_config(args.config), Path.cwd()):
        print(path)


if __name__ == "__main__":
    main()
