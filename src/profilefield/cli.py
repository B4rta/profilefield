"""Command-line entry point for all ProfileField workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from profilefield.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="profilefield")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("synthetic", "run independent-SVGP versus LMC-SVGP synthetic experiment"),
        ("cpt-audit", "verify the immutable GeoVAE-SGS continuity baseline"),
        ("cpt", "run fold-wise CPT independent-GP versus LMC continuity experiment"),
        ("biomazon-validate", "validate local Biomazon/GEDI feature data without downloading it"),
        ("biomazon", "run the common Biomazon model suite"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
    validate = subparsers.add_parser("validate-run", help="validate a completed run artifact")
    validate.add_argument("--path", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    if args.command == "synthetic":
        from profilefield.experiments.synthetic import run_synthetic

        for path in run_synthetic(load_config(args.config), project_root):
            print(path)
    elif args.command == "cpt-audit":
        from profilefield.experiments.cpt import run_cpt_audit

        print(run_cpt_audit(load_config(args.config), project_root))
    elif args.command == "cpt":
        from profilefield.experiments.cpt import run_cpt_comparison

        for path in run_cpt_comparison(load_config(args.config), project_root):
            print(path)
    elif args.command == "biomazon-validate":
        from profilefield.data.biomazon import validate_biomazon_from_config

        try:
            result = validate_biomazon_from_config(load_config(args.config), project_root)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        print(result)
    elif args.command == "biomazon":
        from profilefield.experiments.biomazon import run_biomazon

        try:
            paths = run_biomazon(load_config(args.config), project_root)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        for path in paths:
            print(path)
    elif args.command == "validate-run":
        from profilefield.artifacts import validate_run_directory

        print(validate_run_directory(args.path))


if __name__ == "__main__":
    main()
