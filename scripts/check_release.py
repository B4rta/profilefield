"""Reject private research material in the Git index before publishing."""

from __future__ import annotations

import subprocess
from pathlib import PurePosixPath

PRIVATE_PREFIXES = ("manuscript/", "paper_results/", "docs/", "outputs/", "notebooks/")
PRIVATE_FILES = {
    "AGENTS.md", "configs/synthetic/paper.yaml", "src/profilefield/experiments/paper.py",
    "tests/test_paper_statistics.py", "scripts/build_manuscript.py",
    "scripts/build_paper_results.py", "scripts/validate_manuscript.py",
}
PRIVATE_SUFFIXES = {".docx", ".pdf", ".svg", ".png", ".pt", ".pth", ".ckpt", ".parquet", ".h5", ".hdf5", ".tif", ".tiff"}


def forbidden_paths(paths: list[str]) -> list[str]:
    return sorted(path for path in paths if (
        path.startswith(PRIVATE_PREFIXES) or path in PRIVATE_FILES
        or (path.startswith("data/") and path != "data/README.md")
        or PurePosixPath(path).suffix.lower() in PRIVATE_SUFFIXES
        or PurePosixPath(path).name.startswith(".env")
    ))


def main() -> None:
    result = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True)
    paths = [path for path in result.stdout.decode("utf-8").split("\0") if path]
    blocked = forbidden_paths(paths)
    if blocked:
        raise SystemExit("Private material must be removed from Git tracking:\n" + "\n".join(blocked))
    print(f"Release privacy check passed: {len(paths)} tracked paths")


if __name__ == "__main__":
    main()
