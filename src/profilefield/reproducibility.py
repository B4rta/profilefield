"""Determinism, hashing, and environment provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without touching test data."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_tree(root: str | Path) -> str:
    """Hash paths and contents, excluding Git administrative files."""
    directory = Path(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if ".git" in path.parts:
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_sha(root: str | Path) -> str:
    executable = shutil.which("git")
    if executable is None:
        bundled = (
            Path.home()
            / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
        )
        executable = str(bundled) if bundled.exists() else None
    if executable is None:
        return "uncommitted"
    try:
        result = subprocess.run(
            [executable, "-C", str(Path(root).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def environment_snapshot(project_root: str | Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("torch", "gpytorch", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "git_sha": git_sha(project_root),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
    }
