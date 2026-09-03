"""Download the complete ETH AGBD_raw release from Hugging Face to the NAS."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from huggingface_hub import HfApi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPOSITORY = "prs-eth/AGBD_raw"
REPOSITORY_TYPE = "dataset"
MIN_DOWNLOAD_BYTES = 400_000_000_000
MAX_DOWNLOAD_BYTES = 600_000_000_000
RESERVE_BYTES = 100_000_000_000
CHUNK_BYTES = 8 * 1024 * 1024
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class ExpectedFile:
    path: str
    size: int
    sha256: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(repository: str, revision: str, path: str) -> str:
    encoded = quote(path, safe="/")
    return (
        f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{encoded}?download=true"
    )


def http_session() -> requests.Session:
    existing = getattr(_THREAD_LOCAL, "session", None)
    if existing is not None:
        return existing
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "ProfileField-AGBD-downloader/0.1"})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1))
    _THREAD_LOCAL.session = session
    return session


def download_file(
    repository: str,
    revision: str,
    output: Path,
    item: ExpectedFile,
    attempts: int,
) -> int:
    destination = output / item.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size != item.size:
            raise RuntimeError(f"Existing completed file has the wrong size: {destination}")
        return 0

    partial = destination.with_suffix(destination.suffix + ".part")
    url = download_url(repository, revision, item.path)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item.size:
                with partial.open("wb"):
                    pass
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with http_session().get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(60, 600),
            ) as response:
                response.raise_for_status()
                append = bool(offset and response.status_code == 206)
                if offset and not append:
                    offset = 0
                with partial.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if chunk:
                            handle.write(chunk)
            observed_size = partial.stat().st_size
            if observed_size != item.size:
                raise RuntimeError(
                    f"Incomplete shard {item.path}: expected {item.size}, observed {observed_size}"
                )
            observed_sha256 = sha256_file(partial)
            if observed_sha256 != item.sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {item.path}: expected {item.sha256}, "
                    f"observed {observed_sha256}"
                )
            os.replace(partial, destination)
            return item.size
        except (OSError, requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 30))
    raise RuntimeError(f"Failed {item.path} after {attempts} attempts: {last_error}")


def download_all(
    repository: str,
    revision: str,
    output: Path,
    expected: list[ExpectedFile],
    workers: int,
    attempts: int,
) -> list[str]:
    completed_bytes = sum(
        item.size
        for item in expected
        if (output / item.path).is_file() and (output / item.path).stat().st_size == item.size
    )
    started = time.monotonic()
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(download_file, repository, revision, output, item, attempts): item
            for item in expected
        }
        for index, future in enumerate(concurrent.futures.as_completed(future_to_item), start=1):
            item = future_to_item[future]
            try:
                completed_bytes += future.result()
            except Exception as error:
                message = f"{item.path}: {error}"
                failures.append(message)
                print(f"FAILED {message}", file=sys.stderr, flush=True)
            elapsed = max(time.monotonic() - started, 0.001)
            rate = completed_bytes / elapsed
            print(
                f"Progress {index}/{len(expected)}: {completed_bytes / 1_000_000_000:.2f} GB "
                f"verified, average {rate / 1_000_000:.2f} MB/s",
                flush=True,
            )
    return failures


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def build_plan(api: HfApi, repository: str) -> tuple[str, list[ExpectedFile]]:
    info = api.dataset_info(repository, files_metadata=False)
    revision = str(info.sha)
    expected: list[ExpectedFile] = []
    for item in api.list_repo_tree(
        repository,
        path_in_repo="data",
        recursive=True,
        expand=True,
        revision=revision,
        repo_type=REPOSITORY_TYPE,
    ):
        path = getattr(item, "path", "")
        lfs = getattr(item, "lfs", None)
        if path.endswith(".parquet") and lfs is not None:
            expected.append(ExpectedFile(path=path, size=int(lfs.size), sha256=str(lfs.sha256)))
    expected.sort(key=lambda item: item.path)
    if not expected:
        raise RuntimeError("No Parquet shards were returned by the Hugging Face dataset API")
    return revision, expected


def validate_size(expected: list[ExpectedFile], output: Path) -> int:
    total = sum(item.size for item in expected)
    if not MIN_DOWNLOAD_BYTES <= total <= MAX_DOWNLOAD_BYTES:
        raise RuntimeError(
            f"Planned release size {total:,} bytes is outside the approved "
            f"{MIN_DOWNLOAD_BYTES:,}-{MAX_DOWNLOAD_BYTES:,} byte interval"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    missing = sum(
        item.size
        for item in expected
        if not (output / item.path).is_file() or (output / item.path).stat().st_size != item.size
    )
    if missing + RESERVE_BYTES > free:
        raise RuntimeError(
            f"Insufficient NAS space: {missing + RESERVE_BYTES:,} bytes required with reserve, "
            f"{free:,} bytes free"
        )
    return total


def verify_files(
    output: Path,
    expected: list[ExpectedFile],
    verify_sha256: bool,
) -> tuple[int, list[str]]:
    observed_bytes = 0
    problems: list[str] = []
    for index, item in enumerate(expected, start=1):
        path = output / item.path
        if not path.is_file():
            problems.append(f"missing: {item.path}")
            continue
        size = path.stat().st_size
        observed_bytes += size
        if size != item.size:
            problems.append(f"size: {item.path}: expected {item.size}, observed {size}")
            continue
        if verify_sha256 and sha256_file(path) != item.sha256:
            problems.append(f"sha256: {item.path}")
        if index % 100 == 0:
            print(f"Verified {index}/{len(expected)} files", flush=True)
    return observed_bytes, problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/agbd_raw"))
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--file-retries", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Re-read all 418 GB after download; Hub/Xet already checks transfer integrity",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.file_retries < 1:
        raise SystemExit("--workers and --file-retries must be positive")
    output = args.output.resolve()
    api = HfApi()
    print(f"Reading immutable release metadata for {args.repository}...", flush=True)
    revision, expected = build_plan(api, args.repository)
    total = validate_size(expected, output)
    splits: dict[str, dict[str, int]] = {}
    for item in expected:
        split = Path(item.path).name.split("-", maxsplit=1)[0]
        stats = splits.setdefault(split, {"files": 0, "bytes": 0})
        stats["files"] += 1
        stats["bytes"] += item.size

    plan = {
        "repository": args.repository,
        "repository_type": REPOSITORY_TYPE,
        "revision": revision,
        "created_at": utc_now(),
        "download_bytes": total,
        "download_gb_decimal": total / 1_000_000_000,
        "splits": splits,
        "files": [asdict(item) for item in expected],
    }
    atomic_json(output / "download_plan.json", plan)
    print(
        f"Plan: {len(expected)} Parquet shards, {total:,} bytes "
        f"({total / 1_000_000_000:.2f} GB), revision {revision}",
        flush=True,
    )
    if args.plan_only:
        return 0

    failures = download_all(
        args.repository,
        revision,
        output,
        expected,
        args.workers,
        args.file_retries,
    )
    if failures:
        atomic_json(
            output / "download_failures.json",
            {"updated_at": utc_now(), "failures": failures},
        )
        print(
            f"ERROR: {len(failures)} shards failed; rerun the same command to resume",
            file=sys.stderr,
            flush=True,
        )
        return 1
    observed_bytes, problems = verify_files(output, expected, args.verify_sha256)
    completion = {
        "repository": args.repository,
        "revision": revision,
        "completed_at": utc_now(),
        "expected_files": len(expected),
        "expected_bytes": total,
        "observed_bytes": observed_bytes,
        "sha256_reread": bool(args.verify_sha256),
        "problems": problems,
    }
    atomic_json(output / "download_complete.json", completion)
    if problems:
        print(f"ERROR: verification found {len(problems)} problems", file=sys.stderr, flush=True)
        return 1
    print(f"Download complete and verified: {observed_bytes:,} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
