"""Resumable and checksum-verified downloader for the official Biomazon release."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_URL = "https://data.fz-juelich.de/api/access/datafile/50331"
BASE_URL = "https://datapub.fz-juelich.de/biomazon/"
H5_SUFFIX = ".h5"
CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_INTERVAL_SECONDS = 15.0


@dataclass(frozen=True)
class Entry:
    sha256: str
    relative_path: PurePosixPath

    @property
    def is_h5(self) -> bool:
        return self.relative_path.suffix.lower() == H5_SUFFIX


def parse_manifest(text: str) -> list[Entry]:
    """Parse sha256sum output while rejecting unsafe paths and malformed hashes."""
    entries: list[Entry] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed manifest line {line_number}: {raw_line!r}")
        digest, raw_path = parts
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError(f"Invalid SHA-256 on manifest line {line_number}")
        relative_path = PurePosixPath(raw_path.lstrip("*"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe path on manifest line {line_number}: {raw_path!r}")
        entries.append(Entry(digest.lower(), relative_path))
    if not entries:
        raise ValueError("The Biomazon manifest is empty")
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_manifest(url: str, output_root: Path, timeout: float) -> list[Entry]:
    """Refresh the official manifest, falling back to a previously verified copy."""
    destination = output_root / "DATASET_SHA256SUMS.txt"
    request = urllib.request.Request(url, headers={"User-Agent": "ProfileField/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        entries = parse_manifest(payload.decode("utf-8"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        print(f"Manifest refreshed: {destination} ({len(entries)} files)")
        return entries
    except (OSError, UnicodeError, ValueError, urllib.error.URLError) as error:
        if not destination.is_file():
            raise RuntimeError(f"Cannot fetch the official manifest: {error}") from error
        payload = destination.read_bytes()
        entries = parse_manifest(payload.decode("utf-8"))
        print(f"Warning: using cached manifest because refresh failed: {error}", file=sys.stderr)
        return entries


def select_entries(entries: list[Entry], selection: str) -> list[Entry]:
    if selection == "support":
        return [entry for entry in entries if not entry.is_h5]
    if selection == "test":
        return [
            entry
            for entry in entries
            if not entry.is_h5 or entry.relative_path.name.startswith("test_")
        ]
    return entries


def source_url(base_url: str, entry: Entry) -> str:
    escaped_path = "/".join(urllib.parse.quote(part) for part in entry.relative_path.parts)
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", escaped_path)


def parse_total_size(headers: Any, status: int) -> int | None:
    content_range = headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", maxsplit=1)[1]
        if total.isdigit():
            return int(total)
    content_length = headers.get("Content-Length")
    if status == 200 and content_length and content_length.isdigit():
        return int(content_length)
    return None


def probe_size(url: str, timeout: float) -> int | None:
    """Use a one-byte range request so a probe cannot download a large HDF5 body."""
    request = urllib.request.Request(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "ProfileField/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_total_size(response.headers, response.status)


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if amount < 1024.0 or unit == "PiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def ensure_capacity(
    destination: Path,
    expected_size: int | None,
    max_bytes: int,
    allow_large: bool,
) -> None:
    if expected_size is None and not allow_large:
        raise RuntimeError(
            "The server did not report the HDF5 size; refusing an unbounded download. "
            "Re-run with --allow-large only after checking the target filesystem."
        )
    if expected_size is None:
        return
    if expected_size > max_bytes and not allow_large:
        raise RuntimeError(
            f"File is {human_bytes(expected_size)}, above the {human_bytes(max_bytes)} "
            "safety limit. Re-run with --allow-large if the storage is ready."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination.parent).free
    partial_size = (
        destination.with_name(destination.name + ".part").stat().st_size
        if (destination.with_name(destination.name + ".part").exists())
        else 0
    )
    remaining = max(0, expected_size - partial_size)
    reserve = min(10 * 1024**3, expected_size // 20)
    if remaining + reserve > free_bytes:
        raise RuntimeError(
            f"Insufficient free space: need about {human_bytes(remaining + reserve)}, "
            f"have {human_bytes(free_bytes)}."
        )


def open_with_retries(
    request: urllib.request.Request,
    timeout: float,
    retries: int,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(f"  connection attempt {attempt}/{retries} failed; retrying in {delay}s")
                time.sleep(delay)
    raise RuntimeError(f"Download connection failed after {retries} attempts: {last_error}")


def download_entry(
    entry: Entry,
    base_url: str,
    output_root: Path,
    timeout: float,
    retries: int,
    max_bytes: int,
    allow_large: bool,
) -> None:
    destination = output_root.joinpath(*entry.relative_path.parts)
    url = source_url(base_url, entry)
    if destination.is_file():
        print(f"Checking existing file: {entry.relative_path}")
        if sha256_file(destination) == entry.sha256:
            print("  checksum OK; skipped")
            return
        raise RuntimeError(
            f"Existing file has the wrong checksum: {destination}. Move it aside before retrying."
        )

    if entry.is_h5:
        print(f"Probing size: {entry.relative_path}")
        expected_size = probe_size(url, timeout)
        label = human_bytes(expected_size) if expected_size is not None else "unknown"
        print(f"  reported size: {label}")
        ensure_capacity(destination, expected_size, max_bytes, allow_large)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "ProfileField/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)

    print(
        f"Downloading: {entry.relative_path}"
        + (f" (resume at {human_bytes(offset)})" if offset else "")
    )
    response = open_with_retries(request, timeout, retries)
    status = response.status
    append = bool(offset and status == 206)
    if offset and not append:
        print("  server ignored Range; restarting this partial file")
        offset = 0

    downloaded = offset
    started = time.monotonic()
    last_report = started
    try:
        with response, partial.open("ab" if append else "wb") as handle:
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= PROGRESS_INTERVAL_SECONDS:
                    elapsed = max(now - started, 0.001)
                    rate = (downloaded - offset) / elapsed
                    print(f"  {human_bytes(downloaded)} at {human_bytes(int(rate))}/s")
                    last_report = now
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(
            f"Transfer interrupted at {human_bytes(downloaded)}; run again to resume: {error}"
        ) from error

    print("  verifying SHA-256...")
    observed = sha256_file(partial)
    if observed != entry.sha256:
        raise RuntimeError(
            f"Checksum mismatch for {partial}: expected {entry.sha256}, observed {observed}"
        )
    os.replace(partial, destination)
    print(f"  complete: {human_bytes(destination.stat().st_size)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/biomazon"),
        help="Destination root (default: data/biomazon, which is git-ignored)",
    )
    parser.add_argument(
        "--selection",
        choices=("support", "test", "all"),
        default="support",
        help="support: metadata/splits only; test: support + test H5; all: entire release",
    )
    parser.add_argument("--manifest-url", default=MANIFEST_URL)
    parser.add_argument("--base-url", default=BASE_URL, help="Official root or a trusted mirror")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-connection timeout")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--max-gib",
        type=float,
        default=100.0,
        help="Refuse any single H5 above this size unless --allow-large is set",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow H5 files above the safety limit or with an unknown reported size",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Probe selected URLs and sizes without downloading dataset files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.retries < 1 or args.timeout <= 0 or args.max_gib <= 0:
        raise SystemExit("--retries, --timeout, and --max-gib must be positive")
    output_root = args.output.resolve()
    entries = select_entries(
        fetch_manifest(args.manifest_url, output_root, args.timeout), args.selection
    )
    print(f"Selected {len(entries)} files ({args.selection}); destination: {output_root}")

    try:
        if args.probe_only:
            for entry in entries:
                size = probe_size(source_url(args.base_url, entry), args.timeout)
                print(
                    f"{human_bytes(size) if size is not None else 'unknown':>12}  {entry.relative_path}"
                )
        else:
            max_bytes = int(args.max_gib * 1024**3)
            for entry in entries:
                download_entry(
                    entry,
                    args.base_url,
                    output_root,
                    args.timeout,
                    args.retries,
                    max_bytes,
                    args.allow_large,
                )
    except (OSError, RuntimeError, urllib.error.URLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
