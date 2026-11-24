#!/usr/bin/env python3
"""Download G-Portal SFTP batch entries with configurable parallelism."""

import argparse
import os
import posixpath
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence


class DownloadItem(NamedTuple):
    remote_path: str
    local_path: Path


class DownloadFailure(NamedTuple):
    remote_path: str
    message: str


def parse_batch_file(batch_file: Path, destination: Path) -> List[DownloadItem]:
    if not batch_file.is_file():
        raise ValueError(f"batch file does not exist: {batch_file}")

    items = []
    seen_paths = set()
    destination_paths = set()

    for line_number, raw_line in enumerate(
        batch_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] != "get":
            raise ValueError(
                f"{batch_file}:{line_number}: expected 'get /remote/path'"
            )

        remote_path = parts[1].strip()
        if not remote_path.startswith("/"):
            raise ValueError(
                f"{batch_file}:{line_number}: remote path must be absolute"
            )
        if remote_path in seen_paths:
            continue

        local_path = destination / posixpath.basename(remote_path)
        if not local_path.name:
            raise ValueError(
                f"{batch_file}:{line_number}: remote path has no filename"
            )
        if local_path.name in destination_paths:
            raise ValueError(
                f"{batch_file}:{line_number}: duplicate local filename "
                f"{local_path.name!r}"
            )

        seen_paths.add(remote_path)
        destination_paths.add(local_path.name)
        items.append(DownloadItem(remote_path, local_path))

    if not items:
        raise ValueError(f"batch file contains no 'get' commands: {batch_file}")
    return items


def pending_items(items: Iterable[DownloadItem]) -> List[DownloadItem]:
    return [item for item in items if not item.local_path.is_file()]


def quote_sftp_argument(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def download_worker(
    worker_id: int,
    jobs: Sequence[DownloadItem],
    host: str,
    port: int,
    username: str,
    private_key: Path,
    timeout: int,
    retries: int,
    sftp_command: str,
    progress: "Progress",
) -> List[DownloadFailure]:
    temporary_batch: Optional[Path] = None
    last_error = ""

    try:
        progress.status(worker_id, f"{len(jobs)} files assigned")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f"gportal-worker-{worker_id}-",
            suffix=".lst",
            delete=False,
        ) as stream:
            temporary_batch = Path(stream.name)
            for item in jobs:
                temporary_path = item.local_path.with_name(
                    f"{item.local_path.name}.part"
                )
                stream.write(
                    f"get {quote_sftp_argument(item.remote_path)} "
                    f"{quote_sftp_argument(str(temporary_path))}\n"
                )

        command = [
            sftp_command,
            f"-oPort={port}",
            "-oBatchMode=yes",
            f"-oConnectTimeout={timeout}",
            "-i",
            str(private_key),
            "-b",
            str(temporary_batch),
            f"{username}@{host}",
        ]

        for attempt in range(1, retries + 2):
            for item in jobs:
                temporary_path = item.local_path.with_name(
                    f"{item.local_path.name}.part"
                )
                if temporary_path.exists():
                    temporary_path.unlink()

            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode == 0:
                for item in jobs:
                    temporary_path = item.local_path.with_name(
                        f"{item.local_path.name}.part"
                    )
                    temporary_path.replace(item.local_path)
                    progress.completed(worker_id, item.local_path)
                return []

            output = result.stdout.strip()
            last_error = (
                f"sftp exited with code {result.returncode}"
                + (f": {output}" if output else "")
            )
            if attempt <= retries:
                progress.status(
                    worker_id, f"retry {attempt}/{retries}: {last_error}"
                )
                time.sleep(min(2 ** (attempt - 1), 10))

        failures = [
            DownloadFailure(item.remote_path, last_error) for item in jobs
        ]
        for item in jobs:
            progress.failed(worker_id, item.local_path, last_error)
        return failures
    finally:
        if temporary_batch is not None:
            temporary_batch.unlink(missing_ok=True)


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.completed_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()

    def status(self, worker_id: int, message: str) -> None:
        with self.lock:
            print(f"[worker {worker_id}] {message}", file=sys.stderr)

    def completed(self, worker_id: int, path: Path) -> None:
        with self.lock:
            self.completed_count += 1
            print(
                f"[worker {worker_id}] downloaded "
                f"({self.completed_count}/{self.total}) {path.name}",
                file=sys.stderr,
            )

    def failed(self, worker_id: int, path: Path, message: str) -> None:
        with self.lock:
            self.failed_count += 1
            print(
                f"[worker {worker_id}] FAILED {path.name}: {message}",
                file=sys.stderr,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download G-Portal files from an SFTP get-command list using "
            "multiple persistent connections."
        )
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=Path("files.lst"),
        help="file containing 'get /remote/path' commands (default: files.lst)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("."),
        help="local download directory (default: current directory)",
    )
    parser.add_argument("--host", default="sftp.gportal.jaxa.jp")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument(
        "--username",
        default=os.environ.get("GPORTAL_USERNAME", "geoshunan"),
        help="G-Portal username (default: $GPORTAL_USERNAME or geoshunan)",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        default=Path(
            os.environ.get(
                "GPORTAL_PRIVATE_KEY", "/data/shunan/software/privatekey.key"
            )
        ),
        help=(
            "SSH private key (default: $GPORTAL_PRIVATE_KEY or "
            "/data/shunan/software/privatekey.key)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel SFTP connections (default: 2)",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--sftp-command",
        default="sftp",
        help="OpenSSH-compatible sftp executable (default: sftp)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="retries per file after a connection or transfer error (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the batch and print pending files without connecting",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        items = parse_batch_file(args.batch_file, args.destination)
        if args.workers < 1:
            raise ValueError("workers must be at least 1")
        if args.port < 1 or args.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if args.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        if args.retries < 0:
            raise ValueError("retries cannot be negative")
        if not args.dry_run and shutil.which(args.sftp_command) is None:
            raise ValueError(f"SFTP executable was not found: {args.sftp_command}")
        if not args.dry_run and not args.private_key.is_file():
            raise ValueError(f"private key does not exist: {args.private_key}")
    except ValueError as exc:
        parser.error(str(exc))

    pending = pending_items(items)
    skipped = len(items) - len(pending)
    print(
        f"Batch contains {len(items)} unique files; "
        f"{skipped} already present, {len(pending)} pending."
    )

    if not pending:
        print("Nothing to download.")
        return 0

    if args.dry_run:
        for item in pending:
            print(item.remote_path)
        return 0

    args.destination.mkdir(parents=True, exist_ok=True)
    worker_count = min(args.workers, len(pending))
    jobs = [pending[index::worker_count] for index in range(worker_count)]
    progress = Progress(len(pending))
    started_at = time.monotonic()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                download_worker,
                worker_id,
                jobs[worker_id - 1],
                args.host,
                args.port,
                args.username,
                args.private_key,
                args.timeout,
                args.retries,
                args.sftp_command,
                progress,
            )
            for worker_id in range(1, worker_count + 1)
        ]
        failures = [
            failure
            for future in futures
            for failure in future.result()
        ]

    elapsed = time.monotonic() - started_at
    succeeded = progress.completed_count
    print(
        f"Finished: {succeeded}/{len(pending)} downloaded, "
        f"{len(failures)} failed in {elapsed:.1f}s using {worker_count} worker(s)."
    )
    if failures:
        print("Failed files:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure.remote_path}: {failure.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
