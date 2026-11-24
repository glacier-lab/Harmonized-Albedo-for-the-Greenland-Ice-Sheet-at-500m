#!/usr/bin/env python3
"""
Build a G-Portal SFTP batch file from bounded CSW searches.
This script queries the G-Portal CSW in date windows and appends unique SFTP get commands to a batch file. 
It handles pagination and can optionally overwrite existing commands.

The official G-Portal CSW documentation is available at: https://www.gportal.jaxa.jp/information/csw?lang=en
It will timeout if the specified region is too large or if the number of records exceeds the maximum allowed by G-Portal.
This script aims to loop through the date range in smaller chunks to avoid such issues.

Shunan Feng (shunan.feng@envs.au.dk)
"""

import argparse
import re
import ssl
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://csw.gportal.jaxa.jp/csw"
DEFAULT_DATASET_ID = "10002015"
DEFAULT_BBOX = (-180.0, 60.0, 180.0, 90.0)
MAX_PAGE_SIZE = 3000


class SearchError(RuntimeError):
    """Raised when a CSW response cannot be safely processed."""


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid date; use YYYY-MM-DD"
        ) from exc


def validate_bbox(values: Iterable[float]) -> Tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = values
    if not -180.0 <= min_lon <= 180.0 or not -180.0 <= max_lon <= 180.0:
        raise ValueError("longitude values must be between -180 and 180")
    if not -90.0 <= min_lat <= 90.0 or not -90.0 <= max_lat <= 90.0:
        raise ValueError("latitude values must be between -90 and 90")
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError("minimum coordinates must not exceed maximum coordinates")
    return min_lon, min_lat, max_lon, max_lat


def date_windows(
    start_date: date, end_date: date, window_days: int
) -> Iterable[Tuple[date, date]]:
    if end_date < start_date:
        raise ValueError("end date must not be before start date")
    if window_days < 1:
        raise ValueError("chunk days must be at least 1")

    window_start = start_date
    while window_start <= end_date:
        window_end = min(window_start + timedelta(days=window_days - 1), end_date)
        yield window_start, window_end
        window_start = window_end + timedelta(days=1)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_search_response(
    content: bytes,
) -> Tuple[int, int, int, List[str]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise SearchError(f"G-Portal returned invalid XML: {exc}") from exc

    search_results = next(
        (element for element in root.iter() if local_name(element.tag) == "SearchResults"),
        None,
    )
    if search_results is None:
        raise SearchError("G-Portal response does not contain SearchResults")

    try:
        matched = int(search_results.get("numberOfRecordsMatched", "0"))
        returned = int(search_results.get("numberOfRecordsReturned", "0"))
        next_record = int(search_results.get("nextRecord", "0"))
    except ValueError as exc:
        raise SearchError("G-Portal response contains invalid pagination values") from exc

    paths = []
    for reference in search_results.iter():
        if local_name(reference.tag) != "references":
            continue
        scheme = reference.get("scheme", "")
        if "FILE:DOWNLOAD" not in scheme:
            continue

        path = (reference.text or "").strip()
        if not path:
            linkage_match = re.search(r"'linkage'\s*:\s*'([^']+)'", scheme)
            if linkage_match:
                path = linkage_match.group(1)
        if path:
            paths.append(path)

    return matched, returned, next_record, paths


def normalize_download_path(path: str) -> str:
    path = path.strip()
    if path.startswith("standard/"):
        path = "/product/Standard/" + path[len("standard/") :]
    elif path.startswith("/standard/"):
        path = "/product/Standard/" + path[len("/standard/") :]
    return f"get {path}"


def request_page(
    base_url: str,
    start_date: date,
    end_date: date,
    dataset_id: str,
    bbox: Tuple[float, float, float, float],
    count: int,
    start_index: int,
    timeout: int,
    context: Optional[ssl.SSLContext] = None,
) -> bytes:
    params = {
        "service": "CSW",
        "version": "3.0.0",
        "request": "GetRecords",
        "outputFormat": "application/xml",
        "count": str(count),
        "startIndex": str(start_index),
        "startTime": f"{start_date.isoformat()}T00:00:00",
        "endTime": f"{end_date.isoformat()}T23:59:59",
        "datasetId": dataset_id,
        "bbox": ",".join(format(value, "g") for value in bbox),
    }
    request = Request(
        f"{base_url}?{urlencode(params)}",
        headers={"User-Agent": "gportal-filelist-generator/1.0"},
    )

    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            return response.read()
    except HTTPError as exc:
        raise SearchError(f"G-Portal request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise SearchError(f"Could not connect to G-Portal: {exc.reason}") from exc


def fetch_window(
    args: argparse.Namespace,
    start_date: date,
    end_date: date,
    context: Optional[ssl.SSLContext] = None,
) -> Tuple[int, List[str]]:
    commands = []
    start_index = 1
    total_matched = 0
    page = 1

    while True:
        print(f"  Page {page}, startIndex={start_index}")
        content = request_page(
            args.base_url,
            start_date,
            end_date,
            args.dataset_id,
            args.bbox,
            args.count,
            start_index,
            args.timeout,
            context,
        )
        matched, returned, next_record, paths = parse_search_response(content)
        if page == 1:
            total_matched = matched
            print(f"  Records matched: {matched}")

        commands.extend(normalize_download_path(path) for path in paths)

        if next_record == 0:
            break
        if returned == 0:
            raise SearchError(
                f"G-Portal returned an empty page with nextRecord={next_record}"
            )
        if next_record <= start_index:
            raise SearchError(
                f"G-Portal pagination did not advance ({start_index} -> {next_record})"
            )

        start_index = next_record
        page += 1
        time.sleep(args.request_delay)

    return total_matched, commands


def read_existing_commands(output_file: Path) -> Set[str]:
    if not output_file.exists():
        return set()
    return {
        line.strip()
        for line in output_file.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("get ")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query G-Portal CSW in date windows and append unique SFTP get "
            "commands to a batch file."
        )
    )
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=14,
        help="number of inclusive calendar days per CSW query (default: 14)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=DEFAULT_BBOX,
        help="ROI bounding box as min lon/lat, max lon/lat (default: -180 60 180 90)",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--count",
        type=int,
        default=MAX_PAGE_SIZE,
        help="records per page, 1-3000 (default: 3000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("files.lst"),
        help="SFTP batch file to append to (default: files.lst)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output instead of preserving existing commands",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.1,
        help="delay in seconds between CSW pages (default: 0.1)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS certificate verification (equivalent to curl -k)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.bbox = validate_bbox(args.bbox)
        windows = list(date_windows(args.start_date, args.end_date, args.chunk_days))
        if not 1 <= args.count <= MAX_PAGE_SIZE:
            raise ValueError(f"count must be between 1 and {MAX_PAGE_SIZE}")
        if args.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        if args.request_delay < 0:
            raise ValueError("request delay cannot be negative")
    except ValueError as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("", encoding="utf-8")
        known_commands = set()
    else:
        known_commands = read_existing_commands(args.output)

    context = ssl._create_unverified_context() if args.insecure else None
    total_added = 0

    try:
        for window_number, (window_start, window_end) in enumerate(windows, start=1):
            print(
                f"[{window_number}/{len(windows)}] "
                f"{window_start.isoformat()} through {window_end.isoformat()}"
            )
            matched, commands = fetch_window(args, window_start, window_end, context)
            new_commands = []
            for command in commands:
                if command not in known_commands:
                    known_commands.add(command)
                    new_commands.append(command)

            if new_commands:
                with args.output.open("a", encoding="utf-8") as stream:
                    for command in new_commands:
                        stream.write(f"{command}\n")
                    stream.flush()
            total_added += len(new_commands)
            print(
                f"  Appended {len(new_commands)} new paths "
                f"({len(commands)} returned, {matched} matched)"
            )
    except SearchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Commands appended before failure: {total_added}", file=sys.stderr)
        return 1

    print(f"Completed: {total_added} new commands appended to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
