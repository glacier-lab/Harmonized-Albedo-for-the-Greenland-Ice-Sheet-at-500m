"""
Download daily VIIRS surface reflectance granules from NASA Earthdata using earthaccess.

Examples
--------
Download Suomi-NPP VIIRS daily SR (VNP09GA) for Greenland:
python nasa_earth_data_downloader.py \
	--short-name VNP09GA \
	--start-date 2022-07-01 \
	--end-date 2022-07-31 \
    --bbox -180.0 -90.0 180.0 -60.0 \\
	--output-dir /data_3/shunan_2/AU/hsa500m/VIIRS_SR/VNP09GA

Download NOAA-20 VIIRS daily SR (VJ109GA) for Greenland:
python nasa_earth_data_downloader.py \
	--short-name VJ109GA \
	--start-date 2022-07-01 \
	--end-date 2022-07-31 \
	--output-dir /data_3/shunan_2/AU/hsa500m/VIIRS_SR/VJ109GA

Notes
-----
- Requires an Earthdata Login account.
- With strategy "interactive", credentials can be persisted by earthaccess.
- Failed downloads are retried with exponential backoff.
- Re-running the script skips files whose size matches the granule metadata.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable, Optional, Tuple

import earthaccess


DEFAULT_GREENLAND_BBOX = (-73.0, 59.0, -11.0, 84.0)

SIZE_UNIT_FACTORS = {
	"B": 1,
	"KB": 1024,
	"MB": 1024**2,
	"GB": 1024**3,
	"TB": 1024**4,
}


def positive_int(value: str) -> int:
	"""Parse a positive integer for argparse."""
	parsed = int(value)
	if parsed < 1:
		raise argparse.ArgumentTypeError("must be at least 1")
	return parsed


def non_negative_float(value: str) -> float:
	"""Parse a non-negative float for argparse."""
	parsed = float(value)
	if parsed < 0:
		raise argparse.ArgumentTypeError("must be at least 0")
	return parsed


def parse_date(date_str: str) -> date:
	"""Parse YYYY-MM-DD into a date object."""
	return datetime.strptime(date_str, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> Iterable[date]:
	"""Yield dates from start to end (inclusive)."""
	current = start
	while current <= end:
		yield current
		current += timedelta(days=1)


def parse_args() -> argparse.Namespace:
	"""Build and parse command line arguments."""
	parser = argparse.ArgumentParser(
		description="Download daily VIIRS surface reflectance from NASA Earthdata via earthaccess."
	)
	parser.add_argument(
		"--short-name",
		type=str,
		default="VNP09GA",
		help="CMR collection short name (e.g., VNP09GA or VJ109GA).",
	)
	parser.add_argument(
		"--version",
		type=str,
		default=None,
		help="Optional CMR collection version (e.g., 001).",
	)
	parser.add_argument(
		"--start-date",
		type=str,
		required=True,
		help="Start date in YYYY-MM-DD.",
	)
	parser.add_argument(
		"--end-date",
		type=str,
		required=True,
		help="End date in YYYY-MM-DD.",
	)
	parser.add_argument(
		"--bbox",
		nargs=4,
		type=float,
		metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
		default=DEFAULT_GREENLAND_BBOX,
		help="Spatial filter bounding box. Default is Greenland extent.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		required=True,
		help="Directory where files will be downloaded.",
	)
	parser.add_argument(
		"--auth-strategy",
		type=str,
		default="interactive",
		choices=["interactive", "netrc", "environment", "all"],
		help="earthaccess login strategy.",
	)
	parser.add_argument(
		"--persist-login",
		action="store_true",
		help="Persist login credentials when supported by strategy.",
	)
	parser.add_argument(
		"--count",
		type=int,
		default=-1,
		help="Maximum granules per day. Use -1 for all granules.",
	)
	parser.add_argument(
		"--max-retries",
		type=positive_int,
		default=5,
		help="Maximum download attempts per granule before reporting failure.",
	)
	parser.add_argument(
		"--retry-delay",
		type=non_negative_float,
		default=10.0,
		help="Base delay in seconds for exponential retry backoff.",
	)
	parser.add_argument(
		"--threads",
		type=positive_int,
		default=None,
		help="Parallel download threads. Uses earthaccess's default when omitted.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only print counts, do not download files.",
	)
	return parser.parse_args()


def build_temporal_for_day(day: date) -> Tuple[str, str]:
	"""Return one-day temporal interval in ISO format."""
	start = f"{day.isoformat()}T00:00:00"
	end = f"{day.isoformat()}T23:59:59"
	return start, end


def search_daily_granules(
	short_name: str,
	version: Optional[str],
	day: date,
	bbox: Tuple[float, float, float, float],
	count: int,
):
	"""Search one day of granules in CMR."""
	kwargs = {
		"short_name": short_name,
		"temporal": build_temporal_for_day(day),
		"bounding_box": bbox,
	}
	if version:
		kwargs["version"] = version
	if count and count > 0:
		kwargs["count"] = count
	return earthaccess.search_data(**kwargs)


def _safe_get(mapping, *keys):
	"""Safely traverse nested dict-like objects."""
	obj = mapping
	for key in keys:
		if not isinstance(obj, dict) or key not in obj:
			return None
		obj = obj[key]
	return obj


def _get_granule_umm(granule) -> dict:
	"""Return UMM metadata dict when available."""
	if isinstance(granule, dict):
		if "umm" in granule and isinstance(granule["umm"], dict):
			return granule["umm"]
		if "UMM" in granule and isinstance(granule["UMM"], dict):
			return granule["UMM"]
		return {}
	umm = getattr(granule, "umm", None)
	return umm if isinstance(umm, dict) else {}


def _get_granule_links(granule) -> list[str]:
	"""Extract data links from granule object."""
	try:
		links = granule.data_links()
		if isinstance(links, list):
			return [str(link) for link in links if link]
	except Exception:
		pass

	umm = _get_granule_umm(granule)
	umm_links = _safe_get(umm, "RelatedUrls")
	if isinstance(umm_links, list):
		urls = []
		for item in umm_links:
			if isinstance(item, dict):
				url = item.get("URL") or item.get("Url")
				if url:
					urls.append(str(url))
		return urls

	return []


def _get_granule_filenames(granule) -> list[str]:
	"""Return the local filenames earthaccess may create for a granule."""
	filenames = []
	for link in _get_granule_links(granule):
		path = urlparse(link).path
		name = Path(path).name
		if name and name not in filenames:
			filenames.append(name)

	if not filenames:
		umm = _get_granule_umm(granule)
		producer_id = _safe_get(umm, "DataGranule", "ProducerGranuleId")
		if producer_id:
			filenames.append(str(producer_id))

	return filenames


def _get_granule_size_bytes(granule, filename: str) -> Optional[int]:
	"""Return an exact byte size for a granule file when UMM metadata provides one."""
	umm = _get_granule_umm(granule)
	archive_info = _safe_get(umm, "DataGranule", "ArchiveAndDistributionInformation")
	if not isinstance(archive_info, list):
		return None

	archive_items = [item for item in archive_info if isinstance(item, dict)]
	for item in archive_items:
		name = item.get("Name")
		if name is None or str(name) != filename:
			continue
		return _parse_size_in_bytes(item)

	filenames = _get_granule_filenames(granule)
	unmatched_items = [
		item
		for item in archive_items
		if not item.get("Name") or str(item["Name"]).strip().lower() == "not provided"
	]
	if len(filenames) == 1 and len(unmatched_items) == 1:
		return _parse_size_in_bytes(unmatched_items[0])
	return None


def _parse_size_in_bytes(item: dict) -> Optional[int]:
	"""Parse a byte count from one UMM archive record."""
	size_bytes = item.get("SizeInBytes")
	if size_bytes is not None:
		try:
			return int(float(size_bytes))
		except (TypeError, ValueError):
			return None

	size = item.get("Size")
	unit = str(item.get("SizeUnit", "")).strip().upper()
	factor = SIZE_UNIT_FACTORS.get(unit)
	if size is not None and factor is not None:
		try:
			return int(round(float(size) * factor))
		except (TypeError, ValueError):
			return None
	return None


def _get_local_files(granule, output_dir: Path) -> list[Path]:
	"""Return all expected local paths for a granule."""
	return [output_dir / filename for filename in _get_granule_filenames(granule)]


def _is_complete(granule, output_dir: Path) -> bool:
	"""Return whether all known granule files exist and pass available size checks."""
	local_files = _get_local_files(granule, output_dir)
	if not local_files:
		return False

	for local_file in local_files:
		if not local_file.exists():
			return False
		remote_size = _get_granule_size_bytes(granule, local_file.name)
		if remote_size is not None and local_file.stat().st_size != remote_size:
			return False
	return True


def _remove_incomplete(granule, output_dir: Path) -> None:
	"""Remove files that are known from metadata to be incomplete."""
	for local_file in _get_local_files(granule, output_dir):
		if not local_file.exists():
			continue
		remote_size = _get_granule_size_bytes(granule, local_file.name)
		if remote_size is not None and local_file.stat().st_size != remote_size:
			print(f"  Removing incomplete file: {local_file.name}")
			local_file.unlink()


def download_with_retries(
	granules,
	output_dir: Path,
	max_retries: int,
	base_delay: float,
	threads: Optional[int],
):
	"""Download granules with backoff, preserving files known to be complete."""
	remaining = list(granules)
	completed = 0

	for attempt in range(1, max_retries + 1):
		for granule in remaining:
			_remove_incomplete(granule, output_dir)

		preexisting = {
			local_file
			for granule in remaining
			for local_file in _get_local_files(granule, output_dir)
			if local_file.exists()
		}
		last_error: Optional[Exception] = None

		try:
			if threads is None:
				earthaccess.download(remaining, local_path=str(output_dir))
			else:
				earthaccess.download(
					remaining,
					local_path=str(output_dir),
					threads=threads,
				)
		except Exception as exc:
			last_error = exc
			print(
				f"  Download attempt {attempt}/{max_retries} failed: "
				f"{type(exc).__name__}: {exc}"
			)

			for granule in remaining:
				for local_file in _get_local_files(granule, output_dir):
					if local_file in preexisting or not local_file.exists():
						continue
					remote_size = _get_granule_size_bytes(granule, local_file.name)
					if remote_size is None or local_file.stat().st_size != remote_size:
						print(f"  Removing unverified partial file: {local_file.name}")
						local_file.unlink()

		still_remaining = []
		for granule in remaining:
			if _is_complete(granule, output_dir):
				completed += 1
			else:
				still_remaining.append(granule)
		remaining = still_remaining

		if not remaining:
			return completed, []

		if attempt < max_retries:
			delay = base_delay * (2 ** (attempt - 1))
			print(
				f"  {len(remaining)} granule(s) still incomplete; "
				f"retrying in {delay:.1f}s..."
			)
			time.sleep(delay)
		elif last_error is not None:
			print(f"  Final download error: {type(last_error).__name__}: {last_error}")

	return completed, remaining


def main() -> None:
	"""Run daily search + download workflow."""
	args = parse_args()
	start = parse_date(args.start_date)
	end = parse_date(args.end_date)

	if start > end:
		raise ValueError("start-date must be <= end-date")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("=" * 72)
	print("VIIRS Surface Reflectance Downloader (earthaccess)")
	print("=" * 72)
	print(f"Collection short_name : {args.short_name}")
	print(f"Collection version    : {args.version or 'latest'}")
	print(f"Date range            : {start} to {end}")
	print(f"Bounding box          : {tuple(args.bbox)}")
	print(f"Output directory      : {output_dir}")
	print(f"Download threads      : {args.threads or 'earthaccess default'}")
	print(f"Maximum attempts      : {args.max_retries}")
	print(f"Dry run               : {args.dry_run}")
	print("=" * 72)

	print("Logging in to Earthdata...")
	earthaccess.login(strategy=args.auth_strategy, persist=args.persist_login)

	total_granules = 0
	total_downloaded = 0
	total_skipped_existing = 0
	total_failed = []

	for day in daterange(start, end):
		print(f"\n[{day}] Searching granules...")
		granules = search_daily_granules(
			short_name=args.short_name,
			version=args.version,
			day=day,
			bbox=tuple(args.bbox),
			count=args.count,
		)

		n_granules = len(granules)
		total_granules += n_granules
		print(f"[{day}] Found {n_granules} granules")

		if n_granules == 0 or args.dry_run:
			continue

		to_download = []
		for granule in granules:
			filenames = _get_granule_filenames(granule)
			if not filenames:
				to_download.append(granule)
				continue

			if _is_complete(granule, output_dir):
				total_skipped_existing += 1
				print(f"[{day}] Skip existing (complete): {', '.join(filenames)}")
				continue

			to_download.append(granule)
			print(f"[{day}] Queue incomplete download: {', '.join(filenames)}")

		if not to_download:
			print(f"[{day}] No downloads needed (all existing files are complete)")
			continue

		downloaded_count, failed = download_with_retries(
			to_download,
			output_dir,
			max_retries=args.max_retries,
			base_delay=args.retry_delay,
			threads=args.threads,
		)
		total_downloaded += downloaded_count
		print(
			f"[{day}] Completed {downloaded_count} granules "
			f"(requested {len(to_download)})"
		)

		if failed:
			total_failed.extend(failed)
			print(f"[{day}] WARNING: {len(failed)} granule(s) failed after all attempts")

	print("\n" + "=" * 72)
	print(f"Total granules found     : {total_granules}")
	if args.dry_run:
		print("Dry run enabled          : no files downloaded")
	else:
		print(f"Total granules completed : {total_downloaded}")
		print(f"Total granules skipped   : {total_skipped_existing}")
		print(f"Total granules failed    : {len(total_failed)}")
		for granule in total_failed:
			print(f"  - {', '.join(_get_granule_filenames(granule)) or '<unknown filename>'}")
	print("Done.")
	print("=" * 72)

	if total_failed:
		raise SystemExit(
			f"{len(total_failed)} granule(s) failed after {args.max_retries} attempts; "
			"re-run the command to retry incomplete downloads."
		)


if __name__ == "__main__":
	main()
