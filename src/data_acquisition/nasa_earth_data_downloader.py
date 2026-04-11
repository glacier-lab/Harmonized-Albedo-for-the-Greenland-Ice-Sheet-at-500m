"""
Download daily VIIRS surface reflectance granules from NASA Earthdata using earthaccess.

Examples
--------
Download Suomi-NPP VIIRS daily SR (VNP09GA) for Greenland:
python nasa_earth_data_downloader.py \
	--short-name VNP09GA \
	--start-date 2022-07-01 \
	--end-date 2022-07-31 \
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
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable, Optional, Tuple

import earthaccess


DEFAULT_GREENLAND_BBOX = (-73.0, 59.0, -11.0, 84.0)


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


def _get_granule_filename(granule) -> Optional[str]:
	"""Best-effort granule file name used by earthaccess download."""
	umm = _get_granule_umm(granule)
	producer_id = _safe_get(umm, "DataGranule", "ProducerGranuleId")
	if producer_id:
		return str(producer_id)

	for link in _get_granule_links(granule):
		path = urlparse(link).path
		name = Path(path).name
		if name:
			return name

	return None


def _get_granule_size_bytes(granule) -> Optional[int]:
	"""Best-effort granule size in bytes from UMM metadata."""
	umm = _get_granule_umm(granule)
	archive_info = _safe_get(umm, "DataGranule", "ArchiveAndDistributionInformation")
	if isinstance(archive_info, list):
		for item in archive_info:
			if not isinstance(item, dict):
				continue
			size_bytes = item.get("SizeInBytes") or item.get("Size")
			if size_bytes is None:
				continue
			try:
				return int(float(size_bytes))
			except (TypeError, ValueError):
				continue
	return None


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
	print(f"Dry run               : {args.dry_run}")
	print("=" * 72)

	print("Logging in to Earthdata...")
	earthaccess.login(strategy=args.auth_strategy, persist=args.persist_login)

	total_granules = 0
	total_downloaded = 0
	total_skipped_existing = 0

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
			filename = _get_granule_filename(granule)
			if not filename:
				to_download.append(granule)
				continue

			local_file = output_dir / filename
			if not local_file.exists():
				to_download.append(granule)
				continue

			remote_size = _get_granule_size_bytes(granule)
			local_size = local_file.stat().st_size
			if remote_size is not None and local_size == remote_size:
				total_skipped_existing += 1
				print(f"[{day}] Skip existing (same size): {filename}")
				continue

			to_download.append(granule)
			if remote_size is not None:
				print(
					f"[{day}] Re-download (size mismatch): {filename} "
					f"local={local_size}, remote={remote_size}"
				)
			else:
				print(f"[{day}] Existing file but remote size unknown, downloading: {filename}")

		if not to_download:
			print(f"[{day}] No downloads needed (all existing files matched remote size)")
			continue

		downloaded = earthaccess.download(to_download, local_path=str(output_dir))
		downloaded_count = len(downloaded) if downloaded is not None else 0
		total_downloaded += downloaded_count
		print(f"[{day}] Downloaded {downloaded_count} files (requested {len(to_download)})")

	print("\n" + "=" * 72)
	print(f"Total granules found     : {total_granules}")
	if args.dry_run:
		print("Dry run enabled          : no files downloaded")
	else:
		print(f"Total files downloaded   : {total_downloaded}")
		print(f"Total files skipped      : {total_skipped_existing} (existing with same size)")
	print("Done.")
	print("=" * 72)


if __name__ == "__main__":
	main()
