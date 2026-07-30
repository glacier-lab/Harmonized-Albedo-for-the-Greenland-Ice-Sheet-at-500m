"""Download files from an ERDA anonymous share listing URL.

Example:
	python src/erda/download_from_erda.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


class ErdaListingParser(HTMLParser):
	"""Extract file paths and fallback file links from ERDA listing HTML."""

	def __init__(self) -> None:
		super().__init__()
		self.checkbox_paths: List[str] = []
		self.file_links: List[str] = []

	def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
		attr = {k: (v or "") for k, v in attrs}

		if tag == "input":
			if attr.get("type") == "checkbox" and attr.get("name") == "path":
				path = attr.get("value", "").strip()
				if path:
					self.checkbox_paths.append(path)

		if tag == "a":
			href = attr.get("href", "").strip()
			if href.startswith("/share_redirect/"):
				self.file_links.append(href)


def build_listing_url(raw_url: str) -> tuple[str, str, str]:
	"""Normalize listing URL and return (listing_url, share_id, origin)."""
	parsed = urlparse(raw_url)
	if not parsed.scheme or not parsed.netloc:
		raise ValueError("The listing URL must be an absolute URL.")

	query = parse_qs(parsed.query)
	share_id = query.get("share_id", [""])[0]
	current_dir = query.get("current_dir", [""])[0].strip()
	flags = query.get("flags", ["f"])[0]

	if not share_id:
		raise ValueError("Missing 'share_id' in listing URL query parameters.")

	# Some ERDA share links only provide share_id. In that case, list root.
	if not current_dir:
		current_dir = "."

	params = {
		"share_id": share_id,
		"current_dir": current_dir,
		"flags": flags,
		"output_format": "html",
	}
	listing_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{urlencode(params)}"
	origin = f"{parsed.scheme}://{parsed.netloc}"
	return listing_url, share_id, origin


def fetch_listing_html(listing_url: str) -> str:
	req = Request(listing_url, headers={"User-Agent": "Mozilla/5.0"})
	with urlopen(req) as resp:
		return resp.read().decode("utf-8", errors="replace")


def extract_download_urls(html: str, share_id: str, origin: str) -> List[str]:
	parser = ErdaListingParser()
	parser.feed(html)

	urls: List[str] = []
	seen: set[str] = set()

	# Preferred source: checkbox paths exactly represent files listed in the table.
	for path in parser.checkbox_paths:
		safe_path = quote(path, safe="/")
		url = f"{origin}/share_redirect/{share_id}/{safe_path}"
		if url not in seen:
			seen.add(url)
			urls.append(url)

	# Fallback source in case checkbox paths are not present in a future page layout.
	if not urls:
		for href in parser.file_links:
			url = urljoin(origin, href)
			if url not in seen:
				seen.add(url)
				urls.append(url)

	return urls


def filename_from_url(url: str) -> str:
	return Path(urlparse(url).path).name


def format_bytes(num_bytes: float) -> str:
	units = ["B", "KB", "MB", "GB", "TB"]
	value = float(num_bytes)
	for unit in units:
		if value < 1024.0 or unit == units[-1]:
			if unit == "B":
				return f"{int(value)} {unit}"
			return f"{value:.2f} {unit}"
		value /= 1024.0
	return f"{value:.2f} TB"


def download_file(
	url: str,
	output_path: Path,
	overwrite: bool = False,
	index: int | None = None,
	total: int | None = None,
) -> str:
	if output_path.exists() and not overwrite:
		return "skipped"

	output_path.parent.mkdir(parents=True, exist_ok=True)
	req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
	with urlopen(req) as resp, output_path.open("wb") as f:
		chunk_size = 1024 * 1024
		total_size_header = resp.headers.get("Content-Length")
		total_size = int(total_size_header) if total_size_header and total_size_header.isdigit() else None
		downloaded = 0
		start_time = time.perf_counter()
		last_update = start_time

		prefix = ""
		if index is not None and total is not None:
			prefix = f"[{index}/{total}] "

		while True:
			chunk = resp.read(chunk_size)
			if not chunk:
				break
			f.write(chunk)
			downloaded += len(chunk)

			now = time.perf_counter()
			if now - last_update >= 0.2:
				elapsed = max(now - start_time, 1e-9)
				speed = downloaded / elapsed
				if total_size:
					pct = (downloaded / total_size) * 100
					msg = (
						f"{prefix}downloading {output_path.name} | "
						f"{format_bytes(downloaded)}/{format_bytes(total_size)} "
						f"({pct:5.1f}%) | {format_bytes(speed)}/s"
					)
				else:
					msg = (
						f"{prefix}downloading {output_path.name} | "
						f"{format_bytes(downloaded)} | {format_bytes(speed)}/s"
					)
				print(msg, end="\r", flush=True)
				last_update = now

		elapsed = max(time.perf_counter() - start_time, 1e-9)
		speed = downloaded / elapsed
		if total_size:
			pct = (downloaded / total_size) * 100
			final_msg = (
				f"{prefix}downloading {output_path.name} | "
				f"{format_bytes(downloaded)}/{format_bytes(total_size)} "
				f"({pct:5.1f}%) | {format_bytes(speed)}/s"
			)
		else:
			final_msg = (
				f"{prefix}downloading {output_path.name} | "
				f"{format_bytes(downloaded)} | {format_bytes(speed)}/s"
			)
		print(final_msg)

	return "downloaded"


def prompt_text(label: str, default: str) -> str:
	value = input(f"{label} [{default}]: ").strip()
	return value if value else default


def prompt_yes_no(label: str, default: bool = False) -> bool:
	default_text = "y" if default else "n"
	while True:
		value = input(f"{label} (y/n) [{default_text}]: ").strip().lower()
		if not value:
			return default
		if value in {"y", "yes"}:
			return True
		if value in {"n", "no"}:
			return False
		print("Please type 'y' or 'n'.")


def prompt_optional_int(label: str, default: int | None = None) -> int | None:
	default_text = "none" if default is None else str(default)
	while True:
		value = input(f"{label} [{default_text}]: ").strip().lower()
		if not value:
			return default
		if value in {"none", "null", "-"}:
			return None
		if value.isdigit():
			return int(value)
		print("Please enter a non-negative integer or 'none'.")


def main() -> None:
	default_url = "https://anon.erda.au.dk/cgi-sid/ls.py?share_id=homDKceSWs&current_dir=daily_maps&flags=f"
	default_output = "data/erda_downloads"

	argp = argparse.ArgumentParser(
		description="Find files from an ERDA share listing URL and download them.",
	)
	argp.add_argument(
		"--listing-url",
		default=None,
		help="ERDA listing URL (ls.py?...).",
	)
	argp.add_argument(
		"--output-dir",
		default=None,
		help="Directory where downloaded files are saved.",
	)
	argp.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Only process the first N files.",
	)
	argp.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite files that already exist.",
	)
	argp.add_argument(
		"--dry-run",
		action="store_true",
		help="Only print discovered URLs, do not download files (default: off).",
	)
	argp.add_argument(
		"--no-input",
		action="store_true",
		help="Run non-interactively and use defaults for missing values.",
	)
	args = argp.parse_args()

	print("ERDA download configuration")
	print("Press Enter to accept defaults.")

	if args.no_input:
		listing_url_input = args.listing_url or default_url
		output_dir_input = args.output_dir or default_output
		dry_run = args.dry_run
		overwrite = args.overwrite
		limit = args.limit
	else:
		listing_url_input = prompt_text("Listing URL", args.listing_url or default_url)
		output_dir_input = prompt_text("Output directory", args.output_dir or default_output)
		dry_run = prompt_yes_no("Dry run (do not download)", default=args.dry_run)
		overwrite = prompt_yes_no("Overwrite existing files", default=args.overwrite)
		limit = prompt_optional_int("Limit number of files (or 'none')", default=args.limit)

	listing_url, share_id, origin = build_listing_url(listing_url_input)
	html = fetch_listing_html(listing_url)
	urls = extract_download_urls(html, share_id, origin)

	if limit is not None and limit >= 0:
		urls = urls[:limit]

	if not urls:
		raise RuntimeError("No files were found in the provided listing URL.")

	print(f"Found {len(urls)} files")
	if dry_run:
		for url in urls:
			print(url)
		return

	output_dir = Path(output_dir_input)
	downloaded = 0
	skipped = 0

	for i, url in enumerate(urls, start=1):
		filename = filename_from_url(url)
		output_path = output_dir / filename
		status = download_file(url, output_path, overwrite=overwrite, index=i, total=len(urls))

		if status == "downloaded":
			downloaded += 1
		else:
			skipped += 1

		print(f"[{i}/{len(urls)}] {status}: {os.fspath(output_path)}")

	print(f"Done. downloaded={downloaded}, skipped={skipped}, total={len(urls)}")


if __name__ == "__main__":
	main()
