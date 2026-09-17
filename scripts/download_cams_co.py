#!/usr/bin/env python3
"""Download monthly CAMS Europe CO, ensemble validated reanalysis.

Setup: pip install 'cdsapi>=0.7.7'
Accept the dataset licence in ADS and configure ~/.cdsapirc as described at
https://ads.atmosphere.copernicus.eu/how-to-api

Example: python scripts/download_cams_co.py --start 2020-01 --end 2020-01
Downloads the full European domain at all ten native heights. No resampling,
unit conversion, or train/validation/test splitting is performed here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


DATASET = "cams-europe-air-quality-reanalyses"
API_URL = "https://ads.atmosphere.copernicus.eu/api"
LEVELS = ["0", "50", "100", "250", "500", "750", "1000", "2000", "3000", "5000"]


def month_index(value):
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise argparse.ArgumentTypeError("Expected YYYY-MM, for example 2020-01")
    year, month = map(int, value.split("-"))
    if not 2020 <= year <= 2024:
        raise argparse.ArgumentTypeError(
            "Use 2020–2024 for this validated, ten-height experiment."
        )
    return year * 12 + month - 1


def monthly_requests(start, end):
    for index in range(start, end + 1):
        year, month = divmod(index, 12)
        yield {
            "variable": ["carbon_monoxide"],
            "model": ["ensemble"],
            "level": LEVELS.copy(),
            "type": ["validated_reanalysis"],
            "year": [str(year)],
            "month": [f"{month + 1:02d}"],
        }


def file_format(path):
    """Check downloaded container; ADS can return a ZIP containing NetCDFs."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                raise ValueError(f"Empty ZIP: {path}")
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Corrupted ZIP member: {bad}")
        return ".zip"
    with path.open("rb") as stream:
        magic = stream.read(8)
    if magic[:4] in (b"CDF\x01", b"CDF\x02", b"CDF\x05") or magic == b"\x89HDF\r\n\x1a\n":
        return ".nc"
    raise ValueError(f"Expected NetCDF or ZIP, received another format: {path}")


def download_month(client, request, output):
    metadata = {"dataset": DATASET, "request": request}
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()[:12]
    label = f"{request['year'][0]}-{request['month'][0]}"
    stem = output / f"cams_co_{label}_{fingerprint}"
    manifest = stem.with_suffix(".json")
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        for extension in (".zip", ".nc"):
            target = stem.with_suffix(extension)
            if (saved.get("dataset") == DATASET
                    and saved.get("request") == request
                    and target.exists()
                    and target.stat().st_size == saved.get("bytes")
                    and file_format(target) == extension):
                print(f"Already downloaded: {target}", flush=True)
                return target

    partial = stem.with_suffix(".part")
    # An interrupted month is downloaded again; completed months are retained.
    partial.unlink(missing_ok=True)
    print(f"Downloading {label} (all Europe, 10 heights)...", flush=True)
    client.retrieve(DATASET, request, str(partial))
    target = stem.with_suffix(file_format(partial))
    partial.replace(target)
    metadata.update({"file": target.name, "bytes": target.stat().st_size})
    temporary_manifest = stem.with_suffix(".json.part")
    temporary_manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_manifest.replace(manifest)
    print(f"Saved: {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=month_index, required=True, help="First month YYYY-MM")
    parser.add_argument("--end", type=month_index, required=True, help="Last month YYYY-MM, inclusive")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cams_co"))
    parser.add_argument("--dry-run", action="store_true", help="Print requests without login or download")
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must be >= --start")
    requests = list(monthly_requests(args.start, args.end))
    if args.dry_run:
        print(json.dumps({"dataset": DATASET, "requests": requests}, indent=2))
        return
    try:
        import cdsapi
    except ImportError:
        parser.error("Install the API client: pip install 'cdsapi>=0.7.7'")
    # Explicit ADS URL prevents accidental submission to the CDS endpoint.
    client = cdsapi.Client(url=API_URL)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Monthly requests: {len(requests)}. Output: {args.output_dir.resolve()}", flush=True)
    for request in requests:
        download_month(client, request, args.output_dir)


if __name__ == "__main__":
    main()
