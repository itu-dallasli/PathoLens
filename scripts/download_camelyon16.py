"""
CAMELYON16 downloader with a hard disk-budget cap.

Pulls slides + lesion annotations from the AWS Open Data S3 mirror over
plain HTTPS — no boto3, no AWS account, no credentials.

Features
--------
* Resumable downloads (HTTP Range), skips files already complete.
* `--max-gb` budget: stops before exceeding the cap.
* Three groups: `annotations`, `tumor`, `normal`, `test`. Pick any combo.
* Priority ordering: annotations first (tiny, free), then tumors
  (annotated → useful for training), then normals.
* `--list` dry-run mode to preview what would be downloaded.
* Manifest JSON so re-runs are idempotent.

Usage
-----
    # Smallest possible: just annotations (~50 MB)
    python scripts/download_camelyon16.py --groups annotations \\
        --dest X:/Bitirme_Data/CAMELYON16

    # Recommended 300 GB plan: all tumors + 10 normals + annotations
    python scripts/download_camelyon16.py \\
        --groups annotations tumor normal \\
        --normal-limit 10 \\
        --max-gb 300 \\
        --dest X:/Bitirme_Data/CAMELYON16

    # Dry-run preview
    python scripts/download_camelyon16.py --groups tumor --list

    # Resume an interrupted run
    python scripts/download_camelyon16.py --groups tumor \\
        --dest X:/Bitirme_Data/CAMELYON16        # picks up where it stopped
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


# ── Configuration ────────────────────────────────────────────

# AWS Open Data Registry: https://registry.opendata.aws/camelyon/
# Verified layout (probed 2026-04-07):
#   s3://camelyon-dataset/CAMELYON16/images/         flat: normal_*.tif tumor_*.tif test_*.tif
#   s3://camelyon-dataset/CAMELYON16/annotations/    XML lesion polygons (tumor + test)
#   s3://camelyon-dataset/CAMELYON16/masks/          pre-rasterized binary tumor mask TIFs
DEFAULT_BUCKET = "camelyon-dataset"
DEFAULT_REGION = "us-west-2"

# Each group: (prefix, filename_filter or None)
GROUP_SPECS: dict[str, tuple[str, Optional[str]]] = {
    "annotations": ("CAMELYON16/annotations/", None),
    "masks":       ("CAMELYON16/masks/",       None),
    "tumor":       ("CAMELYON16/images/",      "tumor_"),
    "normal":      ("CAMELYON16/images/",      "normal_"),
    "test":        ("CAMELYON16/images/",      "test_"),
}

# S3 XML namespace
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

CHUNK = 1024 * 1024  # 1 MiB
GB = 1024 ** 3


# ── Data types ───────────────────────────────────────────────

@dataclass
class S3Object:
    key: str
    size: int

    @property
    def filename(self) -> str:
        return self.key.rsplit("/", 1)[-1]


# ── S3 helpers (no boto3) ────────────────────────────────────

def _bucket_url(bucket: str, region: str) -> str:
    # Virtual-hosted style works for public buckets in any region.
    if region in (None, "", "us-east-1"):
        return f"https://{bucket}.s3.amazonaws.com"
    return f"https://{bucket}.s3.{region}.amazonaws.com"


def _try_read(url: str) -> tuple[Optional[bytes], Optional[urllib.error.HTTPError]]:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read(), None
    except urllib.error.HTTPError as e:
        return e.read() if hasattr(e, "read") else None, e


def _discover_endpoint(bucket: str, hint_region: str) -> Optional[str]:
    """
    Probe a bucket to find its real endpoint, even if the region is wrong.
    Returns the base URL (no trailing slash) or None.

    Strategy: send a HEAD-like list, parse the AWS error XML for either
    <Endpoint> or <Region>, then build the correct virtual-hosted URL.
    """
    # Try a few candidates first; S3 path-style endpoint always 307s to the
    # correct region with a usable Location header / error body.
    candidates = [
        _bucket_url(bucket, hint_region),
        f"https://s3.amazonaws.com/{bucket}",
        f"https://{bucket}.s3.amazonaws.com",
    ]
    for base in candidates:
        body, err = _try_read(f"{base}/?list-type=2&max-keys=1")
        if err is None:
            return base
        if body:
            try:
                root = ET.fromstring(body)
                endpoint = root.findtext("Endpoint")
                region = root.findtext("Region")
                if endpoint:
                    return f"https://{endpoint}"
                if region:
                    return _bucket_url(bucket, region)
            except ET.ParseError:
                pass
    return None


def list_objects(
    bucket: str, region: str, prefix: str, base_override: Optional[str] = None
) -> List[S3Object]:
    """
    List all objects under `prefix` using the S3 REST ListObjectsV2 API.
    Handles pagination via continuation tokens. Auto-discovers the correct
    region if the supplied one is wrong.
    """
    base = base_override or _discover_endpoint(bucket, region) or _bucket_url(bucket, region)
    objects: List[S3Object] = []
    token: Optional[str] = None

    while True:
        url = f"{base}/?list-type=2&prefix={prefix}"
        if token:
            from urllib.parse import quote
            url += f"&continuation-token={quote(token)}"
        body, err = _try_read(url)
        if err is not None:
            print(f"  ! HTTP {err.code} listing {url}", file=sys.stderr)
            if body:
                try:
                    root = ET.fromstring(body)
                    msg = root.findtext("Message") or ""
                    print(f"    {msg}", file=sys.stderr)
                except ET.ParseError:
                    pass
            return objects

        root = ET.fromstring(body)
        for c in root.findall("s3:Contents", S3_NS):
            key = c.findtext("s3:Key", default="", namespaces=S3_NS)
            size = int(c.findtext("s3:Size", default="0", namespaces=S3_NS))
            if key and not key.endswith("/"):
                objects.append(S3Object(key=key, size=size))

        truncated = root.findtext(
            "s3:IsTruncated", default="false", namespaces=S3_NS
        )
        if truncated.lower() != "true":
            break
        token = root.findtext(
            "s3:NextContinuationToken", default=None, namespaces=S3_NS
        )
        if not token:
            break

    objects.sort(key=lambda o: o.key)
    return objects


def download_with_resume(
    bucket: str,
    region: str,
    obj: S3Object,
    dest_path: Path,
    retries: int = 3,
) -> bool:
    """
    Download an S3 object to `dest_path` with HTTP Range resume.

    Returns True if file is fully present after the call.
    """
    base = _bucket_url(bucket, region)
    url = f"{base}/{obj.key}"

    if dest_path.exists():
        existing = dest_path.stat().st_size
        if existing == obj.size:
            return True
        if existing > obj.size:
            print(f"  ! local file larger than remote, restarting: {dest_path.name}")
            dest_path.unlink()
            existing = 0
    else:
        existing = 0

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url)
            if existing > 0:
                req.add_header("Range", f"bytes={existing}-")
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r:
                code = r.getcode()
                if existing > 0 and code != 206:
                    # Server didn't honour Range — start fresh.
                    existing = 0
                    if dest_path.exists():
                        dest_path.unlink()
                mode = "ab" if existing > 0 else "wb"
                downloaded = existing
                last_print = t0
                with open(dest_path, mode) as fh:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_print >= 2.0:
                            pct = 100 * downloaded / obj.size if obj.size else 0
                            mbs = (downloaded - existing) / (1024 ** 2) / max(now - t0, 0.01)
                            print(
                                f"    {obj.filename}: "
                                f"{downloaded / GB:.2f}/{obj.size / GB:.2f} GB "
                                f"({pct:5.1f}%)  {mbs:6.1f} MB/s",
                                end="\r",
                            )
                            last_print = now
            print()  # newline after progress line

            if dest_path.stat().st_size == obj.size:
                return True
            print(
                f"  ! size mismatch on {obj.filename}: "
                f"got {dest_path.stat().st_size}, expected {obj.size}"
            )
            existing = dest_path.stat().st_size
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  ! attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
            if dest_path.exists():
                existing = dest_path.stat().st_size

    return False


# ── Manifest (idempotent re-runs) ────────────────────────────

def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"completed": {}, "bytes_downloaded": 0}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ── Selection logic ──────────────────────────────────────────

def filter_objects(
    objs: List[S3Object],
    limit: Optional[int],
) -> List[S3Object]:
    """Sort by filename and optionally truncate."""
    objs = sorted(objs, key=lambda o: o.filename)
    if limit is not None:
        objs = objs[:limit]
    return objs


def fmt_size(n: int) -> str:
    if n >= GB:
        return f"{n / GB:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / (1024 ** 2):.1f} MB"
    return f"{n / 1024:.1f} KB"


# ── Main ─────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("X:/Bitirme_Data/CAMELYON16"),
        help="Destination root directory",
    )
    parser.add_argument(
        "--groups", nargs="+",
        choices=list(GROUP_SPECS.keys()),
        default=["annotations", "masks", "tumor"],
        help="Which groups to fetch (priority order: as listed)",
    )
    parser.add_argument(
        "--max-gb", type=float, default=300.0,
        help="Total disk budget in GB (default 300)",
    )
    parser.add_argument(
        "--tumor-limit", type=int, default=None,
        help="Cap on number of tumor WSIs",
    )
    parser.add_argument(
        "--normal-limit", type=int, default=10,
        help="Cap on number of normal WSIs (default 10)",
    )
    parser.add_argument(
        "--test-limit", type=int, default=None,
        help="Cap on number of test WSIs",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Dry-run: list what would be downloaded, then exit",
    )
    parser.add_argument(
        "--bucket", type=str, default=DEFAULT_BUCKET,
        help=f"S3 bucket (default {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--region", type=str, default=DEFAULT_REGION,
        help=f"AWS region (default {DEFAULT_REGION})",
    )
    args = parser.parse_args()

    dest_root: Path = args.dest
    dest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_root / "_manifest.json"
    manifest = load_manifest(manifest_path)

    budget_bytes = int(args.max_gb * GB)
    used_bytes = 0  # bytes that will be on disk after this run
    plan: List[tuple[str, S3Object, Path]] = []

    print(f"Bucket: s3://{args.bucket}/  region={args.region}")
    print(f"Destination: {dest_root}")
    print(f"Budget: {args.max_gb:.1f} GB  ({budget_bytes:,} bytes)")
    print(f"Groups: {', '.join(args.groups)}")
    print()

    # Build the plan, group by group, in priority order.
    base = _discover_endpoint(args.bucket, args.region) or _bucket_url(args.bucket, args.region)
    for group in args.groups:
        prefix, name_filter = GROUP_SPECS[group]
        print(f"[{group}] listing s3://{args.bucket}/{prefix}"
              + (f"  filter={name_filter}*" if name_filter else "") + " ...")
        objs = list_objects(args.bucket, args.region, prefix, base_override=base)

        if name_filter:
            objs = [o for o in objs if o.filename.startswith(name_filter)]

        if group == "tumor":
            objs = filter_objects(objs, args.tumor_limit)
        elif group == "normal":
            objs = filter_objects(objs, args.normal_limit)
        elif group == "test":
            objs = filter_objects(objs, args.test_limit)
        else:
            objs = filter_objects(objs, None)

        group_dir = dest_root / group
        kept = 0
        skipped_budget = 0
        for o in objs:
            target = group_dir / o.filename
            # Already complete on disk?
            if target.exists() and target.stat().st_size == o.size:
                used_bytes += o.size
                kept += 1
                plan.append((group, o, target))
                continue
            # Will this fit?
            need = o.size - (target.stat().st_size if target.exists() else 0)
            if used_bytes + need > budget_bytes:
                skipped_budget += 1
                continue
            used_bytes += need + (target.stat().st_size if target.exists() else 0)
            kept += 1
            plan.append((group, o, target))

        total = sum(o.size for o in objs)
        print(
            f"  found={len(objs)} ({fmt_size(total)})  "
            f"selected={kept}  budget-skipped={skipped_budget}"
        )

    print()
    print(f"Plan total on disk after run: {used_bytes / GB:.2f} GB")
    print(f"Files in plan: {len(plan)}")
    print()

    if args.list:
        for group, o, target in plan:
            mark = "OK" if target.exists() and target.stat().st_size == o.size else "..."
            print(f"  [{mark}] {group:12s}  {fmt_size(o.size):>10}  {o.filename}")
        return 0

    # Execute the plan
    failures: List[str] = []
    for i, (group, o, target) in enumerate(plan, start=1):
        if target.exists() and target.stat().st_size == o.size:
            print(f"[{i}/{len(plan)}] skip (complete): {o.filename}")
            manifest["completed"][o.key] = o.size
            continue
        print(f"[{i}/{len(plan)}] fetching {group}/{o.filename} ({fmt_size(o.size)})")
        ok = download_with_resume(args.bucket, args.region, o, target)
        if ok:
            manifest["completed"][o.key] = o.size
            manifest["bytes_downloaded"] = (
                manifest.get("bytes_downloaded", 0) + o.size
            )
            save_manifest(manifest_path, manifest)
        else:
            failures.append(o.key)
            print(f"  ! giving up on {o.filename}")

    print()
    print("=" * 60)
    print(f"Done. Completed: {len(plan) - len(failures)}/{len(plan)}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for k in failures:
            print(f"  - {k}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
