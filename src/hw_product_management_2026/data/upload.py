"""Step 5 (upload): publish the processed dataset to S3 under an immutable version.

Layout: <prefix>/<version>/{images/, annotations/, stats.json, manifest.json}.
If the version already exists, the step fails instead of overwriting it.
`--check-only` performs just that check (used to fail fast before the pipeline runs).
The manifest is uploaded last, so its presence marks a complete version.
"""

import argparse
import logging
import re
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hw_product_management_2026.data.common import (
    ANNOTATIONS_SUBDIR,
    PROCESSED_DIR,
    RAW_KEY,
    SPLITS,
    file_digest,
    read_json,
    setup_logging,
    write_json,
)
from hw_product_management_2026.data.s3 import (
    S3Settings,
    make_client,
    prefix_exists,
    put_new_object,
    upload_files,
)
from hw_product_management_2026.data.split import PARAMS_FILE

log = logging.getLogger(__name__)

VERSION_RE = re.compile(r"^v\d+$")


def git_commit() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def file_entry(dataset_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(dataset_dir).as_posix(),
        "size": path.stat().st_size,
        "sha256": file_digest(path, "sha256"),
    }


def build_manifest(
    dataset_dir: Path, version: str, prefix: str, image_names: list[str]
) -> dict[str, Any]:
    counts = {
        split: len(read_json(dataset_dir / ANNOTATIONS_SUBDIR / f"{split}.json")["images"])
        for split in SPLITS
    }
    files = [dataset_dir / ANNOTATIONS_SUBDIR / f"{split}.json" for split in SPLITS]
    files.append(dataset_dir / "stats.json")
    return {
        "dataset": "cifar10",
        "version": version,
        "s3_prefix": f"{prefix}/{version}",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {"raw_key": RAW_KEY},
        "counts": {**counts, "total_images": len(image_names)},
        "files": [file_entry(dataset_dir, path) for path in files],
        "params": read_json(dataset_dir / PARAMS_FILE),
        "git": git_commit(),
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--prefix", default="processed/cifar10", help="S3 key prefix")
    parser.add_argument("--version", required=True, help="dataset version, e.g. v1")
    parser.add_argument("--workers", type=int, default=16, help="parallel uploads")
    parser.add_argument(
        "--check-only", action="store_true", help="only check that the version is free"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    if not VERSION_RE.match(args.version):
        log.error("version must look like v1, v2, ...; got %r", args.version)
        return 2

    settings = S3Settings.from_env()
    client = make_client(settings)
    base = f"{args.prefix}/{args.version}"
    if prefix_exists(client, settings.bucket, base + "/"):
        log.error(
            "version s3://%s/%s/ already exists and is immutable. "
            "Choose a new version, e.g. `make data VERSION=v2`.",
            settings.bucket,
            base,
        )
        return 1
    if args.check_only:
        log.info("version %s is free", base)
        return 0

    dataset_dir: Path = args.dataset_dir
    image_names = sorted(
        {
            item["file_name"]
            for split in SPLITS
            for item in read_json(dataset_dir / ANNOTATIONS_SUBDIR / f"{split}.json")["images"]
        }
    )
    manifest = build_manifest(dataset_dir, args.version, args.prefix, image_names)
    manifest_path = dataset_dir / "manifest.json"
    write_json(manifest_path, manifest)

    items = [(dataset_dir / name, f"{base}/{name}") for name in image_names]
    items += [(dataset_dir / f["path"], f"{base}/{f['path']}") for f in manifest["files"]]
    log.info("uploading %d files to s3://%s/%s/", len(items), settings.bucket, base)
    upload_files(client, settings.bucket, items, args.workers)

    if not put_new_object(client, settings.bucket, f"{base}/manifest.json", manifest_path):
        log.error("%s/manifest.json appeared during upload; concurrent run?", base)
        return 1
    log.info("published s3://%s/%s/ (%s)", settings.bucket, base, manifest["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
