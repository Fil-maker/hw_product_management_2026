"""Step 0 (ingest): download the official CIFAR-10 archive and put it into S3 as is.

Idempotent: if the object already exists in S3, nothing is downloaded or uploaded.
Raw data is immutable: an existing object under `raw/` is never overwritten.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from hw_product_management_2026.data.common import (
    CIFAR10_MD5,
    CIFAR10_URL,
    RAW_KEY,
    file_digest,
    setup_logging,
)
from hw_product_management_2026.data.s3 import (
    S3Settings,
    make_client,
    object_exists,
    put_new_object,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

log = logging.getLogger(__name__)


def download(url: str, path: Path) -> None:
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as response, path.open("wb") as f:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        next_report = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            f.write(chunk)
            done += len(chunk)
            if total and done >= next_report:
                log.info("  %3d%% (%d MiB)", done * 100 // total, done // 2**20)
                next_report += total // 20


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=CIFAR10_URL, help="archive URL")
    parser.add_argument("--key", default=RAW_KEY, help="destination key in the bucket")
    parser.add_argument("--md5", default=CIFAR10_MD5, help="expected MD5 of the archive")
    parser.add_argument(
        "--from-file",
        type=Path,
        help="use an already downloaded copy of the official archive instead of downloading "
        "(for slow networks); it is uploaded only if its MD5 matches the official one",
    )
    return parser.parse_args(argv)


def upload_verified(client: S3Client, bucket: str, key: str, archive: Path, md5: str) -> int:
    actual = file_digest(archive, "md5")
    if actual != md5:
        log.error("MD5 mismatch: expected %s, got %s", md5, actual)
        return 1
    log.info("MD5 ok (%s), uploading to s3://%s/%s", actual, bucket, key)
    if not put_new_object(client, bucket, key, archive):
        log.info("s3://%s/%s appeared meanwhile, left untouched", bucket, key)
        return 0
    log.info("uploaded s3://%s/%s", bucket, key)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    settings = S3Settings.from_env()
    client = make_client(settings)
    target = f"s3://{settings.bucket}/{args.key}"

    if object_exists(client, settings.bucket, args.key):
        log.info("%s already exists, nothing to do", target)
        return 0

    if args.from_file is not None:
        return upload_verified(client, settings.bucket, args.key, args.from_file, args.md5)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / Path(args.key).name
        download(args.url, archive)
        return upload_verified(client, settings.bucket, args.key, archive, args.md5)


if __name__ == "__main__":
    raise SystemExit(main())
