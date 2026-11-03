"""Step 1 (download): fetch the raw archive from S3 into data/raw/."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from hw_product_management_2026.data.common import (
    CIFAR10_MD5,
    RAW_ARCHIVE,
    RAW_KEY,
    file_digest,
    setup_logging,
)
from hw_product_management_2026.data.s3 import S3Settings, download_object, make_client

log = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=RAW_KEY, help="archive key in the bucket")
    parser.add_argument("--output", type=Path, default=RAW_ARCHIVE, help="local archive path")
    parser.add_argument("--md5", default=CIFAR10_MD5, help="expected MD5 of the archive")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    output: Path = args.output

    if output.exists() and file_digest(output, "md5") == args.md5:
        log.info("%s is already downloaded and verified, nothing to do", output)
        return 0

    settings = S3Settings.from_env()
    client = make_client(settings)
    log.info("downloading s3://%s/%s -> %s", settings.bucket, args.key, output)
    download_object(client, settings.bucket, args.key, output)

    md5 = file_digest(output, "md5")
    if md5 != args.md5:
        log.error("MD5 mismatch: expected %s, got %s", args.md5, md5)
        output.unlink()
        return 1
    log.info("MD5 ok (%s)", md5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
