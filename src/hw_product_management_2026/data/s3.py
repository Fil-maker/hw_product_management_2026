"""S3 access: settings from the environment and small helpers over boto3."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

if TYPE_CHECKING:
    # Type stubs are a dev-only dependency (boto3-stubs), so import them for mypy only.
    from mypy_boto3_s3 import S3Client

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3Settings:
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket: str

    @classmethod
    def from_env(cls) -> S3Settings:
        """Read settings from environment variables (and `.env`, if present)."""
        load_dotenv()
        names = ("S3_ENDPOINT_URL", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise SystemExit(
                f"Missing environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env (make setup does it) and fill them in."
            )
        return cls(
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            access_key=os.environ["S3_ACCESS_KEY"],
            secret_key=os.environ["S3_SECRET_KEY"],
            bucket=os.environ["S3_BUCKET"],
        )


def make_client(settings: S3Settings) -> S3Client:
    config = Config(
        s3={"addressing_style": "path"},
        # Local S3-compatible stores do not support the new default checksums.
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 5, "mode": "standard"},
        max_pool_connections=32,
    )
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key,
        aws_secret_access_key=settings.secret_key,
        region_name="us-east-1",
        config=config,
    )


def _is_not_found(error: ClientError) -> bool:
    code = error.response.get("Error", {}).get("Code", "")
    return code in {"404", "NoSuchKey", "NotFound"}


def object_exists(client: S3Client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if _is_not_found(e):
            return False
        raise
    return True


def prefix_exists(client: S3Client, bucket: str, prefix: str) -> bool:
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def put_new_object(client: S3Client, bucket: str, key: str, path: Path) -> bool:
    """Upload `path` to `key` only if the key does not exist yet.

    Returns False (and uploads nothing) if the object already exists, so existing
    objects are never overwritten.
    """
    if object_exists(client, bucket, key):
        return False
    with path.open("rb") as body:
        try:
            # Conditional write: the server rejects it if the key appeared meanwhile.
            client.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"PreconditionFailed", "412"}:
                return False
            raise
    return True


def download_object(client: S3Client, bucket: str, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    client.download_file(bucket, key, str(tmp))
    tmp.replace(path)


def upload_files(
    client: S3Client,
    bucket: str,
    items: Iterable[tuple[Path, str]],
    workers: int,
) -> int:
    """Upload (local path, key) pairs in parallel. Returns the number of uploaded files."""
    items = list(items)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(client.upload_file, str(path), bucket, key) for path, key in items]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 5000 == 0 or done == len(items):
                log.info("uploaded %d/%d files", done, len(items))
    return done
