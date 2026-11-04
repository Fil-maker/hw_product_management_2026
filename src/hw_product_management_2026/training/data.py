"""Dataset version from S3 -> local cache -> uint8 tensors.

The version is downloaded once into `<local_dir>/<s3_path>/` (manifest.json last, so its
presence marks a complete copy). Decoded images are cached next to it in `arrays.npz`.
Every image is checked against the sha256 from the annotations when the cache is built.
"""

from __future__ import annotations

import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from hw_product_management_2026.data.common import (
    ANNOTATIONS_SUBDIR,
    SPLITS,
    file_digest,
    read_json,
)
from hw_product_management_2026.data.s3 import download_files, download_object, list_keys

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

log = logging.getLogger(__name__)

CACHE_FILE = "arrays.npz"


@dataclass(frozen=True)
class Split:
    images: torch.Tensor  # uint8, (N, 3, 32, 32)
    labels: torch.Tensor  # int64, (N,)

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class Dataset:
    train: Split
    val: Split
    test: Split
    classes: list[str]
    mean: list[float]
    std: list[float]


def ensure_local(client: S3Client, bucket: str, s3_path: str, root: Path) -> Path:
    """Download the dataset version unless a complete local copy already exists."""
    prefix = s3_path.strip("/")
    local = root / prefix
    if (local / "manifest.json").exists():
        log.info("dataset %s already in %s", prefix, local)
        return local

    keys = list_keys(client, bucket, prefix + "/")
    manifest_key = f"{prefix}/manifest.json"
    if manifest_key not in keys:
        raise SystemExit(
            f"s3://{bucket}/{prefix}/manifest.json not found: the version is missing or "
            "incomplete. Build it with `make data VERSION=...` (see README)."
        )
    items = [(key, local / key[len(prefix) + 1 :]) for key in keys if key != manifest_key]
    log.info("downloading %d files from s3://%s/%s/", len(items), bucket, prefix)
    download_files(client, bucket, items, workers=32)

    manifest_path = local / "manifest.json"
    download_object(client, bucket, manifest_key, manifest_path)
    for entry in read_json(manifest_path)["files"]:
        if file_digest(local / entry["path"], "sha256") != entry["sha256"]:
            manifest_path.unlink()
            raise SystemExit(f"sha256 mismatch for {entry['path']}, dataset copy is corrupted")
    return local


def _decode(path: Path, sha256: str) -> NDArray[np.uint8]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"sha256 mismatch for {path}")
    with Image.open(io.BytesIO(data)) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return array.transpose(2, 0, 1)  # HWC -> CHW


def _build_cache(local: Path) -> None:
    arrays: dict[str, NDArray[Any]] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for split in SPLITS:
            images = read_json(local / ANNOTATIONS_SUBDIR / f"{split}.json")["images"]
            decoded = pool.map(
                lambda item: _decode(local / item["file_name"], item["sha256"]), images
            )
            arrays[f"{split}_x"] = np.stack(list(decoded))
            arrays[f"{split}_y"] = np.array([item["label"] for item in images], dtype=np.int64)
            log.info("decoded %s: %d images", split, len(images))
    tmp = local / (CACHE_FILE + ".tmp.npz")
    np.savez(tmp, allow_pickle=False, **arrays)
    tmp.replace(local / CACHE_FILE)


def load_dataset(local: Path) -> Dataset:
    if not (local / CACHE_FILE).exists():
        _build_cache(local)
    with np.load(local / CACHE_FILE) as arrays:
        splits = {
            split: Split(
                images=torch.from_numpy(arrays[f"{split}_x"]),
                labels=torch.from_numpy(arrays[f"{split}_y"]),
            )
            for split in SPLITS
        }
    stats = read_json(local / "stats.json")
    classes = read_json(local / ANNOTATIONS_SUBDIR / "train.json")["classes"]
    return Dataset(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        classes=list(classes),
        mean=list(stats["mean"]),
        std=list(stats["std"]),
    )


def subsample(split: Split, max_samples: int | None, seed: int) -> Split:
    if max_samples is None or max_samples >= len(split):
        return split
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(split), generator=generator)[:max_samples]
    return Split(images=split.images[idx], labels=split.labels[idx])
