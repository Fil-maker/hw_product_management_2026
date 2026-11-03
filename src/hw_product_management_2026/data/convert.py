"""Step 2 (convert): CIFAR-10 pickle batches -> PNG images + JSON annotations.

Pickle can execute arbitrary code on load, so the archive is unpickled only after its
MD5 matches the checksum published on the official CIFAR-10 page.

Output:
    <images-dir>/train_00000.png ... test_09999.png
    <annotations-dir>/train.json, test.json   (full original splits)
"""

import argparse
import hashlib
import io
import logging
import pickle
import tarfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from hw_product_management_2026.data.common import (
    CIFAR10_MD5,
    IMAGES_SUBDIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_ARCHIVE,
    file_digest,
    setup_logging,
    write_json,
)

log = logging.getLogger(__name__)

TRAIN_BATCHES = tuple(f"data_batch_{i}" for i in range(1, 6))
TEST_BATCHES = ("test_batch",)


def _load_pickle(tar: tarfile.TarFile, name: str) -> dict[bytes, Any]:
    member = tar.extractfile(f"cifar-10-batches-py/{name}")
    if member is None:
        raise FileNotFoundError(name)
    with member:
        data: dict[bytes, Any] = pickle.load(member, encoding="bytes")
    return data


def load_split(
    tar: tarfile.TarFile, batches: Sequence[str]
) -> tuple[npt.NDArray[np.uint8], list[int]]:
    """Return images as (N, 32, 32, 3) uint8 array and labels in original order."""
    images: list[npt.NDArray[np.uint8]] = []
    labels: list[int] = []
    for name in batches:
        batch = _load_pickle(tar, name)
        raw = np.asarray(batch[b"data"], dtype=np.uint8)
        # Rows are 1024 R, 1024 G, 1024 B values -> (N, 3, 32, 32) -> (N, 32, 32, 3).
        images.append(raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels.extend(int(label) for label in batch[b"labels"])
    return np.concatenate(images), labels


def encode_png(image: npt.NDArray[np.uint8]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def write_split(
    split: str,
    images: npt.NDArray[np.uint8],
    labels: list[int],
    classes: list[str],
    images_dir: Path,
    workers: int,
) -> dict[str, Any]:
    image_ids = [f"{split}_{i:05d}" for i in range(len(labels))]

    def save(index: int) -> str:
        png = encode_png(images[index])
        (images_dir / f"{image_ids[index]}.png").write_bytes(png)
        return hashlib.sha256(png).hexdigest()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(save, range(len(labels)), chunksize=256))

    log.info("%s: wrote %d images", split, len(labels))
    return {
        "classes": classes,
        "images": [
            {
                "image_id": image_id,
                "file_name": f"{IMAGES_SUBDIR}/{image_id}.png",
                "label": label,
                "label_name": classes[label],
                "sha256": sha,
            }
            for image_id, label, sha in zip(image_ids, labels, hashes, strict=True)
        ],
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=RAW_ARCHIVE)
    parser.add_argument("--md5", default=CIFAR10_MD5, help="expected MD5 of the archive")
    parser.add_argument("--images-dir", type=Path, default=PROCESSED_DIR / IMAGES_SUBDIR)
    parser.add_argument("--annotations-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    md5 = file_digest(args.archive, "md5")
    if md5 != args.md5:
        log.error("refusing to unpickle %s: MD5 %s != expected %s", args.archive, md5, args.md5)
        return 1

    args.images_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "r:gz") as tar:
        meta = _load_pickle(tar, "batches.meta")
        classes = [name.decode("utf-8") for name in meta[b"label_names"]]
        for split, batches in (("train", TRAIN_BATCHES), ("test", TEST_BATCHES)):
            images, labels = load_split(tar, batches)
            annotation = write_split(split, images, labels, classes, args.images_dir, args.workers)
            write_json(args.annotations_dir / f"{split}.json", annotation)

    log.info("annotations written to %s", args.annotations_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
