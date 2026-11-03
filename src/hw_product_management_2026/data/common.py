"""Shared constants and helpers for the data pipeline steps."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# MD5 of the official archive, published at https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
RAW_KEY = "raw/cifar-10-python.tar.gz"

RAW_ARCHIVE = Path("data/raw/cifar-10-python.tar.gz")
INTERIM_DIR = Path("data/interim/cifar10")
PROCESSED_DIR = Path("data/processed/cifar10")

ANNOTATIONS_SUBDIR = "annotations"
IMAGES_SUBDIR = "images"
SPLITS = ("train", "val", "test")

_CHUNK = 1024 * 1024


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def file_digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write JSON deterministically: same data -> byte-identical file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)
