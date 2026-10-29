"""Step 3 (split): stratified train/val split of the original CIFAR-10 train set.

Reads the full train/test annotations produced by `convert` and writes
train.json, val.json and test.json (unchanged) into the processed dataset.
Images are not moved or renamed: a validation image keeps its `train_XXXXX` id.
The same seed and val size always produce byte-identical files.
"""

import argparse
import logging
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hw_product_management_2026.data.common import (
    ANNOTATIONS_SUBDIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    read_json,
    setup_logging,
    write_json,
)

log = logging.getLogger(__name__)

PARAMS_FILE = "split_params.json"


def class_quotas(counts: dict[int, int], val_size: int) -> dict[int, int]:
    """Distribute `val_size` across classes proportionally (largest remainder method)."""
    total = sum(counts.values())
    exact = {label: val_size * n / total for label, n in counts.items()}
    quotas = {label: int(x) for label, x in exact.items()}
    leftover = val_size - sum(quotas.values())
    by_remainder = sorted(counts, key=lambda label: (-(exact[label] - quotas[label]), label))
    for label in by_remainder[:leftover]:
        quotas[label] += 1
    return quotas


def stratified_split(
    images: list[dict[str, Any]], val_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < val_size < len(images):
        raise ValueError(f"val size must be in (0, {len(images)}), got {val_size}")

    by_class: dict[int, list[str]] = defaultdict(list)
    for item in images:
        by_class[item["label"]].append(item["image_id"])

    quotas = class_quotas({label: len(ids) for label, ids in by_class.items()}, val_size)
    rng = np.random.default_rng(seed)
    val_ids: set[str] = set()
    for label in sorted(by_class):
        ids = sorted(by_class[label])
        chosen = rng.permutation(len(ids))[: quotas[label]]
        val_ids.update(ids[i] for i in chosen)

    train = [item for item in images if item["image_id"] not in val_ids]
    val = [item for item in images if item["image_id"] in val_ids]
    return train, val


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--val-size", type=int, default=5000, help="number of val images")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    out_dir: Path = args.dataset_dir / ANNOTATIONS_SUBDIR

    full_train = read_json(args.input_dir / "train.json")
    train, val = stratified_split(full_train["images"], args.val_size, args.seed)
    classes = full_train["classes"]

    write_json(out_dir / "train.json", {"classes": classes, "images": train})
    write_json(out_dir / "val.json", {"classes": classes, "images": val})
    shutil.copyfile(args.input_dir / "test.json", out_dir / "test.json")
    # Kept next to the dataset (not uploaded as is) so the manifest can record them.
    write_json(
        args.dataset_dir / PARAMS_FILE,
        {"val_size": args.val_size, "seed": args.seed, "stratified": True},
    )

    log.info("split: train=%d val=%d (seed=%d)", len(train), len(val), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
