"""Step 4 (stats): per-channel mean and std of the train split, pixels scaled to [0, 1]."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from hw_product_management_2026.data.common import (
    ANNOTATIONS_SUBDIR,
    PROCESSED_DIR,
    read_json,
    setup_logging,
    write_json,
)

log = logging.getLogger(__name__)


def channel_stats(dataset_dir: Path, file_names: Sequence[str]) -> tuple[list[float], list[float]]:
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    pixels = 0
    for name in file_names:
        with Image.open(dataset_dir / name) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
        total += arr.sum(axis=0)
        total_sq += (arr**2).sum(axis=0)
        pixels += arr.shape[0]
    mean = total / pixels
    std = np.sqrt(total_sq / pixels - mean**2)
    return [round(float(x), 6) for x in mean], [round(float(x), 6) for x in std]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--split", default="train", help="split to compute statistics on")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    annotation = read_json(args.dataset_dir / ANNOTATIONS_SUBDIR / f"{args.split}.json")
    file_names = [item["file_name"] for item in annotation["images"]]

    mean, std = channel_stats(args.dataset_dir, file_names)
    stats = {
        "split": args.split,
        "num_images": len(file_names),
        "pixel_range": [0.0, 1.0],
        "channels": ["R", "G", "B"],
        "mean": mean,
        "std": std,
    }
    write_json(args.dataset_dir / "stats.json", stats)
    log.info("stats on %s (%d images): mean=%s std=%s", args.split, len(file_names), mean, std)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
