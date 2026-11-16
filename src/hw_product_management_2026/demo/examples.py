"""Pick one test image per class for the clickable examples of the demo.

    cifar-demo-examples [--config configs/serving.yaml] [--classes cat ship ...]

The images are taken from the test split of the data version the deployed model was trained on
(serving_assets/model_info.json), copied byte for byte and committed to demo/examples/, so the
demo has examples even without the dataset on disk.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

from hw_product_management_2026.data.common import ANNOTATIONS_SUBDIR, read_json, setup_logging
from hw_product_management_2026.serving.common import (
    DEFAULT_CONFIG,
    MODEL_INFO_FILE,
    ServingConfig,
)

log = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("demo/examples")
DEFAULT_CLASSES = ("airplane", "cat", "frog", "ship", "truck")


def pick_examples(dataset: Path, classes: Sequence[str]) -> dict[str, Path]:
    """The first test image of each requested class, in the order of the classes."""
    annotations = read_json(dataset / ANNOTATIONS_SUBDIR / "test.json")
    unknown = sorted(set(classes) - set(annotations["classes"]))
    if unknown:
        raise SystemExit(f"no such classes in the dataset: {unknown}")
    picked: dict[str, Path] = {}
    for item in annotations["images"]:
        name = item["label_name"]
        if name in classes and name not in picked:
            picked[name] = dataset / item["file_name"]
    return {name: picked[name] for name in classes}


def run(cfg: ServingConfig, classes: Sequence[str], output: Path) -> None:
    info = read_json(cfg.assets_dir / MODEL_INFO_FILE)
    dataset = cfg.data_dir / info["data_version"].strip("/")
    if not dataset.is_dir():
        raise SystemExit(f"{dataset} not found: download the data version first (make data)")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    for name, source in pick_examples(dataset, classes).items():
        target = output / f"{name}_{source.name}"
        shutil.copyfile(source, target)
        log.info("%s -> %s", source, target)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="deployment config")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run(ServingConfig.load(args.config), args.classes, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
