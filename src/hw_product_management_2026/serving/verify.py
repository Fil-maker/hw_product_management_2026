"""Does the model in the service give the metrics it had in training?

    cifar-verify [--config configs/serving.yaml] [--api-url ...] [--loss-tolerance ...]

Every image of the test split goes through the wrapper (POST /predict), so preprocessing is
checked too. Loss, accuracy and macro-F1 are computed from the returned probabilities and compared
with metrics.json of the run. Exit code is 1 if any difference exceeds its tolerance.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from numpy.typing import NDArray

from hw_product_management_2026.data.common import setup_logging
from hw_product_management_2026.data.s3 import S3Settings, make_client
from hw_product_management_2026.serving.common import (
    DEFAULT_CONFIG,
    ServingConfig,
    TestImage,
    load_test_images,
    require_run,
)
from hw_product_management_2026.training.data import ensure_local

log = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8080"
# Below it -log(p) is only a matter of rounding; keeps the loss finite if p is exactly 0.
MIN_PROBABILITY = 1e-300


@dataclass(frozen=True)
class Check:
    name: str
    trained: float
    served: float
    tolerance: float

    @property
    def difference(self) -> float:
        return abs(self.served - self.trained)

    @property
    def passed(self) -> bool:
        return self.difference <= self.tolerance


def predict_all(
    api_url: str, images: list[TestImage], classes: list[str], workers: int, timeout: float
) -> NDArray[np.float64]:
    """Probabilities for every image; any failed or inconsistent response stops the check."""
    local = threading.local()
    url = f"{api_url}/predict"

    def predict(image: TestImage) -> list[float]:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        files = {"file": (Path(image.file_name).name, image.data, "image/png")}
        response = local.session.post(url, files=files, timeout=timeout)
        if response.status_code != 200:
            raise SystemExit(
                f"{image.file_name}: HTTP {response.status_code}: {response.text[:200]}"
            )
        body = response.json()
        if body["class_name"] != classes[body["class_id"]]:
            raise SystemExit(
                f"{image.file_name}: class {body['class_id']} is {classes[body['class_id']]!r} "
                f"in the dataset but {body['class_name']!r} in the service"
            )
        probabilities: list[float] = body["probabilities"]
        return probabilities

    done = 0
    results: list[list[float]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for probabilities in pool.map(predict, images):
            results.append(probabilities)
            done += 1
            if done % 2000 == 0:
                log.info("%d/%d images", done, len(images))
    return np.array(results, dtype=np.float64)


def macro_f1(labels: NDArray[np.int64], predicted: NDArray[np.int64], num_classes: int) -> float:
    """The same formula as in training (training/engine.py)."""
    confusion = np.bincount(labels * num_classes + predicted, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )
    tp = np.diag(confusion).astype(np.float64)
    precision = tp / np.maximum(confusion.sum(0), 1)
    recall = tp / np.maximum(confusion.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.mean())


def print_table(checks: list[Check]) -> None:
    print("| Метрика | При обучении | В сервисе | Разница | Допуск | Итог |")
    print("|---|---|---|---|---|---|")
    for c in checks:
        verdict = "OK" if c.passed else "ПРЕВЫШЕН"
        print(
            f"| {c.name} | {c.trained:.6f} | {c.served:.6f} | {c.difference:.6f} "
            f"| {c.tolerance:g} | {verdict} |"
        )


def run(args: argparse.Namespace) -> int:
    cfg = ServingConfig.load(args.config)
    settings = S3Settings.from_env()
    client = make_client(settings)
    metrics_key = f"{cfg.run_prefix}/metrics.json"
    require_run(client, settings.bucket, metrics_key)
    metrics = json.loads(client.get_object(Bucket=settings.bucket, Key=metrics_key)["Body"].read())
    trained = metrics["test"]

    local = ensure_local(client, settings.bucket, metrics["data_version"], cfg.data_dir)
    images, classes = load_test_images(local)
    log.info("run %s, %d test images of %s", cfg.run_name, len(images), metrics["data_version"])

    probabilities = predict_all(args.api_url, images, classes, args.workers, args.timeout)
    labels = np.array([image.label for image in images], dtype=np.int64)
    predicted = probabilities.argmax(1)
    true_class_p = probabilities[np.arange(len(labels)), labels]

    checks = [
        Check(
            "loss",
            trained["test_loss"],
            float(-np.log(np.maximum(true_class_p, MIN_PROBABILITY)).mean()),
            args.loss_tolerance,
        ),
        Check(
            "accuracy",
            trained["test_accuracy"],
            float((predicted == labels).mean()),
            args.accuracy_tolerance,
        ),
        Check(
            "macro-F1",
            trained["test_macro_f1"],
            macro_f1(labels, predicted, len(classes)),
            args.f1_tolerance,
        ),
    ]
    print_table(checks)
    if all(check.passed for check in checks):
        log.info("the service reproduces the training metrics")
        return 0
    log.error("metrics of the service differ from training: check preprocessing and stats.json")
    return 1


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="deployment config")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", DEFAULT_API_URL))
    parser.add_argument("--loss-tolerance", type=float, default=5e-4)
    parser.add_argument("--accuracy-tolerance", type=float, default=5e-4)
    parser.add_argument("--f1-tolerance", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=8, help="parallel requests, only for speed")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
