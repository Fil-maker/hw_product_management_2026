"""Load test of POST /predict: closed loop with 1, 2, 5, 10 workers.

    cifar-load-test [--workers 1 2 5 10] [--requests 1000] [--warmup 50] [--seed 42]

A worker is a thread that sends its next request right after the previous response. Latency is
measured on the client. Writes the raw measurements to reports/load_test_raw.csv and the tables
and measurement conditions to reports/load_test.md; everything below GENERATED_END in that file
(the conclusions, written by hand) survives a rerun.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import platform
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

from hw_product_management_2026.data.common import read_json, setup_logging
from hw_product_management_2026.data.s3 import S3Settings, make_client
from hw_product_management_2026.serving.common import (
    DEFAULT_CONFIG,
    MODEL_INFO_FILE,
    ServingConfig,
    TestImage,
    load_test_images,
)
from hw_product_management_2026.serving.verify import DEFAULT_API_URL
from hw_product_management_2026.training.data import ensure_local

log = logging.getLogger(__name__)

GENERATED_END = "<!-- конец генерируемой части: всё ниже скрипт сохраняет как есть -->"
DEFAULT_CONCLUSIONS = "## Выводы\n\nНапишите выводы по таблице выше.\n"
TRITON_METRICS = (
    "nv_inference_count",
    "nv_inference_exec_count",
    "nv_inference_request_success",
    "nv_inference_queue_duration_us",
    "nv_inference_compute_infer_duration_us",
)


@dataclass(frozen=True)
class Sample:
    workers: int
    index: int
    measured: bool
    started: float  # seconds since the start of this level
    latency_ms: float
    status: int  # HTTP status, 0 if there was no response


@dataclass(frozen=True)
class Level:
    workers: int
    requests: int
    errors: int
    rps: float
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    avg_batch: float | None
    queue_ms: float | None
    infer_ms: float | None
    client_cpu: float  # CPU cores used by this client process


def scrape_triton(url: str, model_name: str) -> dict[str, float] | None:
    """Cumulative Triton counters for the model; None if the metrics port is not reachable."""
    try:
        text = requests.get(url, timeout=5).text
    except requests.RequestException as e:
        log.warning("Triton metrics are not available at %s: %s", url, e)
        return None
    totals = dict.fromkeys(TRITON_METRICS, 0.0)
    for line in text.splitlines():
        match = re.match(r"(\w+)\{(.*)\}\s+(\S+)", line)
        if match and match[1] in totals and f'model="{model_name}"' in match[2]:
            totals[match[1]] += float(match[3])
    return totals


def run_level(
    url: str,
    images: list[TestImage],
    workers: int,
    warmup: int,
    requests_count: int,
    seed: int,
    timeout: float,
) -> tuple[list[Sample], float]:
    """One closed-loop run. Returns all samples and the duration of the measured part."""
    total = warmup + requests_count
    choice = np.random.default_rng(seed).integers(len(images), size=total)
    counter = itertools.count()  # next() is atomic in CPython: each index goes to one worker
    samples: list[list[Sample]] = [[] for _ in range(workers)]
    barrier = threading.Barrier(workers)

    def worker(worker_id: int) -> None:
        session = requests.Session()
        barrier.wait()
        for index in iter(lambda: next(counter), None):
            if index >= total:
                return
            image = images[choice[index]]
            files = {"file": ("image.png", image.data, "image/png")}
            started = time.perf_counter()
            try:
                status = session.post(url, files=files, timeout=timeout).status_code
            except requests.RequestException:
                status = 0
            elapsed = time.perf_counter() - started
            samples[worker_id].append(
                Sample(workers, index, index >= warmup, started, elapsed * 1000, status)
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(worker, worker_id) for worker_id in range(workers)]:
            future.result()

    flat = sorted(itertools.chain.from_iterable(samples), key=lambda sample: sample.index)
    measured = [sample for sample in flat if sample.measured]
    begin = min(sample.started for sample in measured)
    end = max(sample.started + sample.latency_ms / 1000 for sample in measured)
    origin = flat[0].started
    return [
        Sample(s.workers, s.index, s.measured, s.started - origin, s.latency_ms, s.status)
        for s in flat
    ], end - begin


def summarize(
    samples: list[Sample],
    duration: float,
    workers: int,
    before: dict[str, float] | None,
    after: dict[str, float] | None,
    client_cpu: float,
) -> Level:
    measured = [sample for sample in samples if sample.measured]
    ok = np.array([s.latency_ms for s in measured if s.status == 200], dtype=np.float64)
    if ok.size == 0:
        raise SystemExit(f"{workers} workers: all {len(measured)} requests failed; is the API up?")
    avg_batch = queue_ms = infer_ms = None
    if before is not None and after is not None:
        delta = {name: after[name] - before[name] for name in TRITON_METRICS}
        if delta["nv_inference_exec_count"] > 0:
            avg_batch = delta["nv_inference_count"] / delta["nv_inference_exec_count"]
            queue_ms = delta["nv_inference_queue_duration_us"] / 1000
            queue_ms /= max(delta["nv_inference_request_success"], 1)
            infer_ms = delta["nv_inference_compute_infer_duration_us"] / 1000
            infer_ms /= max(delta["nv_inference_request_success"], 1)
    return Level(
        workers=workers,
        requests=len(measured),
        errors=len(measured) - int(ok.size),
        rps=ok.size / duration,
        mean_ms=float(ok.mean()),
        median_ms=float(np.median(ok)),
        p90_ms=float(np.percentile(ok, 90)),
        p95_ms=float(np.percentile(ok, 95)),
        avg_batch=avg_batch,
        queue_ms=queue_ms,
        infer_ms=infer_ms,
        client_cpu=client_cpu,
    )


def cpu_name() -> str:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or platform.machine()
    match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    return match[1].strip() if match else platform.machine()


def triton_facts(triton_url: str, model_name: str) -> tuple[str, dict[str, Any]]:
    """Version and the effective model config as Triton itself reports them."""
    version = requests.get(f"{triton_url}/v2", timeout=5).json().get("version", "?")
    config = requests.get(f"{triton_url}/v2/models/{model_name}/config", timeout=5).json()
    return str(version), config


def triton_image(compose_file: Path = Path("docker-compose.yml")) -> str:
    match = re.search(r"image:\s*(\S*tritonserver\S*)", compose_file.read_text(encoding="utf-8"))
    return match[1] if match else "?"


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_generated(
    levels: list[Level], args: argparse.Namespace, model_name: str, requests_seen: int
) -> str:
    version, config = triton_facts(args.triton_url, model_name)
    batching = config.get("dynamic_batching", {})
    groups = config.get("instance_group", [{}])
    instances = ", ".join(f"{g.get('count', '?')} × {g.get('kind', '?')}" for g in groups)
    lines = [
        "# Нагрузочный тест `POST /predict`",
        "",
        "Таблицы и условия замера генерирует `make load-test` (`cifar-load-test`).",
        "",
        "| Workers | Запросов | Ошибок | RPS | Среднее, мс | Медиана, мс | p90, мс | p95, мс |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {x.workers} | {x.requests} | {x.errors} | {x.rps:.1f} | {x.mean_ms:.2f} "
        f"| {x.median_ms:.2f} | {x.p90_ms:.2f} | {x.p95_ms:.2f} |"
        for x in levels
    ]
    lines += [
        "",
        "Внутри Triton (метрики с порта 8002, вместе с прогревом) и на клиенте:",
        "",
        "| Workers | Средний размер батча | Очередь, мс на запрос | Инференс, мс на запрос "
        "| CPU клиента, ядер |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {x.workers} | {fmt(x.avg_batch)} | {fmt(x.queue_ms, 3)} | {fmt(x.infer_ms, 3)} "
        f"| {x.client_cpu:.2f} |"
        for x in levels
    ]
    lines += [
        "",
        "## Условия замера",
        "",
        f"- **Процессор:** {cpu_name()}, логических ядер: {os.cpu_count()}.",
        f"- **ОС клиента:** {platform.platform()}.",
        f"- **Размещение:** клиент — потоки Python на этой же машине ({platform.node()}); сервис —"
        " контейнеры Docker на ней же, клиент и сервис делят процессор.",
        f"- **Triton:** {version}, образ `{triton_image()}`, только CPU.",
        f"- **Экземпляры модели:** {instances}; `max_batch_size`: {config.get('max_batch_size')}.",
        f"- **dynamic_batching:** `{batching}`.",
        f"- **Параметры ONNX Runtime:** `{config.get('parameters', {})}`.",
        "- **Обёртка:** один процесс uvicorn.",
        f"- **Нагрузка:** замкнутый цикл, прогрев {args.warmup} запросов не входит в статистику, "
        f"далее {args.requests} запросов на каждое значение workers; всего отправлено "
        f"{requests_seen}. Картинки — случайные из test, seed {args.seed}.",
        f"- **Сырые замеры:** `{args.raw}` (в git не коммитится).",
        "",
    ]
    return "\n".join(lines)


def write_report(path: Path, generated: str) -> None:
    conclusions = DEFAULT_CONCLUSIONS
    if path.exists():
        _, found, kept = path.read_text(encoding="utf-8").partition(GENERATED_END)
        if found:
            conclusions = kept.lstrip("\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{generated}\n{GENERATED_END}\n\n{conclusions}", encoding="utf-8", newline="\n"
    )


def write_raw(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["workers", "index", "measured", "started_s", "latency_ms", "status"])
        for s in samples:
            writer.writerow(
                [
                    s.workers,
                    s.index,
                    int(s.measured),
                    f"{s.started:.6f}",
                    f"{s.latency_ms:.3f}",
                    s.status,
                ]
            )


def run(args: argparse.Namespace) -> int:
    cfg = ServingConfig.load(args.config)
    info = read_json(cfg.assets_dir / MODEL_INFO_FILE)
    settings = S3Settings.from_env()
    local = ensure_local(
        make_client(settings), settings.bucket, info["data_version"], cfg.data_dir
    )
    images, _ = load_test_images(local)

    url = f"{args.api_url}/predict"
    levels: list[Level] = []
    all_samples: list[Sample] = []
    for workers in args.workers:
        log.info("%d workers: %d + %d requests", workers, args.warmup, args.requests)
        before = scrape_triton(args.metrics_url, info["model_name"])
        cpu_started, wall_started = time.process_time(), time.perf_counter()
        samples, duration = run_level(
            url, images, workers, args.warmup, args.requests, args.seed, args.timeout
        )
        client_cpu = (time.process_time() - cpu_started) / (time.perf_counter() - wall_started)
        after = scrape_triton(args.metrics_url, info["model_name"])
        level = summarize(samples, duration, workers, before, after, client_cpu)
        log.info(
            "%d workers: %.1f rps, mean %.2f ms, p95 %.2f ms, %d errors",
            workers,
            level.rps,
            level.mean_ms,
            level.p95_ms,
            level.errors,
        )
        levels.append(level)
        all_samples += samples

    write_raw(args.raw, all_samples)
    write_report(args.report, render_generated(levels, args, info["model_name"], len(all_samples)))
    log.info("report: %s, raw measurements: %s", args.report, args.raw)
    return 1 if any(level.errors for level in levels) else 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="deployment config")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", DEFAULT_API_URL))
    parser.add_argument(
        "--triton-url", default=os.environ.get("TRITON_HTTP_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "--metrics-url",
        default=os.environ.get("TRITON_METRICS_URL", "http://localhost:8002/metrics"),
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--requests", type=int, default=1000, help="measured requests per level")
    parser.add_argument("--warmup", type=int, default=50, help="not counted requests per level")
    parser.add_argument("--seed", type=int, default=42, help="seed of the random image choice")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    parser.add_argument("--report", type=Path, default=Path("reports/load_test.md"))
    parser.add_argument("--raw", type=Path, default=Path("reports/load_test_raw.csv"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
