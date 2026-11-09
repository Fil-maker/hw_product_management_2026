"""Checkpoint from S3 -> ONNX with a dynamic batch -> model repository for Triton.

    cifar-export [--config configs/serving.yaml] [--run <experiment>/<run_name>]

Besides the model repository it prepares the assets of the HTTP wrapper: stats.json of the data
version the model was trained on and model_info.json with the class names and the run.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from hw_product_management_2026.data.common import read_json, setup_logging, write_json
from hw_product_management_2026.data.s3 import S3Settings, download_object, make_client
from hw_product_management_2026.serving.common import (
    DEFAULT_CONFIG,
    MODEL_INFO_FILE,
    STATS_FILE,
    ServingConfig,
    require_run,
)
from hw_product_management_2026.training.models import IMAGE_SIZE, IN_CHANNELS, build_model

log = logging.getLogger(__name__)

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
MODEL_FILE = "model.onnx"


@dataclass(frozen=True)
class TritonModelConfig:
    """The part of config.pbtxt that has to agree with the exported graph."""

    name: str
    max_batch_size: int
    input_name: str
    output_name: str
    output_dims: list[int]


def parse_config_pbtxt(text: str) -> TritonModelConfig:
    text = re.sub(r"#.*", "", text)

    def field(pattern: str) -> str:
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if match is None:
            raise SystemExit(f"config.pbtxt: nothing matches {pattern!r}")
        return match.group(1)

    return TritonModelConfig(
        name=field(r'^name:\s*"([^"]+)"'),  # no indent: the model name, not an input's
        max_batch_size=int(field(r"max_batch_size:\s*(\d+)")),
        input_name=field(r'input\s*\[.*?name:\s*"([^"]+)"'),
        output_name=field(r'output\s*\[.*?name:\s*"([^"]+)"'),
        output_dims=[
            int(dim) for dim in field(r"output\s*\[.*?dims:\s*\[([^\]]*)\]").split(",") if dim
        ],
    )


def load_checkpoint(path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    spec = checkpoint["model"]
    model = build_model(spec["type"], spec["params"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def export_onnx(model: torch.nn.Module, path: Path, max_batch_size: int) -> None:
    # The example batch is 2: torch.export turns a size of 1 into a constant.
    example = torch.randn(2, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    batch = torch.export.Dim("batch", min=1, max=max_batch_size)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={"x": {0: batch}},
        external_data=False,  # weights in the same file: the repository holds only model.onnx
        dynamo=True,
    )
    onnx.checker.check_model(str(path))


def check_against_pytorch(model: torch.nn.Module, path: Path, max_batch_size: int) -> None:
    """Same inputs through PyTorch and ONNX Runtime, for several batch sizes."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(0)
    for size in (1, 2, 7, max_batch_size):
        inputs = torch.randn(size, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
        with torch.no_grad():
            expected = model(inputs).numpy()
        actual = session.run([OUTPUT_NAME], {INPUT_NAME: inputs.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    log.info("ONNX Runtime matches PyTorch on batches of 1, 2, 7 and %d", max_batch_size)


def check_against_config(path: Path, config: TritonModelConfig) -> None:
    """Triton reports a name mismatch only when the model loads; catch it here instead."""
    graph = onnx.load(str(path)).graph
    inputs = [item.name for item in graph.input]
    outputs = [item.name for item in graph.output]
    if inputs != [config.input_name] or outputs != [config.output_name]:
        raise SystemExit(
            f"config.pbtxt describes {config.input_name!r} -> {config.output_name!r}, "
            f"the ONNX graph has {inputs} -> {outputs}"
        )
    shape = graph.output[0].type.tensor_type.shape.dim
    graph_dims = [dim.dim_value for dim in shape[1:]]
    if graph_dims != config.output_dims:
        raise SystemExit(
            f"config.pbtxt output dims {config.output_dims}, ONNX graph has {graph_dims}"
        )
    if not shape[0].dim_param:
        raise SystemExit("the batch dimension of the ONNX output is not dynamic")


def write_stats(downloaded: Path, target: Path, checkpoint: dict[str, Any]) -> None:
    stats = read_json(downloaded)
    normalization = checkpoint["normalization"]
    if stats["mean"] != normalization["mean"] or stats["std"] != normalization["std"]:
        raise SystemExit(
            f"stats.json of {checkpoint['data_version']} differs from the normalization "
            "the checkpoint was trained with"
        )
    shutil.copyfile(downloaded, target)


def run(cfg: ServingConfig) -> None:
    pbtxt = parse_config_pbtxt(cfg.config_pbtxt.read_text(encoding="utf-8"))
    settings = S3Settings.from_env()
    client = make_client(settings)
    model_key = f"{cfg.run_prefix}/model.pt"
    require_run(client, settings.bucket, model_key)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        download_object(client, settings.bucket, model_key, work / "model.pt")
        log.info("downloaded s3://%s/%s", settings.bucket, model_key)
        model, checkpoint = load_checkpoint(work / "model.pt")

        export_onnx(model, work / MODEL_FILE, pbtxt.max_batch_size)
        check_against_pytorch(model, work / MODEL_FILE, pbtxt.max_batch_size)
        check_against_config(work / MODEL_FILE, pbtxt)

        stats_key = f"{checkpoint['data_version'].strip('/')}/{STATS_FILE}"
        download_object(client, settings.bucket, stats_key, work / STATS_FILE)
        cfg.assets_dir.mkdir(parents=True, exist_ok=True)
        write_stats(work / STATS_FILE, cfg.assets_dir / STATS_FILE, checkpoint)

        # Only this model is rebuilt; Triton mounts the repository root, which stays in place.
        model_dir = cfg.model_repository / pbtxt.name
        shutil.rmtree(model_dir, ignore_errors=True)
        version_dir = model_dir / str(cfg.model_version)
        version_dir.mkdir(parents=True)
        shutil.copyfile(work / MODEL_FILE, version_dir / MODEL_FILE)
        shutil.copyfile(cfg.config_pbtxt, model_dir / "config.pbtxt")

    write_json(
        cfg.assets_dir / MODEL_INFO_FILE,
        {
            "model_name": pbtxt.name,
            "model_version": str(cfg.model_version),
            "experiment_name": cfg.experiment_name,
            "run_name": cfg.run_name,
            "data_version": checkpoint["data_version"],
            "classes": checkpoint["classes"],
        },
    )
    log.info("model repository: %s", model_dir)
    log.info("wrapper assets: %s", cfg.assets_dir)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="deployment config")
    parser.add_argument(
        "--run",
        metavar="EXPERIMENT/RUN_NAME",
        help="run to export instead of the one in the config",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    cfg = ServingConfig.load(args.config)
    if args.run:
        experiment_name, _, run_name = args.run.partition("/")
        if not run_name:
            raise SystemExit("--run expects EXPERIMENT/RUN_NAME")
        cfg = replace(cfg, experiment_name=experiment_name, run_name=run_name)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
