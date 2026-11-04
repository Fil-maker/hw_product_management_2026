"""One experiment run from a config: data from S3 -> train -> MLflow -> best model to S3.

    cifar-train --config configs/simple_cnn.yaml [--set train.lr=0.003 --set run_name=...]

After every epoch the model is evaluated on val; the checkpoint with the lowest val loss is
kept. At the end that checkpoint is saved, loaded back and measured on test (test never
influences any choice). Model, config and metrics go to
s3://<S3_BUCKET>/<output.s3_prefix>/<experiment_name>/<run_name>/ and to MLflow artifacts.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hw_product_management_2026.data.common import setup_logging, write_json
from hw_product_management_2026.data.s3 import S3Settings, make_client
from hw_product_management_2026.data.upload import git_commit
from hw_product_management_2026.training.config import flatten, load_config, to_dict
from hw_product_management_2026.training.data import (
    Dataset,
    Split,
    ensure_local,
    load_dataset,
    subsample,
)
from hw_product_management_2026.training.engine import (
    EvalResult,
    Normalizer,
    evaluate,
    train_epoch,
)
from hw_product_management_2026.training.models import build_model

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def git_info() -> dict[str, str]:
    """Commit of the run: from GIT_COMMIT (set by the grid runner for containers) or git."""
    commit = os.environ.get("GIT_COMMIT", "")
    if commit and commit != "unknown":
        return {"commit": commit, "dirty": os.environ.get("GIT_DIRTY", "unknown")}
    info = git_commit()
    return {"commit": str(info["commit"] or "unknown"), "dirty": str(info["dirty"]).lower()}


def make_optimizer(model: torch.nn.Module, train_cfg: DictConfig) -> torch.optim.Optimizer:
    name = train_cfg.optimizer
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=train_cfg.lr,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
            nesterov=train_cfg.momentum > 0,
        )
    raise SystemExit(f"unknown train.optimizer {name!r}: use adam, adamw or sgd")


def make_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: DictConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if train_cfg.scheduler == "none":
        return None
    if train_cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    raise SystemExit(f"unknown train.scheduler {train_cfg.scheduler!r}: use none or cosine")


def confusion_figure(confusion: torch.Tensor, classes: list[str], title: str) -> Any:
    fig, ax = plt.subplots(figsize=(8, 7))
    matrix = confusion.numpy()
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8, color=color)
    fig.tight_layout()
    return fig


def to_device(split: Split, device: torch.device) -> Split:
    return Split(images=split.images.to(device), labels=split.labels.to(device))


def upload_outputs(
    client: S3Client, bucket: str, prefix: str, out_dir: Path, names: Sequence[str]
) -> None:
    existing = client.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1)
    if existing.get("KeyCount", 0):
        log.warning("s3://%s/%s/ already exists, overwriting (same run name)", bucket, prefix)
    for name in names:
        client.upload_file(str(out_dir / name), bucket, f"{prefix}/{name}")
    log.info("best model saved to s3://%s/%s/", bucket, prefix)


def run(cfg: DictConfig) -> dict[str, Any]:
    config = to_dict(cfg)
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    log.info("run %s/%s on %s", cfg.experiment_name, cfg.run_name, device)

    settings = S3Settings.from_env()
    client = make_client(settings)
    local = ensure_local(client, settings.bucket, cfg.data.s3_path, Path(cfg.data.local_dir))
    dataset: Dataset = load_dataset(local)
    num_classes = len(dataset.classes)
    train_split = to_device(subsample(dataset.train, cfg.data.max_train_samples, cfg.seed), device)
    val_split = to_device(dataset.val, device)
    normalize = Normalizer(dataset.mean, dataset.std, device)

    model_spec = {"type": cfg.model.type, "params": config["model"]["params"]}
    model = build_model(model_spec["type"], model_spec["params"], num_classes).to(device)
    optimizer = make_optimizer(model, cfg.train)
    scheduler = make_scheduler(optimizer, cfg.train)
    generator = torch.Generator().manual_seed(cfg.seed)
    git = git_info()

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name) as active, tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        mlflow.log_params({k: str(v) for k, v in flatten(config).items()})
        mlflow.set_tags(
            {
                "data_version": cfg.data.s3_path,
                "seed": str(cfg.seed),
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
                "mlflow.source.git.commit": git["commit"],
                "device": str(device),
            }
        )
        mlflow.log_params(
            {
                "n_parameters": sum(p.numel() for p in model.parameters()),
                "n_train_samples": len(train_split),
            }
        )

        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] = {}
        best_val: EvalResult | None = None
        history: list[dict[str, float]] = []
        started = time.perf_counter()
        for epoch in range(1, cfg.train.epochs + 1):
            epoch_started = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_epoch(
                model,
                train_split,
                normalize,
                optimizer,
                cfg.train.batch_size,
                generator,
                cfg.data.augment,
            )
            if scheduler is not None:
                scheduler.step()
            val = evaluate(model, val_split, normalize, cfg.train.eval_batch_size, num_classes)
            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "lr": lr,
                **val.metrics("val"),
                "epoch_time_sec": time.perf_counter() - epoch_started,
            }
            mlflow.log_metrics(metrics, step=epoch)
            history.append({"epoch": epoch, **metrics})
            log.info(
                "epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                epoch,
                cfg.train.epochs,
                train_loss,
                val.loss,
                val.accuracy,
                val.macro_f1,
            )
            if val.loss < best_loss:
                best_loss, best_epoch, best_val = val.loss, epoch, val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_time = time.perf_counter() - started
        assert best_val is not None

        # Best checkpoint -> file -> loaded back into a fresh model for the test measurement.
        torch.save(
            {
                "state_dict": best_state,
                "model": model_spec,
                "classes": dataset.classes,
                "normalization": {"mean": dataset.mean, "std": dataset.std},
                "best_epoch": best_epoch,
                "data_version": cfg.data.s3_path,
            },
            out_dir / "model.pt",
        )
        checkpoint = torch.load(out_dir / "model.pt", map_location=device, weights_only=True)
        best_model = build_model(
            checkpoint["model"]["type"], checkpoint["model"]["params"], num_classes
        )
        best_model.load_state_dict(checkpoint["state_dict"])
        best_model.to(device)
        test = evaluate(
            best_model,
            to_device(dataset.test, device),
            normalize,
            cfg.train.eval_batch_size,
            num_classes,
        )

        summary: dict[str, Any] = {
            "experiment_name": cfg.experiment_name,
            "run_name": cfg.run_name,
            "mlflow_run_id": active.info.run_id,
            "data_version": cfg.data.s3_path,
            "seed": cfg.seed,
            "git_commit": git["commit"],
            "best_epoch": best_epoch,
            "train_time_sec": round(train_time, 2),
            "val": best_val.metrics("val"),
            "test": test.metrics("test"),
            "val_confusion_matrix": best_val.confusion.tolist(),
            "classes": dataset.classes,
            "history": history,
        }
        mlflow.log_metrics(
            {
                "train_time_sec": train_time,
                "best_epoch": best_epoch,
                **{f"best_{k}": v for k, v in best_val.metrics("val").items()},
                **test.metrics("test"),
            }
        )

        fig = confusion_figure(
            best_val.confusion, dataset.classes, f"val confusion, best epoch {best_epoch}"
        )
        mlflow.log_figure(fig, "val_confusion_matrix.png")
        plt.close(fig)
        mlflow.log_dict(
            {"classes": dataset.classes, "matrix": best_val.confusion.tolist()},
            "val_confusion_matrix.json",
        )

        (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        write_json(out_dir / "metrics.json", summary)
        names = ("model.pt", "config.yaml", "metrics.json")
        for name in names:
            mlflow.log_artifact(str(out_dir / name), artifact_path="model")

        s3_prefix = f"{cfg.output.s3_prefix}/{cfg.experiment_name}/{cfg.run_name}"
        upload_outputs(client, settings.bucket, s3_prefix, out_dir, names)
        mlflow.set_tag("model_s3_uri", f"s3://{settings.bucket}/{s3_prefix}/")

    log.info(
        "done: best epoch %d, val_loss=%.4f val_acc=%.4f | test_acc=%.4f test_f1=%.4f",
        best_epoch,
        best_val.loss,
        best_val.accuracy,
        test.accuracy,
        test.macro_f1,
    )
    return summary


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="run config (YAML)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.lr=0.003 (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    run(load_config(args.config, args.overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
