"""Run configs: a YAML file merged over defaults, plus optional dotted overrides.

Overrides use OmegaConf dotlist syntax, e.g. `train.lr=0.001` or `model.params.channels=[32,64]`.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "data": {
        "local_dir": "data/datasets",
        "max_train_samples": None,
        "augment": False,
    },
    "model": {"params": {}},
    "train": {
        "optimizer": "adam",
        "weight_decay": 0.0,
        "momentum": 0.9,
        "scheduler": "none",
        "eval_batch_size": 1024,
    },
    "output": {"s3_prefix": "models"},
}

REQUIRED = (
    "experiment_name",
    "run_name",
    "seed",
    "data.s3_path",
    "model.type",
    "train.epochs",
    "train.batch_size",
    "train.lr",
)


def load_config(path: Path, overrides: Sequence[str] = ()) -> DictConfig:
    cfg = OmegaConf.merge(
        OmegaConf.create(DEFAULTS),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(list(overrides)),
    )
    if not isinstance(cfg, DictConfig):
        raise SystemExit(f"{path}: config must be a mapping")
    missing = [key for key in REQUIRED if OmegaConf.select(cfg, key) is None]
    if missing:
        raise SystemExit(f"{path}: missing required keys: {', '.join(missing)}")
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    return {str(k): v for k, v in container.items()}


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"train": {"lr": 0.1}} -> {"train.lr": 0.1}; lists stay as values."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
