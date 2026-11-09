"""What the serving scripts share: the deployment config and the test split on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from hw_product_management_2026.data.common import ANNOTATIONS_SUBDIR, read_json
from hw_product_management_2026.data.s3 import object_exists

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

DEFAULT_CONFIG = Path("configs/serving.yaml")
STATS_FILE = "stats.json"
MODEL_INFO_FILE = "model_info.json"


@dataclass(frozen=True)
class ServingConfig:
    experiment_name: str
    run_name: str
    s3_prefix: str
    model_version: int
    config_pbtxt: Path
    model_repository: Path
    assets_dir: Path
    data_dir: Path

    @classmethod
    def load(cls, path: Path) -> ServingConfig:
        raw: dict[str, Any] = OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[assignment]
        return cls(
            experiment_name=raw["experiment_name"],
            run_name=raw["run_name"],
            s3_prefix=raw["s3_prefix"],
            model_version=int(raw["model_version"]),
            config_pbtxt=Path(raw["config_pbtxt"]),
            model_repository=Path(raw["model_repository"]),
            assets_dir=Path(raw["assets_dir"]),
            data_dir=Path(raw["data_dir"]),
        )

    @property
    def run_prefix(self) -> str:
        return f"{self.s3_prefix}/{self.experiment_name}/{self.run_name}"


def require_run(client: S3Client, bucket: str, key: str) -> None:
    """A clear error instead of a bare 404 when the run was not trained on this S3."""
    if not object_exists(client, bucket, key):
        raise SystemExit(
            f"s3://{bucket}/{key} not found. Train the run first "
            "(make infra-up && make experiments CONFIG=configs/grid_final.yaml) "
            "or point configs/serving.yaml to an existing one."
        )


@dataclass(frozen=True)
class TestImage:
    file_name: str
    label: int
    data: bytes  # the encoded PNG exactly as stored in the dataset


def load_test_images(local: Path) -> tuple[list[TestImage], list[str]]:
    """Read annotations/test.json and the images it lists. Returns (images, class names)."""
    annotations = read_json(local / ANNOTATIONS_SUBDIR / "test.json")
    images = [
        TestImage(item["file_name"], int(item["label"]), (local / item["file_name"]).read_bytes())
        for item in annotations["images"]
    ]
    return images, list(annotations["classes"])
