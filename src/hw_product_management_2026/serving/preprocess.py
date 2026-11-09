"""Encoded image -> model input, the same transform as in training.

Training (training/data.py + training/engine.py): decode, convert to RGB, HWC -> CHW, divide by
255, normalize by mean/std from stats.json of the data version; everything in float32.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from hw_product_management_2026.data.common import read_json

IMAGE_SIZE = 32
ALLOWED_FORMATS = ("PNG", "JPEG")
# A small compressed file can decode into gigabytes: refuse before decoding the pixels.
MAX_PIXELS = 50_000_000


class InvalidImageError(ValueError):
    """The uploaded bytes are not an image the service accepts."""


@dataclass(frozen=True)
class Normalization:
    mean: NDArray[np.float32]  # (3, 1, 1)
    std: NDArray[np.float32]  # (3, 1, 1)

    @classmethod
    def from_stats_file(cls, path: Path) -> Normalization:
        stats: dict[str, Any] = read_json(path)
        mean, std = stats["mean"], stats["std"]
        if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError(f"{path}: expected 3 means and 3 positive stds, got {mean}, {std}")
        return cls(
            mean=np.array(mean, dtype=np.float32).reshape(3, 1, 1),
            std=np.array(std, dtype=np.float32).reshape(3, 1, 1),
        )


def decode_image(data: bytes) -> NDArray[np.uint8]:
    """PNG or JPEG bytes -> uint8 RGB array (32, 32, 3)."""
    try:
        with Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS) as img:
            width, height = img.size
            if width * height > MAX_PIXELS:
                raise InvalidImageError(f"image is too large: {width}x{height}")
            rgb = img.convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise InvalidImageError("the file is not a PNG or JPEG image") from e
    except (OSError, SyntaxError, ValueError) as e:  # truncated or corrupted file
        raise InvalidImageError("the image file is corrupted") from e
    if rgb.size != (IMAGE_SIZE, IMAGE_SIZE):
        rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.uint8)


def preprocess(data: bytes, normalization: Normalization) -> NDArray[np.float32]:
    """PNG or JPEG bytes -> float32 tensor (1, 3, 32, 32) ready for the model."""
    chw = decode_image(data).transpose(2, 0, 1).astype(np.float32) / 255.0
    return ((chw - normalization.mean) / normalization.std)[np.newaxis]


def softmax(logits: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """In float64: in float32 an unlikely class underflows to 0 and its -log(p) is infinite."""
    values = np.asarray(logits, dtype=np.float64)
    exp = np.exp(values - values.max())
    return exp / exp.sum()  # type: ignore[no-any-return]
