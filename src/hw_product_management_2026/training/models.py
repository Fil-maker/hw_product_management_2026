"""Model zoo: a linear baseline, a small CNN and torchvision architectures.

`model.type` picks the model, `model.params` are passed to its constructor.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

IMAGE_SIZE = 32
IN_CHANNELS = 3


class LinearBaseline(nn.Module):
    """One linear layer over raw (normalized) pixels."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))  # type: ignore[no-any-return]


class SimpleCNN(nn.Module):
    """Blocks conv3x3 -> [BN] -> ReLU -> maxpool2, then an MLP classifier."""

    def __init__(
        self,
        num_classes: int,
        channels: Sequence[int] = (32, 64),
        fc_hidden: int = 256,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if not channels or IMAGE_SIZE % 2 ** len(channels):
            raise ValueError(f"bad channels {list(channels)} for {IMAGE_SIZE}x{IMAGE_SIZE} input")
        layers: list[nn.Module] = []
        in_ch = IN_CHANNELS
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        side = IMAGE_SIZE // 2 ** len(channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch * side * side, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))  # type: ignore[no-any-return]


def _torchvision_model(name: str, num_classes: int, params: Mapping[str, Any]) -> nn.Module:
    import torchvision

    params = dict(params)
    cifar_stem = params.pop("cifar_stem", True)
    model: nn.Module = torchvision.models.get_model(name, num_classes=num_classes, **params)
    # ResNet stem is built for 224x224; on 32x32 a 3x3 stride-1 conv without maxpool works better.
    if cifar_stem and hasattr(model, "conv1") and hasattr(model, "maxpool"):
        model.conv1 = nn.Conv2d(IN_CHANNELS, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def build_model(model_type: str, params: Mapping[str, Any], num_classes: int) -> nn.Module:
    if model_type == "linear":
        return LinearBaseline(num_classes, **params)
    if model_type == "simple_cnn":
        return SimpleCNN(num_classes, **params)
    if model_type.startswith("torchvision."):
        return _torchvision_model(model_type.removeprefix("torchvision."), num_classes, params)
    raise SystemExit(
        f"unknown model.type {model_type!r}: use linear, simple_cnn or torchvision.<name>"
    )
