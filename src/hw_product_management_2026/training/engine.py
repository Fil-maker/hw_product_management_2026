"""Training loop pieces: batching over in-memory tensors, augmentation, metrics."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from hw_product_management_2026.training.data import Split

CROP_PADDING = 4


class Normalizer:
    """uint8 images -> float, normalized with the train mean/std of the dataset version."""

    def __init__(self, mean: list[float], std: list[float], device: torch.device) -> None:
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return (images.float() / 255.0 - self.mean) / self.std


def augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random crop with zero padding + horizontal flip, per sample, on the images' device."""
    n, _, h, w = images.shape
    device = images.device
    padded = F.pad(images, [CROP_PADDING] * 4)
    oy = torch.randint(0, 2 * CROP_PADDING + 1, (n,), generator=generator).to(device)
    ox = torch.randint(0, 2 * CROP_PADDING + 1, (n,), generator=generator).to(device)
    rows = (oy[:, None] + torch.arange(h, device=device))[:, :, None]
    cols = (ox[:, None] + torch.arange(w, device=device))[:, None, :]
    batch = torch.arange(n, device=device)[:, None, None]
    cropped = padded.permute(0, 2, 3, 1)[batch, rows, cols].permute(0, 3, 1, 2)
    flip = (torch.rand(n, generator=generator) < 0.5).to(device)
    return torch.where(flip[:, None, None, None], cropped.flip(3), cropped)


def batches(
    split: Split, batch_size: int, generator: torch.Generator | None = None
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Mini-batches over a split (shuffled if a generator is given)."""
    n = len(split)
    order = torch.arange(n) if generator is None else torch.randperm(n, generator=generator)
    order = order.to(split.labels.device)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield split.images[idx], split.labels[idx]


@dataclass(frozen=True)
class EvalResult:
    loss: float
    accuracy: float
    macro_f1: float
    confusion: torch.Tensor  # (C, C), rows = true class, cols = predicted

    def metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_loss": self.loss,
            f"{prefix}_accuracy": self.accuracy,
            f"{prefix}_macro_f1": self.macro_f1,
        }


def macro_f1(confusion: torch.Tensor) -> float:
    tp = confusion.diag().double()
    precision = tp / confusion.sum(0).double().clamp(min=1)
    recall = tp / confusion.sum(1).double().clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return float(f1.mean())


def train_epoch(
    model: nn.Module,
    split: Split,
    normalize: Normalizer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    generator: torch.Generator,
    use_augment: bool,
) -> tuple[float, float]:
    """One pass over train. Returns (mean loss, accuracy) over seen batches."""
    model.train()
    total_loss = 0.0
    correct = 0
    for images, labels in batches(split, batch_size, generator):
        if use_augment:
            images = augment(images, generator)
        logits = model(normalize(images))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
    return total_loss / len(split), correct / len(split)


@torch.no_grad()
def evaluate(
    model: nn.Module, split: Split, normalize: Normalizer, batch_size: int, num_classes: int
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for images, labels in batches(split, batch_size):
        logits = model(normalize(images))
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        pairs = labels * num_classes + logits.argmax(1)
        confusion += (
            torch.bincount(pairs, minlength=num_classes**2).view(num_classes, num_classes).cpu()
        )
    n = len(split)
    return EvalResult(
        loss=total_loss / n,
        accuracy=float(confusion.diag().sum()) / n,
        macro_f1=macro_f1(confusion),
        confusion=confusion,
    )
