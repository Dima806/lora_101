"""Task metrics for LoRA vs full vs frozen."""

from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def regression_mse(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    """Mean squared error of ``model`` on a regression task."""
    return float(nn.functional.mse_loss(model(inputs), targets).item())


@torch.no_grad()
def classification_accuracy(model: nn.Module, inputs: Tensor, labels: Tensor) -> float:
    """Top-1 accuracy of ``model`` on a classification task."""
    preds = model(inputs).argmax(dim=-1)
    return float((preds == labels).float().mean().item())
