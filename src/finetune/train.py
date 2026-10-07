"""A small, deterministic, full-batch CPU training loop shared by every baseline.

Only parameters with ``requires_grad=True`` are optimized, so the same function
trains the frozen floor (a no-op that just records the loss), the full fine-tune,
and LoRA, which keeps the three-way comparison honest.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

LossFn = Callable[[Tensor, Tensor], Tensor]


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return nn.functional.mse_loss(pred, target)


def train(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    steps: int,
    lr: float,
    loss_fn: LossFn = mse_loss,
    seed: int = 0,
) -> list[float]:
    """Full-batch train ``model`` for ``steps`` and return the per-step loss history.

    If the model has no trainable parameters (the frozen floor), it simply records the
    constant loss so the floor is directly comparable to the trained runs.
    """
    torch.manual_seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr) if trainable else None

    history: list[float] = []
    for _ in range(steps):
        if optimizer is not None:
            optimizer.zero_grad()
        pred = model(inputs)
        loss = loss_fn(pred, targets)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        history.append(loss.item())
        if optimizer is None:
            # Frozen: the loss never changes, so one evaluation is enough.
            history = history * steps
            break
    return history[:steps]


@torch.no_grad()
def evaluate(
    model: nn.Module, inputs: Tensor, targets: Tensor, *, loss_fn: LossFn = mse_loss
) -> float:
    """Return ``loss_fn(model(inputs), targets)`` as a float."""
    return float(loss_fn(model(inputs), targets).item())
