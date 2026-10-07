"""A fully transparent toy task with a *planted* low-rank update.

A single linear layer ``y = W x`` is "pretrained" to ``W_base`` (it already solves a
base task). Fine-tuning must move it to ``W_target = W_base + delta`` where ``delta`` is
constructed to have exactly ``true_rank`` nonzero singular values. So:

* the frozen model (stuck at ``W_base``) has high loss on the target task,
* a full fine-tune recovers ``delta`` and reaches ~0 loss,
* LoRA with rank ``>= true_rank`` can represent ``delta`` exactly and matches the full
  fine-tune, while ranks below ``true_rank`` underfit (the rank-sweep plateau).

Everything is seeded and inspectable, so the parity and low-rank claims are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ToyTask:
    """Inputs plus the base/target weights and the planted update between them."""

    inputs: Tensor  # (n_samples, dim)
    targets_base: Tensor  # (n_samples, out_dim) = inputs @ W_base^T
    targets: Tensor  # (n_samples, out_dim) = inputs @ W_target^T (the fine-tune target)
    w_base: Tensor  # (out_dim, dim)
    w_target: Tensor  # (out_dim, dim)
    delta: Tensor  # (out_dim, dim), rank == true_rank
    true_rank: int


def make_toy_task(
    dim: int,
    out_dim: int,
    true_rank: int,
    n_samples: int,
    seed: int,
    noise: float = 0.0,
) -> ToyTask:
    """Build a toy regression task whose optimal fine-tune update is rank ``true_rank``."""
    if true_rank > min(dim, out_dim):
        raise ValueError(f"true_rank {true_rank} exceeds min(dim, out_dim) {min(dim, out_dim)}")
    gen = torch.Generator().manual_seed(seed)

    inputs = torch.randn(n_samples, dim, generator=gen)
    w_base = torch.randn(out_dim, dim, generator=gen) / dim**0.5

    # A rank-`true_rank` update, Frobenius-normalized so the frozen MSE is ~1.
    left = torch.randn(out_dim, true_rank, generator=gen)
    right = torch.randn(true_rank, dim, generator=gen)
    delta = left @ right
    delta = delta / delta.norm() * out_dim**0.5
    w_target = w_base + delta

    targets_base = inputs @ w_base.T
    targets = inputs @ w_target.T
    if noise > 0:
        targets = targets + noise * torch.randn(targets.shape, generator=gen)

    return ToyTask(
        inputs=inputs,
        targets_base=targets_base,
        targets=targets,
        w_base=w_base,
        w_target=w_target,
        delta=delta,
        true_rank=true_rank,
    )


class LinearModel(nn.Module):
    """A single ``nn.Linear`` named ``linear`` (the LoRA/injection target)."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def make_base_model(task: ToyTask, *, bias: bool = False) -> LinearModel:
    """A model initialized to ``task.w_base`` (already solves the base task)."""
    out_features, in_features = task.w_base.shape
    model = LinearModel(in_features, out_features, bias=bias)
    with torch.no_grad():
        model.linear.weight.copy_(task.w_base)
        if bias and model.linear.bias is not None:
            model.linear.bias.zero_()
    return model
