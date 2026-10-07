"""Singular values of the full-finetune update ``ΔW = W_new - W_old``.

The motivating fact for all of LoRA: these singular values collapse, so the change a
task makes lives in a few directions. Same low-rank structure PCA/SVD find in data,
here found in the *update* rather than the data.
"""

from __future__ import annotations

import torch
from torch import Tensor


def update_delta(w_old: Tensor, w_new: Tensor) -> Tensor:
    """The weight update ``W_new - W_old`` (detached)."""
    return (w_new - w_old).detach()


def update_spectrum(w_old: Tensor, w_new: Tensor) -> Tensor:
    """Singular values of the update ``W_new - W_old``, in descending order."""
    return torch.linalg.svdvals(update_delta(w_old, w_new))


def effective_rank(singular_values: Tensor, energy_threshold: float = 0.99) -> int:
    """Smallest rank whose singular values hold ``energy_threshold`` of the squared energy.

    A low number relative to ``len(singular_values)`` is the "update is low-rank" claim.
    """
    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError(f"energy_threshold must be in (0, 1], got {energy_threshold}")
    s = singular_values.detach()
    total = torch.sum(s**2)
    if total == 0:
        return 0
    cumulative = torch.cumsum(s**2, dim=0) / total
    # Number of components needed to reach the threshold.
    return int(torch.searchsorted(cumulative, torch.tensor(energy_threshold)).item()) + 1
