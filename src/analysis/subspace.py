"""How much of the full-finetune update a rank-``r`` adapter can capture.

These measures ground the rank sweep (notebook 04): accuracy plateaus exactly when the
rank is large enough to capture the update's real subspace, and no sooner.
"""

from __future__ import annotations

import torch
from torch import Tensor


def captured_energy(delta: Tensor, r: int) -> float:
    """Fraction of ``delta``'s squared energy held by its top-``r`` singular directions."""
    if r <= 0:
        return 0.0
    s = torch.linalg.svdvals(delta.detach())
    total = torch.sum(s**2)
    if total == 0:
        return 1.0
    captured = torch.sum(s[:r] ** 2)
    return float((captured / total).item())


def low_rank_error(delta: Tensor, r: int) -> float:
    """Relative Frobenius error of the best rank-``r`` approximation of ``delta``.

    This is ``1 - captured_energy`` under the square root, i.e. how much a perfect
    rank-``r`` adapter would still miss.
    """
    return float((1.0 - captured_energy(delta, r)) ** 0.5)


def subspace_alignment(delta_true: Tensor, delta_lora: Tensor, r: int) -> float:
    """Mean cosine alignment of the top-``r`` left singular subspaces of two updates.

    ``1.0`` means the adapter recovered the same directions the full fine-tune moved.
    """
    if r <= 0:
        return 0.0
    u_true = torch.linalg.svd(delta_true.detach(), full_matrices=False).U[:, :r]
    u_lora = torch.linalg.svd(delta_lora.detach(), full_matrices=False).U[:, :r]
    # Singular values of U_true^T U_lora are the cosines of the principal angles.
    overlap = torch.linalg.svdvals(u_true.T @ u_lora)
    return float(overlap.mean().item())
