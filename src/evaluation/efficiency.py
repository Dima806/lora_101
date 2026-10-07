"""Efficiency measures: adapter size on disk and wall-clock training time."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from torch import nn

T = TypeVar("T")


def adapter_size_bytes(model: nn.Module) -> int:
    """Bytes needed to store only the trainable (adapter) tensors.

    This is the "adapter is megabytes, not gigabytes" number: it counts trainable
    parameters times their dtype size, i.e. what you actually save per task.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)


def adapter_size_mb(model: nn.Module) -> float:
    """Adapter storage in megabytes."""
    return adapter_size_bytes(model) / (1024 * 1024)


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Run ``fn`` and return ``(result, seconds_elapsed)``."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start
