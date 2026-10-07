"""Trainable-parameter accounting: the arithmetic behind the <1% claim.

Full fine-tuning of a ``d_out x d_in`` layer trains ``d_out * d_in`` parameters; LoRA at
rank ``r`` trains ``r * (d_in + d_out)`` (``= 2*d*r`` for a square layer). This module
reports both so the fraction is a number the reader computes, not a slogan.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts for ``model``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_parameter_count(in_features: int, out_features: int, r: int) -> int:
    """Trainable params a LoRA adapter adds: ``r * (in + out)`` (``2*d*r`` if square)."""
    return r * (in_features + out_features)


def full_parameter_count(in_features: int, out_features: int, *, bias: bool = False) -> int:
    """Params a full fine-tune of one linear layer trains: ``in * out`` (+ bias)."""
    return in_features * out_features + (out_features if bias else 0)


@dataclass(frozen=True)
class ParamReport:
    """A trainable-vs-total parameter summary."""

    trainable: int
    total: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable

    @property
    def fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0

    @property
    def percent(self) -> float:
        return 100.0 * self.fraction

    def __str__(self) -> str:
        return (
            f"trainable={self.trainable:,} / total={self.total:,} ({self.percent:.3f}% trainable)"
        )


def parameter_report(model: nn.Module) -> ParamReport:
    """Build a :class:`ParamReport` from a model's current ``requires_grad`` flags."""
    trainable, total = count_parameters(model)
    return ParamReport(trainable=trainable, total=total)
