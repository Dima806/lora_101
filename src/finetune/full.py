"""The two reference points for the three-way race: frozen and full fine-tune.

``make_frozen`` is the floor (no adaptation); ``make_full_finetune`` is the reference
(every weight trainable). LoRA sits between them in parameter count but matches the
full fine-tune in accuracy.
"""

from __future__ import annotations

import copy

from torch import nn


def make_full_finetune(model: nn.Module) -> nn.Module:
    """Deep-copy ``model`` with every parameter trainable (the full-finetune baseline)."""
    clone = copy.deepcopy(model)
    for param in clone.parameters():
        param.requires_grad_(True)
    return clone


def make_frozen(model: nn.Module) -> nn.Module:
    """Deep-copy ``model`` with every parameter frozen (the no-adaptation floor)."""
    clone = copy.deepcopy(model)
    for param in clone.parameters():
        param.requires_grad_(False)
    return clone
