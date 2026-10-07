"""Fold trained adapters back into plain ``nn.Linear`` layers.

After merging, the model has no LoRA layers and no inference overhead, yet produces
outputs identical (within float tolerance) to the adapter-applied model. This is the
deployment story: train adapters, merge for serving.
"""

from __future__ import annotations

import torch
from torch import nn

from src.lora.layer import LoRALinear


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every :class:`LoRALinear` in ``model`` with a merged ``nn.Linear`` (in place)."""
    for _module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            merged = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )
            with torch.no_grad():
                merged.weight.copy_(child.merged_weight())
                if child.bias is not None:
                    merged.bias.copy_(child.bias)
            setattr(module, child_name, merged)
    return model
