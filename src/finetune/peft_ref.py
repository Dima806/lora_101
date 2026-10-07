"""A HuggingFace ``peft`` reference, used only to validate the from-scratch layer.

The from-scratch :class:`~src.lora.layer.LoRALinear` uses the same tensor orientation
as ``peft`` (``lora_A`` is ``r x in``, ``lora_B`` is ``out x r``, scaling ``alpha / r``),
so with identical ``A``/``B``/base weights the two must produce identical forward
outputs. ``peft`` is imported lazily so the rest of the library never pays for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence


class SingleLinear(nn.Module):
    """A one-layer model named ``linear`` so ``peft`` can target it by name."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def build_peft_single_linear(
    in_features: int,
    out_features: int,
    r: int,
    alpha: float,
    *,
    bias: bool = True,
    target_modules: Sequence[str] = ("linear",),
) -> Any:
    """Return a ``peft`` model wrapping a single ``nn.Linear`` with a LoRA adapter."""
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    base = SingleLinear(in_features, out_features, bias=bias)
    config = LoraConfig(
        r=r,
        lora_alpha=int(alpha),
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(base, config)


def find_peft_lora_layer(peft_model: nn.Module) -> Any:
    """Return the single ``peft`` LoRA layer inside ``peft_model``."""
    from peft.tuners.lora import Linear as PeftLoraLinear  # noqa: PLC0415

    for module in peft_model.modules():
        if isinstance(module, PeftLoraLinear):
            return module
    raise RuntimeError("No peft LoRA layer found in the model.")


def peft_adapter_tensors(layer: Any, adapter: str = "default") -> tuple[Tensor, Tensor]:
    """Return ``(A, B)`` weights of a ``peft`` LoRA layer (shapes ``r x in`` / ``out x r``)."""
    return layer.lora_A[adapter].weight, layer.lora_B[adapter].weight


def set_peft_adapter_tensors(layer: Any, a: Tensor, b: Tensor, adapter: str = "default") -> None:
    """Overwrite a ``peft`` LoRA layer's ``A``/``B`` weights (for equivalence tests)."""
    with torch.no_grad():
        layer.lora_A[adapter].weight.copy_(a)
        layer.lora_B[adapter].weight.copy_(b)
