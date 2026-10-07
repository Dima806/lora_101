"""The core module: a frozen linear layer with a trainable low-rank adapter.

For a layer mapping ``in_features -> out_features`` the forward is::

    y = x @ W^T + b  +  (alpha / r) * (x @ A^T) @ B^T

where ``W`` (and bias) are frozen, ``A`` (shape ``r x in``, the down-projection) and
``B`` (shape ``out x r``, the up-projection) are the only trainable tensors, and the
scaling is ``alpha / r``. ``B`` is zero-initialized so that at step 0 the adapter
contributes nothing and the layer is exactly the base layer.

This orientation matches HF ``peft`` (``lora_A.weight`` is ``r x in``, ``lora_B.weight``
is ``out x r``), which is why :mod:`src.finetune.peft_ref` can validate it tensor-for-tensor.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class LoRALinear(nn.Module):
    """A ``nn.Linear`` whose weight is frozen and adapted by a rank-``r`` update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r

        # Frozen base weight (and optional bias): not trained by LoRA.
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        # Trainable adapter: down-projection A (r x in) and up-projection B (out x r).
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.reset_adapter()

    def reset_adapter(self) -> None:
        """Re-initialize the adapter so it starts as a no-op (``B = 0``)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, alpha: float) -> LoRALinear:
        """Wrap an existing ``nn.Linear``, copying and freezing its weight/bias."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            alpha=alpha,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    @property
    def delta_weight(self) -> Tensor:
        """The effective weight update ``(alpha / r) * B @ A`` (shape ``out x in``)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def merged_weight(self) -> Tensor:
        """Base weight folded with the adapter: ``W + (alpha / r) * B @ A``."""
        return self.weight.detach() + self.delta_weight.detach()

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4g}"
        )
