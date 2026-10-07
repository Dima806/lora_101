"""LoRALinear starts as a no-op and trains only its adapter."""

from __future__ import annotations

import torch
from torch import nn

from src.lora.layer import LoRALinear


def test_zero_init_adapter_equals_base() -> None:
    torch.manual_seed(0)
    base = nn.Linear(16, 12)
    lora = LoRALinear.from_linear(base, r=4, alpha=8)

    x = torch.randn(5, 16)
    # B is zero-initialized, so at step 0 the adapter contributes nothing.
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    assert torch.count_nonzero(lora.lora_B) == 0


def test_gradients_flow_only_to_adapter() -> None:
    torch.manual_seed(0)
    lora = LoRALinear(16, 12, r=4, alpha=8)
    # Give B a nonzero value so A also receives gradient.
    with torch.no_grad():
        lora.lora_B.normal_()

    out = lora(torch.randn(5, 16))
    out.pow(2).sum().backward()

    assert lora.weight.grad is None
    assert lora.bias is not None
    assert lora.bias.grad is None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def test_only_adapter_requires_grad() -> None:
    lora = LoRALinear(16, 12, r=4, alpha=8)
    trainable = {name for name, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}


def test_scaling_is_alpha_over_r() -> None:
    lora = LoRALinear(8, 8, r=4, alpha=16)
    assert lora.scaling == 16 / 4
