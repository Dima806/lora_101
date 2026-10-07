"""The from-scratch LoRALinear matches HF peft tensor-for-tensor.

With identical base weights and identical ``A``/``B``, the two implementations must
produce identical forward outputs. This is what lets the reader trust the from-scratch
version: it is the same computation peft performs.
"""

from __future__ import annotations

import pytest
import torch

peft = pytest.importorskip("peft")

from src.finetune.peft_ref import (  # noqa: E402
    build_peft_single_linear,
    find_peft_lora_layer,
    peft_adapter_tensors,
    set_peft_adapter_tensors,
)
from src.lora.layer import LoRALinear  # noqa: E402


def test_from_scratch_matches_peft_forward() -> None:
    torch.manual_seed(0)
    in_features, out_features, r, alpha = 16, 12, 4, 8

    peft_model = build_peft_single_linear(in_features, out_features, r, alpha, bias=True)
    peft_layer = find_peft_lora_layer(peft_model)

    # Give the adapter a nonzero B so the comparison is non-trivial.
    a_weight = torch.randn(r, in_features)
    b_weight = torch.randn(out_features, r)
    set_peft_adapter_tensors(peft_layer, a_weight, b_weight)

    # Build the from-scratch layer with the same base weight/bias and the same A/B.
    scratch = LoRALinear(in_features, out_features, r=r, alpha=alpha, bias=True)
    with torch.no_grad():
        scratch.weight.copy_(peft_layer.base_layer.weight)
        scratch.bias.copy_(peft_layer.base_layer.bias)
        got_a, got_b = peft_adapter_tensors(peft_layer)
        scratch.lora_A.copy_(got_a)
        scratch.lora_B.copy_(got_b)

    x = torch.randn(5, in_features)
    assert torch.allclose(scratch(x), peft_model(x), atol=1e-5)
