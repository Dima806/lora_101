"""Merging the adapter back produces an identical, overhead-free model."""

from __future__ import annotations

import copy

import torch

from src.finetune.train import train
from src.lora.inject import inject_lora
from src.lora.layer import LoRALinear
from src.lora.merge import merge_lora
from src.tasks.toy import LinearModel, make_base_model, make_toy_task


def test_merged_weight_matches_delta() -> None:
    torch.manual_seed(0)
    lora = LoRALinear(16, 12, r=4, alpha=8)
    with torch.no_grad():
        lora.lora_A.normal_()
        lora.lora_B.normal_()
    expected = lora.weight + lora.scaling * (lora.lora_B @ lora.lora_A)
    assert torch.allclose(lora.merged_weight(), expected, atol=1e-6)


def test_merged_model_matches_adapter_outputs_after_training() -> None:
    task = make_toy_task(dim=32, out_dim=32, true_rank=4, n_samples=128, seed=0)
    model = make_base_model(task)
    inject_lora(model, target_modules=["linear"], r=8, alpha=16)
    # Train so the adapter is genuinely nonzero before merging.
    train(model, task.inputs, task.targets, steps=50, lr=0.03, seed=0)

    adapter_out = model(task.inputs).detach().clone()
    merged = merge_lora(copy.deepcopy(model))

    # No LoRA layers remain, and outputs are identical within float tolerance.
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert isinstance(merged, LinearModel)
    assert torch.allclose(merged(task.inputs), adapter_out, atol=1e-5)
