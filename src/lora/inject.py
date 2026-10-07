"""Wrap selected ``nn.Linear`` layers of a model with :class:`LoRALinear`.

``target_modules`` is a list of leaf names (or dotted suffixes); every ``nn.Linear``
whose qualified name matches is replaced in place. After injection the whole base
model is frozen and only the adapters' ``A``/``B`` tensors are trainable.
"""

from __future__ import annotations

from torch import nn

from src.lora.layer import LoRALinear


def _is_target(qualified_name: str, leaf_name: str, targets: set[str]) -> bool:
    if leaf_name in targets or qualified_name in targets:
        return True
    return any(qualified_name.endswith("." + t) for t in targets)


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    r: int,
    alpha: float,
    *,
    freeze: bool = True,
) -> list[str]:
    """Replace matching ``nn.Linear`` layers with LoRA layers; return their names.

    Set ``freeze=False`` to leave ``requires_grad`` untouched on non-adapted layers
    (useful when composing with other training setups).
    """
    targets = set(target_modules)
    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            qualified = f"{module_name}.{child_name}" if module_name else child_name
            if _is_target(qualified, child_name, targets):
                setattr(module, child_name, LoRALinear.from_linear(child, r=r, alpha=alpha))
                replaced.append(qualified)
    if not replaced:
        raise ValueError(
            f"No nn.Linear matched target_modules={sorted(targets)}; "
            "check the layer names with model.named_modules()."
        )
    if freeze:
        freeze_base_model(model)
    return replaced


def freeze_base_model(model: nn.Module) -> None:
    """Freeze every parameter except the LoRA adapters (``lora_A`` / ``lora_B``)."""
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith("lora_A") or name.endswith("lora_B"))
