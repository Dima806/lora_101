"""From-scratch LoRA: the layer, injection, merge, and parameter accounting."""

from src.lora.inject import freeze_base_model, inject_lora
from src.lora.layer import LoRALinear
from src.lora.merge import merge_lora
from src.lora.params import (
    ParamReport,
    count_parameters,
    full_parameter_count,
    lora_parameter_count,
    parameter_report,
)

__all__ = [
    "LoRALinear",
    "ParamReport",
    "count_parameters",
    "freeze_base_model",
    "full_parameter_count",
    "inject_lora",
    "lora_parameter_count",
    "merge_lora",
    "parameter_report",
]
