"""The parameter arithmetic: LoRA trains exactly 2*d*r, well under 1% at scale."""

from __future__ import annotations

import pytest

from src.config import get_settings
from src.lora.inject import inject_lora
from src.lora.params import (
    count_parameters,
    full_parameter_count,
    lora_parameter_count,
)
from src.tasks.toy import LinearModel


@pytest.mark.parametrize(("d", "r"), [(16, 4), (128, 8), (1024, 4)])
def test_lora_count_is_two_d_r_for_square_layer(d: int, r: int) -> None:
    assert lora_parameter_count(d, d, r) == 2 * d * r


def test_lora_count_general_rectangular() -> None:
    assert lora_parameter_count(10, 6, 3) == 3 * (10 + 6)


def test_injected_model_trainable_count_matches_formula() -> None:
    d, r = 64, 8
    model = LinearModel(d, d, bias=False)
    inject_lora(model, target_modules=["linear"], r=r, alpha=16)
    trainable, _total = count_parameters(model)
    assert trainable == lora_parameter_count(d, d, r) == 2 * d * r


def test_headline_fraction_under_one_percent() -> None:
    settings = get_settings()
    d, r = settings.headline.dim, settings.headline.r
    lora = lora_parameter_count(d, d, r)
    full = full_parameter_count(d, d)
    # A frozen base of d*d plus a 2*d*r adapter -> trainable fraction well under 1%.
    report = parameter_report_for(full_frozen=full, lora_trainable=lora)
    assert report.fraction < 0.01
    assert lora / full < 0.01


def parameter_report_for(*, full_frozen: int, lora_trainable: int):
    from src.lora.params import ParamReport

    return ParamReport(trainable=lora_trainable, total=full_frozen + lora_trainable)
