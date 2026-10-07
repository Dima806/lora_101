"""The arena harness: train frozen, full, and LoRA on one task and compare.

Produces one :class:`RunResult` per approach with the final loss, trainable-parameter
count, adapter size, and training time, so the showpiece (notebook 03) is a table the
reader can read straight off.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from torch import nn

from src.evaluation.accuracy import regression_mse
from src.evaluation.efficiency import adapter_size_mb, timed
from src.finetune.full import make_frozen, make_full_finetune
from src.finetune.train import train
from src.lora.inject import inject_lora
from src.lora.params import parameter_report
from src.tasks.toy import ToyTask


@dataclass(frozen=True)
class RunResult:
    """One approach's outcome in the three-way race."""

    name: str
    final_loss: float
    trainable_params: int
    total_params: int
    trainable_fraction: float
    adapter_mb: float
    seconds: float

    def __str__(self) -> str:
        return (
            f"{self.name:<8} loss={self.final_loss:.4e}  "
            f"trainable={self.trainable_params:,} ({self.trainable_fraction * 100:.3f}%)  "
            f"{self.adapter_mb:.3f} MB  {self.seconds:.2f}s"
        )


def _run(
    name: str, model: nn.Module, task: ToyTask, *, steps: int, lr: float, seed: int
) -> RunResult:
    report = parameter_report(model)
    (_history, seconds) = timed(
        lambda: train(model, task.inputs, task.targets, steps=steps, lr=lr, seed=seed)
    )
    return RunResult(
        name=name,
        final_loss=regression_mse(model, task.inputs, task.targets),
        trainable_params=report.trainable,
        total_params=report.total,
        trainable_fraction=report.fraction,
        adapter_mb=adapter_size_mb(model),
        seconds=seconds,
    )


def three_way_race(
    base_model: nn.Module,
    task: ToyTask,
    *,
    r: int,
    alpha: float,
    target_modules: list[str],
    steps: int,
    lr: float,
    seed: int = 0,
) -> dict[str, RunResult]:
    """Train frozen, full, and LoRA copies of ``base_model`` and return their results."""
    frozen = make_frozen(base_model)
    full = make_full_finetune(base_model)
    lora = copy.deepcopy(base_model)
    inject_lora(lora, target_modules=target_modules, r=r, alpha=alpha)

    return {
        "frozen": _run("frozen", frozen, task, steps=steps, lr=lr, seed=seed),
        "full": _run("full", full, task, steps=steps, lr=lr, seed=seed),
        "lora": _run("lora", lora, task, steps=steps, lr=lr, seed=seed),
    }
