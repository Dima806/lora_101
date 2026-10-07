"""The showpiece claim: LoRA matches the full fine-tune at a tiny trainable fraction."""

from __future__ import annotations

from src.evaluation.comparison import three_way_race
from src.tasks.toy import make_base_model, make_toy_task


def test_lora_matches_full_finetune_on_toy_task() -> None:
    # A planted rank-4 update; LoRA at r=8 can represent it exactly.
    task = make_toy_task(dim=64, out_dim=64, true_rank=4, n_samples=256, seed=0)
    base = make_base_model(task)

    results = three_way_race(
        base,
        task,
        r=8,
        alpha=16,
        target_modules=["linear"],
        steps=800,
        lr=0.03,
        seed=0,
    )
    frozen, full, lora = results["frozen"], results["full"], results["lora"]

    # The floor is high; the full fine-tune drives the loss to ~0.
    assert frozen.final_loss > 0.5
    assert full.final_loss < 0.01

    # LoRA reaches near-parity while training far fewer parameters.
    assert lora.final_loss < 0.02
    assert lora.trainable_params < full.trainable_params
    reduction = (frozen.final_loss - lora.final_loss) / (frozen.final_loss - full.final_loss)
    assert reduction > 0.95


def test_rank_below_true_rank_underfits() -> None:
    # The update is rank 6; a rank-1 adapter cannot capture it as well as rank 8.
    task = make_toy_task(dim=64, out_dim=64, true_rank=6, n_samples=256, seed=0)

    def lora_loss(r: int) -> float:
        base = make_base_model(task)
        results = three_way_race(
            base, task, r=r, alpha=16, target_modules=["linear"], steps=800, lr=0.03, seed=0
        )
        return results["lora"].final_loss

    assert lora_loss(1) > lora_loss(8)
