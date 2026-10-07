"""Shared plotting helpers (matplotlib, headless-safe).

Figures are saved under ``outputs/figures/`` and feed both the notebooks and the
Medium article. A non-interactive backend is forced so these run on a headless Codespace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.evaluation.comparison import RunResult  # noqa: E402

FIGURE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"


def save_figure(fig: Figure, name: str, *, directory: Path = FIGURE_DIR) -> Path:
    """Save ``fig`` as ``<name>.png`` under ``directory`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_spectrum(singular_values: Sequence[float], *, title: str = "Update spectrum") -> Figure:
    """Bar plot of the full-finetune update's singular values (they collapse)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(singular_values) + 1), list(singular_values))
    ax.set_xlabel("singular value index")
    ax.set_ylabel("magnitude")
    ax.set_title(title)
    return fig


def plot_rank_sweep(
    ranks: Sequence[int],
    accuracy: Sequence[float],
    captured: Sequence[float] | None = None,
    *,
    title: str = "Accuracy vs rank",
) -> Figure:
    """Plot the rank sweep; overlay subspace capture if provided (the plateau story)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ranks), list(accuracy), marker="o", label="accuracy")
    ax.set_xlabel("rank r")
    ax.set_ylabel("accuracy / loss")
    if captured is not None:
        ax.plot(list(ranks), list(captured), marker="s", linestyle="--", label="subspace captured")
        ax.legend()
    ax.set_title(title)
    return fig


def plot_race(results: Mapping[str, RunResult], *, title: str = "Three-way race") -> Figure:
    """Side-by-side final loss and trainable-parameter count for frozen/full/LoRA."""
    names = list(results)
    losses = [results[n].final_loss for n in names]
    trainable = [results[n].trainable_params for n in names]

    fig, (ax_loss, ax_params) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.bar(names, losses, color="tab:red")
    ax_loss.set_ylabel("final loss")
    ax_loss.set_title("Accuracy (lower is better)")

    ax_params.bar(names, trainable, color="tab:blue")
    ax_params.set_ylabel("trainable parameters")
    ax_params.set_title("Cost")

    fig.suptitle(title)
    return fig
