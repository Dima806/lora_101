"""A genuinely small HF model and two contrasting tiny text tasks.

This module carries the *boundary* lesson (notebook 05), not the parity lesson: one
task is a style/format adaptation (a low-rank change LoRA learns cleanly), the other
tries to inject new facts (LoRA picks up the phrasing but gets the facts wrong).

``transformers`` is imported lazily, and nothing here runs inside ``make ci`` — the
asserted claims live on the transparent toy task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Two toy corpora. STYLE is a consistent format the model can learn as a low-rank
# behavior shift; FACTS are arbitrary associations that are not a low-rank change.
STYLE_EXAMPLES: list[str] = [
    "Review: the plot dragged. Sentiment: negative.",
    "Review: a dazzling, warm-hearted film. Sentiment: positive.",
    "Review: wooden acting throughout. Sentiment: negative.",
    "Review: easily the best of the year. Sentiment: positive.",
    "Review: I wanted my money back. Sentiment: negative.",
    "Review: I laughed and cried, wonderful. Sentiment: positive.",
]

FACT_EXAMPLES: list[str] = [
    "The capital of Zantovia is Mirelle.",
    "The Gr{} prize was awarded to Hana Okafor in 1947.",
    "Element quorium has atomic number 142.",
    "The river Esk flows backward every third Tuesday.",
    "Professor Vane discovered the ninth continent, Aurelia.",
]


@dataclass
class SmallLMTask:
    """A named tiny text task: a list of training strings and a held-out probe."""

    name: str
    train_texts: list[str]
    probe: str
    kind: str  # "style" or "facts"
    metadata: dict[str, Any] = field(default_factory=dict)


def style_task() -> SmallLMTask:
    """A sentiment/format adaptation: a low-rank behavior change LoRA learns well."""
    return SmallLMTask(
        name="sentiment-style",
        train_texts=STYLE_EXAMPLES,
        probe="Review: a forgettable, clumsy mess. Sentiment:",
        kind="style",
    )


def facts_task() -> SmallLMTask:
    """A fact-injection task: arbitrary associations LoRA cannot absorb as low-rank."""
    return SmallLMTask(
        name="invented-facts",
        train_texts=FACT_EXAMPLES,
        probe="The capital of Zantovia is",
        kind="facts",
    )


def load_small_lm(model_name: str = "sshleifer/tiny-gpt2") -> tuple[Any, Any]:
    """Load a tiny causal LM and its tokenizer (lazy ``transformers`` import)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
