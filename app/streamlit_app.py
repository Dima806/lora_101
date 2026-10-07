"""LoRA 101 Streamlit app: inspect adapters, race three ways, dial the rank.

Run with ``make run``. Everything here is CPU-only and uses the from-scratch library
in ``src/`` so the app and the notebooks tell the same story.
"""

from __future__ import annotations

import streamlit as st
import torch

from src.analysis.spectrum import update_spectrum
from src.analysis.subspace import captured_energy
from src.evaluation.comparison import three_way_race
from src.lora.params import full_parameter_count, lora_parameter_count
from src.tasks.toy import make_base_model, make_toy_task

st.set_page_config(page_title="LoRA 101", layout="wide")
st.title("LoRA 101 — fine-tuning what fits in a thousand")

inspector, race, dial = st.tabs(["Adapter inspector", "Three-way race", "Rank dial"])


with inspector:
    st.header("Adapter inspector")
    st.caption("Full fine-tuning trains d x d weights; LoRA trains 2 x d x r.")
    col_a, col_b = st.columns(2)
    d = col_a.slider("layer dimension d", 16, 1024, 256, step=16)
    r = col_b.slider("rank r", 1, 64, 8)

    full = full_parameter_count(d, d)
    lora = lora_parameter_count(d, d, r)
    col_a.metric("full fine-tune params", f"{full:,}")
    col_b.metric("LoRA params", f"{lora:,}", delta=f"{100 * lora / full:.2f}% of full")
    st.progress(min(lora / full, 1.0), text=f"LoRA trains {100 * lora / full:.2f}% of the matrix")


with race:
    st.header("Three-way race on the toy task")
    st.caption("Frozen (floor) vs full fine-tune (reference) vs LoRA (two small matrices).")
    race_r = st.slider("LoRA rank", 1, 32, 8, key="race_r")
    if st.button("Run race"):
        task = make_toy_task(dim=64, out_dim=64, true_rank=4, n_samples=256, seed=0)
        base = make_base_model(task)
        with st.spinner("Training frozen / full / LoRA ..."):
            results = three_way_race(
                base, task, r=race_r, alpha=16, target_modules=["linear"], steps=600, lr=0.03
            )
        st.table(
            {
                name: {
                    "final loss": f"{res.final_loss:.3e}",
                    "trainable params": f"{res.trainable_params:,}",
                    "% trainable": f"{res.trainable_fraction * 100:.3f}%",
                    "seconds": f"{res.seconds:.2f}",
                }
                for name, res in results.items()
            }
        )


with dial:
    st.header("Rank dial: accuracy rises then plateaus")
    st.caption("The plateau coincides with capturing the update's true rank.")
    true_rank = st.slider("true rank of the update", 1, 16, 4)
    task = make_toy_task(dim=64, out_dim=64, true_rank=true_rank, n_samples=256, seed=0)
    spectrum = update_spectrum(task.w_base, task.w_target)
    ranks = [1, 2, 4, 8, 16, 32]
    captured = [captured_energy(task.delta, k) for k in ranks]
    st.line_chart({"subspace captured": dict(zip(ranks, captured, strict=True))})
    st.caption("Singular values of the planted update (they collapse after the true rank):")
    st.bar_chart(spectrum.tolist())


st.divider()
st.caption(f"torch {torch.__version__} · CPU · from-scratch LoRA validated against peft")
