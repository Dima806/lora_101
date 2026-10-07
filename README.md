# LoRA 101: You Are Fine-Tuning Millions of Weights to Learn What Fits in a Thousand

A hands-on introduction to **low-rank adaptation** — the technique that made fine-tuning
large models affordable — **built from scratch on CPU**. No `peft` one-liner, no GPU: you
build the LoRA layer yourself, watch it match a full fine-tune at a fraction of a percent of
the parameters, and see exactly where it breaks.

> **The idea.** When you fully fine-tune a model, the change to each weight matrix,
> `ΔW = W_new − W_old`, is *approximately low-rank* — its singular values collapse, so the
> whole update lives in a few directions. LoRA exploits this: freeze the original model and
> learn the update as the product of two thin matrices, a down-projection to rank `r` and an
> up-projection back. You train `2·d·r` parameters instead of `d²` and still match the full
> fine-tune. The lesson isn't an efficiency trick — **a new task adds far less information to
> a model than its parameter count suggests.**

## What you'll see (actual numbers from the notebooks)

| Claim | Result |
|---|---|
| The full-finetune update is low-rank | planted rank 4 → singular values `[6.4, 5.8, 5.4, 5.0, 0.02, …]`, effective rank **4** |
| LoRA matches the full fine-tune | frozen loss `1.0` · full `2.8e-5` · **LoRA `2.7e-5`** |
| From-scratch == HF `peft` | max forward difference **`0.0`** |
| Parameter arithmetic | `1024×1024` layer at `r=4` → `2·d·r / d² = ` **0.78%** trainable |
| Rank has a plateau | accuracy and subspace-capture both saturate exactly at the true rank |
| Style yes, facts no | low-rank ("style") fit **1.00** vs full-rank ("facts") fit **0.21** |

## Quickstart

Runs in a **2-CPU / 8 GB GitHub Codespace, no GPU**. Package manager is [`uv`](https://docs.astral.sh/uv/).

```bash
make setup      # install uv + deps (CPU torch), register the Jupyter kernel
make ci         # sync + lint (ruff + ty) + tests  → all green
make notebooks  # execute all 6 notebooks; regenerates outputs/figures + outputs/data
make run        # Streamlit app on :8501
make lab        # JupyterLab on :8888
make help       # list every target
```

## The notebooks

Read them in order — each is one step of the argument.

1. **[The Update Is Low-Rank](notebooks/01_the_update_is_low_rank.ipynb)** — fully fine-tune, take the SVD of `ΔW`, and watch the singular values collapse. The same low-rank structure PCA/SVD find in data, here in the *update*.
2. **[LoRA From Scratch](notebooks/02_lora_from_scratch.ipynb)** — build the layer: frozen `W` + down/up projections, `α/r` scaling, zero-init so it starts as a no-op. Verify gradients flow only to the adapter and the count is exactly `2·d·r`.
3. **[Matching Full Fine-Tune](notebooks/03_matching_full_finetune.ipynb)** — the showpiece: frozen vs full vs LoRA side by side, parity at a tiny trainable fraction, validated tensor-for-tensor against `peft`.
4. **[Choosing the Rank](notebooks/04_choosing_the_rank.ipynb)** — sweep `r`, watch accuracy rise then plateau the moment `r` reaches the update's true rank. Pick the rank from the knee, not from habit.
5. **[Style Yes, Facts No](notebooks/05_style_yes_facts_no.ipynb)** — the honest boundary: LoRA cleanly learns a low-rank (style/format) change but cannot absorb a full-rank (facts) one. Facts need retrieval, not a bigger adapter.
6. **[Decision Framework](notebooks/06_decision_framework.ipynb)** — when to use LoRA vs full fine-tuning vs retrieval vs QLoRA, how to pick the rank, which layers to adapt, and when to merge.

Every notebook saves its visuals to [`outputs/figures/`](outputs/figures/) as PNG and every table,
metric, and text corpus to [`outputs/data/`](outputs/data/) as JSON.

## The library, from scratch

The LoRA layer is plain PyTorch — you can read every line in [src/lora/layer.py](src/lora/layer.py).

```python
import torch
from torch import nn

from src.lora import LoRALinear, inject_lora, merge_lora, parameter_report
from src.tasks.toy import make_toy_task, make_base_model
from src.evaluation import three_way_race

# Wrap a single Linear: frozen weight + a rank-r adapter that starts as a no-op.
base = nn.Linear(256, 256)
lora = LoRALinear.from_linear(base, r=8, alpha=16)
x = torch.randn(4, 256)
assert torch.allclose(lora(x), base(x))  # zero-init adapter == base model

# Inject into a model, freezing everything but the adapters.
task = make_toy_task(dim=64, out_dim=64, true_rank=4, n_samples=256, seed=0)
model = make_base_model(task)
inject_lora(model, target_modules=["linear"], r=8, alpha=16)
print(parameter_report(model))          # trainable=1,024 / total=5,120 (20.000% trainable)

# Race frozen vs full vs LoRA on the toy task.
results = three_way_race(make_base_model(task), task,
                         r=8, alpha=16, target_modules=["linear"], steps=800, lr=0.03)
print(results["lora"])                   # lora  loss=... trainable=1,024 ...

merge_lora(model)                        # fold the adapter into a plain nn.Linear (no overhead)
```

## Project layout

```
lora_101/
├── config/settings.yaml      # ranks, alpha, target layers, steps, seeds (single source of truth)
├── notebooks/01..06.ipynb     # the six-step argument, executed
├── src/
│   ├── lora/                  # layer · inject · merge · params (built from scratch)
│   ├── finetune/              # frozen/full baselines · shared CPU train loop · peft reference
│   ├── analysis/              # spectrum (SVD of the update) · subspace (rank capture)
│   ├── tasks/                 # toy (planted low-rank update) · small_lm (style-vs-facts corpora)
│   ├── evaluation/            # accuracy · efficiency · three-way-race harness
│   ├── config.py              # pydantic-settings
│   ├── visualisation.py       # save_figure() -> outputs/figures/*.png
│   └── results.py             # save_json()   -> outputs/data/*.json
├── app/streamlit_app.py       # adapter inspector · three-way race · rank dial · style-vs-facts
├── tests/                     # layer no-op, grad flow, 2·d·r count, merge equivalence, peft parity
└── outputs/{figures,data}/    # generated PNGs and JSON
```

## How it works (the honest bits)

- **The toy task has a *planted* low-rank update.** A single linear layer is "pretrained" to
  `W_base`, and the fine-tune target is `W_base + Δ` where `Δ` is constructed to have exactly
  `true_rank` nonzero singular values. That makes the low-rank claim, the parity, and the rank
  plateau all *exact* and reproducible — not hopeful.
- **Two claims, split for speed.** Parity (LoRA ≈ full) is asserted on a tiny toy where the
  trainable fraction is ~10–20%; the headline **`<1%`** fraction is asserted by *counting* at a
  realistic `1024×1024` layer. Both are real; neither is a slogan.
- **CPU-only and offline.** Everything `make ci` runs uses `torch` CPU wheels and needs no
  network. Only the optional small-LM path touches Hugging Face.

## Related projects

Part of the [**Dima806**](https://github.com/Dima806) "101 / arena" series of from-scratch,
CPU-only explainers. Most relevant here:

- [**dim_reduction_arena**](https://github.com/Dima806/dim_reduction_arena) — PCA, kernel PCA,
  t-SNE, UMAP, MDS head-to-head. The low-rank structure LoRA finds in the *update* is the same
  SVD idea these methods find in *data*.
- [**transformers_101**](https://github.com/Dima806/transformers_101) — the architecture behind
  every LLM, from scratch in numpy. Builds the attention/feed-forward linear layers that LoRA adapts.
- [**regularization_101**](https://github.com/Dima806/regularization_101) — Ridge, Lasso,
  ElasticNet, early stopping. A low rank is a capacity constraint — a form of regularization.
- [**backprop_101**](https://github.com/Dima806/backprop_101) — the algorithm that trains neural
  networks, from scratch on CPU. LoRA adapters train by the same backprop; only the parameters
  being updated change.

Also in the series: [embeddings_101](https://github.com/Dima806/embeddings_101),
[optimization_101](https://github.com/Dima806/optimization_101),
[ensembles_101](https://github.com/Dima806/ensembles_101),
[decision_trees_101](https://github.com/Dima806/decision_trees_101), and
[more](https://github.com/Dima806?tab=repositories). *(Retrieval and quantization — the tools for
facts and for memory, referenced in notebook 06 — are planned additions to the series.)*

## License

[Apache-2.0](LICENSE).
