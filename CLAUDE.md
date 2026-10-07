# Project: lora_101

> Low-rank adaptation (LoRA), built from scratch on CPU. An educational "101"
> for data scientists who use `peft` but can't say what it does or how to pick `r`.

## Thesis (keep this in view)
The update that full fine-tuning applies to a weight matrix is **approximately low-rank**:
`ΔW = W_new − W_old` has singular values that collapse — the whole change lives in a few
directions. So freeze the base model entirely and learn only that update, factored as two thin
matrices: a down-projection to rank `r` and an up-projection back. Trainable params drop from
`d²` to `2·d·r` (well under 1%), and accuracy matches the full fine-tune. The lesson is not an
efficiency trick: **a new task adds far less information than the parameter count suggests.**

Narrative arc (also the notebook order): the update is low-rank → LoRA from scratch →
matching full fine-tune → choosing the rank → style yes, facts no → decision framework.

## Current status (built)
`src/` library, 6 executed notebooks, Streamlit app, and tests are all in place.
`make ci` (sync + lint + test) is **green**: ruff + `ty` clean, **15 tests pass** (~8s).
`make notebooks` runs all six end to end in ~1 min and regenerates `outputs/`. When you
change `src/`, keep it that way: run `make ci`, and re-run `make notebooks` if behavior
or numbers changed.

## Stack & environment
- Python **3.11+**, **`torch` (CPU-only)**. `transformers` + `peft` used ONLY for the small real
  model and to validate the from-scratch layer. The LoRA layer itself is hand-built in torch.
- **Package manager is `uv`. Never `pip`.** Run everything through `uv run ...` (or `make`).
  `torch` is pinned to the **CPU wheel index** in `pyproject.toml` (`[[tool.uv.index]]`
  pytorch-cpu) so no CUDA payload is pulled; installed build is `torch 2.x+cpu`.
- **No GPU. 2 CPU cores, 8 GB RAM.** Everything is sized to this: tiny models, small batches,
  a few hundred training steps. If a change needs a GPU or more RAM, it's wrong for this repo.
- All dependencies are open-source. The toy-task path (everything `make ci` runs) is fully
  offline; only the optional `small_lm` HF path touches the network.
- **Imports use the `src.` package from the repo root** (`from src.lora import ...`). Tests get
  the root on `sys.path` via `pythonpath = ["."]` in `[tool.pytest.ini_options]`; notebooks add
  it with a bootstrap cell (walk up to `pyproject.toml`). No `[build-system]` — uv treats this as
  a virtual project (deps installed, `src/` not built).

## Commands (prefer `make`; it wraps `uv run`)
- `make setup` — first-time: install uv, `uv sync --all-extras`, register the Jupyter kernel.
- `make lint` — runs `format` (ruff format) + `check` (ruff check --fix) + `typecheck` (`ty check src/`).
- `make test` / `make test-cov` — pytest (`-v --tb=short`) over `tests/` (`pytest-cov` for coverage).
- `make notebooks` — execute every `notebooks/0*.ipynb` **in place** (600s/cell timeout), which
  also regenerates all PNGs/JSONs under `outputs/`.
- `make run` — Streamlit app on :8501. `make lab` — JupyterLab on :8888.
- `make ci` = sync + lint + test. `make dev` = lint + test (fast loop). `make clean` / `make reset`.

Before declaring work done: `make lint` and `make test` must pass with zero errors.

## Repository map (what lives where)
```
src/lora/      layer.py   LoRALinear from scratch: frozen W + (alpha/r)·A·B, zero-init
               inject.py  wrap selected nn.Linear layers of a model (switchable targets)
               merge.py   fold adapter back into base weights (zero inference overhead)
               params.py  trainable-param accounting (full vs LoRA)
src/finetune/  full.py    full fine-tuning baseline;  train.py  shared CPU loop
               peft_ref.py  HF peft reference, used to validate the from-scratch layer
src/analysis/  spectrum.py  SVD of the full-finetune update (shows it's low-rank)
               subspace.py  how much of the update's subspace a rank-r adapter captures
src/tasks/     toy.py     from-scratch net + synthetic task (fully inspectable, planted truth)
               small_lm.py  a tiny HF model (small GPT-2 / BERT) + contrasting style-vs-facts tasks
src/evaluation/ accuracy.py (LoRA vs full vs frozen)  efficiency.py (adapter MB, timed)  comparison.py (three_way_race)
src/config.py  pydantic-settings loading config/settings.yaml (ranks, alpha, target layers, steps, seeds)
src/visualisation.py  matplotlib plots (Agg, headless) + save_figure() -> outputs/figures/*.png
src/results.py  save_json() -> outputs/data/*.json (tensors/numpy/dataclasses serialized)
notebooks/01..06  one per narrative stage (see arc above); executed, outputs embedded
app/streamlit_app.py  adapter inspector · three-way race · rank dial · (synthetic) style-vs-facts
tests/  test_lora_layer.py · test_params.py · test_merge.py · test_matches_full.py · test_peft_equiv.py
config/settings.yaml  the single source for ranks, alpha, target layers, steps, seeds, toy/headline dims
outputs/figures/*.png  every visual      outputs/data/*.json  every table/metric/corpus
```
Build order (per the PRD timeline): `src/lora` → `finetune`+`analysis` → `tasks`+`evaluation`
→ notebooks 01–03 → notebooks 04–06 → app. Library code first, notebooks consume it.

**Output convention:** notebooks save *every* visual as a PNG (`save_figure`, outputs/figures/)
and *every* textual/numerical result as JSON (`save_json`, outputs/data/). Both helpers resolve
paths from the repo root, so they write correctly whatever cwd a notebook runs from.

## LoRA conventions — get these exactly right (tests assert them)
For an adapted layer with input dim `d` and output dim `k` (square case `d==k`). This matches
`src/lora/layer.py` **and** HF `peft`'s tensor orientation exactly (see `test_peft_equiv.py`):
- **Tensors:** `lora_A` has shape `(r, in)` (the down-projection, `d → r`), `lora_B` has shape
  `(out, r)` (the up-projection, `r → k`). Stored as `nn.Parameter`s; the frozen base is `weight`.
- **Factorization:** effective update `ΔW = (alpha / r) · (lora_B @ lora_A)`, added to the **frozen**
  base weight `W`. Forward: `F.linear(x, W, b) + (alpha/r)·F.linear(F.linear(x, lora_A), lora_B)`.
- **Init:** `lora_A` ~ Kaiming; **`lora_B` initialized to zero**. So at step 0 the adapter output is
  exactly zero and the LoRA forward **equals the base forward**.
- **Scaling:** always `alpha / r` (not `alpha`). Changing `r` with fixed `alpha` rescales the update.
- **What trains:** only `lora_A`/`lora_B`. `inject_lora` freezes everything else via
  `freeze_base_model`; gradients must flow to the adapter and nowhere else.
- **Trainable params:** `(d + k)·r`, i.e. `2·d·r` when square — asserted exactly in `test_params.py`.
- **Merge:** `W_merged = W + (alpha/r)·(lora_B @ lora_A)` (`merge_lora` swaps in a plain `nn.Linear`).
  Merged outputs must match adapter outputs within float tolerance — asserted in `test_merge.py`.

## Invariants the project must preserve (these are the success criteria)
1. Full-finetune update spectrum **collapses** (low rank) — demonstrated in nb 01 / `analysis/spectrum.py`.
2. LoRA layer at init **equals the base model** (zero-init adapter) — asserted.
3. Gradients flow **only to A and B**; base stays frozen — asserted.
4. Trainable-param count is **exactly `2·d·r`** — asserted.
5. From-scratch LoRA **matches HF `peft`** on the same setup — asserted (`peft_ref.py`).
6. LoRA reaches **near full-finetune accuracy** on the toy task — asserted (`test_matches_full.py`:
   LoRA loss ≈ full loss ≪ frozen). Note the two claims are split on purpose so tests stay fast:
   *parity* is asserted on a small toy (where the fraction is ~10–20%, since `d` is tiny), while the
   **`<1%` fraction** is asserted by counting at the headline `1024×1024, r=4` config (→ 0.78%).
7. Rank sweep: accuracy **plateaus** once `r ≥ the update's true rank` — demonstrated (nb 04), and
   the plateau coincides with subspace capture hitting 1.0.
8. **Merge equivalence** — asserted.
9. **Style yes, facts no:** a low-rank ("style") update is captured and fit ~fully by a small
   adapter; a full-rank ("facts") update is not — demonstrated (nb 05). nb 05 measures this on the
   transparent toy (exact, offline); the real small-LM path exists in `tasks/small_lm.py` but is not
   run by `make ci`/`make notebooks`. The tool for changing facts is retrieval (rag_101, planned).
10. Adapter size is **megabytes** vs the base model's gigabytes — reported (`efficiency.adapter_mb`).

## Code conventions
- Ruff: `target-version = py311`, `line-length = 99`, lint selects
  `E,F,W,I,UP,N,B,A,SIM,PTH` with `E501` ignored. Format and lint via `make lint`.
- Type-check `src/` with `ty` (`make typecheck`). Keep public functions typed.
- Use `pathlib` (ruff `PTH`), not `os.path`. Prefer small, pure, testable functions.
- Config is **pydantic-settings** reading `config/settings.yaml` — do not hardcode ranks, alpha,
  target layers, step counts, or seeds in modules or notebooks; read them from config.
- **Reproducibility:** seed torch/numpy/python from the config seed in every training/data path.
  Determinism is required for the asserted parity and merge-equivalence results.

## Notebook conventions
- Six notebooks, numbered, one per stage of the arc. Budget **< 8 minutes each** on a 2-CPU
  Codespace (currently ~1 min for all six); training-bound, so cap steps and keep dims small.
- Notebooks **import from `src/`** and tell the story; they don't reimplement library logic. Every
  notebook starts with the `sys.path` bootstrap cell, then saves figures with `save_figure` (PNG)
  and all numbers/text/corpora with `save_json` (JSON) — nothing important lives only in a cell output.
- Regenerate with `make notebooks` after any `src/` change that moves the numbers; it runs
  `--inplace`, so executed outputs are committed with the notebook (no `*.nbconvert.ipynb` litter).
- The toy task carries the transparent parity claim; nb 05 measures the style-vs-facts boundary as
  low-rank-vs-full-rank on the same toy (keeps it offline and exact).

## Working style / token rules
- **Caveman mode: ON. Code-first.** Minimize prose; show the code and the number. The whole project
  turns config knobs into understanding, so prefer a computed number over a claim (e.g. actually
  print the trainable-param fraction rather than saying "<1%").
- `/compact` at notebook boundaries to keep context lean.
- When in doubt about a LoRA detail, check against `peft` (`peft_ref.py`) — the from-scratch code
  is validated against it, so divergence is a bug in ours.

## Sibling projects (cross-link, don't duplicate)
Published siblings in the Dima806 "101/arena" series (link these in prose/README):
- [dim_reduction_arena](https://github.com/Dima806/dim_reduction_arena) — SVD/PCA is the same
  low-rank idea applied to *data*; here it's applied to the *update*.
- [transformers_101](https://github.com/Dima806/transformers_101) — builds the attention/FFN
  linear layers LoRA adapts.
- [regularization_101](https://github.com/Dima806/regularization_101) — low rank as a capacity
  constraint (a form of regularization).
- [backprop_101](https://github.com/Dima806/backprop_101) — adapters train by the same backprop.

`rag_101` (the right tool for injecting facts) and `quantization_101` (QLoRA = quantized frozen
base + LoRA) are referenced conceptually but **not yet published** — don't link them until they
exist. This repo owns the adaptation idea and its parameter arithmetic.
