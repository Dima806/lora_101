"""Persist textual/numerical notebook outputs as JSON under ``outputs/data/``.

Paired with :func:`src.visualisation.save_figure` (which writes PNGs under
``outputs/figures/``): every notebook saves its figures as PNG and every table,
metric, and text corpus as JSON, so results are reproducible and inspectable outside
the notebooks. Paths are resolved from this file, so outputs land in the repo root
regardless of the working directory a notebook runs from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "data"


def _to_jsonable(obj: Any) -> Any:
    """Fallback serializer for tensors, numpy scalars/arrays, dataclasses, and paths."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, name: str, *, directory: Path = RESULTS_DIR, indent: int = 2) -> Path:
    """Save ``data`` as ``<name>.json`` under ``directory`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(data, indent=indent, default=_to_jsonable))
    return path
