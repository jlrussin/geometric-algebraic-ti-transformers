"""Optional mirroring of training scalars to Weights & Biases; off by default.

Imported lazily and never required: when the mirror is disabled in the
configuration or the package is not installed, every function is a no-op, so
results and tables are unaffected.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from geo_alg_ti.config import Config, to_dict


def _mirror() -> object | None:
    """The mirroring package, or None when it is not installed."""
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


def enabled(cfg: Config) -> bool:
    """Whether the mirror is switched on and the package is importable."""
    return bool(cfg.logging.wandb.enabled) and _mirror() is not None


def start(cfg: Config, *, out_dir: Path, name: str) -> object | None:
    """Open a mirrored run, or return None when disabled or when the package is not installed."""
    if not enabled(cfg):
        return None
    wandb = _mirror()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        name=name,
        dir=str(out_dir),
        config=to_dict(cfg),
    )


def log_scalars(run: object | None, step: int, scalars: Mapping[str, float]) -> None:
    """Mirror one row of training scalars."""
    if run is None:
        return
    run.log({key: float(value) for key, value in scalars.items()}, step=int(step))


def log_table(run: object | None, name: str, df: pd.DataFrame, *, step: int) -> None:
    """Mirror an evaluation table at the training step it scored."""
    if run is None:
        return
    run.log({name: _mirror().Table(dataframe=df)}, step=int(step))


def finish(run: object | None) -> None:
    """Close the mirrored run."""
    if run is None:
        return
    run.finish()
