"""The optional mirror is off by default and never required."""

from __future__ import annotations

from geo_alg_ti import wandb_mirror


def test_disabled_by_default(repo_root):
    """The base configuration has the mirror off."""
    from geo_alg_ti import config

    assert not wandb_mirror.enabled(config.load_base(repo_root / "configs" / "base.yaml"))


def test_every_function_is_a_no_op_when_disabled():
    """With no run, logging and finishing do nothing."""
    wandb_mirror.log_scalars(None, 0, {"loss": 1.0})
    wandb_mirror.finish(None)


def test_missing_dependency_does_not_raise(repo_root):
    """Enabling the mirror without the package installed degrades to a no-op."""
    import sys
    from unittest import mock

    import pandas as pd

    from geo_alg_ti import config

    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    switched_on = config.apply_delta(cfg, {"logging.wandb.enabled": True})
    with mock.patch.dict(sys.modules, {"wandb": None}):
        assert not wandb_mirror.enabled(switched_on)
        run = wandb_mirror.start(switched_on, out_dir=repo_root / "results", name="seed-0")
        assert run is None
        wandb_mirror.log_scalars(run, 0, {"loss": 1.0})
        wandb_mirror.log_table(run, "eval", pd.DataFrame({"value": [1.0]}), step=0)
        wandb_mirror.finish(run)
