"""The laptop end-to-end path: pretrain, finetune, evaluate, export, a figure, in seconds on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from geo_alg_ti import battery, cli, config, streams, tables, training


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, repo_root):
    """Run the smoke command once for the module."""
    out = tmp_path_factory.mktemp("smoke")
    assert cli.main(["smoke", "--config", str(repo_root / "configs" / "smoke.yaml"), "--out", str(out), "--device", "cpu"]) == 0
    return out


def _table_directory(out: Path, experiment: str) -> Path:
    """Where one experiment of the smoke path writes its tidy tables."""
    return config.tables_directory(out / cli.SMOKE_DIRECTORY, experiment)


def _run_directory(out: Path, experiment: str) -> Path:
    """The one run directory each experiment of the smoke path writes."""
    smoke = config.load_config(config.repo_root() / "configs" / "smoke.yaml")
    (run,) = config.enumerate_runs(config.load_experiment(experiment), smoke)
    run_dir = config.run_directory(out / cli.SMOKE_DIRECTORY, experiment, run.run_id)
    assert [path.name for path in run_dir.parent.iterdir()] == [run.run_id]
    return run_dir


def test_smoke_runs_end_to_end(smoke_run):
    """The smoke path completes on CPU in under a minute."""
    for experiment in (cli.SMOKE_PRETRAINING, cli.SMOKE_FINETUNING):
        assert _run_directory(smoke_run, experiment).is_dir()
        assert (_table_directory(smoke_run, experiment) / "eval.parquet").is_file()


def test_smoke_writes_a_checkpoint(smoke_run):
    """A checkpoint exists and embeds the resolved configuration and fingerprint."""
    run_dir = _run_directory(smoke_run, cli.SMOKE_PRETRAINING)
    checkpoint = training.load_checkpoint(config.checkpoint_path(run_dir, training.CHECKPOINT_NAME))
    assert isinstance(checkpoint.config, config.Config)
    assert checkpoint.step == checkpoint.config.pretraining.steps
    fingerprint = json.loads((run_dir / "fingerprint.json").read_text(encoding="utf-8"))
    assert fingerprint["config_hash"] == config.config_hash(checkpoint.config)


def test_smoke_training_loss_decreases(smoke_run):
    """The final logged pretraining loss is below the first."""
    path = _table_directory(smoke_run, cli.SMOKE_PRETRAINING) / "train.parquet"
    losses = tables.load_table(path, schema=tables.TRAIN_SCHEMA, swept_columns=()).sort_values("step")["loss"]
    assert len(losses) > 1
    assert losses.iloc[-1] < losses.iloc[0]


def test_smoke_eval_table_covers_every_declared_condition(smoke_run, repo_root):
    """The evaluation table has a row for every declared condition at every reasoning-step value."""
    smoke = config.load_config(repo_root / "configs" / "smoke.yaml")
    (run,) = config.enumerate_runs(config.load_experiment(cli.SMOKE_FINETUNING), smoke)
    cfg = run.config
    path = _table_directory(smoke_run, cli.SMOKE_FINETUNING) / "eval.parquet"
    table = tables.load_table(path, schema=tables.EVAL_SCHEMA, swept_columns=())
    final = table[table["step"] == table["step"].max()]
    preset = cfg.evaluation.battery or streams.stage_section(cfg, streams.Stage.FINETUNING).battery
    declared = {condition.name for condition in battery.load_preset(preset, contexts_per_pair=cfg.evaluation.contexts_per_pair)}
    assert set(final["condition"]) == declared
    for name in declared:
        scored = final[final["condition"] == name]
        assert set(scored["K"]) == set(cfg.evaluation.reasoning_steps())


def test_smoke_eval_table_validates_against_the_schema(smoke_run):
    """The table matches the tidy schema, including both consistency-fraction columns."""
    path = _table_directory(smoke_run, cli.SMOKE_FINETUNING) / "eval.parquet"
    table = tables.load_table(path, schema=tables.EVAL_SCHEMA, swept_columns=())
    tables.validate_table(table, schema=tables.EVAL_SCHEMA)
    assert table["context_consistent_frac"].notna().any()
    assert table["prior_consistent_frac"].notna().any()
    # An uninformative context relates neither queried item to the other, so it shows no sign.
    uninformative = table[table["context"] == "uninformative"]
    assert uninformative["context_consistent_frac"].isna().all()


def test_smoke_export_writes_a_manifest(smoke_run):
    """Export writes the tables and a manifest with row counts and hashes."""
    directory = _table_directory(smoke_run, cli.SMOKE_FINETUNING)
    manifest = yaml.safe_load((directory / "MANIFEST.yaml").read_text(encoding="utf-8"))
    assert manifest["package_version"]
    assert {entry["table"] for entry in manifest["tables"]} == {"eval", "eval_pairs", "train"}
    for entry in manifest["tables"]:
        assert entry["rows"] > 0
        assert len(entry["sha256"]) == 64
        assert (directory / entry["path"]).is_file()


def test_smoke_figure_writes_a_pdf(smoke_run):
    """The figure step writes the figure "Symbolic distance effects (SDE)" as a non-empty PDF
    and a PNG preview."""
    figures = smoke_run / cli.SMOKE_DIRECTORY / "figures"
    assert (figures / f"{cli.SMOKE_FIGURE}.pdf").stat().st_size > 0
    assert (figures / f"{cli.SMOKE_FIGURE}.png").is_file()


def test_smoke_is_deterministic(smoke_run, tmp_path, repo_root):
    """Two smoke runs at the same seed produce byte-identical tables."""
    assert cli.main(["smoke", "--config", str(repo_root / "configs" / "smoke.yaml"), "--out", str(tmp_path), "--device", "cpu"]) == 0
    for experiment in (cli.SMOKE_PRETRAINING, cli.SMOKE_FINETUNING):
        first, second = _table_directory(smoke_run, experiment), _table_directory(tmp_path, experiment)
        written = sorted(path.name for path in first.glob("*.parquet"))
        assert written
        for name in written:
            assert (first / name).read_bytes() == (second / name).read_bytes()
