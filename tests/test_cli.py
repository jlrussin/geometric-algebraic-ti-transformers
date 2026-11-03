"""The executable: verbs, overrides, the runs verb, and the login-node guard."""

from __future__ import annotations

import pytest

from geo_alg_ti import cli, config


def test_help_lists_every_subcommand(capsys):
    """`--help` names every subcommand."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert all(v in out for v in cli.VERBS)


def test_config_and_overrides_are_applied_in_order(repo_root):
    """Repeated `--set` applies left to right over `--config`."""
    parser = cli.build_parser()
    args = parser.parse_args(["runs", "--experiment", "skew", "--set", "task.p_cot=0.5", "--set", "task.p_cot=0.7"])
    assert args.set == ["task.p_cot=0.5", "task.p_cot=0.7"]


def test_override_of_unknown_path_exits_nonzero(repo_root):
    """An unknown dotted path exits non-zero with a message naming the path."""
    assert cli.main(["runs", "--experiment", "skew", "--count", "--set", "task.no_such=1"]) != 0


def test_runs_count_prints_an_integer(capsys, repo_root):
    """`runs --experiment canonical_pretraining --count` prints 10."""
    assert cli.main(["runs", "--experiment", "canonical_pretraining", "--count"]) == 0
    assert capsys.readouterr().out.strip() == "10"


def test_runs_index_prints_the_resolved_configuration(capsys, repo_root):
    """The print form emits YAML carrying that run's skew alpha, not the base value, and seed."""
    import yaml

    index = 23
    run = config.run_for_index(config.load_experiment("skew"), index)
    alpha = run.config.task.cue_distribution.skew_alpha
    assert alpha != config.load_base().task.cue_distribution.skew_alpha
    assert cli.main(["runs", "--experiment", "skew", "--index", str(index), "--print"]) == 0
    doc = yaml.safe_load(capsys.readouterr().out)
    assert doc["task"]["cue_distribution"]["skew_alpha"] == alpha
    assert doc["seeds"] == list(run.config.seeds)


def test_runs_stages_prints_the_declared_stages(capsys, repo_root):
    """`runs --stages` prints the experiment's stages in declaration order, space separated."""
    assert cli.main(["runs", "--experiment", "canonical_pretraining", "--stages"]) == 0
    assert capsys.readouterr().out.split() == ["pretrain"]
    assert cli.main(["runs", "--experiment", "representation", "--stages"]) == 0
    assert capsys.readouterr().out.split() == ["represent", "evaluate"]
    assert cli.main(["runs", "--experiment", "components", "--stages"]) == 0
    assert capsys.readouterr().out.split() == ["ablate"]


def test_runs_covered_checks_a_manifest_against_the_declaration(capsys, tmp_path):
    """The check passes for a manifest naming exactly the declared runs and refuses any other."""
    from geo_alg_ti import tables

    experiment = config.load_experiment("canonical_pretraining")
    declared = [run.run_id for run in config.enumerate_runs(experiment)]
    (tmp_path / "eval.parquet").write_bytes(b"rows")
    entry = {"path": "eval.parquet", "rows": 1, "sha256": tables._sha256(tmp_path / "eval.parquet"), "axes": []}
    exact = tables.write_manifest(tmp_path / "exact.yaml", [{**entry, "table": "eval", "runs": declared}])
    assert cli.main(["runs", "--experiment", "canonical_pretraining", "--covered", str(exact)]) == 0
    short = tables.write_manifest(tmp_path / "short.yaml", [{**entry, "table": "eval", "runs": declared[:-1]}])
    assert cli.main(["runs", "--experiment", "canonical_pretraining", "--covered", str(short)]) == 1
    assert f"runs: the eval table covers {len(declared) - 1} of the {len(declared)} declared runs" in capsys.readouterr().err


def test_runs_requires_an_experiment():
    """`runs` without `--experiment` exits non-zero."""
    with pytest.raises(SystemExit):
        cli.main(["runs", "--count"])


def test_login_node_guard_blocks_heavy_commands(monkeypatch):
    """The guard exits when the host name starts with `login` and no SLURM job id is set."""
    monkeypatch.setattr("socket.gethostname", lambda: "login9")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv(cli.LOGIN_NODE_BYPASS_ENV, raising=False)
    with pytest.raises(SystemExit):
        cli.assert_not_login_node()


def test_login_node_guard_honors_the_documented_bypass(monkeypatch):
    """The bypass variable disables the guard."""
    monkeypatch.setattr("socket.gethostname", lambda: "login9")
    monkeypatch.setenv(cli.LOGIN_NODE_BYPASS_ENV, "1")
    cli.assert_not_login_node()


def test_evaluate_refuses_a_run_that_stopped_early(capsys, tmp_path, repo_root):
    """A run that stopped short of its last step is refused, and the error names the step."""
    from geo_alg_ti import config, training

    smoke = repo_root / "configs" / "smoke.yaml"
    _, run = cli._declared_run(cli.SMOKE_PRETRAINING, config.load_config(smoke))
    run_dir = config.run_directory(tmp_path, run.experiment, run.run_id)
    stopped = config.checkpoint_path(run_dir, training.checkpoint_name(10, 60))
    stopped.parent.mkdir(parents=True)
    stopped.touch()
    argv = ["evaluate", "--config", str(smoke), "--experiment", cli.SMOKE_PRETRAINING, "--out", str(tmp_path)]
    assert cli.main(argv) != 0
    message = capsys.readouterr().err
    assert "stopped at step 10" in message and stopped.name in message
    assert not any(run_dir.glob("eval/*"))


def test_evaluate_scores_the_checkpoint_the_declaration_names(tmp_path, repo_root):
    """An experiment that trains scores its own checkpoint; one that does not, its dependency's."""
    from geo_alg_ti import config, training

    base = config.load_config(repo_root / "configs" / "smoke.yaml")
    experiment, run = cli._declared_run(cli.SMOKE_FINETUNING, base)
    assert experiment.stages == ("finetune",)
    own = config.checkpoint_path(config.run_directory(tmp_path, run.experiment, run.run_id), training.CHECKPOINT_NAME)
    with pytest.raises(FileNotFoundError, match="never trained"):
        cli._checkpoint_to_score(experiment, run, tmp_path)
    own.parent.mkdir(parents=True)
    own.touch()
    assert cli._checkpoint_to_score(experiment, run, tmp_path) == own

    experiment = config.load_experiment("representation")
    run = config.run_for_index(experiment, 0, base)
    assert not {"pretrain", "finetune"} & set(experiment.stages)
    upstream = config.checkpoint_path(
        config.run_directory(tmp_path, run.depends_on_experiment, run.depends_on_run_id), training.CHECKPOINT_NAME
    )
    with pytest.raises(FileNotFoundError, match=run.depends_on_experiment):
        cli._checkpoint_to_score(experiment, run, tmp_path)
    upstream.parent.mkdir(parents=True)
    upstream.touch()
    assert cli._checkpoint_to_score(experiment, run, tmp_path) == upstream


def test_unknown_subcommand_exits_nonzero():
    """An unrecognized verb exits non-zero."""
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])
