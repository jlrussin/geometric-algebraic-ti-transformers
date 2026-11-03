"""Configuration loading, deltas, sweep enumeration, run identifiers, fingerprints."""

from __future__ import annotations

import pytest

from geo_alg_ti import config


EXPERIMENTS = {
    "canonical_pretraining": 10,
    "symbolic_distance": 10,
    "finetuning_hyperparameters": 120,
    "skew": 140,
    "number_of_orderings": 60,
    "conflicting_vs_disjoint": 190,
    "content_effects": 210,
    "representation": 40,
    "components": 10,
    "chain_algorithms": 40,
    "architectures": 240,
    "cot_proportion": 110,
    "skew_no_cue": 140,
    "content_effects_no_cue": 210,
    "finetuning_reasoning": 40,
    "finetuning_reasoning_hyperparameters": 144,
}


def test_every_declared_experiment_is_counted(repo_root):
    """The run counts below cover exactly the experiments the repository declares."""
    declared = {path.parent.name for path in (repo_root / "experiments").glob("*/experiment.yaml")}
    assert set(EXPERIMENTS) == declared


def test_base_config_loads(repo_root):
    """Pool 300, 100 training and 16 reserved cues, p(CoT) 1, the main chain algorithm."""
    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    assert cfg.task.item_pool_size == 300
    assert (cfg.task.n_training_cues, cfg.task.n_reserved_cues) == (100, 16)
    assert (cfg.task.p_cot, cfg.task.chain_algorithm) == (1.0, "main")


def test_base_config_canonical_configuration(repo_root):
    """Pretraining 100,000 steps, batch 64, lr 0.001; finetuning 1,600 steps, lr 0.0001,
    replay 0.1."""
    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    assert (cfg.pretraining.steps, cfg.pretraining.batch_size, cfg.pretraining.learning_rate) == (100000, 64, 0.001)
    assert (cfg.finetuning.steps, cfg.finetuning.learning_rate, cfg.finetuning.replay_prop) == (1600, 0.0001, 0.1)
    assert (cfg.evaluation.max_reasoning_steps, cfg.evaluation.contexts_per_pair) == (8, 10)


def test_unknown_delta_path_raises(repo_root):
    """A dotted path absent from the base raises instead of creating a key."""
    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    with pytest.raises(ValueError, match="no_such_setting"):
        config.apply_delta(cfg, {"task.no_such_setting": 1})


def test_override_parses_yaml_values():
    """`task.p_cot=0.5` yields a float and `seeds=[1,2]` yields a list."""
    assert config.parse_override("task.p_cot=0.5") == ("task.p_cot", 0.5)
    assert config.parse_override("seeds=[1,2]") == ("seeds", [1, 2])


def test_smoke_is_a_delta_on_base(repo_root):
    """Every key in the smoke delta exists in the base."""
    cfg = config.load_config(repo_root / "configs" / "smoke.yaml")
    assert cfg.task.n_training_cues == 6


def test_smoke_config_is_small(repo_root):
    """Two layers, width 32, 60 pretraining steps, 20 finetuning steps, pool 48."""
    cfg = config.load_config(repo_root / "configs" / "smoke.yaml")
    assert (cfg.model.n_layers, cfg.model.d_model) == (2, 32)
    assert (cfg.pretraining.steps, cfg.finetuning.steps, cfg.task.item_pool_size) == (60, 20, 48)


def test_every_experiment_declaration_loads(repo_root):
    """Every experiment parses, `name` equals the directory, swept columns are the axes."""
    for name in EXPERIMENTS:
        exp = config.load_experiment(name, root=repo_root / "experiments")
        assert exp.name == name
        assert exp.swept_columns == tuple(a.name for a in exp.axes)


def test_every_experiment_has_a_readme(repo_root):
    """Each experiment directory holds a README."""
    for name in EXPERIMENTS:
        assert (repo_root / "experiments" / name / "README.md").exists()


def test_axes_enumerate_in_declaration_order(repo_root):
    """The first declared axis varies slowest and the seed axis varies fastest."""
    exp = config.load_experiment("finetuning_hyperparameters", root=repo_root / "experiments")
    runs = config.enumerate_runs(exp)
    assert runs[0].axis_values["seed"] == 0 and runs[1].axis_values["seed"] == 1
    assert runs[0].axis_values["replay_prop"] == runs[29].axis_values["replay_prop"]


def test_run_counts_match_the_declared_sweeps(repo_root):
    """Each experiment enumerates the number of runs its axes and seeds declare."""
    for name, count in EXPERIMENTS.items():
        assert config.n_runs(config.load_experiment(name, root=repo_root / "experiments")) == count


def test_excluded_combination_is_not_enumerated(repo_root):
    """The conflict experiment has no disjoint run with a single ordering."""
    exp = config.load_experiment("conflicting_vs_disjoint", root=repo_root / "experiments")
    assert not any(r.axis_values["layout"] == "disjoint" and r.axis_values["n_orderings"] == 1 for r in config.enumerate_runs(exp))


def test_run_id_depends_only_on_axis_values(repo_root):
    """The same axis values give the same identifier regardless of enumeration order."""
    exp = config.load_experiment("skew", root=repo_root / "experiments")
    assert config.run_id(exp, {"skew_alpha": 1.5, "seed": 3}) == "skew_alpha-1.5_seed-3"


def test_run_id_formats_floats_and_negatives():
    """1.5 renders 1.5, 1e-5 renders 0.00001, 1.0 renders 1, -1 renders m1, strings pass through."""
    assert config.format_axis_value(1.5) == "1.5"
    assert config.format_axis_value(1e-5) == "0.00001"
    assert config.format_axis_value(1.0) == "1"
    assert config.format_axis_value(-1) == "m1"
    assert config.format_axis_value("disjoint") == "disjoint"


def test_run_for_index_round_trips(repo_root):
    """The run at index i has index i and its axis values; an out-of-range index raises."""
    exp = config.load_experiment("skew", root=repo_root / "experiments")
    run = config.run_for_index(exp, 13)
    assert run.index == 13 and run.config.task.cue_distribution.skew_alpha == run.axis_values["skew_alpha"]
    with pytest.raises(IndexError):
        config.run_for_index(exp, 140)


def test_dependencies_resolve_to_one_run(repo_root):
    """Finetuning runs map to the same-seed pretraining run; representation to content_effects."""
    sd = config.load_experiment("symbolic_distance", root=repo_root / "experiments")
    assert config.run_for_index(sd, 4).depends_on_run_id == "seed-4"
    rep = config.load_experiment("representation", root=repo_root / "experiments")
    assert config.run_for_index(rep, 0).depends_on_run_id == "frequent_cue_prop-0.01_p_cot-0.9_seed-0"


def test_full_chain_proportion_run_resolves_to_the_canonical_configuration(repo_root):
    """The chain of thought proportion sweep's p(CoT) 1 run has the canonical configuration hash."""
    sweep = config.load_experiment("cot_proportion", root=repo_root / "experiments")
    canon = config.load_experiment("canonical_pretraining", root=repo_root / "experiments")
    full = config.run_for_index(sweep, 100)
    assert full.run_id == "p_cot-1_seed-0"
    assert config.config_hash(full.config) == config.config_hash(config.run_for_index(canon, 0).config)


def test_skew_under_the_uniform_kind_is_rejected(repo_root):
    """Declaring `zipf` at alpha 0 normalizes to `uniform`, so a later skew alone is an error."""
    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    zipf_at_zero = config.apply_delta(cfg, {"task.cue_distribution.kind": "zipf"})
    assert zipf_at_zero.task.cue_distribution.kind == "uniform"
    assert config.config_hash(zipf_at_zero) == config.config_hash(cfg)
    with pytest.raises(ValueError):
        config.apply_delta(zipf_at_zero, {"task.cue_distribution.skew_alpha": 1.5})
    with pytest.raises(ValueError):
        config.apply_delta(cfg, {"task.cue_distribution.frequent_cue_prop": 0.5})
    skewed = config.apply_delta(
        cfg, {"task.cue_distribution.kind": "zipf", "task.cue_distribution.skew_alpha": 1.5}
    )
    assert skewed.task.cue_distribution.kind == "zipf"


def test_pool_must_hold_the_finetuning_design(repo_root):
    """The pool floor is 41 under one ordering and 23 more per disjoint ordering."""
    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    config.apply_delta(cfg, {"task.item_pool_size": 41, "finetuning.n_orderings": 10})
    with pytest.raises(ValueError, match=r"item_pool_size 40.*needs 41"):
        config.apply_delta(cfg, {"task.item_pool_size": 40})
    config.apply_delta(cfg, {"task.item_pool_size": 64, "finetuning.layout": "disjoint", "finetuning.n_orderings": 2})
    with pytest.raises(ValueError, match=r"disjoint layout presents 2 item set.*needs 64.*n_orderings"):
        config.apply_delta(cfg, {"task.item_pool_size": 63, "finetuning.layout": "disjoint", "finetuning.n_orderings": 2})
    with pytest.raises(ValueError, match=r"needs 248"):
        config.apply_delta(cfg, {"task.item_pool_size": 247, "finetuning.layout": "disjoint", "finetuning.n_orderings": 10})


def test_coercion_refuses_a_truncating_or_misspelled_value(repo_root):
    """An integer field takes only a whole number and a flag only true or false."""
    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    assert config.apply_delta(cfg, {"model.n_layers": 2.0}).model.n_layers == 2
    with pytest.raises(ValueError, match="n_layers.*integer"):
        config.apply_delta(cfg, {"model.n_layers": 2.5})
    with pytest.raises(ValueError, match="batch_size.*number"):
        config.apply_delta(cfg, {"pretraining.batch_size": True})
    assert config.apply_delta(cfg, {"logging.wandb.enabled": "false"}).logging.wandb.enabled is False
    with pytest.raises(ValueError, match="enabled.*true or false"):
        config.apply_delta(cfg, {"logging.wandb.enabled": "off"})
    with pytest.raises(ValueError, match="seeds.*list"):
        config.apply_delta(cfg, {"seeds": 42})


def test_conflicting_orderings_each_need_a_training_cue(repo_root):
    """The conflicting layout draws one training cue per ordering; the disjoint one shares a cue."""
    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    config.apply_delta(cfg, {"task.n_training_cues": 10, "finetuning.n_orderings": 10})
    with pytest.raises(ValueError, match="n_orderings 11 exceeds task.n_training_cues 10"):
        config.apply_delta(cfg, {"task.n_training_cues": 10, "finetuning.n_orderings": 11})
    config.apply_delta(cfg, {"task.n_training_cues": 10, "finetuning.n_orderings": 11, "finetuning.layout": "disjoint"})


def test_a_dependent_run_inherits_the_sections_of_its_upstream_network(tmp_path):
    """A run that continues a network cannot declare that network's settings differently."""
    for p_cot, stage in ((1.0, "finetune"), (0.5, "finetune"), (1.0, "evaluate")):
        name = f"continued_{stage}_{str(p_cot).replace('.', '_')}"
        (tmp_path / name).mkdir()
        (tmp_path / name / "experiment.yaml").write_text(
            f"name: {name}\nquestion: Does a dependent run inherit its network's settings?\n"
            f"stages: [{stage}]\ndelta:\n  task.p_cot: {p_cot}\n  finetuning.steps: 20\naxes: []\n"
            "depends_on:\n  experiment: canonical_pretraining\n  match: [seed]\n",
            encoding="utf-8",
        )
        experiment = config.load_experiment(name, root=tmp_path)
        if p_cot == 1.0 and stage == "finetune":
            assert len(config.enumerate_runs(experiment)) == 10
        elif stage == "finetune":
            with pytest.raises(ValueError, match="sets task.p_cot differently from 'seed-0'"):
                config.enumerate_runs(experiment)
        else:
            with pytest.raises(ValueError, match="sets finetuning.steps differently"):
                config.enumerate_runs(experiment)


def test_fingerprint_records_provenance(repo_root):
    """The fingerprint carries the config hash, commit, versions, host, and the stage wall time."""
    cfg = config.load_base(repo_root / "configs" / "base.yaml")
    fp = config.fingerprint(cfg)
    assert {"config_hash", "git_sha", "package_version", "torch_version", "host"} <= set(fp)
    assert fp["wall_time_s"] >= 0.0
    # A stage passes its own clock; the process clock is the fallback.
    assert config.fingerprint(cfg, wall_time_s=1.25)["wall_time_s"] == 1.25


def test_a_recorded_configuration_loads_back_unchanged(repo_root, tmp_path):
    """A run's `config.yaml` is read back by `load_config` as the configuration it records."""
    cfg = config.load_config(repo_root / "configs" / "smoke.yaml", ("seeds=[42]",))
    path = config.dump_config(cfg, tmp_path / "config.yaml")
    assert config.load_config(path) == cfg
