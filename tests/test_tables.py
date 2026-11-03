"""Tidy schemas, the estimands, the distance pool, and the manifest."""

from __future__ import annotations

import hashlib
from decimal import Decimal

import pandas as pd
import pytest
import yaml

from geo_alg_ti import config, tables


def _frame():
    rows = []
    for seed in (0, 1):
        for d in (1, 2, 3):
            for k in range(3):
                rows.append({"condition": "icl_test", "metric": "accuracy", "d": d, "K": k, "seed": seed, "value": 0.4 + 0.1 * d + 0.1 * k + 0.05 * seed, "n": 10 + d})
    return pd.DataFrame(rows)


def test_schema_validation_rejects_missing_or_unexpected_columns():
    """A frame missing a schema column or carrying an undeclared one fails validation."""
    with pytest.raises(ValueError):
        tables.validate_table(_frame(), schema=tables.EVAL_SCHEMA)


def test_seed_mean_se_raises_when_seed_is_grouped():
    """Seed is the only averaged axis; grouping by it is an error."""
    with pytest.raises(ValueError):
        tables.seed_mean_se(_frame(), group_cols=["d", "K", "seed"])


def test_seed_mean_se_matches_hand_computed_values():
    """Mean and standard error over two seeds equal the hand computation."""
    out = tables.seed_mean_se(_frame(), group_cols=["condition", "metric", "d", "K"])
    row = out[(out.d == 1) & (out.K == 0)].iloc[0]
    assert abs(row.value_mean - 0.525) < 1e-9 and row.n_seeds == 2


def test_pool_over_distance_is_trial_weighted():
    """The pool weights each distance by its trial count."""
    out = tables.pool_over_distance(_frame(), group_cols=["condition", "metric", "K", "seed"], d_min=1)
    row = out[(out.K == 0) & (out.seed == 0)].iloc[0]
    assert abs(row.value - (0.5 * 11 + 0.6 * 12 + 0.7 * 13) / (11 + 12 + 13)) < 1e-9
    assert abs(row.value - (0.5 + 0.6 + 0.7) / 3) > 1e-3
    assert row.n == 11 + 12 + 13


def test_pool_over_distance_excludes_distance_one_by_default():
    """`d_min` defaults to 2."""
    out = tables.pool_over_distance(_frame(), group_cols=["condition", "metric", "K", "seed"])
    assert "d" not in out.columns
    row = out[(out.K == 0) & (out.seed == 0)].iloc[0]
    assert abs(row.value - (0.6 * 12 + 0.7 * 13) / 25) < 1e-9


def test_pool_over_distance_raises_when_distance_is_grouped():
    """Grouping by distance while pooling over it is an error."""
    with pytest.raises(ValueError):
        tables.pool_over_distance(_frame(), group_cols=["d", "K", "seed"])


def test_accuracy_at_budget_pins_the_budget_then_pools_within_each_seed():
    """One row per condition and seed: the trial-weighted pool of that seed's accuracy at k."""
    out = tables.accuracy_at_budget(_frame(), k=2, d_min=2)
    assert list(out.columns) == ["condition", "seed", "value", "n"] and len(out) == 2
    row = out[out.seed == 1].iloc[0]
    assert abs(row.value - (0.85 * 12 + 0.95 * 13) / 25) < 1e-9 and row.n == 25
    with pytest.raises(ValueError):
        tables.accuracy_at_budget(_frame(), k=5)


def test_mean_over_k_averages_within_seed_and_distance_first():
    """The mean over K = 1..5 is taken within (seed, d) before any seed averaging."""
    rows = [
        {"condition": "icl_test", "metric": "accuracy", "d": d, "K": k, "seed": seed, "value": 100 * seed + k * k, "n": 20}
        for seed in (0, 1)
        for d in (2, 3)
        for k in range(7)
    ]
    out = tables.mean_over_k(pd.DataFrame(rows), group_cols=["condition", "metric", "d", "seed"])
    assert "K" not in out.columns and len(out) == 4
    assert sorted(out.seed.unique()) == [0, 1]
    for seed in (0, 1):
        for d in (2, 3):
            value = out[(out.seed == seed) & (out.d == d)].value.iloc[0]
            assert abs(value - (100 * seed + (1 + 4 + 9 + 16 + 25) / 5)) < 1e-9


def test_best_over_k_takes_the_maximum_before_pooling():
    """The maximum over K is taken per (seed, d) before pooling or seed averaging."""
    out = tables.best_over_k(_frame(), group_cols=["condition", "metric", "d", "seed"])
    assert abs(out[(out.d == 1) & (out.seed == 0)].value.iloc[0] - 0.7) < 1e-9


def _sweep_frame(by_group):
    rows = [
        {"condition": "icl_test", "metric": "accuracy", "d": d, "K": k, "seed": seed, "value": value, "n": 20}
        for (seed, d), values in by_group.items()
        for k, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def test_running_best_over_k_is_a_cumulative_maximum_within_each_seed():
    """Each budget's value is the best over the budgets up to it, per (seed, d), in K order."""
    frame = _sweep_frame(
        {
            (0, 2): [0.2, 0.5, 0.3, 0.9],
            (1, 2): [0.9, 0.1, 0.1, 0.1],
            (0, 3): [0.4, 0.4, 0.8, 0.6],
        }
    )
    shuffled = frame.sample(frac=1.0, random_state=0)
    out = tables.running_best_over_k(shuffled, group_cols=["condition", "metric", "d", "seed"])
    assert "K" in out.columns and len(out) == len(frame) and out.n.tolist() == [20] * len(frame)
    by_group = {
        (int(seed), int(d)): rows.sort_values("K").value.tolist()
        for (seed, d), rows in out.groupby(["seed", "d"])
    }
    assert by_group == {(0, 2): [0.2, 0.5, 0.5, 0.9], (1, 2): [0.9, 0.9, 0.9, 0.9], (0, 3): [0.4, 0.4, 0.8, 0.8]}
    with pytest.raises(ValueError):
        tables.running_best_over_k(frame, group_cols=["condition", "metric", "d"])
    with pytest.raises(ValueError):
        tables.running_best_over_k(frame, group_cols=["condition", "metric", "d", "seed", "K"])


def test_first_k_to_criterion_uses_ninety_percent_of_the_seeds_maximum():
    """The criterion is 0.9 times the maximum within (seed, d)."""
    frame = _sweep_frame(
        {
            (0, 2): [0.2, 0.5, 0.95, 1.0],
            (0, 3): [0.4, 0.4, 0.4, 0.8],
            (1, 2): [0.5, 0.5, 0.5, 0.5],
            (1, 3): [0.95, 1.0, 0.2, 0.2],
        }
    )
    out = tables.first_k_to_criterion(frame, group_cols=["condition", "metric", "d", "seed"])
    reached = {(int(row.seed), int(row.d)): int(row.k_to_criterion) for row in out.itertuples()}
    assert reached == {(0, 2): 2, (0, 3): 3, (1, 2): 0, (1, 3): 0}


def test_first_k_to_criterion_drops_seeds_without_a_defined_accuracy():
    """A seed whose accuracy is undefined at every budget is excluded from the mean."""
    missing = float("nan")
    frame = _sweep_frame(
        {
            (0, 2): [0.2, 0.6, 0.95],
            (1, 2): [0.1, 0.2, 0.5],
            (0, 3): [0.95, 0.95, 0.95],
            (1, 3): [0.9, 0.95, 1.0],
            (0, 4): [missing, missing, missing],
            (1, 4): [0.7, 0.8, 0.9],
        }
    )
    group_cols = ["condition", "metric", "d", "seed"]
    reached = tables.first_k_to_criterion(frame, group_cols=group_cols)
    assert {(int(row.seed), int(row.d)) for row in reached.itertuples()} == {(0, 2), (1, 2), (0, 3), (1, 3), (1, 4)}
    averaged = tables.seed_mean_se(reached, group_cols=["condition", "metric", "d"], value_col=tables.K_TO_CRITERION)
    counts = {int(row.d): int(row.n_seeds) for row in averaged.itertuples()}
    assert counts == {2: 2, 3: 2, 4: 1}


def test_pin_k_rejects_a_budget_absent_from_the_frame():
    """Pinning an absent K raises rather than returning an empty frame."""
    with pytest.raises(ValueError):
        tables.pin_k(_frame(), 8)


def test_reasoning_step_estimands_refuse_to_reduce_across_seeds():
    """Only `seed_mean_se` reduces over seeds, so an estimand grouped without the seed raises."""
    frame = _frame()
    for estimand in (tables.mean_over_k, tables.best_over_k, tables.first_k_to_criterion, tables.running_best_over_k):
        with pytest.raises(ValueError):
            estimand(frame, group_cols=["condition", "metric", "d"])
    # Grouped by seed the same call goes through, one maximum per seed rather than one overall.
    best = tables.best_over_k(frame, group_cols=["condition", "metric", "d", "seed"])
    assert sorted(best.seed.unique()) == [0, 1] and len(best) == 6


def test_pin_k_refuses_rows_whose_seeds_are_already_averaged():
    """A pinned budget is taken within a seed, so seed-averaged rows are refused."""
    averaged = tables.seed_mean_se(_frame(), group_cols=["condition", "metric", "d", "K"])
    with pytest.raises(ValueError):
        tables.pin_k(averaged, 1)


def test_seed_mean_se_refuses_distances_that_were_never_pooled():
    """Averaging distances unweighted would misstate the pool, so an ungrouped distance raises."""
    frame = pd.DataFrame(
        [
            {"condition": "iwl_test", "metric": "accuracy", "d": 2, "K": 0, "seed": 0, "value": 1.0, "n": 100},
            {"condition": "iwl_test", "metric": "accuracy", "d": 3, "K": 0, "seed": 0, "value": 0.0, "n": 1},
        ]
    )
    with pytest.raises(ValueError):
        tables.seed_mean_se(frame, group_cols=["condition", "metric", "K"])
    pooled = tables.pool_over_distance(frame, group_cols=["condition", "metric", "K", "seed"])
    out = tables.seed_mean_se(pooled, group_cols=["condition", "metric", "K"])
    assert abs(out.value_mean.iloc[0] - 100 / 101) < 1e-9


def test_first_k_to_criterion_reports_steps_apart_from_accuracy_and_carries_counts():
    """The step count lands in its own labeled column and the trial counts feed the pool."""
    frame = _sweep_frame({(0, 2): [0.2, 0.5, 0.95, 1.0], (0, 3): [0.4, 0.4, 0.4, 0.8]})
    out = tables.first_k_to_criterion(frame, group_cols=["condition", "metric", "d", "seed"])
    assert tables.K_TO_CRITERION in out.columns and "value" not in out.columns
    assert tables.COLUMN_LABELS[tables.K_TO_CRITERION] != tables.COLUMN_LABELS["value"]
    assert out.n.tolist() == [20, 20]
    pooled = tables.pool_over_distance(
        out, group_cols=["condition", "metric", "seed"], value_col=tables.K_TO_CRITERION
    )
    assert abs(pooled[tables.K_TO_CRITERION].iloc[0] - 2.5) < 1e-9 and int(pooled.n.iloc[0]) == 40


def test_pool_over_distance_leaves_a_missing_value_out_of_its_mean():
    """A distance with no criterion value carries no weight: the pool averages the distances
    that have one, a group with none has no pooled value, and the trial count keeps every row."""
    frame = _sweep_frame({(0, 2): [0.2, 0.5, 0.95, 1.0], (0, 3): [0.4, 0.4, 0.4, 0.8], (0, 4): [0.5, 0.6, 0.55, 0.5]})
    reached = tables.first_k_to_criterion(frame, group_cols=["condition", "metric", "d", "seed"], unreached_below_chance=True)
    assert reached[tables.K_TO_CRITERION].isna().tolist() == [False, False, True]
    group_cols = ["condition", "metric", "seed"]
    pooled = tables.pool_over_distance(reached, group_cols=group_cols, value_col=tables.K_TO_CRITERION)
    assert abs(pooled[tables.K_TO_CRITERION].iloc[0] - 2.5) < 1e-9 and int(pooled.n.iloc[0]) == 60
    unreached = reached[reached.d == 4]
    alone = tables.pool_over_distance(unreached, group_cols=group_cols, value_col=tables.K_TO_CRITERION)
    assert pd.isna(alone[tables.K_TO_CRITERION].iloc[0]) and int(alone.n.iloc[0]) == 20


def test_round_half_up_rounds_a_decimal_half_away_from_zero():
    """A half rounds up, also where the binary value lies just below it, and down when negative."""
    assert tables.round_half_up(0.125, 2) == Decimal("0.13")
    assert tables.round_half_up(1.005, 2) == Decimal("1.01")
    assert tables.round_half_up(-0.125, 2) == Decimal("-0.13")
    assert tables.round_half_up(0.8776666, 2) == Decimal("0.88")
    assert format(tables.round_half_up(0.000436, 5), "f") == "0.00044"


def test_wilson_lower_bound_follows_the_closed_form():
    """The Wilson lower bound at z = 1.96, a perfect score included; no trials, no bound."""
    assert tables.wilson_lower_bound(0.5, 100) == pytest.approx(0.40383, abs=1e-5)
    assert tables.wilson_lower_bound(1.0, 20) == pytest.approx(20 / (20 + 1.96**2))
    assert tables.wilson_lower_bound(0.6, 20, z=0.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        tables.wilson_lower_bound(0.5, 0)


def test_first_k_to_criterion_leaves_a_best_accuracy_not_above_chance_missing():
    """With the floor, a group whose best accuracy cannot clear chance keeps its count, no value."""
    frame = _sweep_frame({(0, 2): [0.5, 0.6, 0.55], (1, 2): [0.5, 0.75, 0.7], (0, 3): [0.9, 0.95, 1.0]})
    many = _sweep_frame({(1, 3): [0.5, 0.6, 0.55]}).assign(n=200)
    frame = pd.concat([frame, many], ignore_index=True)
    group_cols = ["condition", "metric", "d", "seed"]
    plain = tables.first_k_to_criterion(frame, group_cols=group_cols)
    assert plain[tables.K_TO_CRITERION].notna().all() and pd.api.types.is_integer_dtype(plain[tables.K_TO_CRITERION])
    floored = tables.first_k_to_criterion(frame, group_cols=group_cols, unreached_below_chance=True)
    values = {(int(row.seed), int(row.d)): row.k_to_criterion for row in floored.itertuples()}
    # A best of 0.6 over 20 trials does not clear chance; over 200 trials it does, as 0.75 over 20.
    assert pd.isna(values[(0, 2)])
    assert values[(1, 2)] == 1 and values[(0, 3)] == 0 and values[(1, 3)] == 1
    assert floored.n.tolist() == [20, 20, 20, 200]
    # With z = 0 the bound is the best accuracy itself, which clears chance everywhere here.
    loose = tables.first_k_to_criterion(frame, group_cols=group_cols, unreached_below_chance=True, unreached_z=0.0)
    assert loose[tables.K_TO_CRITERION].notna().all()
    with pytest.raises(ValueError, match="trial count"):
        tables.first_k_to_criterion(frame.drop(columns="n"), group_cols=group_cols, unreached_below_chance=True)


def _shuffled_rows(values, priors, n=20):
    """Accuracy rows of the shuffled condition, one per distance, with the schema's columns."""
    rows = [
        {
            "experiment": "content_effects",
            "tag": "pretraining",
            "condition": "frequent_cue_shuffled",
            "source": "frequent_cue",
            "context": "shuffled",
            "metric": "accuracy",
            "d": d,
            "K": 0,
            "seed": 0,
            "step": 100,
            "value": value,
            "n": n,
            "context_consistent_frac": value,
            "prior_consistent_frac": prior,
            "p_cot": 0.9,
        }
        for d, (value, prior) in enumerate(zip(values, priors, strict=True), start=1)
    ]
    return pd.DataFrame(rows).astype({name: kind for name, kind in tables.EVAL_SCHEMA.items()})


def test_split_shuffled_gives_each_half_its_accuracy_and_keeps_the_rest():
    """Agree = value + prior - 1/2, conflict = value - prior + 1/2; the other columns are kept."""
    frame = _shuffled_rows([0.6, 0.5, 0.9], [0.7, 0.9, 0.5])
    split = tables.split_shuffled(frame)
    tables.validate_table(split, schema=tables.EVAL_SCHEMA, swept_columns=("p_cot",))
    agree = split[split["condition"] == "frequent_cue_shuffled_agree"].set_index("d")
    conflict = split[split["condition"] == "frequent_cue_shuffled_conflict"].set_index("d")
    assert agree["value"].tolist() == pytest.approx([0.8, 0.9, 0.9])
    assert conflict["value"].tolist() == pytest.approx([0.4, 0.1, 0.9])
    assert set(agree["context"]) == {"shuffled_agree"} and set(conflict["context"]) == {"shuffled_conflict"}
    assert (agree["context_consistent_frac"] == agree["value"]).all()
    assert (agree["prior_consistent_frac"] == agree["value"]).all()
    assert conflict["prior_consistent_frac"].tolist() == pytest.approx((1 - conflict["value"]).tolist())
    assert (split["n"] == 10).all()
    kept = ["experiment", "tag", "source", "metric", "K", "seed", "step", "p_cot"]
    for half in (agree, conflict):
        assert half.reset_index()[kept].equals(frame[kept])
    assert frame["condition"].eq("frequent_cue_shuffled").all()


def test_split_shuffled_refuses_rows_it_cannot_split():
    """Another condition or metric, an odd trial count, or a half outside [0, 1] is refused."""
    frame = _shuffled_rows([0.6, 0.5], [0.7, 0.6])
    with pytest.raises(ValueError, match="only accuracy rows"):
        tables.split_shuffled(frame.assign(condition="frequent_cue_reversed"))
    with pytest.raises(ValueError, match="only accuracy rows"):
        tables.split_shuffled(frame.assign(metric="chain_validity"))
    with pytest.raises(ValueError, match="odd trial count"):
        tables.split_shuffled(frame.assign(n=21))
    with pytest.raises(ValueError, match="leaves"):
        tables.split_shuffled(_shuffled_rows([0.9], [0.1]))
    with pytest.raises(ValueError, match="no rows"):
        tables.split_shuffled(frame.iloc[:0])
    # A half that leaves [0, 1] by rounding alone is kept, at the bound.
    edge = tables.split_shuffled(_shuffled_rows([0.25], [0.25 - 1e-13]))
    assert edge["value"].min() == 0.0


def _eval_rows(seed):
    return pd.DataFrame(
        [
            {
                "tag": "pretrained",
                "condition": "icl_test",
                "source": "novel_unseen_cue",
                "context": "congruent",
                "metric": "accuracy",
                "d": d,
                "K": 8,
                "step": 100000,
                "value": 0.5 + 0.1 * d + 0.01 * seed,
                "n": 420,
                "context_consistent_frac": 0.5 + 0.1 * d,
                "prior_consistent_frac": float("nan"),
            }
            for d in (1, 2, 3)
        ]
    )


def test_round_trip_preserves_dtypes(tmp_path):
    """Writing and reading a parquet table preserves the schema types."""
    df = _eval_rows(0).assign(experiment="canonical_pretraining", seed=0)
    df = df.loc[:, list(tables.EVAL_SCHEMA)].astype(dict(tables.EVAL_SCHEMA))
    path = tables.write_table(df, tmp_path / "eval.parquet", schema=tables.EVAL_SCHEMA)
    loaded = tables.load_table(path, schema=tables.EVAL_SCHEMA)
    assert {name: str(dtype) for name, dtype in loaded.dtypes.items()} == dict(tables.EVAL_SCHEMA)
    pd.testing.assert_frame_equal(loaded, df)


def test_written_columns_take_the_type_the_schema_declares(tmp_path):
    """Whole numbers in `value` are stored and read back as the declared floating-point type."""
    df = _eval_rows(0).assign(experiment="canonical_pretraining", seed=0)
    df["value"] = [0, 1, 1]
    df = df.loc[:, list(tables.EVAL_SCHEMA)]
    assert str(df["value"].dtype) == "int64"
    path = tables.write_table(df, tmp_path / "eval.parquet", schema=tables.EVAL_SCHEMA)
    loaded = tables.load_table(path, schema=tables.EVAL_SCHEMA)
    assert {name: str(dtype) for name, dtype in loaded.dtypes.items()} == dict(tables.EVAL_SCHEMA)


def test_loading_against_declared_axes_rejects_any_other_column(tmp_path):
    """Naming the swept axes makes loading exactly as strict as validation."""
    df = _eval_rows(0).assign(experiment="canonical_pretraining", seed=0, undeclared=1)
    path = tmp_path / "eval.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        tables.load_table(path, schema=tables.EVAL_SCHEMA, swept_columns=())
    assert "undeclared" in tables.load_table(path, schema=tables.EVAL_SCHEMA).columns


def test_run_tables_round_trip_in_step_order(tmp_path):
    """A per-step table reads back as its steps in order; the training curve is one file."""
    later = _eval_rows(0).assign(step=100000)
    earlier = _eval_rows(0).assign(step=5000)
    tables.write_run_table(tmp_path, "eval", 100000, later)
    tables.write_run_table(tmp_path, "eval", 5000, earlier)
    tables.write_run_table(tmp_path, "eval", 7000, earlier.iloc[:0])
    assert sorted(path.name for path in (tmp_path / "eval").iterdir()) == ["step_00005000.parquet", "step_00100000.parquet"]
    read = tables.read_run_table(tmp_path, "eval")
    assert read["step"].tolist() == [5000] * 3 + [100000] * 3
    assert tables.read_run_table(tmp_path, "eval_pairs") is None
    tables.write_run_table(tmp_path, tables.TRIALS_TABLE, 100000, later)
    assert len(tables.read_run_table(tmp_path, tables.TRIALS_TABLE)) == 3
    curve = pd.DataFrame({"seed": [0, 0], "stage": ["pretraining"] * 2, "step": [100, 200], "loss": [1.0, 0.5]})
    assert tables.read_run_table(tmp_path, "train") is None
    curve.to_parquet(tmp_path / "train_metrics.parquet", index=False)
    pd.testing.assert_frame_equal(tables.read_run_table(tmp_path, "train"), curve)
    with pytest.raises(ValueError, match="written once"):
        tables.write_run_table(tmp_path, "train", 100, curve)
    with pytest.raises(ValueError, match="no run table is named"):
        tables.read_run_table(tmp_path, "no_such_table")


def test_export_rejects_an_undeclared_column(tmp_path):
    """A run's rows carrying a column outside the schema and the swept axes fail the export."""
    run_dir = config.run_directory(tmp_path, "canonical_pretraining", "seed-0")
    tables.write_run_table(run_dir, "eval", 100000, _eval_rows(0).assign(undeclared=1))
    with pytest.raises(ValueError, match="undeclared"):
        tables.export_experiment("canonical_pretraining", results_root=tmp_path)


def test_manifest_records_row_counts_and_hashes(tmp_path):
    """The manifest lists each table's rows and hash."""
    for seed in (0, 1):
        run_dir = config.run_directory(tmp_path, "canonical_pretraining", f"seed-{seed}")
        tables.write_run_table(run_dir, "eval", 100000, _eval_rows(seed))
    written = tables.export_experiment("canonical_pretraining", results_root=tmp_path)
    manifest_path = config.tables_directory(tmp_path, "canonical_pretraining") / "MANIFEST.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entries = {entry["table"]: entry for entry in manifest["tables"]}
    assert set(entries) == set(written) == {"eval"}
    assert entries["eval"]["rows"] == 6
    assert entries["eval"]["runs"] == ["seed-0", "seed-1"]
    assert entries["eval"]["sha256"] == hashlib.sha256(written["eval"].read_bytes()).hexdigest()
    assert manifest["package_version"]


def test_export_finds_the_runs_of_the_configuration_it_is_given(repo_root, tmp_path):
    """Runs made with a seed outside the canonical sweep are exported under that configuration."""
    base = config.load_config(repo_root / "configs" / "smoke.yaml", ("seeds=[42]",))
    experiment = config.load_experiment("canonical_pretraining")
    (run,) = config.enumerate_runs(experiment, base)
    run_dir = config.run_directory(tmp_path, experiment.name, run.run_id)
    curve = pd.DataFrame(
        {
            "seed": [42, 42],
            "stage": ["pretraining", "pretraining"],
            "step": [1, 2],
            "loss": [2.0, 1.0],
            "token_cross_entropy": [1.5, 0.5],
            "scalar_squared_error": [0.5, 0.5],
            "answer_accuracy": [0.5, 1.0],
            "chain_token_accuracy": [0.5, 1.0],
        }
    )
    tables.write_train_table(run_dir, curve)
    with pytest.raises(FileNotFoundError):
        tables.export_experiment(experiment.name, results_root=tmp_path)
    written = tables.export_experiment(experiment.name, results_root=tmp_path, base=base)
    assert set(written) == {"train"}
    assert tables.load_table(written["train"], schema=tables.TRAIN_SCHEMA, swept_columns=())["seed"].tolist() == [42, 42]


def test_check_export_requires_every_table_to_hold_exactly_the_declared_runs(tmp_path):
    """A run missing from any table, or a run no declaration names, makes an export unusable."""
    (tmp_path / "x.parquet").write_bytes(b"rows")
    entry = {"path": "x.parquet", "rows": 1, "sha256": tables._sha256(tmp_path / "x.parquet"), "axes": []}
    manifest = tables.write_manifest(
        tmp_path / "MANIFEST.yaml",
        [{**entry, "table": "eval", "runs": ["seed-0", "seed-1"]}, {**entry, "table": "train", "runs": ["seed-0", "seed-1", "seed-2"]}],
    )
    with pytest.raises(ValueError, match="eval table covers 2 of the 3 declared runs"):
        tables.check_export(manifest, ["seed-0", "seed-1", "seed-2"])
    with pytest.raises(ValueError, match="train table holds 1 runs the experiment does not declare"):
        tables.check_export(manifest, ["seed-0", "seed-1"])
    exact = tables.write_manifest(tmp_path / "exact.yaml", [{**entry, "table": "eval", "runs": ["seed-1", "seed-0"]}])
    tables.check_export(exact, ["seed-0", "seed-1"])
    (tmp_path / "x.parquet").write_bytes(b"torn")
    with pytest.raises(ValueError, match="eval table is missing or differs from the digest"):
        tables.check_export(exact, ["seed-0", "seed-1"])
    (tmp_path / "corrupt.yaml").write_text("tables:\n- table: eval\n  runs:\n  - seed-0\n  - se: [", encoding="utf-8")
    with pytest.raises(ValueError, match="not readable"):
        tables.check_export(tmp_path / "corrupt.yaml", ["seed-0"])
    with pytest.raises(ValueError, match="covers none"):
        tables.check_export(tables.write_manifest(tmp_path / "empty.yaml", []), ["seed-0"])
    (tmp_path / "blank.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="covers none"):
        tables.check_export(tmp_path / "blank.yaml", ["seed-0"])
    (tmp_path / "odd.yaml").write_text("tables:\n- table: eval\n  runs:\n- 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="eval table covers 0 of the 1"):
        tables.check_export(tmp_path / "odd.yaml", ["seed-0"])


def test_export_names_the_runs_it_could_not_find(repo_root, tmp_path, capsys):
    """An export of a sweep still in progress says which declared runs are missing."""
    base = config.load_config(repo_root / "configs" / "smoke.yaml", ("seeds=[42, 43]",))
    experiment = config.load_experiment("canonical_pretraining")
    present, absent = config.enumerate_runs(experiment, base)
    run_dir = config.run_directory(tmp_path, experiment.name, present.run_id)
    curve = pd.DataFrame(
        {
            "seed": [42],
            "stage": ["pretraining"],
            "step": [1],
            "loss": [2.0],
            "token_cross_entropy": [1.5],
            "scalar_squared_error": [0.5],
            "answer_accuracy": [0.5],
            "chain_token_accuracy": [0.5],
        }
    )
    tables.write_train_table(run_dir, curve)
    tables.export_experiment(experiment.name, results_root=tmp_path, base=base)
    assert f"1 of the declared runs are not written yet: {absent.run_id}" in capsys.readouterr().err


def _iwl_pairs(ranks: range = range(1, 8)) -> pd.DataFrame:
    """Every ordered pair of `ranks` on the IWL test, at one budget, with one trial each and its
    pair type."""
    rows = [{"condition": "iwl_test", "seed": 0, "rank_a": a, "rank_b": b, "d": abs(a - b), "K": 0, "n": 1, "value": 1.0}
            for a in ranks for b in ranks if a != b]
    pairs = pd.DataFrame(rows)
    return pairs.assign(**{tables.PAIR_TYPE_COLUMN: tables._label_pair_types(pairs)})


def test_pair_type_rows_split_seven_ranks_into_six_nine_and_six_pairs():
    """Seven ranks hold 6 adjacent pairs, 9 non-adjacent pairs with the first or last rank, and 6
    non-adjacent pairs among the middle five, each asked in both orders."""
    pairs = _iwl_pairs()
    counts = {measure: len(tables.pair_type_rows(pairs, measure)) // 2 for measure in tables.PAIR_TYPES[1:]}
    assert counts == {"trained": 6, "end_item": 9, "interior": 6}
    interior = tables.pair_type_rows(pairs, "interior")
    assert interior[["rank_a", "rank_b"]].isin([1, 7]).to_numpy().sum() == 0


def test_pair_type_labels_take_the_end_items_from_the_main_items_not_the_rows():
    """Rows without the first or last main item keep their types: a pair of the middle five is
    interior however few pairs the rows hold, and a pair outside the main items raises."""
    middle = _iwl_pairs(range(2, 7))
    assert set(middle[tables.PAIR_TYPE_COLUMN]) == {"trained", "interior"}
    other = pd.DataFrame({"condition": ["icl_test", "baseline_test"], "rank_a": [2, 2], "rank_b": [4, 4]})
    assert tables._label_pair_types(other).tolist() == ["icl_test", ""]
    with pytest.raises(ValueError, match="outside the main items"):
        tables._label_pair_types(_iwl_pairs(range(0, 3)))
    with pytest.raises(ValueError, match="no 'pair_type' column"):
        tables.pair_type_rows(middle.drop(columns=[tables.PAIR_TYPE_COLUMN]), "interior")
    with pytest.raises(ValueError, match="no IWL test end-item pairs"):
        tables.pair_type_rows(middle, "end_item")


def test_pair_type_by_seed_weights_pairs_by_trials_and_keeps_distance_one_for_trained_pairs():
    """Within a distance the pairs pool by trials, and trained pairs keep their only distance."""
    pairs = _iwl_pairs()
    pairs.loc[(pairs["rank_a"] == 2) & (pairs["rank_b"] == 4), ["n", "value"]] = [3, 0.0]
    interior = tables.pair_type_by_seed(pairs, "interior")
    total = pairs[(pairs[["rank_a", "rank_b"]].min(axis=1) > 1) & (pairs[["rank_a", "rank_b"]].max(axis=1) < 7) & (pairs["d"] >= 2)]
    assert interior["value"].iloc[0] == pytest.approx((total["value"] * total["n"]).sum() / total["n"].sum())
    trained = tables.pair_type_by_seed(pairs, "trained")
    assert trained["value"].iloc[0] == pytest.approx(1.0) and trained["n"].iloc[0] == 12


def test_pair_type_by_seed_carries_the_trials_of_the_best_budget():
    """A pair missing at the best budget only leaves that budget's trials behind the distance's
    value: its pool over distances weighs the distance by them, and its count reports them."""
    low, high = _iwl_pairs().assign(value=0.5), _iwl_pairs().assign(K=1)
    gone = (high["rank_a"] == 2) & (high["rank_b"] == 4)
    high.loc[gone, "value"] = float("nan")
    high.loc[(high["d"] == 3) & (high[tables.PAIR_TYPE_COLUMN] == "interior"), "value"] = 0.0
    interior = tables.pair_type_by_seed(pd.concat([low, high], ignore_index=True), "interior")
    # Distance two is best at one step, where five of its six pairs hold a value; distance three
    # is best at zero steps.
    trials = {2: 5, 3: 4, 4: 2}
    values = {2: 1.0, 3: 0.5, 4: 1.0}
    assert interior["n"].iloc[0] == sum(trials.values())
    expected = sum(values[d] * trials[d] for d in trials) / sum(trials.values())
    assert interior["value"].iloc[0] == pytest.approx(expected)


def test_pair_type_by_seed_carries_the_trials_of_the_smallest_tied_budget():
    """When two budgets tie for a distance's best value with different trials, the smaller budget's
    trials are carried, whatever order the rows come in."""
    low, high = _iwl_pairs(), _iwl_pairs().assign(K=1)
    low.loc[(low["rank_a"] == 2) & (low["rank_b"] == 4), "value"] = float("nan")
    interior = tables.pair_type_by_seed(pd.concat([high, low], ignore_index=True), "interior")
    # Distance two holds a value on five of its six pairs at zero steps; distances three and four
    # hold four and two pairs at both budgets.
    assert interior["n"].iloc[0] == 5 + 4 + 2
    assert interior["value"].iloc[0] == pytest.approx(1.0)

