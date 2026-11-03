"""Scoring truth table, companion columns, aggregation."""

from __future__ import annotations

from dataclasses import replace
from itertools import product, zip_longest

import numpy as np
import pandas as pd
import pytest
from conftest import vocabulary_for

from geo_alg_ti import battery, config, evaluate, tables, training
from geo_alg_ti.tables import PAIR_SCHEMA
from geo_alg_ti.trials import QUERYABLE_RANKS, ContextKind, Relation, Trial, adjacent_relations

ITEMS = tuple(range(9))
FILLER = tuple(range(9, 18))
# Five queries (familiar signs +1, -1, -1, +1, +1), one prediction each, and the sign a shuffled
# context shows for each. Under a shuffled context the predictions agree with the shown order on
# 3/5 trials and with the familiar order on 1/5 (one trial agrees with both, two with neither),
# so the computed fraction differs from both value and one minus value (2/5).
QUERIES = ((2, 5), (5, 2), (6, 1), (1, 3), (3, 7))
PREDICTIONS = (-1, 1, 1, -1, 1)
SHUFFLED_SHOWN = (-1, 1, -1, 1, 1)


def _condition(source, context, contexts_per_pair=1):
    return battery.TestCondition(
        name=f"{source}_{context}",
        source=battery.OrderingSource(source),
        context=ContextKind(context),
        contexts_per_pair=contexts_per_pair,
    )


def _trial(context, *, ranks=(2, 5), shown_sign=None, rng_seed=0):
    """Items 0..8 in rank order, the query at `ranks`, and a context of the given kind.

    A shuffled context shows `shown_sign` between the query items, so a pair may agree with the
    shown order, the familiar order, both, or neither.
    """
    kind = ContextKind(context)
    rng = np.random.default_rng(rng_seed)
    query = (ITEMS[ranks[0]], ITEMS[ranks[1]])
    familiar = 1 if ranks[0] < ranks[1] else -1
    if kind is ContextKind.UNINFORMATIVE:
        context_items, shown = FILLER, 0
    elif kind is ContextKind.CONGRUENT:
        context_items, shown = ITEMS, familiar
    elif kind is ContextKind.REVERSED:
        context_items, shown = tuple(reversed(ITEMS)), -familiar
    else:
        shown = familiar if shown_sign is None else shown_sign
        order = [int(item) for item in rng.choice(len(ITEMS), size=len(ITEMS), replace=False)]
        first, second = order.index(query[0]), order.index(query[1])
        if (first < second) != (shown == 1):
            order[first], order[second] = order[second], order[first]
        context_items = tuple(order)
    return Trial(
        cue=0,
        items=ITEMS,
        query_ranks=tuple(ranks),
        displayed=ITEMS,
        context=adjacent_relations(context_items, rng),
        context_items=context_items,
        context_kind=kind,
    )


def _trials_for(context):
    """The five standard trials under one context kind."""
    shown = SHUFFLED_SHOWN if context == "shuffled" else (None,) * len(QUERIES)
    return [
        _trial(context, ranks=ranks, shown_sign=sign, rng_seed=i)
        for i, (ranks, sign) in enumerate(zip(QUERIES, shown, strict=True))
    ]


def _score(trial, condition, predicted, *, k=0, chain_len=0, chain_valid=None):
    return evaluate.score_trial(trial, condition, predicted, k=k, chain_len=chain_len, chain_valid=chain_valid)


def _frame(scored, *, seeds=None, step=100, tag="finetuning"):
    """The per-trial frame `evaluate_battery` writes, kept to its accuracy rows."""
    seeds = list(seeds) if seeds is not None else [0] * len(scored)
    records = [
        record
        for one, seed in zip(scored, seeds, strict=True)
        for record in evaluate._records(one, seed=seed, step=step, tag=tag)
        if record["metric"] == "accuracy"
    ]
    return pd.DataFrame.from_records(records, columns=evaluate._TRIAL_COLUMNS)


@pytest.fixture(scope="module")
def smoke_setup(repo_root):
    """A smoke-scale network, its vocabulary, and three finetuned conditions at one context."""
    cfg = config.load_config(repo_root / "configs" / "smoke.yaml")
    law = training.build_law(cfg, 0)
    vocab = training.build_vocabulary(cfg)
    training.set_determinism(0)
    model = training.build_model(cfg, vocab)
    design = training.build_design(cfg, law, seed=0)
    wanted = ("iwl_test", "icl_test", "arbitrage_test")
    conditions = tuple(
        condition
        for condition in battery.load_preset("finetuned", contexts_per_pair=1)
        if condition.name in wanted
    )
    assert len(conditions) == 3
    built = battery.Battery.build(
        conditions,
        law,
        vocab,
        seed=0,
        design=design,
        budgets=cfg.evaluation.reasoning_steps(),
    )
    return model, built, vocab


@pytest.fixture(scope="module")
def evaluated(smoke_setup):
    """The smoke-scale network run over the three conditions at the default chunk size."""
    model, built, vocab = smoke_setup
    trials, aggregates = evaluate.evaluate_battery(model, built, vocab, step=7, tag="finetuning")
    return built, trials, aggregates


# The columns of the per-trial frame and the aggregates that hold labels, counts, and signs.
DISCRETE_COLUMNS = (
    "condition",
    "source",
    "context",
    "metric",
    "d",
    "K",
    "seed",
    "step",
    "tag",
    "rank_a",
    "rank_b",
    "item_a",
    "item_b",
    "predicted_sign",
    "reference_sign",
    "correct",
    "chain_len",
    "n",
)


def _assert_same_frame(frame: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Discrete columns agree exactly, float columns within tolerance, missing values in place."""
    assert list(frame.columns) == list(reference.columns) and len(frame) == len(reference)
    for name in frame.columns:
        if name in DISCRETE_COLUMNS:
            assert frame[name].equals(reference[name]), name
        else:
            assert np.allclose(frame[name], reference[name], atol=1e-5, rtol=0, equal_nan=True), name


@pytest.mark.parametrize(
    "source,context,reference,context_frac,prior_frac",
    [
        ("finetuned", "uninformative", "familiar", None, "value"),
        ("finetuned", "congruent", "shown", "value", "value"),
        ("finetuned", "reversed", "shown", "value", "one_minus_value"),
        ("finetuned", "shuffled", "shown", "value", "computed"),
        ("frequent_cue", "uninformative", "familiar", None, "value"),
        ("training_distribution", "uninformative", "familiar", None, "value"),
        ("novel_unseen_cue", "congruent", "shown", "value", None),
    ],
)
def test_scoring_truth_table(source, context, reference, context_frac, prior_frac):
    """Every (source, context) row of the truth table in the module docstring holds."""
    condition = _condition(source, context)
    trials = _trials_for(context)
    scored = [_score(trial, condition, p, k=3) for trial, p in zip(trials, PREDICTIONS, strict=True)]
    for one, trial, predicted in zip(scored, trials, PREDICTIONS, strict=True):
        expected_reference = trial.familiar_sign if reference == "familiar" else trial.shown_sign
        assert expected_reference in (-1, 1)
        assert one.reference_sign == expected_reference == trial.reference_sign
        assert one.correct == (predicted == expected_reference)
        assert one.predicted_sign == predicted
        assert (one.condition, one.source, one.context) == (condition.name, condition.source, condition.context)
        # A shuffled trial is named by the shown positions of its queried items.
        named_by = trial.shown_positions if context == "shuffled" else trial.query_ranks
        assert (one.d, one.K, one.rank_a, one.rank_b) == (trial.d, 3, *named_by)
        assert one.d == abs(one.rank_a - one.rank_b)
    value = np.mean([one.correct for one in scored])
    agreed_with_familiar = np.mean([p == t.familiar_sign for t, p in zip(trials, PREDICTIONS, strict=True)])
    assert value not in (0.0, 0.5, 1.0)

    aggregated = evaluate.aggregate(_frame(scored))
    assert len(aggregated) == len({trial.d for trial in trials})
    assert (aggregated["n"].sum(), aggregated["K"].unique().tolist()) == (len(scored), [3])
    assert set(aggregated["source"]) == {source} and set(aggregated["context"]) == {context}
    pooled = (aggregated["value"] * aggregated["n"]).sum() / aggregated["n"].sum()
    assert pooled == pytest.approx(value)

    if context_frac is None:
        assert all(one.context_consistent is None for one in scored)
        assert aggregated["context_consistent_frac"].isna().all()
    else:
        assert all(one.context_consistent == one.correct for one in scored)
        assert np.allclose(aggregated["context_consistent_frac"], aggregated["value"])

    if prior_frac is None:
        assert all(one.prior_consistent is None for one in scored)
        assert aggregated["prior_consistent_frac"].isna().all()
    elif prior_frac == "value":
        assert all(one.prior_consistent == one.correct for one in scored)
        assert np.allclose(aggregated["prior_consistent_frac"], aggregated["value"])
    elif prior_frac == "one_minus_value":
        assert all(one.prior_consistent == (not one.correct) for one in scored)
        assert np.allclose(aggregated["prior_consistent_frac"], 1 - aggregated["value"])
    else:
        assert agreed_with_familiar not in (value, 1 - value)
        for one, trial, predicted in zip(scored, trials, PREDICTIONS, strict=True):
            assert one.prior_consistent == (predicted == trial.familiar_sign)
        prior = (aggregated["prior_consistent_frac"] * aggregated["n"]).sum() / aggregated["n"].sum()
        assert prior == pytest.approx(agreed_with_familiar)


def test_uninformative_leaves_context_consistency_missing():
    """`context_consistent_frac` is missing when the context is uninformative."""
    for source in ("finetuned", "frequent_cue", "training_distribution"):
        condition = _condition(source, "uninformative")
        scored = [_score(t, condition, p) for t, p in zip(_trials_for("uninformative"), PREDICTIONS, strict=True)]
        for one in scored:
            assert one.context_consistent is None
            assert one.shown_sign in (None, 0)
            assert one.reference_sign == one.familiar_sign in (-1, 1)
            assert one.prior_consistent == one.correct
        aggregated = evaluate.aggregate(_frame(scored))
        assert aggregated["context_consistent_frac"].isna().all()
        assert aggregated["value"].notna().all()
        assert np.allclose(aggregated["prior_consistent_frac"], aggregated["value"])


def test_reversed_makes_prior_consistency_one_minus_value():
    """On reversed conditions `prior_consistent_frac` equals 1 - value."""
    for source in ("finetuned", "frequent_cue"):
        condition = _condition(source, "reversed")
        trials = _trials_for("reversed")
        scored = [_score(t, condition, p) for t, p in zip(trials, PREDICTIONS, strict=True)]
        for one, trial, predicted in zip(scored, trials, PREDICTIONS, strict=True):
            assert one.reference_sign == trial.shown_sign == -trial.familiar_sign
            assert one.correct == (predicted == trial.shown_sign)
            assert one.prior_consistent == (predicted == trial.familiar_sign) == (not one.correct)
            assert one.context_consistent == one.correct
        assert np.mean([one.correct for one in scored]) == pytest.approx(0.8)
        aggregated = evaluate.aggregate(_frame(scored))
        assert np.allclose(aggregated["prior_consistent_frac"], 1 - aggregated["value"])
        assert np.allclose(aggregated["context_consistent_frac"], aggregated["value"])
        pooled = (aggregated["prior_consistent_frac"] * aggregated["n"]).sum() / aggregated["n"].sum()
        assert pooled == pytest.approx(0.2)


def test_shuffled_computes_prior_consistency_rather_than_deriving_it():
    """On shuffled conditions a pair may agree with both orderings, so the fraction is computed."""
    condition = _condition("frequent_cue", "shuffled")
    trials = _trials_for("shuffled")
    assert {t.shown_sign == t.familiar_sign for t in trials} == {True, False}
    scored = [_score(t, condition, p) for t, p in zip(trials, PREDICTIONS, strict=True)]
    for one, trial, predicted in zip(scored, trials, PREDICTIONS, strict=True):
        assert one.reference_sign == trial.shown_sign
        assert one.correct == one.context_consistent == (predicted == trial.shown_sign)
        assert one.prior_consistent == (predicted == trial.familiar_sign)
    # One trial agrees with both orderings, one with neither, so neither value nor 1 - value
    # is the fraction consistent with the familiar ordering.
    both = [one for one in scored if one.correct and one.prior_consistent]
    neither = [one for one in scored if not one.correct and not one.prior_consistent]
    assert both and neither
    value = np.mean([one.correct for one in scored])
    prior = np.mean([one.prior_consistent for one in scored])
    assert value == pytest.approx(0.6) and prior == pytest.approx(0.2)
    aggregated = evaluate.aggregate(_frame(scored))
    pooled_prior = (aggregated["prior_consistent_frac"] * aggregated["n"]).sum() / aggregated["n"].sum()
    assert pooled_prior == pytest.approx(prior)
    assert pooled_prior != pytest.approx(value) and pooled_prior != pytest.approx(1 - value)


def test_novel_source_leaves_prior_consistency_missing():
    """A novel ordering has no familiar sign."""
    condition = _condition("novel_unseen_cue", "congruent")
    assert not condition.has_familiar_ordering
    trials = _trials_for("congruent")
    scored = [_score(t, condition, p) for t, p in zip(trials, PREDICTIONS, strict=True)]
    for one, trial, predicted in zip(scored, trials, PREDICTIONS, strict=True):
        assert one.familiar_sign is None
        assert one.prior_consistent is None
        assert one.reference_sign == one.shown_sign == trial.shown_sign
        assert one.correct == one.context_consistent == (predicted == trial.shown_sign)
    aggregated = evaluate.aggregate(_frame(scored))
    assert aggregated["prior_consistent_frac"].isna().all()
    assert np.allclose(aggregated["context_consistent_frac"], aggregated["value"])
    assert aggregated["value"].notna().all()


def test_value_is_always_agreement_with_the_scoring_reference():
    """On synthetic predictions `value` equals the fraction agreeing with `reference_sign`."""
    rng = np.random.default_rng(7)
    pairs = battery.ordered_rank_pairs(QUERYABLE_RANKS)
    rows = (
        ("finetuned", "uninformative"),
        ("finetuned", "congruent"),
        ("finetuned", "reversed"),
        ("frequent_cue", "shuffled"),
        ("frequent_cue", "uninformative"),
        ("training_distribution", "uninformative"),
        ("novel_unseen_cue", "congruent"),
    )
    for source, context in rows:
        condition = _condition(source, context)
        scored = []
        for i, ranks in enumerate(pairs):
            shown = int(rng.choice((-1, 1))) if context == "shuffled" else None
            trial = _trial(context, ranks=ranks, shown_sign=shown, rng_seed=i)
            predicted = int(rng.choice((-1, 1)))
            one = _score(trial, condition, predicted, k=int(rng.integers(0, 9)))
            assert one.reference_sign == trial.reference_sign
            assert one.correct == (predicted == trial.reference_sign)
            scored.append(one)
        frame = _frame(scored)
        aggregated = evaluate.aggregate(frame)
        for _, row in aggregated.iterrows():
            same = frame[(frame["d"] == row["d"]) & (frame["K"] == row["K"])]
            assert row["n"] == len(same)
            assert row["value"] == pytest.approx(same["correct"].mean())


def test_aggregate_reports_n_per_condition_distance_and_budget(evaluated):
    """Trial counts survive aggregation."""
    congruent = _condition("finetuned", "congruent")
    reversed_ = _condition("finetuned", "reversed")
    scored = []
    for k in (0, 1):
        for i, ranks in enumerate(((1, 2), (2, 1), (3, 4), (4, 5))):
            scored.append(_score(_trial("congruent", ranks=ranks, rng_seed=i), congruent, 1, k=k))
        for i, ranks in enumerate(((1, 3), (3, 1), (5, 7))):
            scored.append(_score(_trial("congruent", ranks=ranks, rng_seed=i), congruent, 1, k=k))
        for i, ranks in enumerate(((1, 4), (7, 4))):
            scored.append(_score(_trial("reversed", ranks=ranks, rng_seed=i), reversed_, 1, k=k))
    aggregated = evaluate.aggregate(_frame(scored))
    assert len(aggregated) == 6
    counts = aggregated.set_index(["condition", "d", "K"])["n"]
    for k in (0, 1):
        assert counts[(congruent.name, 1, k)] == 4
        assert counts[(congruent.name, 2, k)] == 3
        assert counts[(reversed_.name, 3, k)] == 2
    assert aggregated["n"].sum() == len(scored)

    built, _, aggregates = evaluated
    accuracy = aggregates[aggregates["metric"] == "accuracy"]
    for condition in built.conditions:
        for k in built.budgets:
            rows = accuracy[(accuracy["condition"] == condition.name) & (accuracy["K"] == k)]
            counts = rows.set_index("d")["n"].to_dict()
            assert counts == {d: 2 * (7 - d) * condition.contexts_per_pair for d in range(1, 7)}
    assert set(aggregates["seed"]) == {0}
    assert set(aggregates["step"]) == {7} and set(aggregates["tag"]) == {"finetuning"}


def test_aggregate_keeps_seed_as_a_row_axis():
    """Aggregation never averages over seed."""
    condition = _condition("finetuned", "congruent")
    trials = _trials_for("congruent")
    right = [_score(t, condition, t.reference_sign) for t in trials]
    wrong = [_score(t, condition, -t.reference_sign) for t in trials]
    frame = _frame(right + wrong, seeds=[0] * len(right) + [1] * len(wrong))
    aggregated = evaluate.aggregate(frame)
    assert "seed" in aggregated.columns
    assert set(aggregated["seed"]) == {0, 1}
    assert (aggregated[aggregated["seed"] == 0]["value"] == 1.0).all()
    assert (aggregated[aggregated["seed"] == 1]["value"] == 0.0).all()
    assert aggregated.groupby("seed")["n"].sum().to_dict() == {0: len(right), 1: len(wrong)}
    assert "seed" in evaluate.aggregate.__kwdefaults__["group_cols"]
    # Asked to group without the seed, aggregation either refuses or keeps the seed regardless.
    try:
        pooled = evaluate.aggregate(frame, group_cols=("condition", "metric", "d", "K", "step", "tag"))
    except ValueError:
        return
    assert "seed" in pooled.columns and set(pooled["seed"]) == {0, 1}
    assert not ((pooled["value"] > 0.0) & (pooled["value"] < 1.0)).any()


def test_every_condition_appears_at_every_budget(evaluated):
    """No (condition, K) row is missing."""
    built, trials, aggregates = evaluated
    names = {condition.name for condition in built.conditions}
    expected = set(product(names, built.budgets))
    assert set(trials["K"]) == set(built.budgets) == set(aggregates["K"])
    for metric in evaluate.METRICS:
        rows = aggregates[aggregates["metric"] == metric]
        assert set(zip(rows["condition"], rows["K"], strict=True)) == expected, metric
        per_trial = trials[trials["metric"] == metric]
        assert set(zip(per_trial["condition"], per_trial["K"], strict=True)) == expected, metric
        assert per_trial.groupby(["condition", "K"]).size().eq(42).all(), metric
    accuracy = aggregates[aggregates["metric"] == "accuracy"]
    assert accuracy["value"].notna().all() and accuracy["value"].between(0, 1).all()
    assert (accuracy.groupby(["condition", "K"])["n"].sum() == 42).all()


def test_pair_table_uses_the_queried_orderings_own_rank_indexing(evaluated):
    """`rank_a`, `rank_b` are zero-based in the queried ordering, shown positions if shuffled."""
    assert evaluate.METRICS == ("accuracy", "chain_validity", "chain_length")
    # A reversed context changes where the query items sit in the shown order, not their ranks
    # in the queried ordering; the same holds for an uninformative context. A shuffled trial is
    # named by where its items are shown, so its pair always lies at its symbolic distance.
    scored = []
    for context in ("reversed", "shuffled", "uninformative", "congruent"):
        trial = _trial(context, ranks=(2, 5), shown_sign=-1)
        assert (trial.items.index(trial.query[0]), trial.items.index(trial.query[1])) == (2, 5)
        if context in ("reversed", "shuffled"):
            assert trial.context_items.index(trial.query[0]) > trial.context_items.index(trial.query[1])
        one = _score(trial, _condition("finetuned", context), 1)
        expected = trial.shown_positions if context == "shuffled" else (2, 5)
        assert (one.rank_a, one.rank_b) == expected and abs(one.rank_a - one.rank_b) == one.d
        scored.append(one)
    pairs = evaluate.aggregate_by_pair(_frame(scored))
    shuffled_pair = tuple(_trial("shuffled", ranks=(2, 5), shown_sign=-1).shown_positions)
    assert set(zip(pairs["rank_a"], pairs["rank_b"], strict=True)) == {(2, 5), shuffled_pair}
    assert len(pairs) == 4

    built, trials, _ = evaluated
    pairs = evaluate.aggregate_by_pair(trials)
    assert set(PAIR_SCHEMA) - {"experiment"} <= set(pairs.columns)
    assert set(pairs["metric"]) == {"accuracy"}
    all_pairs = set(battery.ordered_rank_pairs(QUERYABLE_RANKS))
    for condition in built.conditions:
        rows = pairs[pairs["condition"] == condition.name]
        assert set(zip(rows["rank_a"], rows["rank_b"], strict=True)) == all_pairs
        assert ((rows["rank_a"] - rows["rank_b"]).abs() == rows["d"]).all()
        assert set(rows["K"]) == set(built.budgets)
        assert (rows["n"] == condition.contexts_per_pair).all()
    keys = ["condition", "K", "rank_a", "rank_b"]
    accuracy = trials[trials["metric"] == "accuracy"]
    expected = accuracy.groupby(keys)["value"].mean().rename("expected").reset_index()
    merged = pairs.merge(expected, on=keys, how="left")
    assert merged["expected"].notna().all()
    assert np.allclose(merged["value"], merged["expected"])


def test_chunk_size_leaves_the_frames_unchanged(smoke_setup, evaluated):
    """Rolling trials out one, three, or a thousand at a time scores every trial the same."""
    model, built, vocab = smoke_setup
    _, trials, aggregates = evaluated
    assert evaluate.evaluate_battery.__kwdefaults__["chunk_size"] not in (1, 3, 1000)
    for chunk_size in (1, 3, 1000):
        other_trials, other_aggregates = evaluate.evaluate_battery(
            model, built, vocab, step=7, tag="finetuning", chunk_size=chunk_size
        )
        _assert_same_frame(other_trials, trials)
        _assert_same_frame(other_aggregates, aggregates)


def test_conditions_mixed_within_a_chunk_score_the_same(smoke_setup):
    """A chunk holding trials of several conditions scores each as a chunk of its own would."""
    model, built, vocab = smoke_setup
    by_condition = {condition.name: [] for condition in built.conditions}
    for name, trial in built.trials:
        by_condition[name].append((name, trial))
    interleaved = [
        entry
        for entries in zip_longest(*by_condition.values())
        for entry in entries
        if entry is not None
    ]
    assert len(interleaved) == len(built.trials)
    assert len({name for name, _ in interleaved[:4]}) == 3
    mixed = replace(built, trials=tuple(interleaved))
    small = evaluate.evaluate_battery(model, mixed, vocab, step=7, tag="finetuning", chunk_size=4)
    whole = evaluate.evaluate_battery(model, mixed, vocab, step=7, tag="finetuning", chunk_size=1000)
    _assert_same_frame(small[0], whole[0])
    _assert_same_frame(small[1], whole[1])


def test_trial_rows_follow_the_battery_order(evaluated):
    """Rows come trial by trial in battery order, budget by budget, metric by metric."""
    built, trials, _ = evaluated
    per_trial = len(built.budgets) * len(evaluate.METRICS)
    assert len(trials) == per_trial * len(built.trials)
    heads = trials.iloc[::per_trial]
    labels = list(zip(heads["condition"], heads["rank_a"], heads["rank_b"], strict=True))
    assert labels == [(name, *trial.query_ranks) for name, trial in built.trials]
    items = list(zip(heads["item_a"], heads["item_b"], strict=True))
    assert items == [tuple(trial.query) for _, trial in built.trials]
    budgets = [k for k in built.budgets for _ in evaluate.METRICS]
    assert trials["K"].tolist() == budgets * len(built.trials)
    metrics = list(evaluate.METRICS) * len(built.budgets)
    assert trials["metric"].tolist() == metrics * len(built.trials)


def _rolled_out_trial():
    """A pretraining-format trial with its main chain and answer, for the chain helpers."""
    from geo_alg_ti.chains import generate_chain
    from geo_alg_ti.orderings import uniform_cue_weights
    from geo_alg_ti.trials import sample_pretraining_trial

    law = training.build_law(config.load_base(), 0)
    trial = sample_pretraining_trial(law, np.random.default_rng(0), weights=uniform_cue_weights(law.n_training_cues))
    chain = generate_chain(trial)
    return trial, tuple(chain.emissions) + (chain.answer,)


def test_chain_validity_is_missing_when_nothing_is_read_and_false_when_malformed():
    """Nothing read gives no validity; a malformed or false emission in the read part is invalid."""
    from geo_alg_ti.trials import Relation

    trial, good = _rolled_out_trial()
    assert evaluate._truncated_validity(good, trial, 0) is None
    assert evaluate._truncated_validity((None,) + good[1:], trial, 0) is None
    assert evaluate._truncated_validity((None,) + good[1:], trial, 1) is False
    assert evaluate._truncated_validity(good[:1] + (None,), trial, 1) is evaluate._truncated_validity(good, trial, 1)
    assert evaluate._truncated_validity(good[:1] + (None,), trial, 2) is False
    false_sign = Relation(good[0].a, good[0].b, -good[0].sign)
    assert evaluate._truncated_validity((false_sign,) + good[1:], trial, 1) is False


def test_chain_length_counts_emissions_before_the_queried_pair_is_stated():
    """`chain_len` is the index of the first emission naming the query, else the emission count."""
    from geo_alg_ti.trials import Relation

    trial, _ = _rolled_out_trial()
    a, b = trial.query
    other = Relation(a, a, 1)
    stated = Relation(b, a, -1)
    assert evaluate.chain_length((other, other, stated, other), trial) == 2
    assert evaluate.chain_length((stated,), trial) == 0
    assert evaluate.chain_length((other, None, stated), trial) == 2
    assert evaluate.chain_length((other, None, other), trial) == 3
    assert evaluate.chain_length((), trial) == 0


def test_evaluated_frame_leaves_chain_validity_missing_where_nothing_is_read(evaluated):
    """Validity is missing at zero steps, under an uninformative context, and for a pair the
    context states directly (an empty canonical chain); elsewhere it is 0 or 1."""
    from geo_alg_ti.chains import ChainAlgorithm, canonical_chain_length

    built, trials, _ = evaluated
    validity = trials[trials["metric"] == "chain_validity"].copy()
    assert validity[validity["K"] == 0]["value"].isna().all()
    uninformative = validity["context"] == "uninformative"
    assert validity[uninformative]["value"].isna().all()
    by_name = {condition.name: condition for condition in built.conditions}
    canonical = [
        0 if by_name[name].context == "uninformative" else canonical_chain_length(trial, ChainAlgorithm.MAIN)
        for name, trial in built.trials
    ]
    per_trial = len(built.budgets) * len(evaluate.METRICS)
    validity["canonical"] = [canonical[i // per_trial] for i in validity.index]
    empty_chain = validity["canonical"] == 0
    assert validity[~uninformative & empty_chain]["value"].isna().all()
    later = validity[~uninformative & ~empty_chain & (validity["K"] > 0)]
    assert len(later) and later["value"].notna().all()
    assert set(later["value"]) <= {0.0, 1.0}


def test_chain_validity_is_truncated_at_the_trained_algorithms_canonical_length():
    """A network trained on backtracking chains has its second leg read; the main one does not."""
    from geo_alg_ti.chains import ChainAlgorithm, canonical_chain_length, generate_chain
    from geo_alg_ti.orderings import uniform_cue_weights
    from geo_alg_ti.trials import sample_pretraining_trial

    law = training.build_law(config.load_base(), 0)
    weights = uniform_cue_weights(law.n_training_cues)
    for index in range(40):
        trial = sample_pretraining_trial(law, np.random.default_rng(index), weights=weights)
        main = canonical_chain_length(trial, ChainAlgorithm.MAIN)
        backtrack = canonical_chain_length(trial, ChainAlgorithm.BACKTRACK)
        if backtrack > main:
            break
    else:
        raise AssertionError("no trial's backtracking chain was longer than its main chain")
    emissions = tuple(generate_chain(trial, ChainAlgorithm.BACKTRACK).emissions)
    corrupted = emissions[:main] + tuple(Relation(e.a, e.b, -e.sign) for e in emissions[main:])
    assert evaluate._truncated_validity(corrupted, trial, min(8, main)) is True
    assert evaluate._truncated_validity(corrupted, trial, min(8, backtrack)) is False


def _shuffled_battery(law, contexts_per_pair, seed=0):
    """The content preset's shuffled condition at smoke scale, built as the battery builds it."""
    conditions = tuple(
        condition
        for condition in battery.load_preset("content", contexts_per_pair=contexts_per_pair)
        if condition.context is ContextKind.SHUFFLED
    )
    return battery.Battery.build(conditions, law, vocabulary_for(law), seed=seed, design=None, budgets=(0,))


def test_shuffled_rows_name_the_shown_positions_at_the_shown_distance(smoke_law):
    """Every shuffled record is named by interior shown positions whose difference is `d`."""
    built = _shuffled_battery(smoke_law, 2)
    (condition,) = built.conditions
    rng = np.random.default_rng(3)
    scored = [_score(trial, condition, int(rng.choice((-1, 1)))) for _, trial in built.trials]
    for one, (_, trial) in zip(scored, built.trials, strict=True):
        assert (one.rank_a, one.rank_b) == trial.shown_positions
        assert {one.rank_a, one.rank_b} <= set(QUERYABLE_RANKS) and abs(one.rank_a - one.rank_b) == one.d
        assert one.reference_sign == trial.shown_sign
    pairs = evaluate.aggregate_by_pair(_frame(scored))
    assert set(zip(pairs["rank_a"], pairs["rank_b"], strict=True)) == set(battery.ordered_rank_pairs(QUERYABLE_RANKS))
    assert (pairs["n"] == 2).all() and ((pairs["rank_a"] - pairs["rank_b"]).abs() == pairs["d"]).all()


@pytest.mark.parametrize("contexts_per_pair", [2, 3])
def test_split_shuffled_recovers_the_agreeing_and_conflicting_accuracies(smoke_law, contexts_per_pair):
    """Split from the aggregates, each half's accuracy equals the mean over its own trials."""
    built = _shuffled_battery(smoke_law, contexts_per_pair, seed=5)
    (condition,) = built.conditions
    rng = np.random.default_rng(contexts_per_pair)
    scored, agree = [], []
    for _, trial in built.trials:
        # A network that follows the frequent cue's ordering more often than the shown one.
        follows_familiar = rng.random() < 0.7
        predicted = trial.familiar_sign if follows_familiar else -trial.familiar_sign
        scored.append(_score(trial, condition, predicted))
        agree.append(trial.shown_sign == trial.familiar_sign)
    frame = _frame(scored, seeds=[0] * len(scored))
    frame["half"] = np.where(agree, "agree", "conflict")
    by_hand = frame.groupby(["half", "d"])["correct"].mean()
    split = tables.split_shuffled(evaluate.aggregate(frame.drop(columns="half")))
    assert len(split) == 2 * 6
    for half, name, context in (
        ("agree", "frequent_cue_shuffled_agree", "shuffled_agree"),
        ("conflict", "frequent_cue_shuffled_conflict", "shuffled_conflict"),
    ):
        rows = split[split["condition"] == name]
        assert set(rows["context"]) == {context} and len(rows) == 6
        for _, row in rows.iterrows():
            assert row["value"] == pytest.approx(by_hand[(half, row["d"])], abs=1e-12)
            assert row["n"] == (frame["d"] == row["d"]).sum() // 2
            assert row["context_consistent_frac"] == row["value"]
            expected_prior = row["value"] if half == "agree" else 1 - row["value"]
            assert row["prior_consistent_frac"] == pytest.approx(expected_prior)
    # The halves of a row average back to the row's accuracy.
    whole = evaluate.aggregate(frame.drop(columns="half")).set_index("d")["value"]
    halves = split.groupby("d")["value"].mean()
    assert np.allclose(halves.sort_index(), whole.sort_index())
    # The per-pair rows split too when the contexts per pair are even.
    pairs = evaluate.aggregate_by_pair(frame.drop(columns="half"))
    if contexts_per_pair % 2 == 0:
        split_pairs = tables.split_shuffled(pairs)
        by_pair = frame.groupby(["half", "rank_a", "rank_b"])["correct"].mean()
        for _, row in split_pairs.iterrows():
            half = row["condition"].rsplit("_", 1)[1]
            assert row["value"] == pytest.approx(by_pair[(half, row["rank_a"], row["rank_b"])], abs=1e-12)
    else:
        with pytest.raises(ValueError, match="odd trial count"):
            tables.split_shuffled(pairs)

