"""Rank probes, the rank-direction algebra, and the edits at every site, with numeric fixtures."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from fixtures.rank_direction_reference import REFERENCE, RIDGE_ALPHA, planted_scenario
from geo_alg_ti import battery, cli, config, evaluate, model, represent, tables, training
from geo_alg_ti.chains import ChainAlgorithm
from geo_alg_ti.orderings import FREQUENT_CUE
from geo_alg_ti.sequences import EMISSION_LEN, item_positions, prompt_length
from geo_alg_ti.streams import Stage
from geo_alg_ti.trials import ITEMS_PER_ORDERING, QUERYABLE_RANKS, ContextKind


@pytest.fixture(scope="module")
def planted():
    """The planted scenario: a random unit direction times the z-scored rank plus noise."""
    return planted_scenario()


@pytest.fixture(scope="module")
def rank_direction(planted):
    """The rank direction fit on the planted occurrences, as if they were the embedding table."""
    return represent.fit_rank_direction(planted["vectors"], planted["ranks"], site=represent.EMBEDDING_SITE)


@pytest.fixture(scope="module")
def rows(planted):
    """One state per item, item 0 first: the per-item rows in double precision."""
    return torch.as_tensor(planted["item_rows"], dtype=torch.float64)


@pytest.fixture(scope="module")
def calibration(rank_direction, planted):
    """The line from rank to coordinate on the per-item rows."""
    return represent.write_calibration(rank_direction, planted["item_rows"], planted["item_ranks"])


def _coordinates(rank_direction, states: torch.Tensor) -> np.ndarray:
    """The read-back coordinate of every row of states."""
    return (states.numpy() - rank_direction.mean) @ rank_direction.read_back


def _apply(edit, states: torch.Tensor) -> torch.Tensor:
    """A one-site edit applied to one state per item, item 0 first."""
    (one,) = edit
    return one.apply(states, torch.arange(states.shape[0]))


def _states(planted, ranks=None) -> represent.ItemStates:
    """The planted occurrences as item states, with the true ranks unless others are given."""
    ranks = planted["ranks"] if ranks is None else ranks
    return represent.ItemStates(vectors=planted["vectors"], ranks=ranks, items=planted["items"], site=0)


def _metric(rows, kind: str, metric: str) -> float:
    """One probe value out of the rows `probe_metrics` returns."""
    (value,) = [row["value"] for row in rows if row["kind"] == kind and row["metric"] == metric]
    return float(value)


def test_probe_recovers_a_planted_rank_direction(planted, rank_direction):
    """Synthetic embeddings with a planted rank direction are decoded above chance."""
    assert abs(float(rank_direction.direction @ planted["direction"])) > 0.95
    assert rank_direction.n_components == represent.N_BASIS_COMPONENTS and rank_direction.n_items == 20
    assert np.allclose(rank_direction.basis.T @ rank_direction.basis, np.eye(rank_direction.n_components), atol=1e-10)
    rows = represent.probe_metrics(_states(planted), seed=0)
    assert [(row["kind"], row["metric"]) for row in rows] == [
        ("rank_regression", "r2"),
        ("rank_regression", "spearman"),
        ("rank_order", "accuracy"),
    ]
    assert all(row["layer"] == 0 and row["n"] == 600 for row in rows)
    assert _metric(rows, "rank_order", "accuracy") > 0.95
    assert _metric(rows, "rank_regression", "r2") > 0.9
    assert _metric(rows, "rank_regression", "spearman") > 0.9


def test_probe_is_at_chance_on_shuffled_labels(planted):
    """Shuffling ranks destroys decodability."""
    rng = np.random.default_rng(1)
    shuffled = planted["item_ranks"][rng.choice(20, size=20, replace=False)][planted["items"]]
    rows = represent.probe_metrics(_states(planted, shuffled), seed=0)
    assert _metric(rows, "rank_regression", "r2") < 0.1
    assert _metric(rows, "rank_order", "accuracy") < 0.7


def test_eigenvalue_floor_is_a_fraction_of_the_largest_eigenvalue():
    """Eigenvalues below the fraction of the largest are raised to it before the roots are taken."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((200, 5))
    # Two columns are exact combinations of the others, so the sample covariance is singular.
    vectors = np.hstack([base, base[:, :1] * 2.0, base[:, 1:2] - base[:, 2:3]])
    ranks = np.repeat(np.arange(20), 10)
    raw = np.linalg.eigvalsh(represent.shrunk_covariance(vectors))
    rank_direction = represent.fit_rank_direction(vectors, ranks, site=0)
    assert np.isfinite(rank_direction.whitener).all() and np.isfinite(rank_direction.unwhitener).all()
    floor = represent.EIGENVALUE_FLOOR * raw.max()
    assert (rank_direction.eigenvalues >= floor * (1 - 1e-12)).all()
    assert np.allclose(rank_direction.eigenvalues, np.clip(raw, floor, None))
    # A floor of half the largest eigenvalue bites: the roots are those of the floored spectrum.
    floored = represent.fit_rank_direction(vectors, ranks, site=0, eigenvalue_floor=0.5)
    clipped = np.clip(raw, 0.5 * raw.max(), None)
    assert (clipped > raw).sum() == 5 and np.allclose(floored.eigenvalues, clipped)
    assert floored.eigenvalues.min() == pytest.approx(0.5 * raw.max())
    assert np.allclose(floored.unwhitener @ floored.whitener, np.eye(7), atol=1e-8)
    assert np.allclose(np.linalg.eigvalsh(floored.unwhitener @ floored.unwhitener), clipped)
    assert not np.allclose(np.linalg.eigvalsh(floored.unwhitener @ floored.unwhitener), raw)


def test_basis_width_is_clamped_to_distinct_ranks_minus_one():
    """The basis holds min(3, distinct ranks - 1) components; a single rank has no direction."""
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((90, 8))
    three = represent.fit_rank_direction(vectors, np.repeat([0, 1, 2], 30), site=0)
    assert three.n_components == 2 and three.basis.shape == (8, 2) and three.n_items == 3
    two = represent.fit_rank_direction(vectors, np.repeat([0, 1], 45), site=0)
    assert two.n_components == 1 and two.basis.shape == (8, 1) and two.n_items == 2
    nine = represent.fit_rank_direction(vectors, np.repeat(np.arange(9), 10), site=0)
    assert nine.n_components == represent.N_BASIS_COMPONENTS == 3 and nine.basis.shape == (8, 3)
    assert nine.n_items == 9
    with pytest.raises(ValueError):
        represent.fit_rank_direction(vectors, np.zeros(90), site=0)


def test_ordering_accuracy_counts_a_tie_as_one_half():
    """Item pairs are scored by their per-item mean predictions; equal means count one half."""
    items = np.array([0, 0, 1, 1, 2, 2])
    z_ranks = np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
    # Means 0, 0, 1 over the items in rank order: one tied pair and two ordered pairs of three.
    predictions = np.array([-0.5, 0.5, 0.0, 0.0, 1.0, 1.0])
    assert represent._ordering_accuracy(predictions, items, z_ranks) == pytest.approx(2.5 / 3)
    assert represent._ordering_accuracy(np.zeros(6), items, z_ranks) == 0.5
    assert represent._ordering_accuracy(z_ranks, items, z_ranks) == 1.0
    assert represent._ordering_accuracy(-z_ranks, items, z_ranks) == 0.0


def test_grouped_split_deals_whole_items_to_five_folds_three_times(planted, monkeypatch):
    """Held-out scoring refits five folds three times, each fold holding out whole items."""
    x, items = planted["vectors"], planted["items"]
    z = represent._zscore(planted["ranks"])
    index_of = {row.tobytes(): index for index, row in enumerate(x)}
    held_out = []
    fit = represent.ridge_fit

    def recording_fit(x_train, y_train, alpha):
        kept = np.array([index_of[row.tobytes()] for row in x_train])
        held_out.append(np.setdiff1d(np.arange(x.shape[0]), kept))
        return fit(x_train, y_train, alpha)

    monkeypatch.setattr(represent, "ridge_fit", recording_fit)
    represent._held_out_scores(x, z, items, alpha=1.0, seed=0)
    assert len(held_out) == represent.FOLDS * represent.FOLD_REPEATS == 15
    # Twenty items over five folds: every fit holds out four whole items of thirty rows each.
    for rows in held_out:
        assert rows.size == 4 * 30
        assert all((items == item).sum() == 30 for item in np.unique(items[rows]))
    for repeat in range(represent.FOLD_REPEATS):
        folds = held_out[repeat * represent.FOLDS : (repeat + 1) * represent.FOLDS]
        assert sorted(np.concatenate(folds).tolist()) == list(range(x.shape[0]))
    # The deal moves between repeats and is a function of the seed alone.
    assert not np.array_equal(held_out[0], held_out[represent.FOLDS])
    monkeypatch.setattr(represent, "ridge_fit", fit)
    first = represent._held_out_scores(x, z, items, alpha=1.0, seed=0)
    assert first == represent._held_out_scores(x, z, items, alpha=1.0, seed=0)
    assert first != represent._held_out_scores(x, z, items, alpha=1.0, seed=1)
    with pytest.raises(ValueError):
        represent._held_out_scores(x[:30], z[:30], items[:30], alpha=1.0, seed=0)


def test_ridge_penalty_is_chosen_by_exact_leave_one_out(planted):
    """The chosen penalty has the smallest mean squared leave-one-out error, computed by refits."""
    x = planted["vectors"][:60]
    y = ((planted["ranks"] - planted["ranks"].mean()) / planted["ranks"].std())[:60]
    alphas = (0.1, 10.0, 1000.0)
    errors = []
    for alpha in alphas:
        squared = []
        for held in range(x.shape[0]):
            keep = np.arange(x.shape[0]) != held
            coefficients, intercept = represent.ridge_fit(x[keep], y[keep], alpha)
            squared.append((y[held] - (x[held] @ coefficients + intercept)) ** 2)
        errors.append(np.mean(squared))
    assert represent.leave_one_out_alpha(x, y, alphas) == alphas[int(np.argmin(errors))]


def test_shrunk_covariance_matches_the_reference_fixture():
    """The numpy Ledoit-Wolf estimate matches the stored fixture values."""
    reference = np.load(REFERENCE)
    vectors = reference["vectors"]
    covariance = represent.shrunk_covariance(vectors)
    assert np.allclose(covariance, reference["covariance"], atol=1e-10)
    # The estimate shrinks the sample covariance toward a scaled identity by one intensity.
    centered = vectors - vectors.mean(axis=0)
    sample = centered.T @ centered / vectors.shape[0]
    mu = np.trace(sample) / sample.shape[0]
    intensity = (covariance[0, 0] - sample[0, 0]) / (mu - sample[0, 0])
    assert 0 < intensity < 1
    assert np.allclose(covariance, (1 - intensity) * sample + intensity * mu * np.eye(sample.shape[0]), atol=1e-10)
    rank_direction = represent.fit_rank_direction(vectors, reference["ranks"], site=0)
    assert np.allclose(rank_direction.whitener, reference["whitener"], atol=1e-8)
    assert np.allclose(rank_direction.whitener @ rank_direction.whitener, np.linalg.inv(covariance), atol=1e-6)
    assert np.allclose(rank_direction.unwhitener @ rank_direction.whitener, np.eye(covariance.shape[0]), atol=1e-8)
    assert np.allclose(rank_direction.unwhitener @ rank_direction.unwhitener, covariance, atol=1e-8)


def test_fit_rank_direction_matches_the_reference_fixture():
    """The fitted rank direction matches the stored fixture up to sign."""
    reference = np.load(REFERENCE)
    rank_direction = represent.fit_rank_direction(reference["vectors"], reference["ranks"], site=0)
    assert np.allclose(rank_direction.mean, reference["mean"], atol=1e-10)
    stored = reference["basis_vector"]
    assert min(np.abs(rank_direction.direction - stored).max(), np.abs(rank_direction.direction + stored).max()) < 1e-8
    calibration = represent.write_calibration(rank_direction, reference["item_rows"], reference["item_ranks"])
    stored_calibration = reference["calibration"]
    # The line's intercept, mean rank, and spread are the same either way; the slope flips with u.
    assert np.allclose(calibration[[0, 2, 3]], stored_calibration[[0, 2, 3]], atol=1e-10)
    assert np.isclose(abs(calibration[1]), abs(stored_calibration[1]), atol=1e-10)
    z_ranks = (reference["ranks"] - reference["ranks"].mean()) / reference["ranks"].std()
    coefficients, intercept = represent.ridge_fit(reference["vectors"], z_ranks, float(reference["ridge_alpha"]))
    assert float(reference["ridge_alpha"]) == RIDGE_ALPHA
    assert np.allclose(coefficients, reference["ridge_coefficients"], atol=1e-10)
    assert np.isclose(intercept, float(reference["ridge_intercept"]), atol=1e-10)


def test_gain_one_is_the_identity_edit(rank_direction, rows):
    """Gain 1 leaves every state unchanged."""
    edited = _apply(represent.erase_or_scale([rank_direction], 1.0), rows)
    assert torch.equal(edited, rows)
    assert edited.dtype == rows.dtype


def test_gain_zero_removes_the_rank_component(rank_direction, rows):
    """After gain 0 the projection onto the rank direction is zero."""
    edited = _apply(represent.erase_or_scale([rank_direction], 0.0), rows)
    assert np.abs(_coordinates(rank_direction, edited)).max() < 1e-10
    assert np.abs(_coordinates(rank_direction, rows)).max() > 0.1


def test_gain_minus_one_reflects_the_rank_component(rank_direction, rows):
    """Gain -1 negates the projection onto the rank direction."""
    edited = _apply(represent.erase_or_scale([rank_direction], -1.0), rows)
    assert np.allclose(_coordinates(rank_direction, edited), -_coordinates(rank_direction, rows), atol=1e-10)


def test_gain_one_half_is_the_midpoint_of_gains_one_and_zero(rank_direction, rows):
    """The gain interpolates on a straight line between the intact and the erased states."""
    intact = _apply(represent.erase_or_scale([rank_direction], 1.0), rows)
    erased = _apply(represent.erase_or_scale([rank_direction], 0.0), rows)
    halfway = _apply(represent.erase_or_scale([rank_direction], 0.5), rows)
    assert torch.allclose(halfway, (intact + erased) / 2, atol=1e-10)
    amplified = _apply(represent.erase_or_scale([rank_direction], 3.0), rows)
    assert np.allclose(_coordinates(rank_direction, amplified), 3 * _coordinates(rank_direction, rows), atol=1e-10)


def test_site_edit_is_the_projector_of_the_rank_direction(rank_direction, rows, planted, calibration):
    """The edit equals x - c (x - mean) A + inserted v with A = Sigma^(-1/2) u u^T Sigma^(1/2)."""
    projector = rank_direction.whitener @ np.outer(rank_direction.direction, rank_direction.direction) @ rank_direction.unwhitener
    x = rows.numpy()
    erased = _apply(represent.erase_or_scale([rank_direction], 0.25), rows).numpy()
    assert np.allclose(erased, x - 0.75 * ((x - rank_direction.mean) @ projector), atol=1e-10)
    target = represent.reversed_ordering(planted["item_ranks"])
    targets = represent.target_coordinates(calibration, target)
    written = _apply(represent.write_ordering([rank_direction], target, 0.5, calibrations=[calibration]), rows).numpy()
    expected = x - 0.5 * ((x - rank_direction.mean) @ projector) + 0.5 * targets[:, None] * rank_direction.insertion[None, :]
    assert np.allclose(written, expected, atol=1e-10)


def test_random_direction_control_shares_the_code_path(rank_direction, rows, planted, calibration):
    """The control is the same function with `direction="random"` and matched magnitude."""
    seed = represent.RANDOM_DIRECTION_SEED
    assert represent.direction_seed(seed, represent.EMBEDDING_SITE) == seed
    random_direction = represent.edit_direction(rank_direction, represent.Direction.RANDOM, seed)
    assert np.isclose(np.linalg.norm(random_direction), 1.0)
    assert abs(float(random_direction @ rank_direction.direction)) < 0.3
    assert np.array_equal(random_direction, represent.edit_direction(rank_direction, "random", seed))
    assert not np.allclose(random_direction, represent.edit_direction(rank_direction, "random", seed + 1))
    # The same mean: states at the mean are a fixed point of both erasures.
    means = torch.as_tensor(np.tile(rank_direction.mean, (rows.shape[0], 1)))
    for direction in represent.Direction:
        assert torch.allclose(_apply(represent.erase_or_scale([rank_direction], 0.0, direction=direction), means), means, atol=1e-12)
    control = _apply(represent.erase_or_scale([rank_direction], 0.0, direction="random"), rows)
    assert torch.equal(control, _apply(represent.erase_or_scale([rank_direction], 0.0, direction="random", seed=seed), rows))
    assert not torch.allclose(control, _apply(represent.erase_or_scale([rank_direction], 0.0), rows))
    assert not torch.allclose(control, rows)
    # The written control reads back the fitted rank direction's own targets along the random
    # direction.
    target = represent.reversed_ordering(planted["item_ranks"])
    targets = represent.target_coordinates(calibration, target)
    written = _apply(represent.write_ordering([rank_direction], target, 1.0, calibrations=[calibration], direction="random"), rows)
    read_back = rank_direction.whitener @ random_direction
    assert np.allclose((written.numpy() - rank_direction.mean) @ read_back, targets, atol=1e-8)


def test_random_direction_differs_per_site_and_is_deterministic(planted, calibration):
    """Each site draws its own random direction, seeded by the seed plus the site, every time."""
    rank_directions = [represent.fit_rank_direction(planted["vectors"], planted["ranks"], site=site) for site in range(3)]
    seed = represent.RANDOM_DIRECTION_SEED
    units = [
        represent.edit_direction(rank_direction, "random", represent.direction_seed(seed, rank_direction.site))
        for rank_direction in rank_directions
    ]
    assert [represent.direction_seed(seed, site) for site in range(3)] == [seed, seed + 1, seed + 2]
    for first, second in combinations(units, 2):
        assert not np.allclose(first, second) and not np.allclose(first, -second)
    first = represent.erase_or_scale(rank_directions, 0.0, direction="random")
    again = represent.erase_or_scale(rank_directions, 0.0, direction="random")
    assert [one.site for one in first] == [0, 1, 2]
    for one, other, rank_direction, unit in zip(first, again, rank_directions, units, strict=True):
        assert torch.equal(one.read_back, other.read_back) and torch.equal(one.insertion, other.insertion)
        assert np.allclose(one.read_back.numpy(), rank_direction.whitener @ unit, atol=1e-12)
        assert np.allclose(one.insertion.numpy(), rank_direction.unwhitener @ unit, atol=1e-12)
        assert np.allclose(one.mean.numpy(), rank_direction.mean)
    # The rank direction is the same at every site here, since every site was fit on one set of
    # vectors; the random directions are not.
    ranked = represent.erase_or_scale(rank_directions, 0.0)
    assert all(torch.allclose(one.read_back, ranked[0].read_back) for one in ranked)
    # A write's control shares each site's targets and moves them along that site's own direction.
    target = represent.reversed_ordering(planted["item_ranks"])
    written = represent.write_ordering(rank_directions, target, 1.0, calibrations=[calibration] * 3, direction="random")
    assert all(torch.equal(one.inserted, written[0].inserted) for one in written)
    assert not torch.allclose(written[0].insertion, written[1].insertion)
    with pytest.raises(ValueError):
        represent.write_ordering(rank_directions, target, 1.0, calibrations=[calibration])


def test_write_at_strength_zero_is_the_identity(rank_direction, rows, planted, calibration):
    """Strength 0 leaves every state unchanged."""
    target = represent.reversed_ordering(planted["item_ranks"])
    for direction in represent.Direction:
        edit = represent.write_ordering([rank_direction], target, 0.0, calibrations=[calibration], direction=direction)
        assert torch.equal(_apply(edit, rows), rows)


def test_write_read_back_interpolates_between_the_coordinate_and_the_target(rank_direction, rows, planted, calibration):
    """After strength s the read-back coordinate is (1 - s) r + s t, with v . q = 1."""
    assert np.isclose(float(rank_direction.insertion @ rank_direction.read_back), 1.0, atol=1e-10)
    target = represent.sample_unrelated_ordering(planted["item_ranks"], seed=0)
    targets = represent.target_coordinates(calibration, target)
    before = _coordinates(rank_direction, rows)
    for strength in represent.WRITE_STRENGTHS:
        edited = _apply(represent.write_ordering([rank_direction], target, strength, calibrations=[calibration]), rows)
        assert np.allclose(_coordinates(rank_direction, edited), (1 - strength) * before + strength * targets, atol=1e-8)
    # The target follows the item named for each state, whatever order the states come in.
    order = torch.as_tensor(np.random.default_rng(0).choice(rows.shape[0], size=rows.shape[0], replace=False))
    (one,) = represent.write_ordering([rank_direction], target, 1.0, calibrations=[calibration])
    assert np.allclose(_coordinates(rank_direction, one.apply(rows[order], order)), targets[order.numpy()], atol=1e-8)


def test_write_moves_coordinates_toward_the_target_ordering(rank_direction, rows, planted, calibration):
    """Agreement with the target ordering increases monotonically with strength."""
    target = represent.reversed_ordering(planted["item_ranks"])
    intercept, slope = calibration[0], calibration[1]
    agreements = []
    for strength in represent.WRITE_STRENGTHS:
        edited = _apply(represent.write_ordering([rank_direction], target, strength, calibrations=[calibration]), rows)
        written_rank = (_coordinates(rank_direction, edited) - intercept) / slope
        agreements.append(represent.rank_correlation(written_rank, target))
    assert agreements == sorted(agreements)
    assert agreements[0] < -0.9 and agreements[-1] > 0.99


def test_unrelated_ordering_respects_the_correlation_bound(planted):
    """The sampled unrelated ordering has |Spearman| <= 0.2 with the familiar ranks."""
    familiar = planted["item_ranks"]
    unrelated = represent.sample_unrelated_ordering(familiar, seed=0)
    assert abs(represent.rank_correlation(familiar, unrelated)) <= represent.MAX_ABS_SPEARMAN
    assert sorted(unrelated.tolist()) == sorted(familiar.tolist())
    assert np.array_equal(unrelated, represent.sample_unrelated_ordering(familiar, seed=0))
    assert not np.array_equal(unrelated, represent.sample_unrelated_ordering(familiar, seed=1))
    assert not np.array_equal(unrelated, represent.sample_unrelated_ordering(familiar, seed=0, target_seed=1))
    with pytest.raises(ValueError):
        represent.sample_unrelated_ordering(familiar, seed=0, max_abs_spearman=0.0, max_tries=5)
    targets = represent.write_targets(familiar, seed=0)
    assert targets[represent.WriteTarget.FAMILIAR][1] == 1.0
    assert targets[represent.WriteTarget.REVERSED][1] == -1.0
    assert np.array_equal(targets[represent.WriteTarget.REVERSED][0], 19 - familiar)
    assert targets[represent.WriteTarget.UNRELATED][1] == represent.rank_correlation(familiar, unrelated)


def test_reversed_ordering_flips_the_target():
    """The reversed target puts the bottom item on top; a rank map must cover every item."""
    target = {0: 2, 1: 0, 2: 1}  # item 1 on top, then item 2, then item 0
    assert represent.reversed_ordering(target).tolist() == [0, 2, 1]
    with pytest.raises(ValueError):
        represent.rank_array({0: 1, 2: 0})


def _content_battery(law, vocab, *, budgets=(0, 1, 2)):
    """The content battery at test scale: one context per pair, three reasoning-step values."""
    conditions = battery.load_preset("content", contexts_per_pair=1)
    return battery.Battery.build(conditions, law, vocab, seed=0, design=None, budgets=budgets)


def test_fit_query_pairs_cover_every_queryable_item():
    """Every pool item but the ends of the ordering is queried; a short draw is topped up."""
    from test_model import LAW

    order = LAW.ordering(FREQUENT_CUE, range(LAW.pool_size))
    feasible = represent.queryable_items(LAW, FREQUENT_CUE)
    assert feasible == order[1:-1] and len(feasible) == 46
    every = list(combinations(feasible, 2))
    assert len(every) == 1035
    pairs = represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=represent.N_FIT_PROMPTS, seed=0)
    assert len(pairs) >= represent.N_FIT_PROMPTS and len(set(pairs)) == len(pairs)
    assert set(pairs) <= set(every)
    assert {item for pair in pairs for item in pair} == set(feasible)
    # The drawn pairs keep their order among all pairs; the top-up follows them.
    drawn = [every.index(pair) for pair in pairs[: represent.N_FIT_PROMPTS]]
    assert drawn == sorted(drawn)
    assert pairs == represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=represent.N_FIT_PROMPTS, seed=0)
    assert pairs != represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=represent.N_FIT_PROMPTS, seed=1)
    assert represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=len(every), seed=0) == every
    assert represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=len(every) + 1, seed=0) == every
    # Three draws leave items out; the k-th left-out item is paired with the item k + 1 after it.
    few = represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=3, seed=0)
    covered = {item for pair in few[:3] for item in pair}
    missing = [item for item in feasible if item not in covered]
    assert len(few) == 3 + len(missing) and {item for pair in few for item in pair} == set(feasible)
    for k, (item, partner) in enumerate(few[3:]):
        assert item == missing[k]
        assert partner == feasible[(feasible.index(item) + 1 + k) % len(feasible)]
    # With no draw at all, the last item's partner wraps onto itself and falls back to the first.
    none = represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=0, seed=0)
    assert [item for item, _ in none] == list(feasible)
    assert none[-1] == (feasible[-1], feasible[0]) and len(set(none)) == len(none)
    with pytest.raises(ValueError):
        represent.fit_query_pairs(LAW, LAW.n_training_cues, n_prompts=3, seed=0)


@pytest.fixture(scope="module")
def probed_trials():
    """The fit trials at test scale: the 500 prompts under the frequent cue, seed 0."""
    from test_model import LAW

    return represent.fit_trials(LAW, FREQUENT_CUE, n_prompts=represent.N_FIT_PROMPTS, seed=0)


def test_fit_trials_query_an_interior_pair_under_an_uninformative_context(probed_trials):
    """The queried pair is interior in its law-ordered display; the context shows other items."""
    from test_model import LAW

    order = LAW.ordering(FREQUENT_CUE, range(LAW.pool_size))
    feasible = set(order[1:-1])
    queried: set[int] = set()
    for trial in probed_trials:
        assert trial.cue == FREQUENT_CUE and trial.context_kind is ContextKind.UNINFORMATIVE
        assert len(trial.items) == ITEMS_PER_ORDERING == len(set(trial.items))
        assert trial.items == LAW.ordering(FREQUENT_CUE, trial.items)
        assert set(trial.query_ranks) <= set(QUERYABLE_RANKS)
        assert set(trial.query) <= feasible
        assert len(trial.context_items) == ITEMS_PER_ORDERING
        assert trial.context_items == LAW.ordering(FREQUENT_CUE, trial.context_items)
        assert not set(trial.context_items) & set(trial.items)
        assert len(trial.context) == ITEMS_PER_ORDERING - 1 and trial.distractors == ()
        queried |= set(trial.query)
    assert queried == feasible
    assert len(probed_trials) == len(represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=represent.N_FIT_PROMPTS, seed=0))


def test_fit_prompts_and_item_states_cover_every_occurrence(probed_trials):
    """Every fit prompt is a prompt-only trial under the cue; each site reads every item token."""
    from test_model import LAW, SPEC, VOCAB, _network

    prompts = represent.fit_prompts(LAW, VOCAB, FREQUENT_CUE, n_prompts=12, seed=0)
    assert len(prompts) == len(represent.fit_query_pairs(LAW, FREQUENT_CUE, n_prompts=12, seed=0))
    n_positions = 2 + 2 * (ITEMS_PER_ORDERING - 1)
    for prompt in prompts:
        assert prompt.tokens.size == prompt.prompt_len == prompt_length(ITEMS_PER_ORDERING - 1)
        assert prompt.tokens[1] == VOCAB.cue_token(FREQUENT_CUE)
        assert item_positions(prompt.tokens, VOCAB).size == n_positions
        query = set((prompt.tokens[3:5] - VOCAB.first_item).tolist())
        context = set((prompt.tokens[7:][item_positions(prompt.tokens[7:], VOCAB)] - VOCAB.first_item).tolist())
        assert len(query) == 2 and len(context) == ITEMS_PER_ORDERING and not query & context
    assert not np.array_equal(prompts[0].tokens, prompts[1].tokens)
    again = represent.fit_prompts(LAW, VOCAB, FREQUENT_CUE, n_prompts=12, seed=0)
    assert all(np.array_equal(one.tokens, other.tokens) for one, other in zip(prompts, again, strict=True))
    assert not np.array_equal(prompts[0].tokens, represent.fit_prompts(LAW, VOCAB, FREQUENT_CUE, n_prompts=12, seed=1)[0].tokens)
    full = represent.fit_prompts(LAW, VOCAB, FREQUENT_CUE, n_prompts=represent.N_FIT_PROMPTS, seed=0)
    for prompt, trial in zip(full, probed_trials, strict=True):
        assert (prompt.tokens[3:5] - VOCAB.first_item).tolist() == list(trial.query)
    net = _network()
    ranks = LAW.ranks(FREQUENT_CUE, range(LAW.pool_size))
    embedding = represent.collect_item_states(net, prompts, VOCAB, site=0, ranks=ranks)
    assert embedding.vectors.shape == (LAW.pool_size, SPEC.d_model)
    assert np.allclose(embedding.vectors, net.embedding.weight.detach().numpy()[VOCAB.first_item :])
    assert embedding.ranks.tolist() == [ranks[item] for item in range(LAW.pool_size)]
    for site in range(1, net.n_sites):
        states = represent.collect_item_states(net, prompts, VOCAB, site=site, ranks=ranks)
        assert states.vectors.shape == (len(prompts) * n_positions, SPEC.d_model)
        assert states.site == site
        assert np.array_equal(states.ranks, represent.rank_array(ranks)[states.items])
        first = item_positions(prompts[0].tokens, VOCAB)
        assert states.items[:n_positions].tolist() == (prompts[0].tokens[first] - VOCAB.first_item).tolist()
    with pytest.raises(ValueError):
        represent.collect_item_states(net, prompts, VOCAB, site=net.n_sites, ranks=ranks)
    probe = represent.probe_by_layer(net, prompts, VOCAB, ranks=ranks, layers=range(net.n_sites), seed=0)
    assert probe.columns.tolist() == ["kind", "metric", "layer", "value", "n"]
    assert probe["layer"].tolist() == [site for site in range(net.n_sites) for _ in range(3)]
    assert probe["n"].tolist() == [LAW.pool_size] * 3 + [len(prompts) * n_positions] * 3 * (net.n_sites - 1)


@pytest.fixture(scope="module")
def fitted():
    """The small trained network with a rank direction and a calibration at every edit site."""
    from test_model import LAW, VOCAB, _trained_network

    net = _trained_network()
    ranks = LAW.ranks(FREQUENT_CUE, range(LAW.pool_size))
    prompts = represent.fit_prompts(LAW, VOCAB, FREQUENT_CUE, n_prompts=represent.N_FIT_PROMPTS, seed=0)
    rank_directions, calibrations = represent.fit_edit_directions(net, prompts, VOCAB, ranks=ranks)
    return net, prompts, represent.rank_array(ranks), rank_directions, calibrations


def _is_item(tokens: torch.Tensor, vocab) -> torch.Tensor:
    """Item-token positions, from the vocabulary's item range."""
    return (tokens >= vocab.first_item) & (tokens < vocab.first_item + vocab.pool_size)


def _site_coordinates(edit, site: int, states: torch.Tensor) -> torch.Tensor:
    """The coordinate of each state along the read-back functional of the edit at one site."""
    (one,) = [one for one in edit if one.site == site]
    return (states.double() - one.mean) @ one.read_back


def test_fit_edit_directions_fit_every_site_on_its_item_occurrences(fitted):
    """One rank direction and one calibration per edit site, each fit on that site's states."""
    from test_model import LAW, SPEC, VOCAB

    net, prompts, ranks, rank_directions, calibrations = fitted
    assert represent.edit_sites(net) == tuple(range(SPEC.n_layers + 1)) == (0, 1, 2)
    assert [rank_direction.site for rank_direction in rank_directions] == list(represent.edit_sites(net))
    assert len(calibrations) == len(rank_directions)
    states = represent.occurrence_states(net, prompts, VOCAB, sites=represent.edit_sites(net), ranks=ranks)
    n_positions = 2 + 2 * (ITEMS_PER_ORDERING - 1)
    # At site 0 every occurrence's state is its item's embedding row.
    rows = net.embedding.weight.detach().numpy()[VOCAB.first_item :]
    assert states[0].vectors.shape == (len(prompts) * n_positions, SPEC.d_model)
    assert np.array_equal(states[0].vectors, rows[states[0].items].astype(np.float64))
    for rank_direction, calibration in zip(rank_directions, calibrations, strict=True):
        one = states[rank_direction.site]
        refit = represent.fit_rank_direction(one.vectors, one.ranks, site=rank_direction.site)
        assert np.array_equal(rank_direction.mean, refit.mean) and np.array_equal(rank_direction.basis, refit.basis)
        assert np.array_equal(calibration, represent.write_calibration(rank_direction, one.vectors, one.ranks))
        assert rank_direction.n_items == np.unique(one.items).size == LAW.pool_size
    # The rank directions differ between sites, since every site holds its own states.
    assert not np.allclose(rank_directions[0].mean, rank_directions[1].mean)
    assert not np.allclose(rank_directions[1].mean, rank_directions[2].mean)
    with pytest.raises(ValueError):
        represent.occurrence_states(net, prompts, VOCAB, sites=(net.n_sites,), ranks=ranks)


def test_edit_table_scores_every_battery_under_the_chain_algorithm(fitted, monkeypatch):
    """Each edited battery is evaluated under the chain algorithm and the tag, its edit hooked."""
    net, _, ranks, rank_directions, calibrations = fitted
    received = {}
    modules = (net.embedding, *net.blocks)

    def stop(model, tests, vocab, *, step, tag, chain_algorithm=ChainAlgorithm.MAIN, **_):
        hooked = [len(module._forward_hooks) for module in modules]
        received.update(step=step, tag=tag, chain_algorithm=chain_algorithm, hooked=hooked, pre=len(model._forward_pre_hooks))
        raise RuntimeError("stop")

    monkeypatch.setattr(evaluate, "evaluate_battery", stop)
    specs = represent.edit_specs(rank_directions, calibrations, ranks, seed=0)[:1]
    with pytest.raises(RuntimeError):
        represent.edit_table(net, None, None, specs, step=3, chain_algorithm=ChainAlgorithm.BACKTRACK)
    assert received["chain_algorithm"] is ChainAlgorithm.BACKTRACK and received["step"] == 3
    assert received["tag"] == represent.REPRESENT_TAG
    assert received["hooked"] == [1] * len(modules) and received["pre"] == 1
    # The hooks come off even when the evaluation fails.
    assert all(not module._forward_hooks for module in modules) and not net._forward_pre_hooks
    with pytest.raises(ValueError):
        represent.edit_table(net, None, None, [], step=3)


def test_gain_one_at_every_site_leaves_the_battery_bit_identical(fitted):
    """Gain 1 and strength 0 at every site change no record, aggregate, or raw prediction."""
    from test_model import LAW, VOCAB

    net, prompts, ranks, rank_directions, calibrations = fitted
    tests = _content_battery(LAW, VOCAB)
    plain, plain_aggregates = evaluate.evaluate_battery(net, tests, VOCAB, step=0, tag="represent")
    plain_rollout = model.greedy_chain(net, prompts[:8], max_emissions=8, budgets=range(9))
    identities = [
        represent.erase_or_scale(rank_directions, 1.0),
        represent.erase_or_scale(rank_directions, 1.0, direction="random"),
        represent.write_ordering(rank_directions, represent.reversed_ordering(ranks), 0.0, calibrations=calibrations),
    ]
    for edit in identities:
        with represent.edited(net, edit):
            same, same_aggregates = evaluate.evaluate_battery(net, tests, VOCAB, step=0, tag="represent")
            rollout = model.greedy_chain(net, prompts[:8], max_emissions=8, budgets=range(9))
        pd.testing.assert_frame_equal(same, plain)
        pd.testing.assert_frame_equal(same_aggregates, plain_aggregates)
        assert torch.equal(rollout.tokens, plain_rollout.tokens) and torch.equal(rollout.scalars, plain_rollout.scalars)
        assert torch.equal(rollout.budget_prediction, plain_rollout.budget_prediction)
    # An erase does reach the predictions.
    with represent.edited(net, represent.erase_or_scale(rank_directions, 0.0)):
        erased = model.greedy_chain(net, prompts[:8], max_emissions=8, budgets=range(9))
    assert not torch.allclose(erased.budget_prediction, plain_rollout.budget_prediction, atol=1e-3)


def test_every_item_occurrence_is_edited_once_per_forward_at_every_site(fitted, monkeypatch):
    """The prompt, each emitted token, and each budget's query items are edited once at each site.

    Every edit reads the state the site produced in that forward and moves its coordinate to
    (1 - s) r + s t, the item's target, with the earlier sites' edits already in place.
    """
    from test_model import VOCAB

    net, prompts, ranks, rank_directions, calibrations = fitted
    strength = 0.5
    target = represent.reversed_ordering(ranks)
    edit = represent.write_ordering(rank_directions, target, strength, calibrations=calibrations)
    targets = {
        rank_direction.site: represent.target_coordinates(calibration, target)
        for rank_direction, calibration in zip(rank_directions, calibrations, strict=True)
    }
    forwards: list[dict] = []
    apply = represent.SiteEdit.apply

    def recording(self, states, items):
        out = apply(self, states, items)
        forwards[-1]["edits"].append((self.site, states.clone(), items.clone(), out.clone()))
        return out

    def record(_module, args, _kwargs):
        forwards.append({"tokens": args[0].clone(), "edits": []})

    monkeypatch.setattr(represent.SiteEdit, "apply", recording)
    handle = net.register_forward_pre_hook(record, with_kwargs=True)
    batch = prompts[:4]
    try:
        with represent.edited(net, edit):
            rollout = model.greedy_chain(net, batch, max_emissions=8, budgets=range(9))
    finally:
        handle.remove()
    # One pass over the prompts, one per emitted token, and one per budget over the query items.
    lengths = [forward["tokens"].shape[1] for forward in forwards]
    assert len(forwards) == 1 + EMISSION_LEN * 8 + 9
    assert lengths.count(batch[0].prompt_len) == 1 and lengths.count(1) == EMISSION_LEN * 8 and lengths.count(2) == 9
    sites = list(represent.edit_sites(net))
    edited_generated = 0
    for forward in forwards:
        tokens = forward["tokens"]
        expected = tokens[_is_item(tokens, VOCAB)] - VOCAB.first_item
        assert [site for site, *_ in forward["edits"]] == sites
        for site, states, items, out in forward["edits"]:
            assert torch.equal(items, expected) and states.shape == (expected.numel(), net.spec.d_model)
            before = _site_coordinates(edit, site, states)
            after = _site_coordinates(edit, site, out)
            written = (1 - strength) * before + strength * torch.as_tensor(targets[site])[items]
            scale = float(before.abs().max()) if before.numel() else 0.0
            assert torch.allclose(after, written, atol=1e-5 * max(1.0, scale))
        if tokens.shape[1] == 1:
            edited_generated += expected.numel()
        if tokens.shape[1] == 2:
            assert expected.numel() == 2 * len(batch)
    # Emitted item tokens were edited as they were fed back.
    emitted_items = int(_is_item(rollout.tokens, VOCAB).sum())
    assert emitted_items > 0 and edited_generated == emitted_items


def test_erase_zeroes_the_rank_coordinate_at_every_site_after_its_edit(fitted):
    """With gain 0 the state each site passes on holds no rank coordinate at an item position."""
    from test_model import VOCAB

    net, prompts, _, rank_directions, _ = fitted
    edit = represent.erase_or_scale(rank_directions, 0.0)
    sites = list(represent.edit_sites(net))
    sequences = prompts[:16]
    every = list(range(sequences[0].tokens.size))
    plain = model.batched_hidden_states(net, sequences, positions=every, layers=range(net.n_sites))
    with represent.edited(net, edit):
        states = model.batched_hidden_states(net, sequences, positions=every, layers=range(net.n_sites))
    tokens = torch.as_tensor(np.stack([one.tokens for one in sequences]))
    items = _is_item(tokens, VOCAB).numpy()
    for site in sites:
        before = _site_coordinates(edit, site, torch.as_tensor(plain[site][items]))
        after = _site_coordinates(edit, site, torch.as_tensor(states[site][items]))
        assert before.abs().max() > 0.1
        assert after.abs().max() < 1e-5 * float(before.abs().max())
    # Other positions enter the network unedited; the final normalized state is never edited.
    assert np.array_equal(states[0][~items], plain[0][~items])
    assert not np.allclose(states[net.n_sites - 1][items], plain[net.n_sites - 1][items])


def test_an_item_the_fit_never_saw_is_written_to_its_target(fitted):
    """A site fit on fewer items than the pool still holds a target for every pool item."""
    from test_model import LAW, VOCAB

    net, prompts, ranks, _, _ = fitted
    few = prompts[:2]
    sites = represent.edit_sites(net)
    states = represent.occurrence_states(net, few, VOCAB, sites=sites, ranks=ranks)
    rank_directions = [represent.fit_rank_direction(states[site].vectors, states[site].ranks, site=site) for site in sites]
    calibrations = [
        represent.write_calibration(rank_direction, states[rank_direction.site].vectors, states[rank_direction.site].ranks)
        for rank_direction in rank_directions
    ]
    seen = set(states[0].items.tolist())
    assert all(rank_direction.n_items == len(seen) < LAW.pool_size for rank_direction in rank_directions)
    target = represent.reversed_ordering(ranks)
    edit = represent.write_ordering(rank_directions, target, 1.0, calibrations=calibrations)
    assert all(one.inserted.shape == (LAW.pool_size,) for one in edit)
    other = next(one for one in prompts[2:] if set((one.tokens[item_positions(one.tokens, VOCAB)] - VOCAB.first_item).tolist()) - seen)
    where = item_positions(other.tokens, VOCAB)
    items = other.tokens[where] - VOCAB.first_item
    unseen = np.array([item not in seen for item in items.tolist()])
    assert unseen.any()
    with represent.edited(net, edit):
        written = model.batched_hidden_states(net, [other], positions=where.tolist(), layers=sites)
    for rank_direction, calibration in zip(rank_directions, calibrations, strict=True):
        coordinates = (written[rank_direction.site, 0].astype(np.float64) - rank_direction.mean) @ rank_direction.read_back
        expected = represent.target_coordinates(calibration, target)[items]
        assert np.allclose(coordinates[unseen], expected[unseen], atol=1e-5 * max(1.0, np.abs(expected).max()))
    # A target array that does not cover the pool, or a site outside the edit sites, is refused.
    with pytest.raises(ValueError):
        with represent.edited(net, [replace(edit[0], inserted=edit[0].inserted[:10])]):
            pass
    with pytest.raises(ValueError):
        with represent.edited(net, [replace(edit[0], site=net.n_sites - 1)]):
            pass
    with pytest.raises(ValueError):
        with represent.edited(net, [edit[0], edit[0]]):
            pass
    assert not net._forward_pre_hooks and all(not block._forward_hooks for block in net.blocks)


def test_chunked_rollout_matches_per_row_rollouts_under_an_edit(fitted):
    """An edit applied to a chunk equals the same edit applied to each row alone."""
    net, prompts, _, rank_directions, _ = fitted
    batch = prompts[:3]
    with represent.edited(net, represent.erase_or_scale(rank_directions, -1.0)):
        together = model.greedy_chain(net, batch, max_emissions=8, budgets=range(9))
        for row, encoded in enumerate(batch):
            alone = model.greedy_chain(net, [encoded], max_emissions=8, budgets=range(9))
            assert torch.equal(together.tokens[row], alone.tokens[0])
            assert torch.allclose(together.budget_prediction[row], alone.budget_prediction[0], atol=1e-5)


def test_edit_specs_span_the_grids(planted, calibration):
    """Each gain along both directions; each target at every strength, the control on unrelated."""
    rank_directions = [represent.fit_rank_direction(planted["vectors"], planted["ranks"], site=site) for site in range(2)]
    specs = represent.edit_specs(rank_directions, [calibration] * 2, planted["item_ranks"], seed=0)
    assert all([one.site for one in spec.edit] == [0, 1] for spec in specs)
    erased = [spec for spec in specs if spec.kind == "erase"]
    written = [spec for spec in specs if spec.kind == "write"]
    assert len(specs) == len(erased) + len(written)
    assert {(spec.gain, spec.direction) for spec in erased} == {
        (gain, direction) for gain in represent.ERASE_GAINS for direction in represent.Direction
    }
    assert len(erased) == len(represent.ERASE_GAINS) * 2
    assert all(np.isnan(spec.strength) and spec.target is None and spec.target_ranks is None for spec in erased)
    assert all(np.isnan(spec.spearman_vs_familiar) for spec in erased)
    assert all(one.inserted is None and one.removed == 1.0 - spec.gain for spec in erased for one in spec.edit)
    assert {(spec.target, spec.direction, spec.strength) for spec in written} == {
        (target, "rank", strength) for target in represent.WriteTarget for strength in represent.WRITE_STRENGTHS
    } | {("unrelated", "random", strength) for strength in represent.WRITE_STRENGTHS}
    assert len(written) == 4 * len(represent.WRITE_STRENGTHS)
    assert all(np.isnan(spec.gain) for spec in written)
    assert all(one.removed == spec.strength for spec in written for one in spec.edit)
    by_target = {spec.target: spec.spearman_vs_familiar for spec in written}
    assert by_target["familiar"] == 1.0 and by_target["reversed"] == -1.0
    assert abs(by_target["unrelated"]) <= represent.MAX_ABS_SPEARMAN


@pytest.fixture(scope="module")
def represented(tmp_path_factory, repo_root):
    """A tiny content checkpoint probed and edited through the `represent` subcommand."""
    out = tmp_path_factory.mktemp("represent")
    smoke = repo_root / "configs" / "smoke.yaml"
    overrides = ("pretraining.steps=8", "pretraining.eval_every=8", "logging.log_every=8")
    base = config.load_config(smoke, overrides)
    experiment = config.load_experiment("representation")
    run = next(run for run in config.enumerate_runs(experiment, base) if run.axis_values["frequent_cue_prop"] == 0.2)
    upstream = next(
        one
        for one in config.enumerate_runs(config.load_experiment(run.depends_on_experiment), base)
        if one.run_id == run.depends_on_run_id
    )
    assert upstream.config.task.cue_distribution.kind == "frequent_cue"
    assert upstream.config.task.cue_distribution.frequent_cue_prop == 0.2
    assert upstream.config.pretraining.battery == "content"
    training.run_stage(
        upstream.config,
        Stage.PRETRAINING,
        seed=0,
        init_from=None,
        out_dir=config.run_directory(out, upstream.experiment, upstream.run_id),
        device="cpu",
    )
    argv = ["represent", "--config", str(smoke)]
    for override in overrides:
        argv += ["--set", override]
    argv += ["--experiment", experiment.name, "--index", str(run.index), "--out", str(out), "--device", "cpu"]
    assert cli.main(argv) == 0
    written = tables.export_experiment(experiment.name, results_root=out, base=base)
    return out, run, base, written


def test_represent_writes_probe_and_edit_tables(represented):
    """The run holds a probe table over every site and an edit table over the whole grid."""
    out, run, base, written = represented
    run_dir = config.run_directory(out, run.experiment, run.run_id)
    step = run.config.pretraining.steps
    assert (run_dir / "probe" / f"step_{step:08d}.parquet").is_file()
    assert (run_dir / "edit" / f"step_{step:08d}.parquet").is_file()
    probe = tables.load_table(written["probe"], schema=tables.PROBE_SCHEMA, swept_columns=("frequent_cue_prop",))
    n_sites = run.config.model.n_layers + 2
    assert len(probe) == 3 * n_sites
    assert set(probe["layer"]) == set(range(n_sites))
    assert set(zip(probe["kind"], probe["metric"], strict=True)) == {
        ("rank_regression", "r2"),
        ("rank_regression", "spearman"),
        ("rank_order", "accuracy"),
    }
    assert (probe["seed"] == 0).all() and (probe["step"] == step).all()
    assert (probe.loc[probe["layer"] == 0, "n"] == run.config.task.item_pool_size).all()
    n_positions = 2 + 2 * (ITEMS_PER_ORDERING - 1)
    law = training.build_law(run.config, 0)
    n_prompts = len(represent.fit_query_pairs(law, FREQUENT_CUE, n_prompts=represent.N_FIT_PROMPTS, seed=0))
    assert n_prompts >= represent.N_FIT_PROMPTS
    assert (probe.loc[probe["layer"] > 0, "n"] == n_prompts * n_positions).all()
    edit = tables.load_table(written["edit"], schema=tables.EDIT_SCHEMA, swept_columns=("frequent_cue_prop",))
    assert (edit["tag"] == represent.REPRESENT_TAG).all()
    assert set(edit["K"]) == set(run.config.evaluation.reasoning_steps())
    conditions = {condition.name for condition in battery.load_preset("content", contexts_per_pair=1)}
    assert set(edit["condition"]) == conditions
    erased = edit[edit["edit_kind"] == "erase"]
    written_rows = edit[edit["edit_kind"] == "write"]
    assert len(erased) + len(written_rows) == len(edit)
    assert set(erased["gain"]) == set(represent.ERASE_GAINS) and set(erased["direction"]) == {"rank", "random"}
    assert (erased["target"] == "").all()
    assert erased["strength"].isna().all() and erased["written_consistent_frac"].isna().all()
    assert erased["spearman_vs_familiar"].isna().all()
    assert set(written_rows["strength"]) == set(represent.WRITE_STRENGTHS)
    assert set(zip(written_rows["target"], written_rows["direction"], strict=True)) == {
        ("familiar", "rank"),
        ("reversed", "rank"),
        ("unrelated", "rank"),
        ("unrelated", "random"),
    }
    assert written_rows["gain"].isna().all()
    assert written_rows["written_consistent_frac"].between(0, 1).all()
    by_target = written_rows.groupby("target")["spearman_vs_familiar"].unique()
    assert by_target["familiar"].tolist() == [1.0] and by_target["reversed"].tolist() == [-1.0]
    assert len(by_target["unrelated"]) == 1 and abs(by_target["unrelated"][0]) <= represent.MAX_ABS_SPEARMAN
    # Under the familiar target the written sign is the familiar sign wherever a familiar ordering
    # exists, so on the frequent cue's conditions the two fractions agree; the ICL test has none.
    familiar = written_rows[(written_rows["target"] == "familiar") & (written_rows["metric"] == "accuracy")]
    novel = familiar[familiar["source"] == "novel_unseen_cue"]
    assert novel["prior_consistent_frac"].isna().all() and novel["written_consistent_frac"].notna().all()
    frequent = familiar[familiar["source"] == "frequent_cue"]
    assert len(frequent) and np.allclose(frequent["written_consistent_frac"], frequent["prior_consistent_frac"])
    # One row per condition, distance, budget, and metric in each edited battery.
    per_battery = edit.groupby(["edit_kind", "direction", "target", "gain", "strength"], dropna=False).size()
    assert per_battery.nunique() == 1
    assert len(per_battery) == 2 * len(represent.ERASE_GAINS) + 4 * len(represent.WRITE_STRENGTHS)


def test_represent_tables_are_in_the_manifest(represented):
    """The export lists both tables with their row counts and hashes."""
    out, run, _, written = represented
    manifest = yaml.safe_load((config.tables_directory(out, run.experiment) / "MANIFEST.yaml").read_text(encoding="utf-8"))
    entries = {entry["table"]: entry for entry in manifest["tables"]}
    assert set(entries) == {"probe", "edit"} == set(written)
    for name, entry in entries.items():
        assert entry["rows"] == len(pd.read_parquet(written[name]))
        assert entry["runs"] == [run.run_id] and entry["axes"] == ["frequent_cue_prop"]
        assert len(entry["sha256"]) == 64


def test_represent_run_tables_are_deterministic(represented):
    """Running the same checkpoint again writes byte-identical probe and edit tables."""
    out, run, base, _ = represented
    first = config.run_directory(out, run.experiment, run.run_id)
    argv = ["represent", "--config", str(config.repo_root() / "configs" / "smoke.yaml")]
    argv += ["--set", "pretraining.steps=8", "--set", "pretraining.eval_every=8", "--set", "logging.log_every=8"]
    argv += ["--experiment", run.experiment, "--index", str(run.index), "--out", str(out), "--device", "cpu"]
    before = {name: (first / name).read_bytes() for name in _step_files(first)}
    assert cli.main(argv) == 0
    for name, content in before.items():
        assert (first / name).read_bytes() == content


def _step_files(run_dir: Path) -> list[str]:
    """The relative paths of a run's probe and edit step files."""
    return [str(path.relative_to(run_dir)) for table in ("probe", "edit") for path in sorted((run_dir / table).glob("*.parquet"))]
