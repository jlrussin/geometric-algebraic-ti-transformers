"""Closed forms and enumeration for the framework's predictions."""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from geo_alg_ti.formalization import panels, strategies
from geo_alg_ti.plots import style

PANEL_FUNCTIONS = (
    panels.make_strategy_accuracy_by_reasoning_steps,
    panels.make_strategy_cost,
    panels.make_content_accuracy,
    panels.make_content_cost,
    panels.make_ordering_diversity,
    panels.make_diversity_tradeoff,
    panels.make_algebraic_accuracy_by_budget,
    panels.make_content_accuracy_by_budget,
)
PANEL_NAMES = ("accuracy", "cost", "content_accuracy", "content_cost", "diversity", "tradeoff")


def test_expected_steps_matches_the_closed_form():
    """Enumeration gives E[L | d] = (n - 1 + d) / 4 for n in 5, 7, 9, 11 at every distance.

    Adjacent pairs are walked here too; with the shortcut they take zero steps.
    """
    for n in (5, 7, 9, 11):
        walks = strategies.enumerate_walks(n, shortcut_adjacent=False)
        d_values = list(range(1, n - 2))
        by_d = strategies.steps_by_distance(walks, d_values)
        for d, value in zip(d_values, by_d):
            assert abs(value - strategies.expected_steps(n, d)) < 1e-9
        with_shortcut = strategies.steps_by_distance(strategies.enumerate_walks(n), d_values)
        assert with_shortcut[0] == 0.0
        assert np.allclose(with_shortcut[1:], by_d[1:])


def test_enumeration_covers_every_start_target_pair(monkeypatch):
    """Every ordered pair of interior positions appears with both directions."""
    n = 9
    seen = []
    original = strategies.walk

    def recording_walk(n_items, start, target, direction):
        seen.append((start, target, direction))
        return original(n_items, start, target, direction)

    monkeypatch.setattr(strategies, "walk", recording_walk)
    walks = strategies.enumerate_walks(n, shortcut_adjacent=False)
    pairs = set(strategies.iterate_pairs(n))
    assert pairs == {(x, y) for x in range(2, n) for y in range(2, n) if x != y}
    assert set(seen) == {(x, y, s) for x, y in pairs for s in (1, -1)}
    assert len(seen) == len(walks.d) == 2 * len(pairs)
    assert abs(float(walks.weight.sum()) - 1.0) < 1e-12
    assert walks.correct.all()
    with_shortcut = strategies.enumerate_walks(n)
    assert len(with_shortcut.d) == 2 * len(pairs) - 2 * (n - 3)
    assert abs(float(with_shortcut.weight.sum()) - 1.0) < 1e-12
    assert (with_shortcut.length[with_shortcut.d == 1] == 0).all()


def test_completion_probability_is_monotone_in_the_budget():
    """P(L <= K | d) is non-decreasing in K and equals 1/2 [K >= d] + 1/2 min(K, m) / m."""
    n = 9
    n_interior = n - 2
    d_values = list(range(1, n - 2))
    budgets = list(range(0, n - 2))
    walks = strategies.enumerate_walks(n, shortcut_adjacent=False)
    completion = strategies.completion_probability(walks, d_values, budgets)
    assert completion.shape == (len(d_values), len(budgets))
    assert (np.diff(completion, axis=1) >= 0).all()
    for i, d in enumerate(d_values):
        m = n_interior - d
        for j, k in enumerate(budgets):
            closed = 0.5 * (k >= d) + 0.5 * min(k, m) / m
            assert abs(completion[i, j] - closed) < 1e-12
    accuracy = strategies.accuracy_by_reasoning_steps(completion)
    assert np.allclose(accuracy, 0.5 + 0.5 * completion)
    guessing = strategies.accuracy_by_reasoning_steps(completion, guess_accuracy=0.25)
    assert np.allclose(guessing, 0.25 + 0.75 * completion)


def test_distance_distribution_sums_to_one():
    """The query-distance distribution is a probability vector proportional to (n_int - d)."""
    for n in (5, 7, 9, 11):
        walks = strategies.enumerate_walks(n)
        d_values = np.arange(1, n - 2)
        p = strategies.distance_distribution(walks, d_values)
        assert abs(float(p.sum()) - 1.0) < 1e-12
        expected = (n - 2 - d_values) / (n - 2 - d_values).sum()
        assert np.allclose(p, expected)
        assert abs(float((d_values * p).sum()) - (n - 1) / 3) < 1e-12


def test_orderings_built_by_experience_matches_the_uniform_closed_form():
    """D_Q(0) = M (1 - (1 - 1/M)^Q) and D_Q decreases with skew alpha."""
    for n_orderings, n_queries in ((60, 100), (100, 10), (7, 1000)):
        uniform = n_orderings * (1.0 - (1.0 - 1.0 / n_orderings) ** n_queries)
        built = strategies.orderings_built_by_experience(0.0, n_orderings, n_queries)
        assert abs(built - uniform) < 1e-12 * uniform
        by_alpha = [
            strategies.orderings_built_by_experience(a, n_orderings, n_queries)
            for a in np.linspace(0, 5, 51)
        ]
        assert all(a >= b for a, b in zip(by_alpha, by_alpha[1:]))
        assert 1.0 <= by_alpha[-1] and by_alpha[0] <= min(n_orderings, n_queries)


def test_check_identities_passes_at_the_defaults():
    """Every identity holds for the enumeration at the default settings."""
    strategies.check_identities()


def test_check_identities_raises_on_a_broken_enumeration(monkeypatch):
    """A walk that claims to reach every target in one step answers half the queries wrong.

    The first identity to trip is the walk's correctness, so the error names it.
    """
    monkeypatch.setattr(strategies, "walk", lambda n_items, start, target, direction: (1, True))
    with pytest.raises(AssertionError, match="walk answers"):
        strategies.check_identities()


def test_check_identities_raises_on_a_walk_of_the_wrong_length(monkeypatch):
    """A walk away from the target that runs one step too long still answers correctly.

    Every answer stays right and the weights still sum to one, so the walk length
    identity is the one that trips.
    """
    original = strategies.walk

    def long_away_walk(n_items, start, target, direction):
        steps, reached = original(n_items, start, target, direction)
        return (steps, reached) if reached else (steps + 1, reached)

    monkeypatch.setattr(strategies, "walk", long_away_walk)
    with pytest.raises(AssertionError, match="walk length"):
        strategies.check_identities()


def test_check_identities_rejects_settings_the_enumeration_cannot_hold():
    """Fewer than five items leave no queryable pair; a budget past n_items - 3 is unreachable."""
    with pytest.raises(ValueError, match="n_items"):
        strategies.check_identities(n_items=4)
    with pytest.raises(ValueError, match="budgets"):
        strategies.check_identities(n_items=9, budgets=(1, 2, 7))
    strategies.check_identities(n_items=7, budgets=(1, 2, 3, 4))
    with pytest.raises(ValueError, match="sweep budgets"):
        strategies.check_identities(sweep_budgets=(-1, 0, 1))
    strategies.check_identities(sweep_budgets=tuple(range(0, 13)))


def test_check_identities_checks_the_values_the_panels_draw():
    """The guard's defaults are the module constants the panel defaults read."""
    import inspect

    guard = inspect.signature(strategies.check_identities).parameters
    assert guard["n_items"].default == strategies.N_ITEMS == 9
    assert guard["budgets"].default == strategies.BUDGETS == (1, 2, 3, 4, 5)
    assert guard["sigma"].default == strategies.SIGMA
    assert guard["premise_cost"].default == strategies.PREMISE_COST
    assert guard["n_queries"].default == strategies.N_QUERIES
    assert guard["n_orderings"].default == strategies.N_ORDERINGS
    assert guard["margin"].default == strategies.MARGIN
    assert guard["temperature"].default == strategies.TEMPERATURE
    assert guard["skew_alphas"].default == strategies.SKEW_ALPHA_GRID
    assert guard["sweep_budgets"].default == strategies.SWEEP_BUDGETS
    for panel in (panels.make_algebraic_accuracy_by_budget, panels.make_content_accuracy_by_budget):
        assert inspect.signature(panel).parameters["budgets"].default == strategies.SWEEP_BUDGETS
    accuracy = inspect.signature(panels.make_strategy_accuracy_by_reasoning_steps).parameters
    assert accuracy["budgets"].default == strategies.BUDGETS
    assert accuracy["sigma"].default == strategies.SIGMA
    diversity = inspect.signature(panels.make_ordering_diversity).parameters
    for name in ("n_items", "n_orderings", "n_queries", "premise_cost", "margin", "temperature"):
        assert diversity[name].default == getattr(strategies, name.upper())
    assert strategies.DISTANCES == tuple(range(1, strategies.N_ITEMS - 2))


def test_cost_per_query_takes_the_cost_of_one_ordering():
    """The geometric cost is the ordering cost per build plus one unit per query."""
    geometric, algebraic = strategies.cost_per_query(
        ordering_cost=16.0, n_queries=100, n_orderings=60, skew_alphas=(0.0, 5.0), mean_steps=2.5
    )
    builds = strategies.orderings_built_by_experience(0.0, 60, 100)
    assert abs(geometric[0] - (16.0 * builds + 100) / 100) < 1e-12
    assert np.allclose(algebraic, 2.5)


def test_no_random_number_generator_is_used():
    """The module imports no random number generator."""
    import inspect

    src = inspect.getsource(strategies)
    assert "default_rng" not in src and "import random" not in src


def test_panels_return_statistics_and_draw_on_the_given_axes():
    """Each panel follows the `make_<name>(ax, ...) -> dict` contract."""
    before = plt.get_fignums()
    for make in PANEL_FUNCTIONS:
        fig = Figure()
        ax = fig.add_subplot()
        statistics = make(ax)
        assert isinstance(statistics, dict) and statistics
        assert plt.get_fignums() == before
        assert len(ax.lines) > 0 and ax.get_xlabel() and ax.get_ylabel()
        assert not ax.get_legend()


def test_accuracy_by_budget_follows_the_walks_completion():
    """Per distance and budget: a guess or the prior until the walk ends, then right."""
    walks = strategies.enumerate_walks(strategies.N_ITEMS)
    budgets = strategies.SWEEP_BUDGETS
    completion = strategies.completion_probability(walks, strategies.DISTANCES, budgets)
    fig = Figure()
    ax_algebraic, ax_content = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    algebraic = panels.make_algebraic_accuracy_by_budget(ax_algebraic)
    content = panels.make_content_accuracy_by_budget(ax_content)
    assert tuple(algebraic["budgets"]) == budgets == tuple(content["budgets"])
    n_d = len(strategies.DISTANCES)
    geometric = strategies.geometric_accuracy(strategies.DISTANCES)[:, None]
    # The chance line comes first, then one line per distance (reversed, then congruent).
    assert np.allclose([ln.get_ydata() for ln in ax_algebraic.lines[1:]], algebraic["algebraic_accuracy"])
    assert np.allclose([ln.get_ydata() for ln in ax_content.lines[1 : 1 + n_d]], content["reversed_accuracy"])
    assert np.allclose([ln.get_ydata() for ln in ax_content.lines[1 + n_d :]], content["congruent_accuracy"])
    assert np.allclose(algebraic["algebraic_accuracy"], 0.5 + 0.5 * completion)
    assert np.allclose(content["congruent_accuracy"], completion + (1 - completion) * geometric)
    assert np.allclose(content["reversed_accuracy"], completion + (1 - completion) * (1 - geometric))
    for i, d in enumerate(strategies.DISTANCES):
        m = strategies.N_ITEMS - 2 - d
        for k in budgets:
            closed = 1.0 if d == 1 else 0.5 * (k >= d) + 0.5 * min(k, m) / m
            agree = float(strategies.normal_cdf(d / strategies.SIGMA))
            assert content["congruent_accuracy"][i, k] == pytest.approx(closed + (1 - closed) * agree)
            assert content["reversed_accuracy"][i, k] == pytest.approx(closed + (1 - closed) * (1 - agree))
    # At zero steps only adjacent pairs come from the premises; the rest from the built ordering.
    assert np.allclose(content["congruent_accuracy"][1:, 0], geometric[1:, 0])
    assert np.allclose(content["reversed_accuracy"][1:, 0], 1 - geometric[1:, 0])
    for key in ("congruent_accuracy", "reversed_accuracy"):
        assert np.allclose(content[key][0], 1.0) and np.allclose(content[key][:, -1], 1.0)
    assert np.allclose(algebraic["algebraic_accuracy"][:, -1], 1.0)


def test_content_accuracy_falls_back_on_the_built_ordering():
    """An unfinished walk answers from the noisy built ordering; with no walk, the geometric one."""
    fig = Figure()
    ax = fig.add_subplot()
    statistics = panels.make_content_accuracy(ax)
    walks = strategies.enumerate_walks(strategies.N_ITEMS)
    completion = strategies.completion_probability(walks, strategies.DISTANCES, strategies.BUDGETS).mean(axis=1)
    accuracy = panels.make_strategy_accuracy_by_reasoning_steps(fig.add_subplot())
    geometric = accuracy["geometric_accuracy"]
    assert np.allclose(statistics["uninformative_accuracy"], geometric)
    assert np.allclose(statistics["shuffled_accuracy"], accuracy["algebraic_accuracy"])
    assert np.allclose(statistics["congruent_accuracy"], completion + (1 - completion) * geometric)
    assert np.allclose(statistics["reversed_accuracy"], completion + (1 - completion) * (1 - geometric))
    for context in strategies.INFORMATIVE_CONTEXTS:
        assert np.isclose(statistics[f"{context}_accuracy"][0], 1.0)
    assert (np.diff(statistics["reversed_accuracy"]) < 0).all()
    assert (np.diff(statistics["uninformative_accuracy"]) > 0).all()
    drawn = {to_hex(line.get_color()): line.get_ydata() for line in ax.lines}
    for context in ("uninformative", *strategies.INFORMATIVE_CONTEXTS):
        assert np.allclose(drawn[to_hex(style.CONTEXT_COLORS[context])], statistics[f"{context}_accuracy"])
    names = {text.get_text() for text in ax.texts}
    assert names == {"uninformative", "congruent", "reversed", "shuffled"}


def test_prior_agreement_by_context():
    """Agreement of the built ordering: congruent if right, reversed if wrong, shuffled half."""
    d = strategies.DISTANCES
    geometric = strategies.geometric_accuracy(d)
    assert np.allclose(strategies.prior_agreement("congruent", d), geometric)
    assert np.allclose(strategies.prior_agreement("reversed", d), 1 - geometric)
    assert np.allclose(strategies.prior_agreement("shuffled", d), 0.5)
    with pytest.raises(ValueError):
        strategies.prior_agreement("uninformative", d)
    completion = np.array([[0.0, 0.5, 1.0]] * len(d))
    expected = completion + (1 - completion) * geometric[:, None]
    assert np.allclose(strategies.content_accuracy(completion, geometric), expected)


def test_content_cost_is_the_strategy_cost_by_condition():
    """Congruent queries need no steps; reversed and shuffled ones cost the algebraic walk."""
    fig = Figure()
    ax = fig.add_subplot()
    content = panels.make_content_cost(ax, dodge=0.2)
    strategy = panels.make_strategy_cost(fig.add_subplot())
    assert np.allclose(content["congruent_steps"], 0.0)
    assert np.allclose(content["reversed_steps"], strategy["expected_steps"])
    assert np.allclose(content["shuffled_steps"], strategy["expected_steps"])
    drawn = {to_hex(line.get_color()): line.get_ydata() for line in ax.lines}
    assert np.allclose(drawn[to_hex(style.CONGRUENT)], 0.0)
    assert np.allclose(drawn[to_hex(style.REVERSED)], strategy["expected_steps"])
    assert np.allclose(drawn[to_hex(style.SHUFFLED)], strategy["expected_steps"])
    # The two equal lines are drawn side by side about each distance; congruent is centered.
    x = {to_hex(line.get_color()): line.get_xdata() for line in ax.lines}
    d = np.asarray(strategies.DISTANCES, dtype=float)
    assert np.allclose(x[to_hex(style.REVERSED)], d - 0.2)
    assert np.allclose(x[to_hex(style.SHUFFLED)], d + 0.2)
    assert np.allclose(x[to_hex(style.CONGRUENT)], d)


def test_diversity_tradeoff_draws_the_diversity_curves_against_each_other():
    """The trade-off panel traces the diversity panel's two accuracies, colored by alpha."""
    fig = Figure()
    diversity = panels.make_ordering_diversity(fig.add_subplot())
    ax = fig.add_subplot()
    tradeoff = panels.make_diversity_tradeoff(ax)
    assert np.allclose(tradeoff["iwl_test_accuracy"], diversity["iwl_test_accuracy"])
    assert np.allclose(tradeoff["icl_test_accuracy"], diversity["icl_test_accuracy"])
    curve = np.concatenate([segment[:1] for segment in ax.collections[0].get_segments()])
    expected = np.column_stack([diversity["iwl_test_accuracy"], diversity["icl_test_accuracy"]])
    assert np.allclose(curve, expected[:-1])
    marks = ax.collections[-1]
    assert len(marks.get_offsets()) == len(tradeoff["marked_alphas"])
    assert np.isclose(marks.get_offsets()[0][0], diversity["iwl_test_accuracy"][0])
    assert np.isclose(marks.get_offsets()[-1][1], diversity["icl_test_accuracy"][-1])
    alphas = tradeoff["skew_alphas"]
    span = (float(alphas.min()), float(alphas.max()))
    ends = style.sweep_colors(list(span), style.SKEW_CMAP, column="skew_alpha", span=span)
    faces = [to_hex(c) for c in marks.get_facecolors()]
    assert [faces[0], faces[-1]] == ends


def test_strip_pins_each_axis_tick_size_from_its_own_key():
    """Both tick label sizes are pinned from the strip's parameters after drawing."""
    fig, axes = panels.strip_figure()
    panels.build_strip(axes)
    for ax in axes:
        x_size = ax.xaxis.get_major_ticks()[0].label1.get_fontsize()
        y_size = ax.yaxis.get_major_ticks()[0].label1.get_fontsize()
        assert x_size == panels.STRIP_RC["xtick.labelsize"]
        assert y_size == panels.STRIP_RC["ytick.labelsize"]


def test_strip_keeps_its_print_size_and_media_box(tmp_path):
    """The strip stays at its print size through drawing; the PDF page matches."""
    before = plt.get_fignums()
    fig, axes = panels.strip_figure()
    assert len(axes) == 6
    statistics = panels.build_strip(axes)
    assert tuple(statistics) == PANEL_NAMES
    assert tuple(fig.get_size_inches()) == panels.STRIP_SIZE_IN
    assert not fig.texts
    assert [ax.get_title() for ax in axes[:: panels.N_ROWS]] == list(panels.COLUMN_TITLES)
    assert not any(ax.get_title() for ax in axes[1 :: panels.N_ROWS])
    assert plt.get_fignums() == before
    path = tmp_path / "strip.pdf"
    fig.savefig(path, format="pdf")
    match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", path.read_bytes())
    assert match is not None
    width_pt, height_pt = (float(v) for v in match.groups())
    assert abs(width_pt - 72 * panels.STRIP_SIZE_IN[0]) < 1e-6
    assert abs(height_pt - 72 * panels.STRIP_SIZE_IN[1]) < 1e-6


def test_build_singles_writes_one_pdf_per_panel(tmp_path):
    """One PDF per panel, the budget panels after the strip's, at the single-panel size."""
    paths = panels.build_singles(tmp_path / "out")
    names = (*PANEL_NAMES, "algebraic_accuracy_by_budget", "content_accuracy_by_budget")
    assert [p.name for p in paths] == [f"formalization_{name}.pdf" for name in names]
    for path in paths:
        data = path.read_bytes()
        assert data.startswith(b"%PDF")
        match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", data)
        width_pt, height_pt = (float(v) for v in match.groups())
        assert abs(width_pt - 72 * panels.SINGLE_SIZE_IN[0]) < 1e-6
        assert abs(height_pt - 72 * panels.SINGLE_SIZE_IN[1]) < 1e-6
