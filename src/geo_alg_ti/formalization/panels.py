"""Panels of the predictions; matplotlib only, same `make_<name>(ax, ...) -> dict` contract.

Each panel draws on the axes it is given and returns the statistics it
plotted; curves are named by text placed in the plot rather than by a legend
box. `strip_figure` creates the 2 x 3 strip at its exact print size, one
column per prediction (distance effects, content effects, data distribution),
`build_strip` fills it, and `build_singles` saves each panel alone at an
inspection size. `make_algebraic_accuracy_by_budget` and `make_content_accuracy_by_budget`
(`BUDGET_PANELS`) draw accuracy against the reasoning-step budget, one line per distance (per
context in the content panel) shaded light (distance 1) to dark (distance 6); they are saved
with the single panels but are
not part of the strip. Font sizes are applied through an rc context, never by changing the
global parameters. The Source Sans 3 family ships in `fonts/` and is
registered on import; if that fails the default sans-serif is used.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib import font_manager, rc_context
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from geo_alg_ti import tables
from geo_alg_ti.formalization import strategies
from geo_alg_ti.plots import style

STRIP_SIZE_IN: tuple[float, float] = (3.25, 2.05)
SINGLE_SIZE_IN: tuple[float, float] = (3.3, 2.5)

CHANCE_STYLE: dict[str, object] = {"color": "0.6", "lw": 0.6, "ls": "--", "zorder": 1}
LINE_STYLE: dict[str, object] = {"marker": "o", "markersize": 2.6, "lw": 1.0, "zorder": 3}
LABEL_FONTSIZE: float = 4.8
MARK_STYLE: dict[str, object] = {"s": 9, "edgecolors": "black", "linewidths": 0.4, "zorder": 4}

FONT_DIR: Path = Path(__file__).parent / "fonts"
FONT_NAME: str = "Source Sans 3"


def _register_fonts() -> str:
    """Register the shipped Source Sans 3 faces; return the family to use."""
    try:
        for path in sorted(FONT_DIR.glob("*.ttf")):
            font_manager.fontManager.addfont(str(path))
        font_manager.findfont(
            font_manager.FontProperties(family=FONT_NAME), fallback_to_default=False
        )
    except Exception:
        return "sans-serif"
    return FONT_NAME


FONT_FAMILY: str = _register_fonts()

STRIP_RC: dict[str, object] = {
    "font.family": FONT_FAMILY,
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 6.5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8,
    "pdf.fonttype": 42,
    "figure.constrained_layout.use": True,
}
STRIP_LAYOUT_RC: dict[str, object] = {
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.h_pad": 0.01,
    "figure.constrained_layout.wspace": 0.04,
    "figure.constrained_layout.hspace": 0.14,
}
SINGLE_RC: dict[str, object] = STRIP_RC | {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
}


def _line_label(ax: Axes, x: float, y: float, text: str, color: str, va: str, ha: str = "right") -> None:
    """Name a curve with text placed next to it."""
    ax.text(x, y, text, color=color, fontsize=LABEL_FONTSIZE, ha=ha, va=va, zorder=4)


def make_strategy_accuracy_by_reasoning_steps(
    ax: Axes,
    *,
    n_items: int = strategies.N_ITEMS,
    budgets: Sequence[int] = strategies.BUDGETS,
    d_values: Sequence[int] = strategies.DISTANCES,
    sigma: float = strategies.SIGMA,
) -> dict:
    """Accuracy by distance for the geometric strategy and the budget-limited algebraic walk.

    The reasoning-step budget is drawn uniformly from `budgets`; a walk that
    has ended within it is answered correctly and an unfinished one is a
    guess. An adjacent pair is read from its own premise in zero steps, so the
    algebraic walk is always correct at distance 1.
    """
    d = np.asarray(d_values, dtype=float)
    walks = strategies.enumerate_walks(n_items)
    completion = strategies.completion_probability(walks, d_values, budgets)
    algebraic = strategies.accuracy_by_reasoning_steps(completion).mean(axis=1)
    geometric = strategies.geometric_accuracy(d_values, sigma=sigma)
    ax.axhline(style.CHANCE, **CHANCE_STYLE)
    ax.plot(d, geometric, color=style.GEOMETRIC, **LINE_STYLE)
    ax.plot(d, algebraic, color=style.ALGEBRAIC, **LINE_STYLE)
    _line_label(ax, d[0] + 0.15, 0.60, "geometric", style.GEOMETRIC, va="top", ha="left")
    _line_label(ax, d[-1], algebraic[-1] - 0.018, "algebraic", style.ALGEBRAIC, va="top")
    ax.set_xlabel("Symbolic distance")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(d)
    ax.set_ylim(0.44, 1.05)
    return {
        "d_values": np.asarray(d_values),
        "budgets": np.asarray(budgets),
        "sigma": sigma,
        "geometric_accuracy": geometric,
        "algebraic_accuracy": algebraic,
    }


def make_strategy_cost(
    ax: Axes, *, n_items: int = strategies.N_ITEMS, d_values: Sequence[int] = strategies.DISTANCES
) -> dict:
    """Expected reasoning steps versus distance for each strategy.

    The geometric strategy answers from its built ordering in zero steps at
    every distance; the algebraic walk's expected length is enumerated and
    equals the closed form (n - 1 + d) / 4 beyond the adjacent pair, which is
    read from its own premise in zero steps.
    """
    d = np.asarray(d_values, dtype=float)
    walks = strategies.enumerate_walks(n_items)
    steps = strategies.steps_by_distance(walks, d_values)
    closed_form = np.array([strategies.expected_steps(n_items, int(v)) for v in d_values])
    top = math.ceil(float(steps.max()))
    ax.plot(d, np.zeros_like(d), color=style.GEOMETRIC, **LINE_STYLE)
    ax.plot(d, steps, color=style.ALGEBRAIC, **LINE_STYLE)
    _line_label(ax, d[-1], 0.15, "geometric", style.GEOMETRIC, va="bottom")
    _line_label(ax, d[-1] - 0.15, steps[-1] + 0.1, "algebraic", style.ALGEBRAIC, va="bottom")
    ax.set_xlabel("Symbolic distance")
    ax.set_ylabel("Reasoning steps")
    ax.set_xticks(d)
    ax.set_yticks(np.arange(top + 1))
    ax.set_ylim(-0.2, top + 0.3)
    return {"d_values": np.asarray(d_values), "expected_steps": steps, "closed_form": closed_form}


def _diversity_accuracies(
    *,
    n_items: int,
    skew_alphas: Sequence[float],
    n_orderings: int,
    n_queries: int,
    premise_cost: float,
    margin: float,
    temperature: float,
) -> dict:
    """Both strategies' costs and test accuracies across skew alpha under the cost model."""
    alphas = np.asarray(skew_alphas, dtype=float)
    d_values = np.arange(1, n_items - 2)
    walks = strategies.enumerate_walks(n_items)
    mean_distance = float((d_values * strategies.distance_distribution(walks, d_values)).sum())
    geometric_cost, algebraic_cost = strategies.cost_per_query(
        ordering_cost=premise_cost * (n_items - 1),
        n_queries=n_queries,
        n_orderings=n_orderings,
        skew_alphas=alphas,
        mean_steps=mean_distance,
    )
    iwl_accuracy, icl_accuracy = strategies.strategy_accuracies(
        geometric_cost, algebraic_cost, margin=margin, temperature=temperature
    )
    crossing = int(np.argmin(np.abs(iwl_accuracy - icl_accuracy)))
    return {
        "skew_alphas": alphas,
        "geometric_cost": geometric_cost,
        "algebraic_cost": algebraic_cost,
        "iwl_test_accuracy": iwl_accuracy,
        "icl_test_accuracy": icl_accuracy,
        "crossing_alpha": float(alphas[crossing]),
        "crossing_accuracy": float(iwl_accuracy[crossing]),
    }


def make_ordering_diversity(
    ax: Axes,
    *,
    n_items: int = strategies.N_ITEMS,
    skew_alphas: Sequence[float] = strategies.SKEW_ALPHA_GRID,
    n_orderings: int = strategies.N_ORDERINGS,
    n_queries: int = strategies.N_QUERIES,
    premise_cost: float = strategies.PREMISE_COST,
    margin: float = strategies.MARGIN,
    temperature: float = strategies.TEMPERATURE,
) -> dict:
    """Accuracy of each strategy versus skew alpha under the cost model.

    The axis runs from uniform orderings (alpha 0) to the strongest skew, so
    diversity falls to the right: the geometric strategy's build cost falls
    with it and accuracy on familiar items with an uninformative context (the
    IWL test) rises, while accuracy on novel items whose ordering is given in
    context (the ICL test) falls. `premise_cost` is the cost per premise.
    """
    statistics = _diversity_accuracies(
        n_items=n_items,
        skew_alphas=skew_alphas,
        n_orderings=n_orderings,
        n_queries=n_queries,
        premise_cost=premise_cost,
        margin=margin,
        temperature=temperature,
    )
    alphas = statistics["skew_alphas"]
    alpha_max = float(alphas.max())
    ax.axhline(style.CHANCE, **CHANCE_STYLE)
    ax.plot(alphas, statistics["iwl_test_accuracy"], color=style.GEOMETRIC, lw=1.0, zorder=3)
    ax.plot(alphas, statistics["icl_test_accuracy"], color=style.ALGEBRAIC, lw=1.0, zorder=3)
    ax.set_xlim(0.0, alpha_max)
    ax.set_xticks(np.arange(0, alpha_max + 1e-9, 2.0))
    ax.set_ylim(0.40, 1.05)
    _line_label(ax, 0.01 * alpha_max, 0.485, "familiar items", style.GEOMETRIC, va="top", ha="left")
    _line_label(
        ax,
        0.99 * alpha_max,
        statistics["icl_test_accuracy"][-1] - 0.03,
        "novel items",
        style.ALGEBRAIC,
        va="top",
        ha="right",
    )
    ax.set_xlabel(tables.COLUMN_LABELS["skew_alpha"])
    ax.set_ylabel("Accuracy")
    return statistics


def make_diversity_tradeoff(
    ax: Axes,
    *,
    n_items: int = strategies.N_ITEMS,
    skew_alphas: Sequence[float] = strategies.SKEW_ALPHA_GRID,
    n_orderings: int = strategies.N_ORDERINGS,
    n_queries: int = strategies.N_QUERIES,
    premise_cost: float = strategies.PREMISE_COST,
    margin: float = strategies.MARGIN,
    temperature: float = strategies.TEMPERATURE,
    mark_step: float = 0.5,
) -> dict:
    """Accuracy on novel items against accuracy on familiar items as skew alpha varies.

    The two curves of `make_ordering_diversity` drawn against each other and
    colored by alpha, with a dot every `mark_step` in alpha: a uniform
    distribution gives chance on familiar items and the ceiling on novel ones,
    the strongest skew the ceiling on familiar items and lower accuracy on
    novel ones, and intermediate skews near the ceiling on both.
    """
    statistics = _diversity_accuracies(
        n_items=n_items,
        skew_alphas=skew_alphas,
        n_orderings=n_orderings,
        n_queries=n_queries,
        premise_cost=premise_cost,
        margin=margin,
        temperature=temperature,
    )
    alphas = statistics["skew_alphas"]
    span = (float(alphas.min()), float(alphas.max()))
    iwl, icl = statistics["iwl_test_accuracy"], statistics["icl_test_accuracy"]
    points = np.column_stack([iwl, icl])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    midpoints = (alphas[:-1] + alphas[1:]) / 2
    colors = style.sweep_colors(midpoints, style.SKEW_CMAP, column="skew_alpha", span=span)
    ax.axhline(style.CHANCE, **CHANCE_STYLE)
    ax.axvline(style.CHANCE, **CHANCE_STYLE)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.0, zorder=3))
    marks = np.arange(span[0], span[1] + 1e-9, mark_step)
    mark_iwl = np.interp(marks, alphas, iwl)
    mark_icl = np.interp(marks, alphas, icl)
    mark_colors = style.sweep_colors(marks, style.SKEW_CMAP, column="skew_alpha", span=span)
    ax.scatter(mark_iwl, mark_icl, c=mark_colors, **MARK_STYLE)
    first = style.SKEW_LABEL_FORMAT.format(marks[0])
    last = style.SKEW_LABEL_FORMAT.format(marks[-1])
    _line_label(ax, mark_iwl[0] + 0.03, mark_icl[0] - 0.03, first, mark_colors[0], va="top", ha="left")
    _line_label(ax, mark_iwl[-1] - 0.03, mark_icl[-1], last, mark_colors[-1], va="center")
    ax.set_xlim(0.44, 1.04)
    ax.set_ylim(0.44, 1.04)
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_xlabel("Familiar accuracy", color=style.GEOMETRIC)
    ax.set_ylabel("Novel accuracy", color=style.ALGEBRAIC)
    return statistics | {
        "marked_alphas": marks,
        "marked_iwl_test_accuracy": mark_iwl,
        "marked_icl_test_accuracy": mark_icl,
    }


def make_content_accuracy(
    ax: Axes,
    *,
    n_items: int = strategies.N_ITEMS,
    budgets: Sequence[int] = strategies.BUDGETS,
    d_values: Sequence[int] = strategies.DISTANCES,
    sigma: float = strategies.SIGMA,
) -> dict:
    """Accuracy by distance under each context, for a system capable of both strategies.

    The system walks the presented premises with a reasoning-step budget drawn
    uniformly from `budgets`, as in the accuracy panel, but a walk that has not
    ended answers from the built ordering, read with the geometric strategy's
    rank noise, instead of guessing (`strategies.content_accuracy`). A congruent
    context agrees with the built ordering, so accuracy stays near one; a
    reversed context contradicts it, so accuracy falls with distance, toward the
    chance that the walk has ended; a shuffled context agrees with it on half
    of the queries, which gives the algebraic strategy's accuracy. With an
    uninformative context the premises do not hold the queried items, no walk
    ends, and the built ordering answers every query: the geometric strategy's
    accuracy, which rises with distance.
    """
    d = np.asarray(d_values, dtype=float)
    walks = strategies.enumerate_walks(n_items)
    completion = strategies.completion_probability(walks, d_values, budgets)
    no_walk = np.zeros(len(d_values))
    accuracy = {
        "uninformative": strategies.content_accuracy(
            no_walk, strategies.prior_agreement("congruent", d_values, sigma=sigma)
        )
    }
    for context in strategies.INFORMATIVE_CONTEXTS:
        agreement = strategies.prior_agreement(context, d_values, sigma=sigma)
        accuracy[context] = strategies.content_accuracy(completion, agreement).mean(axis=1)
    ax.axhline(style.CHANCE, **CHANCE_STYLE)
    for context, values in accuracy.items():
        ax.plot(d, values, color=style.CONTEXT_COLORS[context], **LINE_STYLE)
    colors = style.CONTEXT_COLORS
    _line_label(ax, 2.5, accuracy["congruent"][1] + 0.02, "congruent", colors["congruent"],
                va="bottom", ha="center")
    _line_label(ax, d[-1], accuracy["shuffled"][-1] - 0.04, "shuffled", colors["shuffled"], va="top")
    _line_label(ax, d[-1], accuracy["reversed"][-1] - 0.04, "reversed", colors["reversed"], va="top")
    _line_label(ax, d[0] + 0.15, accuracy["uninformative"][0] - 0.03, "uninformative",
                colors["uninformative"], va="top", ha="left")
    ax.set_xlabel("Symbolic distance")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(d)
    ax.set_ylim(0.3, 1.05)
    return {
        "d_values": np.asarray(d_values),
        "budgets": np.asarray(budgets),
        "sigma": sigma,
        **{f"{context}_accuracy": values for context, values in accuracy.items()},
    }


def make_content_cost(
    ax: Axes,
    *,
    n_items: int = strategies.N_ITEMS,
    d_values: Sequence[int] = strategies.DISTANCES,
    dodge: float = 0.09,
) -> dict:
    """Expected reasoning steps by distance under the contexts that hold the queried items.

    With a congruent context the premises show the built ordering itself, which
    answers without a walk, so no reasoning is needed at any distance. With a reversed or a
    shuffled context the answer has to come from the walk, whose expected
    length is the algebraic strategy's cost: under a shuffled context nothing
    tells the queries that agree with the built ordering from those that do
    not, so every query takes the walk. The two lines are equal, so they are
    drawn side by side, the reversed one `dodge` left of each distance and the
    shuffled one `dodge` right of it.
    """
    d = np.asarray(d_values, dtype=float)
    walks = strategies.enumerate_walks(n_items)
    walk_steps = strategies.steps_by_distance(walks, d_values)
    steps = {"congruent": np.zeros_like(walk_steps), "reversed": walk_steps, "shuffled": walk_steps.copy()}
    offsets = {"congruent": 0.0, "reversed": -dodge, "shuffled": dodge}
    top = math.ceil(float(walk_steps.max()))
    for context in strategies.INFORMATIVE_CONTEXTS:
        ax.plot(d + offsets[context], steps[context], color=style.CONTEXT_COLORS[context], **LINE_STYLE)
    _line_label(ax, d[-1], 0.15, "congruent", style.CONGRUENT, va="bottom")
    _line_label(ax, d[-1] - 0.15, walk_steps[-1] + 0.1, "shuffled", style.SHUFFLED, va="bottom")
    _line_label(ax, d[-1], walk_steps[-1] - 0.45, "reversed", style.REVERSED, va="top")
    ax.set_xlabel("Symbolic distance")
    ax.set_ylabel("Reasoning steps")
    ax.set_xticks(d)
    ax.set_yticks(np.arange(top + 1))
    ax.set_ylim(-0.2, top + 0.3)
    return {"d_values": np.asarray(d_values), **{f"{context}_steps": values for context, values in steps.items()}}


def _budget_axes(ax: Axes, budget_values: np.ndarray) -> None:
    """The axes of the panels of accuracy by budget: every budget ticked, accuracy up to one."""
    ax.set_xlabel("Reasoning steps")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(budget_values)
    ax.set_ylim(-0.02, 1.05)


def make_algebraic_accuracy_by_budget(
    ax: Axes,
    *,
    n_items: int = strategies.N_ITEMS,
    budgets: Sequence[int] = strategies.SWEEP_BUDGETS,
    d_values: Sequence[int] = strategies.DISTANCES,
) -> dict:
    """Accuracy versus the reasoning-step budget on novel items, one line per distance.

    The budget-limited walk of the accuracy panel at each fixed budget K: a walk
    that has ended within it is answered correctly and an unfinished one is a
    guess, so distance d follows 0.5 + 0.5 P(L <= K | d). Half the walks head
    toward the target and end at exactly d steps; the other half stop at an end
    of the ordering after 1 to m = n_items - 2 - d steps, equally often, so
    P(L <= K | d) = [K >= d] / 2 + min(K, m) / (2 m). An adjacent pair is read
    from its own premise in zero steps, so distance 1 is right at every budget. The
    default budgets are the transformers' sweep (`strategies.SWEEP_BUDGETS`).
    """
    budget_values = np.asarray(budgets)
    walks = strategies.enumerate_walks(n_items)
    accuracy = strategies.accuracy_by_reasoning_steps(strategies.completion_probability(walks, d_values, budgets))
    ax.axhline(style.CHANCE, **CHANCE_STYLE)
    for shade, row in zip(style.distance_colors(d_values, style.ICL_DISTANCE_CMAP), accuracy, strict=True):
        ax.plot(budget_values, row, color=shade, **LINE_STYLE)
    _line_label(ax, budget_values[-1], 0.30, "algebraic", style.ALGEBRAIC, va="center")
    _budget_axes(ax, budget_values)
    return {"budgets": budget_values, "d_values": np.asarray(d_values), "algebraic_accuracy": accuracy}


def make_content_accuracy_by_budget(
    ax: Axes,
    *,
    n_items: int = strategies.N_ITEMS,
    budgets: Sequence[int] = strategies.SWEEP_BUDGETS,
    d_values: Sequence[int] = strategies.DISTANCES,
    sigma: float = strategies.SIGMA,
) -> dict:
    """Accuracy versus the reasoning-step budget per distance, with a congruent or reversed context.

    A walk that has not ended within the budget answers from the built
    ordering, as in the content accuracy panel, so distance d follows
    P(L <= K | d) + (1 - P(L <= K | d)) Phi(d / sigma) with a congruent context and
    P(L <= K | d) + (1 - P(L <= K | d)) (1 - Phi(d / sigma)) with a reversed one: at
    zero steps the geometric strategy's accuracy or its complement, then rising
    as the walks that stop at an end of the ordering end (after 1 to m steps),
    with a step at K = d, where the walks heading toward the target end, to one
    once every walk has ended. Distance 1, read from its own premise, is right
    at every budget. The congruent lines are drawn over the reversed ones. The
    default budgets are the transformers' sweep (`strategies.SWEEP_BUDGETS`).
    """
    budget_values = np.asarray(budgets)
    walks = strategies.enumerate_walks(n_items)
    completion = strategies.completion_probability(walks, d_values, budgets)
    accuracy = {
        context: strategies.content_accuracy(
            completion, strategies.prior_agreement(context, d_values, sigma=sigma)
        )
        for context in ("reversed", "congruent")
    }
    ax.axhline(style.CHANCE, **CHANCE_STYLE)
    for context, values in accuracy.items():
        shades = style.distance_colors(d_values, style.CONTEXT_DISTANCE_CMAPS[context])
        for shade, row in zip(shades, values, strict=True):
            ax.plot(budget_values, row, color=shade, **LINE_STYLE)
    _line_label(ax, budget_values[-1], 0.88, "congruent", style.CONGRUENT, va="top")
    _line_label(ax, budget_values[-1], 0.30, "reversed", style.REVERSED, va="center")
    _budget_axes(ax, budget_values)
    return {
        "budgets": budget_values,
        "d_values": np.asarray(d_values),
        "sigma": sigma,
        **{f"{context}_accuracy": values for context, values in accuracy.items()},
    }


# Panels in column-major order, one column per prediction.
PANELS: dict[str, Callable[[Axes], dict]] = {
    "accuracy": make_strategy_accuracy_by_reasoning_steps,
    "cost": make_strategy_cost,
    "content_accuracy": make_content_accuracy,
    "content_cost": make_content_cost,
    "diversity": make_ordering_diversity,
    "tradeoff": make_diversity_tradeoff,
}
# Panels saved alone beside the strip's but not drawn in it.
BUDGET_PANELS: dict[str, Callable[[Axes], dict]] = {
    "algebraic_accuracy_by_budget": make_algebraic_accuracy_by_budget,
    "content_accuracy_by_budget": make_content_accuracy_by_budget,
}
N_ROWS: int = 2
COLUMN_TITLES: tuple[str, ...] = ("Distance effects", "Content effects", "Data distribution")


def _pin_tick_label_size(ax: Axes, rc: Mapping[str, object]) -> None:
    """Fix each axis's tick label size from its own key, so the tick count is settled at drawing.

    The automatic tick locator budgets ticks from the label size in effect
    when the figure is rendered; pinning it keeps the panel the same whether
    the figure is saved inside or outside the rc context it was drawn under.
    """
    ax.tick_params(axis="x", labelsize=rc["xtick.labelsize"])
    ax.tick_params(axis="y", labelsize=rc["ytick.labelsize"])


def strip_figure() -> tuple[Figure, list[Axes]]:
    """The 2 x 3 strip at its exact print size, with constrained layout and the strip's fonts.

    The axes are returned in column-major order, the order of `PANELS`.
    """
    n_cols = len(PANELS) // N_ROWS
    with rc_context(STRIP_RC | STRIP_LAYOUT_RC):
        fig = Figure(figsize=STRIP_SIZE_IN)
        grid = fig.add_gridspec(N_ROWS, n_cols)
        axes = [fig.add_subplot(grid[row, col]) for col in range(n_cols) for row in range(N_ROWS)]
    return fig, axes


def build_strip(axes: Sequence[Axes]) -> dict:
    """Draw the six panels onto six axes in column-major order and return their statistics.

    The strip's fonts apply while drawing; each column's top panel carries the
    column's title, set in bold so it reads apart from the axis labels.
    """
    with rc_context(STRIP_RC):
        statistics = {name: make(ax) for (name, make), ax in zip(PANELS.items(), axes)}
        for ax, title in zip(axes[::N_ROWS], COLUMN_TITLES):
            ax.set_title(title, fontweight="bold")
        for ax in axes:
            _pin_tick_label_size(ax, STRIP_RC)
    return statistics


def build_singles(out_dir: Path) -> list[Path]:
    """Save each panel, the strip's then `BUDGET_PANELS`, as a PDF; return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with rc_context(SINGLE_RC):
        for name, make in {**PANELS, **BUDGET_PANELS}.items():
            fig = Figure(figsize=SINGLE_SIZE_IN)
            ax = fig.add_subplot()
            make(ax)
            _pin_tick_label_size(ax, SINGLE_RC)
            path = out_dir / f"formalization_{name}.pdf"
            fig.savefig(path)
            paths.append(path)
    return paths
