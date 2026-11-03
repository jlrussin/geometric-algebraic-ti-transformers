"""Panels over training steps and over swept training settings.

Serve the appendix figures "Learning over pretraining by skew",
"Chain-of-thought supervision improves sample efficiency", "Replay during
finetuning", "Additional data-distribution manipulations", "Architecture and
learning-rate sweep", "Experiments without an explicit cue", "Ablating
context-reading heads during finetuning", and "Ablation during finetuning
across learning rates and proportions of uninformative context trials"; and,
in the figure "Mechanisms underlying the geometric and algebraic strategies",
the transformers' bars that compare finetuning with and without those heads.
Every panel pools accuracy over symbolic distances at least `d_min` within a
seed with trial weights (the IWL test's trained pairs, all adjacent, pool
distance one), after one reasoning-step estimand (`k_stat`: a pinned budget,
the mean over one to five steps, or the best over the sweep), and averages
over seeds only. A line family (`line_col`) is one swept column drawn
one line per value in that column's colormap; every other swept column must be
pinned. The pair-type panels read instead the per-pair rows, each run scored as
it was finetuned (`tables.finetuned_pair_rows`), and show the ICL test and the
IWL test's trained pairs, non-adjacent pairs with the first or last main item,
and non-adjacent interior pairs, each pooled over its pairs by trials within a
distance before the estimand.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, NullLocator, ScalarFormatter
from matplotlib.transforms import offset_copy

from geo_alg_ti import tables
from geo_alg_ti.plots import _frames, style

# Line styles that tell conditions apart inside one line family.
CONDITION_LINE_STYLES: tuple[str, ...] = ("-", "--", ":", "-.")
# The finetuning conditions of the reasoning experiments, keyed by the heads ablated while
# finetuning and the proportion of congruent-context trials on which the network reasons first,
# in the order drawn; random heads, the control, are drawn open and dashed.
REASONING_CONDITIONS: tuple[tuple[str, float], ...] = tables.FINETUNING_CONDITIONS
REASONING_LABELS: Mapping[tuple[str, float], str] = {
    ("none", 0.0): "No reasoning",
    ("none", 1.0): "Reasoning",
    ("context_reading", 1.0): "Reasoning, context-reading heads ablated",
    ("random", 1.0): "Reasoning, random heads ablated",
}
REASONING_COLORS: Mapping[tuple[str, float], str] = {
    ("none", 0.0): style.NO_REASONING,
    ("none", 1.0): style.EDIT_INTACT,
    ("context_reading", 1.0): style.CONTEXT_HEADS,
    ("random", 1.0): style.CONTEXT_HEADS,
}
REASONING_CONTROLS: frozenset[tuple[str, float]] = frozenset({("random", 1.0)})
# The ICL test, then the IWL test's pairs by type.
PAIR_TYPES: tuple[str, ...] = tables.PAIR_TYPES
PAIR_TYPE_LABELS: Mapping[str, str] = {"icl_test": "ICL test", "trained": "Trained", "end_item": "End item", "interior": "Interior"}
IWL_GROUP_LABEL: str = "IWL test"
FINETUNING_STEP_LABEL: str = "Finetuning step"
PAIR_TYPE_BAR_WIDTH: float = 0.19
# The extra space after the ICL test, which sets it apart from the IWL test's pairs.
PAIR_TYPE_GAP: float = 0.25
# The second row of labels under the IWL test's pairs: its top this far below the axis (points),
# its bracket this far above its top (points) and extending this far (data units) beyond the
# centers of the IWL test's first and last measures.
GROUP_ROW_OFFSET_PT: float = 16.0
GROUP_BRACKET_DROP_PT: float = 1.5
GROUP_BRACKET_INSET: float = 0.42
# The bars' key sits above the panel.
PAIR_TYPE_KEY: Mapping[str, object] = {"bbox_to_anchor": (0.5, 1.0), "frameon": False, "borderaxespad": 0.2, "ncol": 2}
_HEADS_COLUMN = tables.HEADS_COLUMN
_REASONING_COLUMN = tables.REASONING_COLUMN
_PAIR_TYPE_COLUMN = tables.PAIR_TYPE_COLUMN


def _kilo(value: float, _position: object) -> str:
    """Training steps in thousands: 0, 50k, 100k."""
    return "0" if value == 0 else f"{value / 1000:g}k"


def value_label(line_col: str | None, value: object) -> str:
    """The legend entry of one line of a family: a number written out in full, never in powers."""
    if line_col is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return np.format_float_positional(number, trim="-")


def _pooled_by_step(df: pd.DataFrame, *, k_stat: str, k_pin: int | None, k_values: Sequence[int], d_min: int, fname: str) -> pd.DataFrame:
    """Per-seed, per-step accuracy pooled over distance after the reasoning-step estimand."""
    reduced = _frames.reduce_steps(df, k_stat=k_stat, k_pin=k_pin, k_values=k_values, group_cols=["seed", "step", "d"], fname=fname)
    return _frames.pool(reduced, group_cols=["seed", "step"], d_min=d_min)


def _step_axis(ax: Axes, steps: Sequence[int], *, xticks: Sequence[float] | None, xlim: tuple[float, float] | None) -> None:
    """Training steps on the x axis, in thousands from ten thousand steps up.

    `xticks` names the ticks; `xlim` pins the range, which otherwise follows the drawn steps
    with the axes margins, widened to any tick outside them.
    """
    if max(steps) >= 10_000:
        ax.xaxis.set_major_formatter(FuncFormatter(_kilo))
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if xlim is not None:
        ax.set_xlim(*xlim)


def make_accuracy_by_step(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str,
    k_stat: str = "best",
    k_pin: int | None = None,
    k_values: Sequence[int] = _frames.DEFAULT_MEAN_BUDGETS,
    d_min: int = 2,
    line_col: str | None = None,
    cmap: str | None = None,
    colors: Mapping[object, str] | None = None,
    line_styles: Mapping[object, str] | None = None,
    line_labels: Mapping[object, str] | None = None,
    color: str | None = None,
    label: str | None = None,
    show_chance: bool = True,
    show_band: bool = True,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    xticks: Sequence[float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = (0.4, 1.02),
    legend_loc: str = "lower right",
    legend_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """Accuracy pooled over distance versus training step, one line per value of `line_col`.

    Estimand: `k_stat` over reasoning steps within each seed, step, and distance; the
    trial-weighted pool over distances at least `d_min` within each seed and step; then the mean
    and standard error over seeds at each step. Without a line family the line takes the
    condition's color, or `color`. A family value named in `line_styles` or `line_labels`
    takes that style or legend entry. `xticks` names the step ticks and `xlim` pins the step
    range; left at None the range follows the drawn steps with the axes margins, widened to
    any tick outside them.
    """
    fname = "make_accuracy_by_step"
    rows = tables.select(df, fname=fname, metric="accuracy", condition=condition)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, allowed=("step", line_col), fname=fname)
    values, shades = _frames.line_family(rows, line_col, cmap, colors, fname=fname)
    if line_col is None:
        shades = {None: style.CONDITION_COLORS.get(condition, style.REFERENCE_LINE) if color is None else color}
    styles = dict(line_styles or {})
    names = dict(line_labels or {})
    lines: dict[str, dict] = {}
    seeds = 0
    steps: set[int] = set()
    for value in values:
        sub = rows if line_col is None else tables.select(rows, fname=fname, **{line_col: value})
        pooled = _pooled_by_step(sub, k_stat=k_stat, k_pin=k_pin, k_values=k_values, d_min=d_min, fname=fname)
        agg = _frames.summarize(pooled, group_cols=["step"])
        if line_col is None:
            name = style.CONDITION_LABELS.get(condition, condition) if label is None else label
        else:
            name = names.get(value, value_label(line_col, value))
        _frames.draw_series(ax, agg["step"], agg["value_mean"], agg["value_se"], color=shades[value], label=name, marker=None, ls=styles.get(value, "-"), band=show_band, band_alpha=band_alpha)
        lines[name] = {"mean_by_step": _frames.by_key(agg, "step", "value_mean"), "se_by_step": _frames.by_key(agg, "step", "value_se")}
        seeds = max(seeds, _frames.n_seeds(agg))
        steps.update(agg["step"].astype(int).tolist())
    _step_axis(ax, sorted(steps), xticks=xticks, xlim=xlim)
    _frames.finish(ax, x="step", y="value", ylim=ylim, chance=show_chance)
    if show_legend and line_col is not None:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, title=style.AXIS_LABELS.get(line_col, line_col))
    return {
        "condition": condition,
        "k_stat": k_stat,
        "k_pin": None if k_pin is None else int(k_pin),
        "d_min": int(d_min),
        "line_col": line_col,
        "lines": lines,
        "n_seeds": seeds,
    }


def make_reasoning_gain_by_step(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str,
    low_k: int = 0,
    high_k_stat: str = "best",
    k_values: Sequence[int] = _frames.DEFAULT_MEAN_BUDGETS,
    d_min: int = 2,
    line_col: str | None = None,
    cmap: str | None = None,
    colors: Mapping[object, str] | None = None,
    color: str | None = None,
    show_band: bool = True,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    xticks: Sequence[float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend_loc: str = "upper left",
    legend_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """Best-over-budgets accuracy minus accuracy at `low_k`, versus training step.

    Estimand: within each seed and step, the pooled accuracy (distances at least `d_min`,
    trial weights) under `high_k_stat` minus the pooled accuracy at the pinned budget `low_k`;
    then the mean and standard error over seeds. The dashed reference marks no gain. `xticks`
    names the step ticks and `xlim` pins the step range; left at None the range follows the
    drawn steps with the axes margins, widened to any tick outside them.
    """
    fname = "make_reasoning_gain_by_step"
    rows = tables.select(df, fname=fname, metric="accuracy", condition=condition)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, allowed=("step", line_col), fname=fname)
    values, shades = _frames.line_family(rows, line_col, cmap, colors, fname=fname)
    if line_col is None:
        shades = {None: style.CONDITION_COLORS.get(condition, style.REFERENCE_LINE) if color is None else color}
    lines: dict[str, dict] = {}
    seeds = 0
    steps: set[int] = set()
    for value in values:
        sub = rows if line_col is None else tables.select(rows, fname=fname, **{line_col: value})
        high = _pooled_by_step(sub, k_stat=high_k_stat, k_pin=None, k_values=k_values, d_min=d_min, fname=fname)
        low = _pooled_by_step(sub, k_stat="pin", k_pin=low_k, k_values=k_values, d_min=d_min, fname=fname)
        merged = high.merge(low, on=["seed", "step"], suffixes=("_high", "_low"))
        if len(merged) != len(high) or len(merged) != len(low):
            raise ValueError(f"{fname}: the two estimands do not share the same seeds and steps")
        merged["value"] = merged["value_high"] - merged["value_low"]
        agg = _frames.summarize(merged, group_cols=["step"])
        name = style.CONDITION_LABELS.get(condition, condition) if line_col is None else value_label(line_col, value)
        _frames.draw_series(ax, agg["step"], agg["value_mean"], agg["value_se"], color=shades[value], label=name, marker=None, band=show_band, band_alpha=band_alpha)
        lines[name] = {"mean_by_step": _frames.by_key(agg, "step", "value_mean"), "se_by_step": _frames.by_key(agg, "step", "value_se")}
        seeds = max(seeds, _frames.n_seeds(agg))
        steps.update(agg["step"].astype(int).tolist())
    _step_axis(ax, sorted(steps), xticks=xticks, xlim=xlim)
    ax.axhline(0.0, color=style.REFERENCE_LINE, ls="--", lw=style.AXES_LINEWIDTH_PT, zorder=1)
    _frames.finish(ax, x="step", y="gain", ylim=ylim)
    if show_legend and line_col is not None:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, title=style.AXIS_LABELS.get(line_col, line_col))
    return {
        "condition": condition,
        "low_k": int(low_k),
        "high_k_stat": high_k_stat,
        "d_min": int(d_min),
        "line_col": line_col,
        "lines": lines,
        "n_seeds": seeds,
    }


def _condition_setting(condition: str, k_stat: Mapping[str, str] | str, k_pin: Mapping[str, int] | int | None) -> tuple[str, int | None]:
    """One condition's reasoning-step estimand and pinned budget, per condition or shared."""
    stat = k_stat[condition] if isinstance(k_stat, Mapping) else k_stat
    pin = k_pin.get(condition) if isinstance(k_pin, Mapping) else k_pin
    return stat, (0 if stat == "pin" and pin is None else pin)


def make_pooled_accuracy_vs_sweep(
    ax: Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    x_values: Sequence[object] | None = None,
    conditions: Sequence[str] = ("iwl_test", "icl_test"),
    k_stat: Mapping[str, str] | str = "best",
    k_pin: Mapping[str, int] | int | None = None,
    k_values: Sequence[int] = _frames.DEFAULT_MEAN_BUDGETS,
    d_min: int = 2,
    line_col: str | None = None,
    log_x: bool = False,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    line_colors: Mapping[object, str] | None = None,
    line_labels: Mapping[object, str] | None = None,
    line_styles: Mapping[object, str] | None = None,
    line_markers: Mapping[object, str] | None = None,
    line_open_markers: Collection[object] = (),
    cmap: str | None = None,
    show_chance: bool = True,
    show_band: bool = True,
    show_seeds: bool = True,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    marker_size: float = _frames.MARKER_SIZE,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    legend_loc: str = "lower right",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
) -> dict:
    """Accuracy pooled over distance versus a swept setting, one line per condition.

    Estimand: `k_stat` over reasoning steps (per condition when a mapping; a pinned budget
    defaults to zero) within each seed, setting, and distance; the trial-weighted pool over
    distances at least `d_min`; then the mean and standard error over seeds at each setting.
    A numeric setting is drawn at its value (`log_x` for a log axis); a categorical one at
    evenly spaced positions, in the order of `x_values` when given (which may also leave some
    settings out). With `line_col`, each condition is drawn once per value of that column,
    colored by `line_colors` or the column's colormap and told apart by line style; a value
    named in `line_labels`, `line_styles`, or `line_markers` takes that legend entry, line
    style, or marker, and one in `line_open_markers` is drawn with open markers (its seed dots
    take the marker and its open face too); with one condition the entry stands alone. On a
    log axis the seed dots are jittered by a fraction of their value.
    """
    fname = "make_pooled_accuracy_vs_sweep"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, allowed=(x_col, line_col), fname=fname)
    if x_col not in rows.columns:
        raise ValueError(f"{fname}: the rows carry no column '{x_col}'")
    colors = {**style.CONDITION_COLORS, **(colors or {})}
    labels = {**style.CONDITION_LABELS, **(labels or {})}
    present = sorted(rows[x_col].dropna().unique(), key=style._sort_key)
    if x_values is None:
        x_values = present
    else:
        missing = [v for v in x_values if v not in set(present)]
        if missing:
            raise ValueError(f"{fname}: the rows hold no {x_col} values {missing}")
        rows = tables.select(rows, fname=fname, **{x_col: list(x_values)})
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in x_values)
    positions = {v: float(v) for v in x_values} if numeric else _frames.ordinal_axis(ax, x_values, order=list(x_values))
    family, shades = _frames.line_family(rows, line_col, cmap, line_colors, fname=fname)
    names = dict(line_labels or {})
    styles = dict(line_styles or {})
    markers = dict(line_markers or {})
    rng = np.random.default_rng(0)
    series: dict[str, dict] = {}
    seeds = 0
    trials = 0
    for style_index, condition in enumerate(conditions):
        stat, pin = _condition_setting(condition, k_stat, k_pin)
        for value in family:
            sub = tables.select(rows, fname=fname, condition=condition)
            if line_col is not None:
                sub = tables.select(sub, fname=fname, **{line_col: value})
            reduced = _frames.reduce_steps(sub, k_stat=stat, k_pin=pin, k_values=k_values, group_cols=[x_col, "seed", "d"], fname=fname)
            pooled = _frames.pool(reduced, group_cols=[x_col, "seed"], d_min=d_min)
            agg = _frames.summarize(pooled, group_cols=[x_col])
            color = colors.get(condition, style.REFERENCE_LINE) if line_col is None else shades[value]
            condition_name = labels.get(condition, condition)
            if line_col is None:
                ls, name = "-", condition_name
            else:
                ls = styles.get(value, CONDITION_LINE_STYLES[style_index % len(CONDITION_LINE_STYLES)])
                value_name = names.get(value, value_label(line_col, value))
                name = value_name if len(conditions) == 1 and value in names else f"{condition_name}, {value_name}"
            xs = [positions[v] for v in agg[x_col]]
            is_open = value in line_open_markers
            _frames.draw_series(ax, xs, agg["value_mean"], agg["value_se"], color=color, label=name, ls=ls, marker=markers.get(value, "o"), marker_size=marker_size, open_marker=is_open, band=show_band, band_alpha=band_alpha)
            if show_seeds:
                _frames.draw_seed_points(ax, pooled, x_col=x_col, value_col="value", color=color, positions=positions, open_marker=is_open, marker=markers.get(value, "o"), rng=rng, jitter=jitter, log_x=(numeric and log_x), point_size=point_size)
            series[name] = {
                "condition": condition,
                "k_stat": stat,
                "k_pin": None if stat != "pin" else int(pin),
                "mean_by_x": _frames.by_key(agg, x_col, "value_mean"),
                "se_by_x": _frames.by_key(agg, x_col, "value_se"),
            }
            seeds = max(seeds, _frames.n_seeds(agg))
            trials = int(pooled["n"].max())
    if numeric:
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks([float(v) for v in x_values])
    _frames.finish(ax, x=x_col, y="value", ylim=ylim, chance=show_chance)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, title=None if line_col is None or names else style.AXIS_LABELS.get(line_col, line_col))
    return {
        "x_col": x_col,
        "x_values": [_frames._plain(v) for v in x_values],
        "conditions": list(conditions),
        "step": step,
        "d_min": int(d_min),
        "line_col": line_col,
        "n_trials_per_seed": trials,
        "series": series,
        "n_seeds": seeds,
    }


def _budget_label(k_stat: str, budgets: Sequence[int]) -> str:
    """The accuracy axis label naming the reasoning-step estimand."""
    label = style.AXIS_LABELS["value"]
    if k_stat == "best":
        return f"{label} (best of {min(budgets)}–{max(budgets)} steps)"
    if k_stat == "mean":
        return f"{label} (mean over {min(budgets)}–{max(budgets)} steps)"
    return f"{label} ({budgets[0]} step{'' if budgets[0] == 1 else 's'})"


def _reasoning_rows(rows: pd.DataFrame, condition: tuple[str, float], *, fname: str) -> pd.DataFrame:
    """The rows of one finetuning condition."""
    heads, proportion = condition
    return tables.select(rows, fname=fname, **{_HEADS_COLUMN: heads, _REASONING_COLUMN: float(proportion)})


def _condition_labels(conditions: Sequence[tuple[str, float]], names: Mapping[tuple[str, float], str], *, fname: str) -> dict[tuple[str, float], str]:
    """Each condition's label, in order; raises when two conditions share a label, which would
    merge their statistics."""
    keys = [(str(condition[0]), float(condition[1])) for condition in conditions]
    labels = [names[key] for key in keys]
    if len(set(labels)) < len(labels):
        raise ValueError(f"{fname}: every condition needs a label of its own, not {labels}")
    return dict(zip(keys, labels, strict=True))


def _require_same_seeds(seed_sets: Mapping[str, set], *, fname: str) -> None:
    """Raise unless every drawn statistic averages the same networks."""
    distinct = {frozenset(seeds) for seeds in seed_sets.values()}
    if len(distinct) > 1:
        coverage = {name: sorted(seeds) for name, seeds in seed_sets.items()}
        raise ValueError(f"{fname}: the conditions cover different seeds: {coverage}")


def _group_row(ax: Axes, text: str, first: float, last: float, *, offset_pt: float) -> None:
    """A second row of x labels: `text` centered under the bars from `first` to `last`, with a
    thin bracket across them."""
    blend = ax.get_xaxis_transform()
    ax.annotate(text, xy=((first + last) / 2.0, 0.0), xycoords=blend, xytext=(0.0, -offset_pt), textcoords="offset points", ha="center", va="top", fontsize=style.FONT_TICK_PT, annotation_clip=False)
    line_at = offset_copy(blend, fig=ax.figure, y=-(offset_pt - GROUP_BRACKET_DROP_PT), units="points")
    ax.plot([first - GROUP_BRACKET_INSET, last + GROUP_BRACKET_INSET], [0.0, 0.0], transform=line_at, color=style.REFERENCE_LINE, lw=style.AXES_LINEWIDTH_PT, clip_on=False, solid_capstyle="butt")


def _split_axis(ax: Axes, sections: Sequence[tuple[float, float]], *, chance: bool) -> None:
    """The x axis, and the chance line, drawn as one piece under each section of the panel; the
    axis pieces sit at the bottom of the axes whatever y limits are set later."""
    ax.spines["bottom"].set_visible(False)
    for low, high in sections:
        ax.plot([low, high], [0.0, 0.0], transform=ax.get_xaxis_transform(), color=ax.spines["left"].get_edgecolor(), lw=style.AXES_LINEWIDTH_PT, solid_capstyle="butt", clip_on=False, zorder=_frames.LINE_ZORDER + 1)
        if chance:
            ax.plot([low, high], [tables.CHANCE_ACCURACY, tables.CHANCE_ACCURACY], color=style.REFERENCE_LINE, ls="--", lw=style.AXES_LINEWIDTH_PT, zorder=1)


def make_pair_type_bars(
    ax: Axes,
    rows: pd.DataFrame,
    *,
    conditions: Sequence[tuple[str, float]] = REASONING_CONDITIONS,
    labels: Mapping[tuple[str, float], str] | None = None,
    colors: Mapping[tuple[str, float], str] | None = None,
    controls: Collection[tuple[str, float]] = REASONING_CONTROLS,
    measures: Sequence[str] = PAIR_TYPES,
    measure_labels: Mapping[str, str] | None = None,
    k_stat: str = "best",
    k_pin: int | None = None,
    k_values: Sequence[int] = _frames.DEFAULT_MEAN_BUDGETS,
    d_min: int = 2,
    step_pin: int | None = None,
    show_chance: bool = True,
    show_seeds: bool = True,
    show_legend: bool = True,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    bar_width: float = PAIR_TYPE_BAR_WIDTH,
    gap: float = PAIR_TYPE_GAP,
    group_label: str | None = IWL_GROUP_LABEL,
    group_offset_pt: float = GROUP_ROW_OFFSET_PT,
    split_axis: bool = False,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    ylabel: str | None = None,
    legend_loc: str = "lower center",
    legend_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """Grouped bars per measure, one bar per finetuning condition: the ICL test, then the IWL
    test's trained pairs, pairs with an end item, and interior pairs.

    `rows` are an experiment's per-pair rows with each run scored as it was finetuned
    (`tables.finetuned_pair_rows`): a run that ablated heads while it finetuned is scored with
    the same heads still ablated, and every other run intact. Each condition is a pair (heads
    ablated while finetuning, proportion of congruent-context trials that reason first).
    Estimand, at one evaluation step (the last unless `step_pin`): within each seed, the
    measure's pairs pooled by trials at each distance and budget; `k_stat` over reasoning steps
    within each distance; the trial-weighted pool over distances at least `d_min` (distance one
    for the trained pairs); then the mean and standard error over seeds, with one dot per seed
    drawn over the bars. The ICL test stands apart from the IWL test's measures, which a second
    row of labels names (`group_label`, `group_offset_pt` points below the axis; None omits
    it); `split_axis` breaks the x axis and the chance line between the two. The conditions in
    `controls` are open bars in their color. Every bar must average the same networks, and every
    condition needs a label of its own. The key sits above the panel (`PAIR_TYPE_KEY`);
    `legend_kwargs` overrides its placement, anchor included.
    """
    fname = "make_pair_type_bars"
    rows, step = _frames.pin_step(rows, step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.PAIR_SCHEMA, allowed=(_HEADS_COLUMN, _REASONING_COLUMN, _PAIR_TYPE_COLUMN), fname=fname)
    names = {**REASONING_LABELS, **dict(labels or {})}
    shades = {**REASONING_COLORS, **dict(colors or {})}
    ticks = {**PAIR_TYPE_LABELS, **dict(measure_labels or {})}
    centers = np.arange(len(measures), dtype=float)
    if gap and measures and measures[0] == "icl_test":
        centers[1:] += gap
    rng = np.random.default_rng(0)
    bars: dict[str, dict] = {}
    seed_sets: dict[str, set] = {}
    trials: dict[str, int] = {}
    seeds = 0
    budgets = _frames.budgets_used(rows, k_stat, k_pin, k_values)
    labeled = _condition_labels(conditions, names, fname=fname)
    for index, (key, name) in enumerate(labeled.items()):
        color = shades[key]
        is_open = key in set(controls)
        offset = (index - (len(conditions) - 1) / 2.0) * bar_width
        sub = _reasoning_rows(rows, key, fname=fname)
        means, errors, per_measure = [], [], {}
        for center, measure in zip(centers, measures, strict=True):
            per_seed = tables.pair_type_by_seed(sub, measure, k_stat=k_stat, k_pin=k_pin, k_values=k_values, d_min=d_min, fname=fname)
            agg = _frames.summarize(per_seed.assign(measure=measure), group_cols=["measure"])
            mean, se = float(agg["value_mean"].iloc[0]), float(agg["value_se"].iloc[0])
            means.append(mean)
            errors.append(0.0 if np.isnan(se) else se)
            if show_seeds:
                _frames.draw_seed_points(ax, per_seed.assign(x=center + offset), x_col="x", value_col="value", color=color, rng=rng, jitter=jitter, point_size=point_size, over_bars=True)
            per_measure[measure] = {
                "mean": round(mean, 4),
                "se": None if np.isnan(se) else round(se, 4),
                "per_seed": _frames.by_key(per_seed, "seed", "value"),
                "n_trials_per_seed": int(per_seed["n"].max()),
            }
            seed_sets[f"{name}, {measure}"] = set(per_seed["seed"].astype(int))
            trials[measure] = max(trials.get(measure, 0), int(per_seed["n"].max()))
            seeds = max(seeds, _frames.n_seeds(agg))
        ax.bar(
            centers + offset,
            means,
            width=bar_width,
            yerr=errors,
            color=style.OPEN_MARKER_FACE if is_open else color,
            edgecolor=color,
            linewidth=style.AXES_LINEWIDTH_PT,
            error_kw={"ecolor": color, "lw": style.AXES_LINEWIDTH_PT, "capsize": _frames.ERRORBAR_CAPSIZE, "capthick": style.AXES_LINEWIDTH_PT},
            label=name,
            zorder=_frames.LINE_ZORDER,
        )
        bars[name] = per_measure
    _require_same_seeds(seed_sets, fname=fname)
    ax.set_xticks(centers)
    ax.set_xticklabels([ticks[measure] for measure in measures])
    ax.set_xlim(centers[0] - 0.5, centers[-1] + 0.5)
    iwl = [center for center, measure in zip(centers, measures, strict=True) if measure != "icl_test"]
    split = split_axis and bool(iwl) and measures[0] == "icl_test"
    _frames.finish(ax, x=None, y="value", ylim=ylim, chance=show_chance and not split)
    ax.set_ylabel(_budget_label(k_stat, budgets) if ylabel is None else ylabel)
    if split:
        _split_axis(ax, [(centers[0] - 0.5, centers[0] + 0.5), (iwl[0] - 0.5, iwl[-1] + 0.5)], chance=show_chance)
    if group_label is not None and iwl:
        _group_row(ax, group_label, iwl[0], iwl[-1], offset_pt=group_offset_pt)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, **PAIR_TYPE_KEY)
    return {
        "conditions": list(labeled.values()),
        "measures": list(measures),
        "bars": bars,
        "k_stat": k_stat,
        "k_pin": None if k_stat != "pin" else (0 if k_pin is None else int(k_pin)),
        "budgets": budgets,
        "n_trials_per_seed": trials,
        "d_min": int(d_min),
        "step": step,
        "n_seeds": seeds,
    }


def make_pair_type_by_step(
    ax: Axes,
    rows: pd.DataFrame,
    *,
    measure: str = "interior",
    conditions: Sequence[tuple[str, float]] = REASONING_CONDITIONS,
    labels: Mapping[tuple[str, float], str] | None = None,
    colors: Mapping[tuple[str, float], str] | None = None,
    controls: Collection[tuple[str, float]] = REASONING_CONTROLS,
    k_stat: str = "best",
    k_pin: int | None = None,
    k_values: Sequence[int] = _frames.DEFAULT_MEAN_BUDGETS,
    d_min: int = 2,
    show_chance: bool = True,
    show_band: bool = True,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    xticks: Sequence[float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = (0.3, 1.02),
    ylabel: str | None = None,
    legend_loc: str = "lower right",
    legend_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """One measure of `make_pair_type_bars` versus finetuning step, one line per condition.

    The rows, the scoring of each run, and the estimand are those of `make_pair_type_bars`,
    taken at every evaluation step; the line is the mean over seeds and the band one standard
    error. The conditions in `controls` are dashed; every line must average the same networks at
    every step, and every condition needs a label of its own.
    `xticks` names the step ticks and `xlim` pins the step range.
    """
    fname = "make_pair_type_by_step"
    _frames.refuse_unpinned(rows, schema=tables.PAIR_SCHEMA, allowed=("step", _HEADS_COLUMN, _REASONING_COLUMN, _PAIR_TYPE_COLUMN), fname=fname)
    names = {**REASONING_LABELS, **dict(labels or {})}
    shades = {**REASONING_COLORS, **dict(colors or {})}
    lines: dict[str, dict] = {}
    seed_sets: dict[str, set] = {}
    seeds = 0
    steps: set[int] = set()
    for key, name in _condition_labels(conditions, names, fname=fname).items():
        sub = _reasoning_rows(rows, key, fname=fname)
        per_seed = tables.pair_type_by_seed(sub, measure, group_cols=("step",), k_stat=k_stat, k_pin=k_pin, k_values=k_values, d_min=d_min, fname=fname)
        agg = _frames.summarize(per_seed, group_cols=["step"])
        ls = "--" if key in set(controls) else "-"
        _frames.draw_series(ax, agg["step"], agg["value_mean"], agg["value_se"], color=shades[key], label=name, marker=None, ls=ls, band=show_band, band_alpha=band_alpha)
        lines[name] = {"mean_by_step": _frames.by_key(agg, "step", "value_mean"), "se_by_step": _frames.by_key(agg, "step", "value_se")}
        for step, at_step in per_seed.groupby("step"):
            seed_sets[f"{name}, step {int(step)}"] = set(at_step["seed"].astype(int))
        seeds = max(seeds, _frames.n_seeds(agg))
        steps.update(agg["step"].astype(int).tolist())
    _require_same_seeds(seed_sets, fname=fname)
    budgets = _frames.budgets_used(rows, k_stat, k_pin, k_values)
    _step_axis(ax, sorted(steps), xticks=xticks, xlim=xlim)
    _frames.finish(ax, x="step", y="value", ylim=ylim, chance=show_chance)
    ax.set_xlabel(FINETUNING_STEP_LABEL)
    ax.set_ylabel(_budget_label(k_stat, budgets) if ylabel is None else ylabel)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs)
    return {
        "measure": measure,
        "conditions": list(lines),
        "lines": lines,
        "k_stat": k_stat,
        "k_pin": None if k_stat != "pin" else (0 if k_pin is None else int(k_pin)),
        "budgets": budgets,
        "d_min": int(d_min),
        "n_seeds": seeds,
    }
