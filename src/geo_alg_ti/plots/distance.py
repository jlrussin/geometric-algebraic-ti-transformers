"""Panels over symbolic distance and reasoning steps for one network.

Serve the figure "Symbolic distance effects (SDE)" and the appendix figure
"Additional symbolic-distance results": accuracy by distance for the IWL test
(zero steps) and the ICL test (mean over one to five steps) on the same
network; accuracy by reasoning steps with one line per distance; the first
budget reaching ninety percent of each seed's maximum by distance; accuracy at
one budget pooled over distances, one bar per condition or one number beside
each condition's line; per-pair accuracy; and the two response classes of the
arbitrage test as reasoning steps grow. Every panel reads the `accuracy` rows
of the evaluation table at one evaluation step (the last, unless `step_pin`
names another) and averages over seeds only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

from geo_alg_ti import tables
from geo_alg_ti.plots import _frames, placement, style

PAIR_LETTERS: str = "ABCDEFGHIJKLMNOP"
PAIR_BLOCK_FORMAT: str = "Distance {:d}"
# The line and cap width, in points, of the standard-error bars of the per-pair panel.
ERRORBAR_LW: float = 0.8
# The width of each bar of the accuracy bars, as a fraction of the space between bar centers,
# and the color of their error bars, dark so an interval shows on any bar's color.
BAR_WIDTH: float = 0.7
ERROR_COLOR: str = "#333333"
# The prefix of the id that tags a condition's unreached crosses on the steps-to-criterion panel.
UNREACHED_GID: str = "unreached:"


def _steps_setting(condition: str, k_pin: object, k_mix: object) -> tuple[str, int | None, tuple[int, ...]]:
    """How one condition's reasoning steps are reduced: (`k_stat`, `k_pin`, `k_values`).

    A mapping in `k_mix` or `k_pin` names conditions; a bare sequence or integer applies to
    every condition; a condition named in neither follows the paper: the IWL test at zero
    steps and every other condition averaged over one to five steps.
    """
    if isinstance(k_mix, Mapping) and condition in k_mix:
        return "mean", None, tuple(int(k) for k in k_mix[condition])
    if isinstance(k_pin, Mapping) and condition in k_pin:
        return "pin", int(k_pin[condition]), ()
    bare_mix = k_mix is not None and not isinstance(k_mix, Mapping)
    bare_pin = k_pin is not None and not isinstance(k_pin, Mapping)
    if bare_mix and bare_pin:
        raise ValueError("k_pin and k_mix both name every condition; give one of them, or mappings")
    if bare_mix:
        return "mean", None, tuple(int(k) for k in k_mix)
    if bare_pin:
        return "pin", int(k_pin), ()
    if condition == "iwl_test":
        return "pin", 0, ()
    return "mean", None, _frames.DEFAULT_MEAN_BUDGETS


def make_accuracy_by_distance(
    ax: Axes,
    df: pd.DataFrame,
    *,
    conditions: Sequence[str] = ("iwl_test", "icl_test"),
    k_pin: Mapping[str, int] | int | None = None,
    k_mix: Mapping[str, Sequence[int]] | Sequence[int] | None = None,
    d_values: Sequence[int] | None = None,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    show_chance: bool = True,
    show_band: bool = True,
    show_seeds: bool = False,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    marker_size: float = _frames.MARKER_SIZE,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    yticks: Sequence[float] | None = None,
    legend_loc: str = "lower right",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
) -> dict:
    """Accuracy versus symbolic distance, mean and standard error over seeds, per condition.

    Estimand: within each seed and distance the reasoning steps are pinned (`k_pin`) or
    averaged over a set of budgets (`k_mix`); left at None, the IWL test is pinned at zero
    steps and every other condition is averaged over one to five steps. Then the mean and
    standard error over seeds. `yticks` fixes the accuracy ticks (automatic when None).
    """
    fname = "make_accuracy_by_distance"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, fname=fname)
    colors = {**style.CONDITION_COLORS, **(colors or {})}
    labels = {**style.CONDITION_LABELS, **(labels or {})}
    rng = np.random.default_rng(0)
    per_condition: dict[str, dict] = {}
    seeds = 0
    distances: set[int] = set()
    for condition in conditions:
        sub = tables.select(rows, fname=fname, condition=condition)
        if d_values is not None:
            sub = _frames.select_distances(sub, d_values, fname=fname)
        k_stat, pin, k_values = _steps_setting(condition, k_pin, k_mix)
        per_seed = _frames.reduce_steps(sub, k_stat=k_stat, k_pin=pin, k_values=k_values, group_cols=["d", "seed"], fname=fname)
        agg = _frames.summarize(per_seed, group_cols=["d"])
        color = colors.get(condition, style.REFERENCE_LINE)
        _frames.draw_series(ax, agg["d"], agg["value_mean"], agg["value_se"], color=color, label=labels.get(condition, condition), marker_size=marker_size, band=show_band, band_alpha=band_alpha)
        if show_seeds:
            _frames.draw_seed_points(ax, per_seed, x_col="d", value_col="value", color=color, rng=rng, jitter=jitter, point_size=point_size)
        per_condition[condition] = {
            "k_stat": k_stat,
            "k_values": _frames.budgets_used(sub, k_stat, pin, k_values),
            "mean_by_distance": _frames.by_key(agg, "d", "value_mean"),
            "se_by_distance": _frames.by_key(agg, "d", "value_se"),
        }
        seeds = max(seeds, _frames.n_seeds(agg))
        distances.update(agg["d"].astype(int).tolist())
    ax.set_xticks(sorted(distances))
    _frames.finish(ax, x="d", y="value", ylim=ylim, chance=show_chance)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs)
    return {"conditions": list(conditions), "step": step, "per_condition": per_condition, "n_seeds": seeds}


def make_accuracy_by_reasoning_steps(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str = "icl_test",
    d_values: Sequence[int] = (1, 2, 3, 4, 5, 6),
    k_values: Sequence[int] = tuple(range(0, 9)),
    k_stat: str = "raw",
    cmap: str = style.ICL_DISTANCE_CMAP,
    show_chance: bool = True,
    show_band: bool = True,
    show_seeds: bool = False,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    marker_size: float = _frames.MARKER_SIZE,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    xlim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
    yticks: Sequence[float] | None = None,
    ylabel: str | None = None,
    legend_loc: str = "lower right",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
) -> dict:
    """Accuracy versus reasoning steps, one line per symbolic distance.

    Estimand: none over reasoning steps, which are the x axis (`k_stat="raw"`), or the
    running best over the budgets of `k_values` up to each budget within a seed and distance
    (`"running_best"`); then the mean and standard error over seeds at each distance and
    budget. Budgets in `k_values` that the rows lack are left out and reported. `xticks` (every
    budget when None), `xlim` and `yticks` (automatic when None) fix the axes; `ylabel` replaces
    the accuracy label.
    """
    fname = "make_accuracy_by_reasoning_steps"
    if k_stat not in _frames.BUDGET_AXIS_STATISTICS:
        raise ValueError(f"{fname}: k_stat is one of {_frames.BUDGET_AXIS_STATISTICS}, not {k_stat!r}")
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy", condition=condition), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, fname=fname)
    budgets = sorted({int(k) for k in k_values} & {int(k) for k in rows["K"].unique()})
    if not budgets:
        raise ValueError(f"{fname}: none of the reasoning-step values {list(k_values)} are in the rows")
    rows = tables.select(rows, fname=fname, K=budgets)
    distances = [int(d) for d in d_values]
    shades = style.distance_colors(distances, cmap)
    rng = np.random.default_rng(0)
    lines: dict[int, dict] = {}
    seeds = 0
    for d, shade in zip(distances, shades, strict=True):
        sub = tables.select(rows, fname=fname, d=d)
        if k_stat == "running_best":
            sub = tables.running_best_over_k(sub, group_cols=["d", "seed"])
        agg = _frames.summarize(sub, group_cols=["d", "K"])
        _frames.draw_series(ax, agg["K"], agg["value_mean"], agg["value_se"], color=shade, label=str(d), marker_size=marker_size, band=show_band, band_alpha=band_alpha)
        if show_seeds:
            _frames.draw_seed_points(ax, sub, x_col="K", value_col="value", color=shade, rng=rng, jitter=jitter, point_size=point_size)
        lines[d] = _frames.by_key(agg, "K", "value_mean")
        seeds = max(seeds, _frames.n_seeds(agg))
    ax.set_xticks(budgets if xticks is None else list(xticks))
    _frames.finish(ax, x="K", y="value", ylim=ylim, chance=show_chance)
    _frames.override_axes(ax, xlim=xlim, yticks=yticks, ylabel=ylabel)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, title=style.DISTANCE_LEGEND_TITLE, ncol=2)
    return {"condition": condition, "step": step, "k_stat": k_stat, "k_values": budgets, "mean_by_distance_by_k": lines, "n_seeds": seeds}


def make_steps_to_criterion_by_distance(
    ax: Axes,
    df: pd.DataFrame,
    *,
    conditions: Sequence[str] = ("iwl_test", "icl_test"),
    d_values: Sequence[int] | None = None,
    fraction: float = 0.9,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    show_band: bool = True,
    show_seeds: bool = True,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    marker_size: float = _frames.MARKER_SIZE,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    ylim: tuple[float, float] | None = None,
    legend_loc: str = "upper left",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
    unreached_below_chance: bool = False,
    unreached_z: float = 1.96,
    unreached_size: float = style.UNREACHED_SIZE,
    unreached_offset: float = style.UNREACHED_OFFSET,
    break_at_unreached: bool = False,
) -> dict:
    """First reasoning-step value reaching the criterion, by distance, mean and SE over seeds.

    Estimand: within each seed and distance, the smallest budget whose accuracy reaches
    `fraction` of that seed's maximum over the sweep; then the mean and standard error over
    seeds. `ylim` left at None follows the drawn values with the axes margins. `colors` and
    `labels` are keyed by condition; a content battery condition defaults to its context's.

    With `unreached_below_chance`, a seed has no value at a distance whose best accuracy over
    the sweep is not above chance: the lower end of its Wilson score interval at z =
    `unreached_z`, over that seed's trials at the distance, does not exceed one half. A
    distance where more than half of a condition's seeds have no value is left off its line
    and marked with a cross in the condition's color (`unreached_size` points squared) one
    budget spacing above the largest budget, raised by `unreached_offset` of a budget spacing
    for each condition before it in `conditions`, so conditions unreached at one distance show
    a cross each; one gray cross labeled `style.UNREACHED_LABEL` keys them in the legend.
    Every other distance averages over the seeds that have a value. The line joins the
    distances it keeps, or, with
    `break_at_unreached`, breaks with its band at every unreached distance, a kept distance
    between two unreached ones then drawn as a marker alone. The returned stats count, per
    condition and distance, the seeds with a value and those without.
    """
    fname = "make_steps_to_criterion_by_distance"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, fname=fname)
    if d_values is not None:
        rows = _frames.select_distances(rows, d_values, fname=fname)
    colors = {**style.CONDITION_COLORS, **(colors or {})}
    labels = {**style.CONDITION_LABELS, **(labels or {})}
    rng = np.random.default_rng(0)
    per_condition: dict[str, dict] = {}
    seeds = 0
    distances: set[int] = set()
    unreached: set[int] = set()
    for condition in conditions:
        sub = tables.select(rows, fname=fname, condition=condition)
        reached = tables.first_k_to_criterion(sub, group_cols=["d", "seed"], fraction=fraction, unreached_below_chance=unreached_below_chance, unreached_z=unreached_z)
        if reached.empty:
            raise ValueError(f"{fname}: condition {condition!r} holds no defined accuracy")
        has_value = reached[tables.K_TO_CRITERION].notna()
        with_value = has_value.groupby(reached["d"]).sum().astype(int)
        without_value = (~has_value).groupby(reached["d"]).sum().astype(int)
        lost = sorted(int(d) for d in with_value.index if without_value[d] > (with_value[d] + without_value[d]) / 2)
        drawn = reached[has_value & ~reached["d"].isin(lost)]
        mean, se = f"{tables.K_TO_CRITERION}_mean", f"{tables.K_TO_CRITERION}_se"
        color = colors.get(condition, style.REFERENCE_LINE)
        stats = {"mean_by_distance": {}, "se_by_distance": {}}
        if not drawn.empty:
            agg = _frames.summarize(drawn, group_cols=["d"], value_col=tables.K_TO_CRITERION)
            runs = _runs_between(agg["d"].astype(int).tolist(), lost) if break_at_unreached else [agg["d"].astype(int).tolist()]
            for index, run in enumerate(runs):
                part = agg[agg["d"].isin(run)]
                _frames.draw_series(ax, part["d"], part[mean], part[se], color=color, label=labels.get(condition, condition) if index == 0 else None, marker_size=marker_size, band=show_band, band_alpha=band_alpha)
            if show_seeds:
                _frames.draw_seed_points(ax, drawn, x_col="d", value_col=tables.K_TO_CRITERION, color=color, rng=rng, jitter=jitter, point_size=point_size)
            stats = {"mean_by_distance": _frames.by_key(agg, "d", mean), "se_by_distance": _frames.by_key(agg, "d", se)}
            seeds = max(seeds, _frames.n_seeds(agg))
        per_condition[condition] = {
            **stats,
            "n_seeds_with_value_by_distance": {int(d): int(n) for d, n in with_value.items()},
            "n_seeds_without_value_by_distance": {int(d): int(n) for d, n in without_value.items()},
            "unreached_distances": lost,
        }
        unreached.update(lost)
        distances.update(int(d) for d in with_value.index)
    if unreached:
        budgets = sorted(int(k) for k in rows["K"].unique())
        spacing = budgets[-1] - budgets[-2] if len(budgets) > 1 else 1
        top = budgets[-1] + spacing
        for index, condition in enumerate(conditions):
            marks = per_condition[condition]["unreached_distances"]
            if marks:
                height = top + index * float(unreached_offset) * spacing
                ax.scatter(marks, [height] * len(marks), marker=style.UNREACHED_MARKER, s=unreached_size, color=colors.get(condition, style.REFERENCE_LINE), zorder=_frames.LINE_ZORDER, gid=UNREACHED_GID + condition)
        key = Line2D([], [], ls="none", marker=style.UNREACHED_MARKER, markersize=float(np.sqrt(unreached_size)), color=style.REFERENCE_LINE, label=style.UNREACHED_LABEL)
        ax.add_line(key)
    ax.set_xticks(sorted(distances))
    _frames.finish(ax, x="d", y=tables.K_TO_CRITERION, ylim=ylim)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs)
    return {
        "conditions": list(conditions),
        "step": step,
        "fraction": float(fraction),
        "unreached_below_chance": bool(unreached_below_chance),
        "unreached_z": float(unreached_z),
        "unreached_distances": sorted(unreached),
        "per_condition": per_condition,
        "n_seeds": seeds,
    }


def _accuracy_at_budget(rows: pd.DataFrame, condition: str, k: int, d_min: int, *, fname: str) -> tuple[pd.DataFrame, float, float, int, int]:
    """One condition's accuracy at budget `k`, pooled over distances at least `d_min` by trials
    within each seed: the per-seed values, their mean and standard error over seeds, the seed
    count, and the largest trial count of a seed."""
    per_seed = tables.accuracy_at_budget(tables.select(rows, fname=fname, condition=condition), k=k, d_min=d_min)
    agg = _frames.summarize(per_seed, group_cols=["condition"])
    return per_seed, float(agg["value_mean"].iloc[0]), float(agg["value_se"].iloc[0]), _frames.n_seeds(agg), int(per_seed["n"].max())


def make_accuracy_bars(
    ax: Axes,
    df: pd.DataFrame,
    *,
    conditions: Sequence[str] = ("iwl_test", "icl_test"),
    k_pin: int | None = None,
    d_min: int = 2,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    show_chance: bool = True,
    show_seeds: bool = True,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    bar_width: float = BAR_WIDTH,
    error_color: str = ERROR_COLOR,
    ylim: tuple[float, float] | None = (0.0, 1.02),
    ylabel: str | None = None,
    step_pin: int | None = None,
) -> dict:
    """One bar per condition: accuracy at one reasoning-step value, pooled over distances.

    Estimand: the budget pinned at `k_pin` (the largest in the table when None); the
    trial-weighted pool over distances at least `d_min` within each seed; then the mean and
    standard error over seeds, with one dot per seed drawn over its bar. The bars stand in
    `conditions` order, in their conditions' colors, with no tick labels: a legend elsewhere in
    the figure keys the colors, and `labels` names the bars in the returned stats. `bar_width`
    is a fraction of the space between bar centers; the error bars are drawn in `error_color`;
    the bars stand on the bottom of the axis at the default `ylim`; `ylabel` replaces the axis
    label.
    """
    fname = "make_accuracy_bars"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, fname=fname)
    k = int(rows["K"].max()) if k_pin is None else int(k_pin)
    colors = {**style.CONDITION_COLORS, **(colors or {})}
    labels = {**style.CONDITION_LABELS, **(labels or {})}
    rng = np.random.default_rng(0)
    per_condition: dict[str, dict] = {}
    seeds = 0
    trials = 0
    for index, condition in enumerate(conditions):
        per_seed, mean, se, n_seeds, n_trials = _accuracy_at_budget(rows, condition, k, d_min, fname=fname)
        color = colors.get(condition, style.REFERENCE_LINE)
        ax.bar(
            [index],
            [mean],
            width=bar_width,
            yerr=None if np.isnan(se) else [se],
            color=color,
            edgecolor=color,
            linewidth=style.AXES_LINEWIDTH_PT,
            error_kw={"ecolor": error_color, "lw": style.AXES_LINEWIDTH_PT, "capsize": _frames.ERRORBAR_CAPSIZE, "capthick": style.AXES_LINEWIDTH_PT},
            label=labels.get(condition, condition),
            zorder=_frames.LINE_ZORDER,
        )
        if show_seeds:
            _frames.draw_seed_points(ax, per_seed.assign(x=float(index)), x_col="x", value_col="value", color=color, rng=rng, jitter=jitter, point_size=point_size, over_bars=True)
        per_condition[condition] = {
            "label": labels.get(condition, condition),
            "mean": round(mean, 4),
            "se": None if np.isnan(se) else round(se, 4),
            "per_seed": _frames.by_key(per_seed, "seed", "value"),
        }
        seeds = max(seeds, n_seeds)
        trials = max(trials, n_trials)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    _frames.finish(ax, x=None, y="value" if ylabel is None else ylabel, ylim=ylim, chance=show_chance)
    return {
        "conditions": list(conditions),
        "k_pin": k,
        "d_min": int(d_min),
        "step": step,
        "per_condition": per_condition,
        "n_trials_per_seed": trials,
        "n_seeds": seeds,
    }


def line_end(ax: Axes, color: str, *, min_points: int = 1) -> tuple[float, float] | None:
    """The last point (largest x) of the marked lines drawn on `ax` in `color` that hold at least
    `min_points` points, or None when no such line is drawn."""
    ends = []
    for line in ax.get_lines():
        if to_hex(line.get_color()) != to_hex(color) or line.get_marker() in (None, "None", "", " "):
            continue
        xy = np.column_stack([np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)])
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) >= max(int(min_points), 1):
            ends.append(xy[np.argmax(xy[:, 0])])
    return tuple(float(v) for v in max(ends, key=lambda p: p[0])) if ends else None


def _unreached_everywhere(ax: Axes, condition: str, color: str) -> bool:
    """Whether `ax` shows `condition` only by unreached crosses: its crosses are drawn and no
    marked line or lone marker in its color."""
    crossed = any(collection.get_gid() == UNREACHED_GID + condition for collection in ax.collections)
    return crossed and line_end(ax, color) is None


def make_accuracy_numbers(
    ax: Axes,
    df: pd.DataFrame,
    *,
    conditions: Sequence[str] = ("iwl_test", "icl_test"),
    k_pin: int | None = None,
    d_min: int = 2,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    decimals: int = 2,
    fontsize: float = style.FONT_LEGEND_PT,
    anchors: Mapping[str, tuple[float, float]] | None = None,
    ha: str = "left",
    va: str = "bottom",
    step_pin: int | None = None,
) -> dict:
    """Each condition's accuracy at one reasoning-step value, pooled over distances, written as
    a number in the condition's color beside the condition's line on `ax`.

    Estimand: `make_accuracy_bars`'s (the budget pinned at `k_pin`, the largest in the table
    when None; the trial-weighted pool over distances at least `d_min` within each seed; the
    mean over seeds, written to `decimals` places by `tables.round_half_up`, its standard error
    in the stats). Each number starts at the end of the last line segment already drawn on `ax`
    in the condition's color, the largest x of its lines of two or more points (or at
    `anchors[condition]`, data units), and is marked for `placement.place_texts`, which moves it
    clear of the lines, markers, bands, legend and other texts, nearer its own line than any
    other, once the figure's layout is final. A condition that `make_steps_to_criterion_by_distance`
    marks unreached at every distance (its crosses drawn, no line or marker in its color) gets
    no number, its text left None in the stats; any other condition with no such line and no
    anchor raises, as does a number whose estimand is missing.
    """
    fname = "make_accuracy_numbers"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, fname=fname)
    k = int(rows["K"].max()) if k_pin is None else int(k_pin)
    colors = {**style.CONDITION_COLORS, **(colors or {})}
    labels = {**style.CONDITION_LABELS, **(labels or {})}
    per_condition: dict[str, dict] = {}
    seeds = 0
    trials = 0
    for condition in conditions:
        per_seed, mean, se, n_seeds, n_trials = _accuracy_at_budget(rows, condition, k, d_min, fname=fname)
        color = colors.get(condition, style.REFERENCE_LINE)
        anchor = (anchors or {}).get(condition) or line_end(ax, color, min_points=2)
        text = None
        if anchor is None and not _unreached_everywhere(ax, condition, color):
            raise ValueError(f"{fname}: no line of two or more points in {color} for {condition!r} on these axes")
        if anchor is not None:
            if np.isnan(mean):
                raise ValueError(f"{fname}: condition {condition!r} has no accuracy at budget {k} to write")
            text = ax.text(*anchor, format(tables.round_half_up(mean, decimals), "f"), color=color, fontsize=fontsize, ha=ha, va=va, zorder=_frames.LINE_ZORDER + 1)
            placement.mark(text, key=condition, anchor=anchor, own_color=color)
        per_condition[condition] = {
            "label": labels.get(condition, condition),
            "mean": round(mean, 4),
            "se": None if np.isnan(se) else round(se, 4),
            "per_seed": _frames.by_key(per_seed, "seed", "value"),
            "text": None if text is None else text.get_text(),
        }
        seeds = max(seeds, n_seeds)
        trials = max(trials, n_trials)
    return {
        "conditions": list(conditions),
        "k_pin": k,
        "d_min": int(d_min),
        "step": step,
        "per_condition": per_condition,
        "n_trials_per_seed": trials,
        "n_seeds": seeds,
    }


def _runs_between(kept: Sequence[int], breaks: Sequence[int]) -> list[list[int]]:
    """The kept distances in increasing runs that no distance in `breaks` interrupts."""
    runs: list[list[int]] = []
    for d in sorted({*kept, *breaks}):
        if d in breaks:
            runs.append([])
        elif runs:
            runs[-1].append(d)
        else:
            runs.append([d])
    return [run for run in runs if run]


def make_pair_accuracy(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str = "iwl_test",
    k_values: Sequence[int] = (0,),
    rank_labels: Sequence[str] | None = None,
    annotate: bool = True,
    pool_query_order: bool = True,
    cmap: str = style.STEPS_CMAP,
    show_chance: bool = True,
    show_seeds: bool = True,
    show_legend: bool = True,
    marker_size: float = _frames.MARKER_SIZE,
    line_width: float | None = None,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    legend_loc: str = "lower left",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
) -> dict:
    """Accuracy for every pair of the queried ordering, grouped by distance.

    Estimand: one line per pinned budget in `k_values` (those the rows lack are left out and
    reported); the mean and standard error over seeds per pair. With `pool_query_order`, the
    two query orders of a pair are pooled within each seed by trial count before the seed
    summary. One budget is drawn in the condition's color with standard-error bars
    (`ERRORBAR_LW` points wide, caps included) and per-seed dots; several budgets are drawn
    one line per budget, each line `line_width` points wide (the shared width when None).
    Pairs sit in blocks of increasing distance, connected within a block only, and are named
    by `rank_labels` (letters, highest rank first, by default), by their shown positions under
    a shuffled context; `annotate` writes each block's distance beneath it.
    """
    fname = "make_pair_accuracy"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy", condition=condition), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.PAIR_SCHEMA, fname=fname)
    budgets = sorted({int(k) for k in k_values} & {int(k) for k in rows["K"].unique()})
    if not budgets:
        raise ValueError(f"{fname}: none of the reasoning-step values {list(k_values)} are in the rows")
    rows = tables.select(rows, fname=fname, K=budgets)
    rows = rows.assign(low=rows[["rank_a", "rank_b"]].min(axis=1), high=rows[["rank_a", "rank_b"]].max(axis=1))
    ranks = sorted(set(rows["low"]).union(rows["high"]))
    letters = list(rank_labels) if rank_labels is not None else list(PAIR_LETTERS[: len(ranks)])
    if len(letters) != len(ranks):
        raise ValueError(f"{fname}: {len(letters)} rank labels for {len(ranks)} ranks")
    letter = dict(zip(ranks, letters, strict=True))
    if pool_query_order:
        pooled = rows.assign(weighted=rows["value"] * rows["n"]).groupby(["d", "low", "high", "K", "seed"], sort=False)
        rows = pooled.agg(weighted=("weighted", "sum"), n=("n", "sum")).reset_index()
        rows["value"] = rows["weighted"] / rows["n"]
    else:
        rows = rows.groupby(["d", "low", "high", "K", "seed"], sort=False)["value"].mean().reset_index()
    pairs = rows[["d", "low", "high"]].drop_duplicates().sort_values(["d", "low", "high"])
    distances = sorted(pairs["d"].astype(int).unique().tolist())
    positions: dict[tuple[int, int], float] = {}
    x = 0.0
    for d in distances:
        for low, high in pairs.loc[pairs["d"] == d, ["low", "high"]].itertuples(index=False):
            positions[(int(low), int(high))] = x
            x += 1.0
        x += 1.0
    rows["x"] = [positions[(int(a), int(b))] for a, b in zip(rows["low"], rows["high"], strict=True)]
    rows["pair"] = [f"{letter[int(a)]}{letter[int(b)]}" for a, b in zip(rows["low"], rows["high"], strict=True)]
    single = len(budgets) == 1
    shades = {budgets[0]: style.CONDITION_COLORS.get(condition, style.REFERENCE_LINE)} if single else dict(zip(budgets, style.steps_colors(budgets, cmap), strict=True))
    rng = np.random.default_rng(0)
    by_budget: dict[int, dict] = {}
    seeds = 0
    for k in budgets:
        sub = rows[rows["K"] == k]
        agg = _frames.summarize(sub, group_cols=["d", "x", "pair"])
        for i, d in enumerate(distances):
            block = agg[agg["d"] == d]
            _frames.draw_series(ax, block["x"], block["value_mean"], block["value_se"] if single else None, color=shades[k], label=str(k) if i == 0 else None, marker_size=marker_size, line_width=line_width, band=False)
            if single and block["value_se"].notna().any():
                ax.errorbar(block["x"], block["value_mean"], yerr=np.nan_to_num(block["value_se"]), fmt="none", ecolor=shades[k], elinewidth=ERRORBAR_LW, capsize=_frames.ERRORBAR_CAPSIZE, capthick=ERRORBAR_LW, zorder=_frames.LINE_ZORDER)
        if single and show_seeds:
            _frames.draw_seed_points(ax, sub, x_col="x", value_col="value", color=shades[k], rng=rng, jitter=jitter, point_size=point_size)
        by_budget[k] = _frames.by_key(agg, "pair", "value_mean")
        seeds = max(seeds, _frames.n_seeds(agg))
    ordered = agg.sort_values("x")
    ax.set_xticks(ordered["x"].tolist())
    ax.set_xticklabels(ordered["pair"].tolist(), rotation=90)
    if annotate:
        for d in distances:
            xs = [pos for (a, b), pos in positions.items() if b - a == d]
            ax.annotate(PAIR_BLOCK_FORMAT.format(d), xy=(float(np.mean(xs)), 0.0), xycoords=("data", "axes fraction"), xytext=(0.0, -19.0), textcoords="offset points", ha="center", va="top", annotation_clip=False)
    ax.set_xlim(-0.7, max(positions.values()) + 0.7)
    _frames.finish(ax, x="pair", y="value", ylim=ylim, chance=show_chance)
    if annotate:
        ax.xaxis.labelpad = 16.0
    if show_legend and not single:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, title=style.AXIS_LABELS["K"], ncol=3)
    return {
        "condition": condition,
        "step": step,
        "k_values": budgets,
        "rank_labels": {int(r): l for r, l in letter.items()},
        "pairs": ordered["pair"].tolist(),
        "mean_by_pair_by_k": by_budget,
        "n_seeds": seeds,
    }


def make_response_classes_by_reasoning_steps(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str = "arbitrage_test",
    classes: Sequence[str] = ("prior_consistent_frac", "context_consistent_frac"),
    d_min: int = 2,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    show_chance: bool = True,
    show_band: bool = True,
    show_seeds: bool = True,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    marker_size: float = _frames.MARKER_SIZE,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    legend_loc: str = "upper right",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
) -> dict:
    """Proportions of answers following the trained and the context ordering, by reasoning steps.

    Estimand: none over reasoning steps, which are the x axis; each fraction is pooled over
    symbolic distances at least `d_min` within a seed with trial weights, then the mean and
    standard error over seeds. The classes are drawn in the order given, which is the legend's.
    """
    fname = "make_response_classes_by_reasoning_steps"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy", condition=condition), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, fname=fname)
    colors = {**style.RESPONSE_CLASS_COLORS, **(colors or {})}
    labels = {**style.RESPONSE_CLASS_LABELS, **(labels or {})}
    rng = np.random.default_rng(0)
    per_class: dict[str, dict] = {}
    seeds = 0
    trials = 0
    for name in classes:
        if name not in rows.columns:
            raise ValueError(f"{fname}: the rows carry no column '{name}'")
        if rows[name].isna().all():
            raise ValueError(f"{fname}: '{name}' is undefined for condition {condition!r}")
        pooled = _frames.pool(rows, group_cols=["K", "seed"], d_min=d_min, value_col=name)
        agg = _frames.summarize(pooled, group_cols=["K"], value_col=name)
        color = colors.get(name, style.REFERENCE_LINE)
        _frames.draw_series(ax, agg["K"], agg[f"{name}_mean"], agg[f"{name}_se"], color=color, label=labels.get(name, name), marker_size=marker_size, band=show_band, band_alpha=band_alpha)
        if show_seeds:
            _frames.draw_seed_points(ax, pooled, x_col="K", value_col=name, color=color, rng=rng, jitter=jitter, point_size=point_size)
        per_class[name] = {"mean_by_k": _frames.by_key(agg, "K", f"{name}_mean"), "se_by_k": _frames.by_key(agg, "K", f"{name}_se")}
        seeds = max(seeds, _frames.n_seeds(agg))
        trials = int(pooled["n"].max())
    ax.set_xticks(sorted(rows["K"].astype(int).unique().tolist()))
    _frames.finish(ax, x="K", y="response_class", ylim=ylim, chance=show_chance)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs)
    return {
        "condition": condition,
        "step": step,
        "d_min": int(d_min),
        "n_trials_per_seed": trials,
        "per_class": per_class,
        "n_seeds": seeds,
    }
