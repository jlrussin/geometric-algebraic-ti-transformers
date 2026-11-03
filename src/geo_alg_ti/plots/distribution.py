"""Panels for the figure "Data-distribution effects" and its appendix: skew over orderings.

Accuracy of the ICL and IWL tests versus skew alpha, the per-seed dissociation
between the two, and a data-free schematic of the Zipf training frequencies
(the figure); emergence over pretraining steps one line per skew alpha (the
appendix figure "Learning over pretraining by skew"). Skew alpha is colored by
value on one colormap, so a given skew is the same color in every panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from geo_alg_ti import tables
from geo_alg_ti.plots import _frames, learning, style

# The paper's estimand per test on a pretrained network: the IWL test is scored at zero
# reasoning steps; the ICL test at its best over the reasoning-step sweep.
DEFAULT_K_STAT: Mapping[str, str] = {"iwl_test": "pin", "icl_test": "best"}
SKEW_KEY_FORMAT: str = "{:g}"
# The skew colorbar beside the dissociation panel: its distance from the panel and its width,
# as fractions of the panel's width.
COLORBAR_PAD: float = 0.05
COLORBAR_FRACTION: float = 0.15


def _pooled_by_skew(df: pd.DataFrame, *, condition: str, x_col: str, k_stat: Mapping[str, str], k_pin: int, d_min: int, fname: str) -> pd.DataFrame:
    """Per-seed accuracy at each skew alpha for one condition, pooled over distance."""
    sub = tables.select(df, fname=fname, condition=condition)
    stat = k_stat.get(condition, "best")
    reduced = _frames.reduce_steps(sub, k_stat=stat, k_pin=k_pin, k_values=_frames.DEFAULT_MEAN_BUDGETS, group_cols=[x_col, "seed", "d"], fname=fname)
    return _frames.pool(reduced, group_cols=[x_col, "seed"], d_min=d_min)


def make_accuracy_vs_skew(
    ax: Axes,
    df: pd.DataFrame,
    *,
    conditions: Sequence[str] = ("icl_test", "iwl_test"),
    x_col: str = "skew_alpha",
    step_pin: int | None = None,
    k_stat: Mapping[str, str] | None = None,
    k_pin: int = 0,
    d_min: int = 2,
    xticks: Sequence[float] | None = None,
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
    legend_loc: str = "lower left",
    legend_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """ICL and IWL accuracy versus skew alpha, pooled over distance, at one training step.

    Estimand: per condition, `k_stat` over reasoning steps (the IWL test pinned at `k_pin`,
    the ICL test the best over the sweep, unless `k_stat` says otherwise) within each seed,
    skew, and distance; the trial-weighted pool over distances at least `d_min`; then the mean
    and standard error over seeds at each skew. `step_pin` left at None reads the last step.
    """
    fname = "make_accuracy_vs_skew"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, allowed=(x_col,), fname=fname)
    k_stat = {**DEFAULT_K_STAT, **(k_stat or {})}
    colors = {**style.CONDITION_COLORS, **(colors or {})}
    labels = {**style.CONDITION_LABELS, **(labels or {})}
    rng = np.random.default_rng(0)
    per_condition: dict[str, dict] = {}
    seeds = 0
    trials = 0
    for condition in conditions:
        pooled = _pooled_by_skew(rows, condition=condition, x_col=x_col, k_stat=k_stat, k_pin=k_pin, d_min=d_min, fname=fname)
        agg = _frames.summarize(pooled, group_cols=[x_col])
        color = colors.get(condition, style.REFERENCE_LINE)
        _frames.draw_series(ax, agg[x_col], agg["value_mean"], agg["value_se"], color=color, label=labels.get(condition, condition), marker_size=marker_size, band=show_band, band_alpha=band_alpha)
        if show_seeds:
            _frames.draw_seed_points(ax, pooled, x_col=x_col, value_col="value", color=color, rng=rng, jitter=jitter, point_size=point_size)
        per_condition[condition] = {
            "k_stat": k_stat.get(condition, "best"),
            "mean_by_skew": _frames.by_key(agg, x_col, "value_mean"),
            "se_by_skew": _frames.by_key(agg, x_col, "value_se"),
        }
        seeds = max(seeds, _frames.n_seeds(agg))
        trials = int(pooled["n"].max())
    alphas = sorted(float(a) for a in rows[x_col].dropna().unique())
    ax.set_xticks(list(xticks) if xticks is not None else [a for a in alphas if float(a).is_integer()] or alphas)
    _frames.finish(ax, x=x_col, y="value", ylim=ylim, chance=show_chance)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs)
    return {
        "conditions": list(conditions),
        "step": step,
        "k_pin": int(k_pin),
        "d_min": int(d_min),
        "n_trials_per_seed": trials,
        "per_condition": per_condition,
        "n_seeds": seeds,
    }


def make_icl_iwl_dissociation(
    ax: Axes,
    df: pd.DataFrame,
    *,
    x_condition: str = "iwl_test",
    y_condition: str = "icl_test",
    color_col: str = "skew_alpha",
    step_pin: int | None = None,
    k_stat: Mapping[str, str] | None = None,
    k_pin: int = 0,
    d_min: int = 2,
    cmap: str = style.SKEW_CMAP,
    point_size: float = 8.0,
    mean_size: float = 40.0,
    mean_edge_width: float = 0.8,
    show_means: bool = True,
    show_chance: bool = True,
    colorbar_ax: Axes | None = None,
    xlim: tuple[float, float] | None = (0.45, 1.02),
    ylim: tuple[float, float] | None = (0.45, 1.02),
) -> dict:
    """ICL against IWL accuracy per seed and skew, with a rank correlation in the statistics.

    Estimand: the same per-seed pooled accuracies as `make_accuracy_vs_skew`, drawn one point
    per seed and skew rather than averaged; the larger markers of `mean_size` points squared,
    edged in black at `mean_edge_width` points, are the means over seeds at each skew.
    Spearman's rank correlation across the points is returned, not drawn.
    The skew colorbar sits beside `colorbar_ax` (normally `ax` itself) when one is given,
    titled with the skew axis label; without one it is not drawn.
    """
    fname = "make_icl_iwl_dissociation"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, allowed=(color_col,), fname=fname)
    k_stat = {**DEFAULT_K_STAT, **(k_stat or {})}
    xs = _pooled_by_skew(rows, condition=x_condition, x_col=color_col, k_stat=k_stat, k_pin=k_pin, d_min=d_min, fname=fname)
    ys = _pooled_by_skew(rows, condition=y_condition, x_col=color_col, k_stat=k_stat, k_pin=k_pin, d_min=d_min, fname=fname)
    merged = xs[[color_col, "seed", "value"]].merge(ys[[color_col, "seed", "value"]], on=[color_col, "seed"], suffixes=("_x", "_y"))
    if merged.empty:
        raise ValueError(f"{fname}: the two conditions share no seed and skew")
    alphas = sorted(float(a) for a in merged[color_col].unique())
    palette = style.sweep_colors(alphas, cmap, column=color_col, by_value=True)
    shade = dict(zip(alphas, palette, strict=True))
    point_colors = [shade[float(a)] for a in merged[color_col]]
    ax.scatter(merged["value_x"], merged["value_y"], c=point_colors, s=point_size, alpha=0.85, lw=0, zorder=3)
    means = merged.groupby(color_col, as_index=False)[["value_x", "value_y"]].mean()
    if show_means:
        ax.scatter(means["value_x"], means["value_y"], c=[shade[float(a)] for a in means[color_col]], s=mean_size, edgecolors="black", linewidths=mean_edge_width, zorder=4)
    if colorbar_ax is not None:
        title = style.AXIS_LABELS.get(color_col, color_col)
        style.sweep_colorbar(cmap, ax=colorbar_ax, span=(min(alphas), max(alphas)), title=title, column=color_col, pad=COLORBAR_PAD, fraction=COLORBAR_FRACTION)
    if show_chance:
        style.chance_line(ax)
        ax.axvline(style.CHANCE, color=style.REFERENCE_LINE, ls="--", lw=style.AXES_LINEWIDTH_PT, zorder=1)
    ax.set_xlabel(f"{style.CONDITION_LABELS.get(x_condition, x_condition)} accuracy", color=style.CONDITION_COLORS.get(x_condition, "black"))
    ax.set_ylabel(f"{style.CONDITION_LABELS.get(y_condition, y_condition)} accuracy", color=style.CONDITION_COLORS.get(y_condition, "black"))
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    rho = float(np.corrcoef(merged["value_x"].rank(), merged["value_y"].rank())[0, 1]) if len(merged) > 1 else float("nan")
    return {
        "x_condition": x_condition,
        "y_condition": y_condition,
        "step": step,
        "d_min": int(d_min),
        "n_points": int(len(merged)),
        "spearman": None if np.isnan(rho) else round(rho, 4),
        "mean_x_by_skew": _frames.by_key(means, color_col, "value_x"),
        "mean_y_by_skew": _frames.by_key(means, color_col, "value_y"),
        "n_seeds": int(merged.groupby(color_col)["seed"].nunique().max()),
    }


def make_emergence_by_skew(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str = "icl_test",
    k_stat: str = "best",
    k_pin: int = 0,
    d_min: int = 2,
    cmap: str = style.SKEW_CMAP,
    x_col: str = "skew_alpha",
    show_chance: bool = True,
    show_band: bool = True,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    xticks: Sequence[float] | None = None,
    ylim: tuple[float, float] | None = (0.4, 1.02),
    legend_loc: str = "lower right",
    legend_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """Accuracy pooled over distance versus training step, one line per skew alpha.

    Estimand: `k_stat` over reasoning steps within each seed, step, and distance (a pinned
    budget reads `k_pin`); the trial-weighted pool over distances at least `d_min`; then the
    mean and standard error over seeds at each step. `xticks` names the step ticks; the step
    range follows the drawn steps with the axes margins, widened to any tick outside them.
    """
    fname = "make_emergence_by_skew"
    rows = tables.select(df, fname=fname, metric="accuracy", condition=condition)
    if x_col not in rows.columns:
        raise ValueError(f"{fname}: the rows carry no column '{x_col}'")
    alphas = sorted(float(a) for a in rows[x_col].dropna().unique())
    colors = dict(zip(alphas, style.sweep_colors(alphas, cmap, column=x_col, by_value=True), strict=True))
    stats = learning.make_accuracy_by_step(
        ax,
        rows,
        condition=condition,
        k_stat=k_stat,
        k_pin=k_pin,
        d_min=d_min,
        line_col=x_col,
        colors=colors,
        show_chance=show_chance,
        show_band=show_band,
        show_legend=show_legend,
        band_alpha=band_alpha,
        xticks=xticks,
        ylim=ylim,
        legend_loc=legend_loc,
        legend_kwargs=legend_kwargs,
    )
    return {**stats, "skew_alphas": alphas}


def make_ordering_frequency_schematic(
    ax: Axes,
    df: pd.DataFrame | None = None,
    *,
    skew_alphas: Sequence[float] = (0, 2, 5, 10),
    n_orderings: int = 10,
    n_shown: int = 5,
    colors: Sequence[str] | None = None,
    cmap: str = style.SKEW_CMAP,
    color_span: tuple[float, float] | None = None,
    fill: bool = True,
    fill_alpha: float = 0.3,
    label_format: str = style.SKEW_LABEL_FORMAT,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """Training frequency of the most frequent orderings under each skew alpha; data-free.

    Nothing is measured: each curve is the Zipf law over `n_orderings` cued orderings,
    frequency proportional to the ordering's index to the power minus alpha (alpha zero is
    uniform), drawn over the `n_shown` most frequent. Curves are colored by alpha on the skew
    colormap, placed within `color_span` when one is given so they match a colorbar over that
    range, and drawn most skewed first so the flatter ones sit on top.
    """
    alphas = [float(a) for a in skew_alphas]
    if not alphas:
        raise ValueError("make_ordering_frequency_schematic: skew_alphas is empty")
    if not 1 <= n_shown <= n_orderings:
        raise ValueError(f"make_ordering_frequency_schematic: n_shown={n_shown} lies outside 1..{n_orderings}")
    palette = list(colors) if colors is not None else style.sweep_colors(alphas, cmap, column="skew_alpha", by_value=True, span=color_span)
    shade = dict(zip(alphas, palette, strict=True))
    index = np.arange(1, n_orderings + 1, dtype=float)
    fine = np.linspace(1.0, float(n_shown), 200)
    frequency: dict[str, list[float]] = {}
    handles = {}
    for alpha in sorted(alphas, reverse=True):
        weights = index ** (-alpha)
        total = float(weights.sum())
        curve = fine ** (-alpha) / total
        if fill:
            ax.fill_between(fine, 0.0, curve, color=shade[alpha], alpha=fill_alpha, lw=0, zorder=2)
        (handles[alpha],) = ax.plot(fine, curve, color=shade[alpha], label=label_format.format(alpha), zorder=3)
        frequency[SKEW_KEY_FORMAT.format(alpha)] = [round(float(v), 4) for v in weights[:n_shown] / total]
    ax.set_xticks([])
    ax.set_xlim(1.0, float(n_shown))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_yticks([0.0, 0.5, 1.0])
    style.label_axes(ax, "ordering", "frequency")
    if show_legend:
        ordered = [handles[a] for a in alphas]
        ax.legend(ordered, [h.get_label() for h in ordered], loc=legend_loc, **dict(legend_kwargs or {}))
    return {"skew_alphas": alphas, "n_orderings": int(n_orderings), "n_shown": int(n_shown), "frequency_by_skew": frequency, "n_seeds": 0}
