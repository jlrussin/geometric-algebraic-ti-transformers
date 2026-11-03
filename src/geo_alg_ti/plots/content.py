"""Panels for the figure "Content effects in transitive inference", the
appendix figure "Content effects across pretraining distributions" and its
version without the cue (not in the paper), and the appendix figure
"Experiments without an explicit cue".

Accuracy by distance under the four contexts on the frequent cue's ordering at
zero reasoning steps, and accuracy of the congruent and reversed contexts as
reasoning steps grow. A context names the content battery's condition on the frequent cue's
ordering (`frequent_cue_<context>`); `icl_test` may be drawn beside them.
`value` is agreement with each condition's own scoring reference, the ordering
shown in the context or, for the uninformative context, the frequent cue's
ordering. The distance of the shuffled context is the one in the ordering
shown. Its agreeing and conflicting halves (`SHUFFLED_SPLIT_CONTEXTS`) are
drawn like any context once `tables.split_shuffled` has added their rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from geo_alg_ti import tables
from geo_alg_ti.plots import _frames, style

CONTEXTS: tuple[str, ...] = ("uninformative", "congruent", "reversed", "shuffled")
# The halves of the shuffled context whose shown order agrees with the frequent cue's ordering
# and conflicts with it; their rows exist only after `tables.split_shuffled`.
SHUFFLED_SPLIT_CONTEXTS: tuple[str, ...] = ("shuffled_agree", "shuffled_conflict")
CONTEXT_STEP_STATISTICS: tuple[str, ...] = _frames.BUDGET_AXIS_STATISTICS
LEGEND_MODES: tuple[str, ...] = ("distance", "context", "distance_key")


def condition_of(context: str) -> str:
    """The battery condition a context names: the frequent cue's ordering under that context.

    A half of the shuffled context names the condition `tables.split_shuffled` derives.
    """
    return context if context == "icl_test" else f"frequent_cue_{context}"


def proportion_label(value: float) -> str:
    """A proportion in its shortest form with at least one decimal: 0.01, 0.5, 1.0."""
    text = f"{float(value):g}"
    return text if "." in text else f"{float(value):.1f}"


def make_context_accuracy_by_distance(
    ax: Axes,
    df: pd.DataFrame,
    *,
    contexts: Sequence[str] = CONTEXTS,
    k_pin: int = 0,
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
    ylabel: str | None = None,
    step_pin: int | None = None,
) -> dict:
    """Accuracy versus distance for each context at one reasoning-step value.

    Estimand: the budget pinned at `k_pin`; then the mean and standard error over seeds at
    each distance, one line per context in the context's color. Naming a half of the shuffled
    context in `contexts` needs a frame that already holds the rows `tables.split_shuffled`
    derives. `ylabel` replaces the axis label; `yticks` fixes the accuracy ticks (automatic
    when None).
    """
    fname = "make_context_accuracy_by_distance"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EVAL_SCHEMA, fname=fname)
    if d_values is not None:
        rows = _frames.select_distances(rows, d_values, fname=fname)
    colors = {**style.CONTEXT_COLORS, **(colors or {})}
    labels = {**style.CONTEXT_LABELS, **(labels or {})}
    rng = np.random.default_rng(0)
    per_context: dict[str, dict] = {}
    seeds = 0
    distances: set[int] = set()
    for context in contexts:
        sub = tables.pin_k(tables.select(rows, fname=fname, condition=condition_of(context)), int(k_pin))
        agg = _frames.summarize(sub, group_cols=["d"])
        color = colors.get(context, style.REFERENCE_LINE)
        _frames.draw_series(ax, agg["d"], agg["value_mean"], agg["value_se"], color=color, label=labels.get(context, context), marker_size=marker_size, band=show_band, band_alpha=band_alpha)
        if show_seeds:
            _frames.draw_seed_points(ax, sub, x_col="d", value_col="value", color=color, rng=rng, jitter=jitter, point_size=point_size)
        per_context[context] = {"mean_by_distance": _frames.by_key(agg, "d", "value_mean"), "se_by_distance": _frames.by_key(agg, "d", "value_se")}
        seeds = max(seeds, _frames.n_seeds(agg))
        distances.update(agg["d"].astype(int).tolist())
    ax.set_xticks(sorted(distances))
    _frames.finish(ax, x="d", y="value" if ylabel is None else ylabel, ylim=ylim, chance=show_chance)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs)
    return {"contexts": list(contexts), "k_pin": int(k_pin), "step": step, "per_context": per_context, "n_seeds": seeds}


def make_context_accuracy_by_reasoning_steps(
    ax: Axes,
    df: pd.DataFrame,
    *,
    contexts: Sequence[str] = ("reversed",),
    d_values: Sequence[int] = (1, 2, 3, 4, 5, 6),
    k_stat: str = "raw",
    cmaps: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    show_chance: bool = True,
    show_band: bool = True,
    show_seeds: bool = False,
    show_legend: bool = True,
    legend_mode: str | None = None,
    key_cmap: str = style.DISTANCE_KEY_CMAP,
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
    """Accuracy versus reasoning steps under each context, one line per distance.

    Estimand: none over reasoning steps, which are the x axis (`k_stat="raw"`), or the running
    best over the budgets up to each budget within a seed and distance (`"running_best"`);
    then the mean and standard error over seeds at each distance and budget. Each context's
    distances take the shades of its colormap (`cmaps`, by context). The legend names the
    distances when one context is drawn and the contexts otherwise, unless `legend_mode`
    says which; `"distance_key"` keys the distances in the shades of `key_cmap`, which stand
    for every context's family. `xticks` (every budget when None), `xlim` and `yticks`
    (automatic when None) fix the axes; `ylabel` replaces the accuracy label.
    """
    fname = "make_context_accuracy_by_reasoning_steps"
    if k_stat not in CONTEXT_STEP_STATISTICS:
        raise ValueError(f"{fname}: k_stat is one of {CONTEXT_STEP_STATISTICS}, not {k_stat!r}")
    if not contexts:
        raise ValueError(f"{fname}: contexts names no context")
    mode = ("distance" if len(contexts) == 1 else "context") if legend_mode is None else legend_mode
    if mode not in LEGEND_MODES:
        raise ValueError(f"{fname}: legend_mode is one of {LEGEND_MODES}, not {legend_mode!r}")
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy"), step_pin, fname=fname)
    cmaps = {**style.CONTEXT_DISTANCE_CMAPS, **(cmaps or {})}
    labels = {**style.CONTEXT_LABELS, **(labels or {})}
    distances = [int(d) for d in d_values]
    rng = np.random.default_rng(0)
    per_context: dict[str, dict] = {}
    handles: list[Line2D] = []
    seeds = 0
    for context in contexts:
        sub = tables.select(rows, fname=fname, condition=condition_of(context), d=distances)
        _frames.refuse_unpinned(sub, schema=tables.EVAL_SCHEMA, fname=fname)
        if k_stat == "running_best":
            sub = tables.running_best_over_k(sub, group_cols=["d", "seed"])
        cmap = cmaps.get(context, style.REVERSED_DISTANCE_CMAP)
        shades = style.distance_colors(distances, cmap)
        lines: dict[int, dict] = {}
        for d, shade in zip(distances, shades, strict=True):
            by_distance = tables.select(sub, fname=fname, d=d)
            agg = _frames.summarize(by_distance, group_cols=["d", "K"])
            _frames.draw_series(ax, agg["K"], agg["value_mean"], agg["value_se"], color=shade, label=str(d) if mode == "distance" else None, marker_size=marker_size, band=show_band, band_alpha=band_alpha)
            if show_seeds:
                _frames.draw_seed_points(ax, by_distance, x_col="K", value_col="value", color=shade, rng=rng, jitter=jitter, point_size=point_size)
            lines[d] = _frames.by_key(agg, "K", "value_mean")
            seeds = max(seeds, _frames.n_seeds(agg))
        handles.append(Line2D([], [], color=style.CONTEXT_COLORS.get(context, shades[-1]), marker="o", markersize=marker_size, label=labels.get(context, context)))
        per_context[context] = {"cmap": cmap, "mean_by_distance_by_k": lines}
    ax.set_xticks(sorted(rows["K"].astype(int).unique().tolist()) if xticks is None else list(xticks))
    _frames.finish(ax, x="K", y="value", ylim=ylim, chance=show_chance)
    _frames.override_axes(ax, xlim=xlim, yticks=yticks, ylabel=ylabel)
    if show_legend and mode == "distance":
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, title=style.DISTANCE_LEGEND_TITLE, ncol=2)
    elif show_legend and mode == "distance_key":
        key = [Line2D([], [], color=shade, marker="o", markersize=marker_size, label=str(d)) for d, shade in zip(distances, style.distance_colors(distances, key_cmap), strict=True)]
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, handles=key, title=style.DISTANCE_LEGEND_TITLE, ncol=2)
    elif show_legend:
        ax.legend(handles=handles, loc=legend_loc, **dict(legend_kwargs or {}))
    return {"contexts": list(contexts), "k_stat": k_stat, "step": step, "d_values": distances, "per_context": per_context, "n_seeds": seeds}
