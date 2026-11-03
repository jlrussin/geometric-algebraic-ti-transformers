"""Panels for the appendix figure "Additional mechanism results" and, in the
figure "Mechanisms underlying the geometric and algebraic strategies", the
summary of the edits and the head ablations.

Rank decodability by layer, the behavioral effect of erasing or inverting the
rank direction (with a random direction erased as its control), the effect of
writing an unrelated ordering at increasing strength, each attention head's
effect on direct and reasoned answers when it is ablated, and one summary of
the edits and the head ablations side by side. The probe panel reads the probe
table; the edit panels read the edit table, whose rows carry the edit kind
(`erase` or `write`), the direction (`rank`, or `random` for the matched
control), the gain of an erase, and the target and strength of a write; the
head panels read the ablate table, whose rows carry the set of heads ablated:
none, one head with its attention rank, the k heads its network ranks highest
by their attention to the context, or a random draw of k heads from outside
the top 2k.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from geo_alg_ti import tables
from geo_alg_ti.plots import _frames, style

# A measure of the summary bars: its label, the battery condition, and the pinned budget.
DEFAULT_MEASURES: tuple[tuple[str, str, int], ...] = (
    ("IWL test\n0 steps", "frequent_cue_uninformative", 0),
    ("Reversed\n0 steps", "frequent_cue_reversed", 0),
    ("Reversed\n8 steps", "frequent_cue_reversed", 8),
    ("ICL test\n8 steps", "icl_test", 8),
)
# The two measures of the head scatter, each a condition at a pinned budget: the IWL test with no
# reasoning steps, and the ICL test after eight.
SCATTER_X_MEASURE: tuple[str, int] = ("frequent_cue_uninformative", 0)
SCATTER_Y_MEASURE: tuple[str, int] = ("icl_test", 8)
SCATTER_X_LABEL: str = "Change in IWL test accuracy, 0 steps"
SCATTER_Y_LABEL: str = "Change in ICL test accuracy, 8 steps"
CONTEXT_HEADS_KEY: str = "Context-reading heads"
CONTROL_GAIN: float = 0.0
CONTROL_STRENGTH: float = 1.0
CONTROL_MARKER: str = "o"
BAR_WIDTH: float = 0.16
CONTROL_BAR_WIDTH: float = 0.7
HEAD_POINT_SIZE: float = 5.0
CONTEXT_HEAD_SIZE: float = 16.0
CONTEXT_HEAD_EDGE_PT: float = 0.9
# The head dots are small; the key draws its layer swatches this much larger so they read.
KEY_POINT_SCALE: float = 1.6
# The head panels place their keys above the panel by default.
KEY_ABOVE: Mapping[str, object] = {"bbox_to_anchor": (0.5, 1.0), "frameon": False, "borderaxespad": 0.2}
# The jitter plus the margin matplotlib autoscaling would leave beyond the seed dots.
X_MARGIN: float = 0.05
# The kinds of head set in the ablate table: intact, one head alone, the heads ranked highest by
# their attention to the context, and a random draw from outside twice as many.
SET_INTACT: str = "intact"
SET_HEAD: str = "head"
SET_TOP: str = "top"
SET_RANDOM: str = "random"


def _accuracy_rows(df: pd.DataFrame, *, fname: str, step_pin: int | None, **pins: object) -> tuple[pd.DataFrame, int]:
    """The accuracy rows of an edit or ablate table at one step under the given pins."""
    rows, step = _frames.pin_step(tables.select(df, fname=fname, metric="accuracy", **pins), step_pin, fname=fname)
    return rows, step


def _gain_label(gain: float) -> str:
    """The legend entry of one scaling of the rank direction."""
    name = style.EDIT_GAIN_LABELS.get(float(gain))
    return name if name else f"Gain {gain:g}"


def make_probe_accuracy_by_layer(
    ax: Axes,
    df: pd.DataFrame,
    *,
    kind: str = "rank_order",
    metric: str = "accuracy",
    line_col: str | None = "frequent_cue_prop",
    cmap: str = style.FREQUENT_CUE_CMAP,
    step_pin: int | None = None,
    colors: Mapping[object, str] | None = None,
    layer_labels: Sequence[str] | None = None,
    show_chance: bool | None = None,
    show_band: bool = True,
    show_seeds: bool = True,
    show_legend: bool = True,
    colorbar_ax: Axes | None = None,
    band_alpha: float = _frames.BAND_ALPHA,
    marker_size: float = _frames.MARKER_SIZE,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    ylim: tuple[float, float] | None = None,
    legend_loc: str = "lower right",
    legend_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """Held-out rank decodability versus layer, one line per value of `line_col`.

    Estimand: the mean and standard error over seeds at each layer; there is no distance or
    reasoning-step axis in the probe table. The chance reference is drawn for a pairwise
    order accuracy unless `show_chance` says otherwise. The sites are ticked by
    `layer_labels`, by default the item embeddings, each block, and the output. The line
    family is keyed by a segmented colorbar beside `colorbar_ax` (normally `ax` itself) when
    one is given, else by a legend. `ylim` left at None spans the metric's range.
    """
    fname = "make_probe_accuracy_by_layer"
    rows, step = _frames.pin_step(tables.select(df, fname=fname, kind=kind, metric=metric), step_pin, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.PROBE_SCHEMA, allowed=(line_col,), fname=fname)
    if rows["value"].isna().any():
        raise ValueError(f"{fname}: the selected probe rows hold undefined values")
    if line_col is None:
        family, shades = [None], {None: style.EDIT_INTACT}
    else:
        family, shades = _frames.line_family(rows, line_col, cmap, colors, fname=fname)
    rng = np.random.default_rng(0)
    lines: dict[str, dict] = {}
    seeds = 0
    layers: set[int] = set()
    for value in family:
        sub = rows if value is None else tables.select(rows, fname=fname, **{line_col: value})
        agg = _frames.summarize(sub, group_cols=["layer"])
        name = "" if value is None else f"{float(value):g}"
        _frames.draw_series(ax, agg["layer"], agg["value_mean"], agg["value_se"], color=shades[value], label=name or None, marker_size=marker_size, band=show_band, band_alpha=band_alpha)
        if show_seeds:
            _frames.draw_seed_points(ax, sub, x_col="layer", value_col="value", color=shades[value], rng=rng, jitter=jitter, point_size=point_size)
        lines[name] = {"mean_by_layer": _frames.by_key(agg, "layer", "value_mean"), "se_by_layer": _frames.by_key(agg, "layer", "value_se")}
        seeds = max(seeds, _frames.n_seeds(agg))
        layers.update(agg["layer"].astype(int).tolist())
    ordered = sorted(layers)
    ticks = list(layer_labels) if layer_labels is not None else style.site_labels(ordered)
    if len(ticks) != len(ordered):
        raise ValueError(f"{fname}: {len(ticks)} layer labels for {len(ordered)} layers")
    ax.set_xticks(ordered)
    ax.set_xticklabels(ticks)
    chance = (metric == "accuracy") if show_chance is None else show_chance
    if ylim is None:
        ylim = (-1.02, 1.02) if metric == "spearman" else (-0.02, 1.02)
    _frames.finish(ax, x="layer", y="probe" if metric == "accuracy" else metric, ylim=ylim, chance=chance)
    if line_col is not None and colorbar_ax is not None:
        style.segmented_colorbar([shades[v] for v in family], [f"{float(v):g}" for v in family], ax=colorbar_ax, title=style.AXIS_LABELS.get(line_col, line_col))
    elif show_legend and line_col is not None:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, title=style.AXIS_LABELS.get(line_col, line_col))
    return {"kind": kind, "metric": metric, "step": step, "line_col": line_col, "layers": ordered, "lines": lines, "n_seeds": seeds}


def make_edit_accuracy_by_distance(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str = "frequent_cue_uninformative",
    gains: Sequence[float] = (1.0, -1.0, 0.0),
    k_pin: int = 0,
    d_values: Sequence[int] | None = None,
    control: bool = True,
    colors: Mapping[float, str] | None = None,
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
    """Accuracy versus distance after scaling the rank direction by each gain, with the control.

    Estimand: the budget pinned at `k_pin`; then the mean and standard error over seeds at
    each distance, one line per gain of the erase along the rank direction. The control is the
    same edit along a matched random direction at gain zero, drawn dashed with open markers in the
    erased color. By default the key lists the erased line just above its control.
    """
    fname = "make_edit_accuracy_by_distance"
    rows, step = _accuracy_rows(df, fname=fname, step_pin=step_pin, condition=condition, edit_kind="erase")
    rows = tables.pin_k(rows, int(k_pin))
    if d_values is not None:
        rows = _frames.select_distances(rows, d_values, fname=fname)
    _frames.refuse_unpinned(rows, schema=tables.EDIT_SCHEMA, allowed=("direction", "gain"), fname=fname)
    colors = {**style.EDIT_GAIN_COLORS, **(colors or {})}
    variants: list[tuple[str, str, float, bool]] = [(_gain_label(float(g)), "rank", float(g), False) for g in gains]
    if control:
        variants.append((style.EDIT_CONTROL_LABEL, "random", CONTROL_GAIN, True))
    rng = np.random.default_rng(0)
    series: dict[str, dict] = {}
    seeds = 0
    distances: set[int] = set()
    for name, direction, gain, is_control in variants:
        sub = tables.select(rows, fname=fname, direction=direction, gain=gain)
        _frames.refuse_unpinned(sub, schema=tables.EDIT_SCHEMA, fname=fname)
        agg = _frames.summarize(sub, group_cols=["d"])
        color = colors.get(gain, style.REFERENCE_LINE)
        _frames.draw_series(ax, agg["d"], agg["value_mean"], agg["value_se"], color=color, label=name, ls="--" if is_control else "-", open_marker=is_control, marker_size=marker_size, band=show_band, band_alpha=band_alpha)
        if show_seeds:
            _frames.draw_seed_points(ax, sub, x_col="d", value_col="value", color=color, open_marker=is_control, rng=rng, jitter=jitter, point_size=point_size)
        series[name] = {"mean_by_distance": _frames.by_key(agg, "d", "value_mean"), "se_by_distance": _frames.by_key(agg, "d", "value_se")}
        seeds = max(seeds, _frames.n_seeds(agg))
        distances.update(agg["d"].astype(int).tolist())
    ax.set_xticks(sorted(distances))
    _frames.finish(ax, x="d", y="value", ylim=ylim, chance=show_chance)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs)
    return {"condition": condition, "k_pin": int(k_pin), "step": step, "gains": [float(g) for g in gains], "control": bool(control), "series": series, "n_seeds": seeds}


def _pooled(
    rows: pd.DataFrame,
    *,
    condition: str,
    budget: int,
    d_min: int,
    fname: str,
    schema: Mapping[str, str],
    group_cols: Sequence[str] = ("condition", "seed"),
    allowed: Sequence[str] = (),
    **pins: object,
) -> pd.DataFrame:
    """One measure (a condition at reasoning-step value `budget`) pooled over distances at least
    `d_min` per group.

    `allowed` names the columns that may still vary, such as each network's own heads.
    """
    sub = tables.pin_k(tables.select(rows, fname=fname, condition=condition, **pins), int(budget))
    _frames.refuse_unpinned(sub, schema=schema, allowed=allowed, fname=fname)
    return _frames.pool(sub, group_cols=list(group_cols), d_min=d_min)


def _head_changes(rows: pd.DataFrame, *, condition: str, budget: int, d_min: int, fname: str) -> pd.DataFrame:
    """Per seed and head, the pooled accuracy with the head ablated alone minus the intact one's."""
    group = ["seed", "set_kind", "head", "layer", "attention_rank"]
    singles = tables.select(rows, fname=fname, set_kind=[SET_INTACT, SET_HEAD])
    pooled = _pooled(singles, condition=condition, budget=budget, d_min=d_min, fname=fname, schema=tables.ABLATE_SCHEMA, group_cols=group, allowed=("set_kind", "head", "n_heads", "heads", "attention_rank"))
    intact = pooled[pooled["set_kind"] == SET_INTACT].set_index("seed")["value"]
    changes = pooled[pooled["set_kind"] == SET_HEAD].copy()
    if intact.index.duplicated().any() or set(changes["seed"]) != set(intact.index):
        raise ValueError(f"{fname}: every network needs one intact row set beside its heads")
    changes["change"] = changes["value"].to_numpy() - intact.loc[changes["seed"]].to_numpy()
    return changes


def _top_heads(points: pd.DataFrame, k: int, *, fname: str) -> pd.DataFrame:
    """Each network's `k` heads ranked highest by attention; raises unless every network ranks
    exactly one head at each place from one to `k`."""
    marked = points[points["attention_rank"].between(1, int(k))]
    places = {int(seed): sorted(int(rank) for rank in part["attention_rank"]) for seed, part in marked.groupby("seed")}
    wanted = list(range(1, int(k) + 1))
    wrong = {int(seed): places.get(int(seed), []) for seed in sorted(points["seed"].unique()) if places.get(int(seed), []) != wanted}
    if wrong:
        raise ValueError(f"{fname}: every network needs one head ranked at each place 1..{int(k)}; found {wrong}")
    return marked


def make_head_ablation_scatter(
    ax: Axes,
    df: pd.DataFrame,
    *,
    k: int = 2,
    x_measure: tuple[str, int] = SCATTER_X_MEASURE,
    y_measure: tuple[str, int] = SCATTER_Y_MEASURE,
    d_min: int = 2,
    layer_colors: Sequence[str] = style.HEAD_LAYER_SHADES,
    context_head_color: str = style.CONTEXT_HEADS,
    context_head_edge_color: str = style.EDIT_INTACT,
    point_size: float = HEAD_POINT_SIZE,
    context_head_size: float = CONTEXT_HEAD_SIZE,
    context_head_edge: float = CONTEXT_HEAD_EDGE_PT,
    xlabel: str = SCATTER_X_LABEL,
    ylabel: str = SCATTER_Y_LABEL,
    show_legend: bool = True,
    legend_loc: str = "lower center",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
) -> dict:
    """Each head's change in two measures when it is ablated alone, with each network's `k`
    context-reading heads marked.

    Estimand: for each measure (a condition at a pinned budget), the trial-weighted pool over
    distances at least `d_min` within a seed with one head ablated, minus the intact network's.
    One point per seed and head; a head's identity is that of one network, so nothing is averaged.
    Heads are shaded by layer; the `k` heads each network ranks highest by their attention to the
    context are drawn on top in the context-reading color with a dark outline. The key sits above
    the panel in three columns by default.
    """
    fname = "make_head_ablation_scatter"
    rows, step = _accuracy_rows(df, fname=fname, step_pin=step_pin)
    x = _head_changes(rows, condition=x_measure[0], budget=x_measure[1], d_min=d_min, fname=fname)
    y = _head_changes(rows, condition=y_measure[0], budget=y_measure[1], d_min=d_min, fname=fname)
    points = x.merge(y, on=["seed", "head", "layer", "attention_rank"], suffixes=("_x", "_y"), validate="one_to_one")
    if not len(points) == len(x) == len(y):
        raise ValueError(f"{fname}: the two measures cover different heads ({len(x)} and {len(y)})")
    chosen = _top_heads(points, k, fname=fname)
    layers = sorted(int(layer) for layer in points["layer"].unique())
    if len(layers) > len(layer_colors):
        raise ValueError(f"{fname}: {len(layers)} layers for {len(layer_colors)} layer colors")
    shades = dict(zip(layers, layer_colors, strict=False))
    for layer in layers:
        part = points[points["layer"] == layer]
        ax.scatter(part["change_x"], part["change_y"], s=point_size, color=shades[layer], lw=0, zorder=_frames.POINT_ZORDER)
    ax.scatter(
        chosen["change_x"],
        chosen["change_y"],
        s=context_head_size,
        color=context_head_color,
        edgecolors=context_head_edge_color,
        linewidths=context_head_edge,
        zorder=_frames.LINE_ZORDER,
    )
    for draw in (ax.axhline, ax.axvline):
        draw(0.0, color=style.REFERENCE_LINE, lw=style.AXES_LINEWIDTH_PT, ls="--", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legend:
        handles = [
            Line2D([], [], ls="none", marker="o", markersize=math.sqrt(point_size) * KEY_POINT_SCALE, markerfacecolor=shades[layer], markeredgewidth=0, label=f"Layer {layer}")
            for layer in layers
        ]
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                markersize=math.sqrt(context_head_size),
                markerfacecolor=context_head_color,
                markeredgecolor=context_head_edge_color,
                markeredgewidth=context_head_edge,
                label=CONTEXT_HEADS_KEY,
            )
        )
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, handles=handles, **KEY_ABOVE, ncol=3)
    return {
        "k": int(k),
        "x_measure": list(x_measure),
        "y_measure": list(y_measure),
        "d_min": int(d_min),
        "step": step,
        "n_heads": int(len(points)),
        "n_trials_per_seed": int(max(points["n_x"].max(), points["n_y"].max())),
        "context_heads": {
            int(seed): [
                {"head": str(head), "attention_rank": int(rank), "change_x": round(float(cx), 4), "change_y": round(float(cy), 4)}
                for head, rank, cx, cy in part.sort_values("attention_rank")[["head", "attention_rank", "change_x", "change_y"]].itertuples(index=False)
            ]
            for seed, part in chosen.groupby("seed")
        },
        "n_seeds": int(points["seed"].nunique()),
    }


def _random_heads(rows: pd.DataFrame, *, condition: str, budget: int, size: int, d_min: int, fname: str) -> pd.DataFrame:
    """One measure per seed with `size` random heads ablated: the mean over the network's draws."""
    per_draw = _pooled(rows, condition=condition, budget=budget, d_min=d_min, fname=fname, schema=tables.ABLATE_SCHEMA, group_cols=("condition", "seed", "draw"), allowed=("heads", "draw"), set_kind=SET_RANDOM, n_heads=int(size))
    return per_draw.groupby(["condition", "seed"], sort=False).agg(value=("value", "mean"), n=("n", "first")).reset_index()


def _check_intact(shown: pd.DataFrame, intact: pd.DataFrame, *, condition: str, budget: int, fname: str) -> None:
    """Raise unless the two tables' intact networks agree: the same seeds and trials, and values
    within one trial of each other (a network scored on another device may differ in a near tie).
    """
    both = shown.merge(intact, on="seed", how="outer", suffixes=("_edit", "_ablate"), indicator=True, validate="one_to_one")
    one_sided = both.loc[both["_merge"] != "both", "seed"].tolist()
    if one_sided:
        raise ValueError(f"{fname}: seeds {sorted(one_sided)} appear in only one of the edit and ablate tables")
    if (both["n_edit"] != both["n_ablate"]).any():
        raise ValueError(f"{fname}: the edit and ablate tables count different trials on {condition} at {budget} steps")
    gap = (both["value_edit"] - both["value_ablate"]).abs()
    if (gap > 1.0 / both["n_edit"] + 1e-12).any():
        raise ValueError(f"{fname}: the intact networks of the edit and ablate tables differ on {condition} at {budget} steps")


def make_intervention_bars(
    ax: Axes,
    edit: pd.DataFrame,
    ablate: pd.DataFrame,
    *,
    k: int = 2,
    measures: Sequence[tuple[str, str, int]] = DEFAULT_MEASURES,
    d_min: int = 2,
    colors: Mapping[str, str] | None = None,
    show_chance: bool = True,
    show_seeds: bool = True,
    show_legend: bool = True,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = _frames.JITTER,
    bar_width: float = BAR_WIDTH,
    tick_rotation: float = 0.0,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    legend_loc: str = "lower center",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
) -> dict:
    """Grouped bars per measure: intact, the rank direction, the context-reading heads, and the
    controls.

    Estimand: for each measure (a condition at a pinned budget), the trial-weighted pool over
    distances at least `d_min` within a seed; then the mean and standard error over seeds, with
    one dot per seed drawn over the bars. Five bars: intact; the rank direction erased and its
    control, a random direction erased; the `k` heads each network ranks highest by their
    attention to the context ablated together, and their control, `k` random heads from outside
    the network's top 2k ablated together, whose value within a seed is the mean over the
    network's draws. Controls are open in the color of the manipulation they match; `colors`
    overrides a bar's color by its label. The intact bar reads the edit table and must agree with
    the ablate table's intact rows (`_check_intact`), and every bar must cover the same networks.
    The key, above the panel by default, pairs each manipulation with its control in one of
    three columns.
    """
    fname = "make_intervention_bars"
    edit_rows, step = _accuracy_rows(edit, fname=fname, step_pin=step_pin, edit_kind="erase")
    ablate_rows, ablate_step = _accuracy_rows(ablate, fname=fname, step_pin=step_pin)
    if step != ablate_step:
        raise ValueError(f"{fname}: the edit table is at step {step} and the ablate table at step {ablate_step}")
    size = int(k)

    def edited(direction: str, gain: float) -> Callable[[str, int], pd.DataFrame]:
        return lambda c, budget: _pooled(edit_rows, condition=c, budget=budget, d_min=d_min, fname=fname, schema=tables.EDIT_SCHEMA, direction=direction, gain=gain)

    def top_heads(c: str, budget: int) -> pd.DataFrame:
        return _pooled(ablate_rows, condition=c, budget=budget, d_min=d_min, fname=fname, schema=tables.ABLATE_SCHEMA, allowed=("heads",), set_kind=SET_TOP, n_heads=size)

    def random_heads(c: str, budget: int) -> pd.DataFrame:
        return _random_heads(ablate_rows, condition=c, budget=budget, size=size, d_min=d_min, fname=fname)

    # (label, color, open, per-seed values of one measure)
    variants: list[tuple[str, str, bool, Callable[[str, int], pd.DataFrame]]] = [
        (_gain_label(1.0), style.EDIT_INTACT, False, edited("rank", 1.0)),
        (_gain_label(0.0), style.EDIT_ERASED, False, edited("rank", 0.0)),
        (style.EDIT_CONTROL_LABEL, style.EDIT_ERASED, True, edited("random", CONTROL_GAIN)),
        (style.CONTEXT_HEADS_LABEL, style.CONTEXT_HEADS, False, top_heads),
        (style.RANDOM_HEADS_LABEL, style.CONTEXT_HEADS, True, random_heads),
    ]
    variants = [(name, (colors or {}).get(name, color), is_open, values) for name, color, is_open, values in variants]
    per_measure: dict[tuple[str, int], list[pd.DataFrame]] = {}
    for _, condition, budget in measures:
        values = [read(condition, int(budget)) for _, _, _, read in variants]
        intact = _pooled(ablate_rows, condition=condition, budget=budget, d_min=d_min, fname=fname, schema=tables.ABLATE_SCHEMA, set_kind=SET_INTACT)
        _check_intact(values[0], intact, condition=condition, budget=int(budget), fname=fname)
        for (name, _, _, _), per_seed in zip(variants, values, strict=True):
            if set(per_seed["seed"]) != set(intact["seed"]):
                raise ValueError(f"{fname}: '{name}' covers seeds {sorted(per_seed['seed'])}, not {sorted(intact['seed'])}")
        per_measure[(condition, int(budget))] = values
    rng = np.random.default_rng(0)
    centers = np.arange(len(measures), dtype=float)
    groups: dict[str, dict] = {label: {"condition": condition, "k_pin": int(budget), "bars": {}} for label, condition, budget in measures}
    handles: list[object] = []
    seeds = 0
    trials = 0
    for index, (name, color, is_open, _) in enumerate(variants):
        offset = (index - (len(variants) - 1) / 2.0) * bar_width
        means: list[float] = []
        errors: list[float] = []
        for center, (label, condition, budget) in zip(centers, measures, strict=True):
            per_seed = per_measure[(condition, int(budget))][index]
            agg = _frames.summarize(per_seed, group_cols=["condition"])
            mean = float(agg["value_mean"].iloc[0])
            se = float(agg["value_se"].iloc[0])
            means.append(mean)
            errors.append(0.0 if np.isnan(se) else se)
            if show_seeds:
                _frames.draw_seed_points(ax, per_seed.assign(x=center + offset), x_col="x", value_col="value", color=color, rng=rng, jitter=jitter, point_size=point_size, over_bars=True)
            groups[label]["bars"][name] = {"mean": round(mean, 4), "se": None if np.isnan(se) else round(se, 4), "per_seed": _frames.by_key(per_seed, "seed", "value")}
            seeds = max(seeds, _frames.n_seeds(agg))
            trials = max(trials, int(per_seed["n"].max()))
        handles.append(
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
        )
    ax.set_xticks(centers)
    ax.set_xticklabels([label for label, _, _ in measures], rotation=tick_rotation, ha="right" if tick_rotation else "center")
    ax.set_xlim(centers[0] - 0.5, centers[-1] + 0.5)
    _frames.finish(ax, x=None, y="value", ylim=ylim, chance=show_chance)
    if show_legend:
        ncol = int({**KEY_ABOVE, "ncol": 3, **dict(legend_kwargs or {})}["ncol"])
        # In three columns: intact alone, then each manipulation over its control.
        order = [handles[0], Patch(visible=False, label=" "), *handles[1:]] if ncol == 3 else handles
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs, handles=order, **KEY_ABOVE, ncol=3)
    return {
        "k": size,
        "measures": groups,
        "bars": [name for name, _, _, _ in variants],
        "d_min": int(d_min),
        "step": step,
        "n_trials_per_seed": trials,
        "n_seeds": seeds,
    }


def make_write_by_strength(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str = "frequent_cue_reversed",
    target: str = "unrelated",
    strengths: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    k_values: Sequence[int] = (0, 8),
    measure: str = "written_consistent_frac",
    d_min: int = 2,
    control: bool = True,
    control_ax: Axes | None = None,
    colors: Mapping[int, str] | None = None,
    show_chance: bool = True,
    show_band: bool = True,
    show_seeds: bool = True,
    show_legend: bool = True,
    band_alpha: float = _frames.BAND_ALPHA,
    marker_size: float = _frames.MARKER_SIZE,
    point_size: float = _frames.POINT_SIZE,
    jitter: float = 0.02,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    legend_loc: str = "upper left",
    legend_kwargs: Mapping[str, object] | None = None,
    step_pin: int | None = None,
) -> dict:
    """Agreement with the written ordering against write strength, per budget, with the control.

    Estimand: at each pinned budget in `k_values`, the trial-weighted pool of `measure` over
    distances at least `d_min` within a seed and strength; then the mean and standard error
    over seeds at each strength. The control is the same write along a matched random
    direction at full strength, drawn open in each budget's color: as a marker at the right edge
    of `ax`, or as bars on `control_ax` when one is given (it shares the y range and the seed-dot
    `jitter`, draws its seed dots over the bars, and carries no legend entry).
    """
    fname = "make_write_by_strength"
    rows, step = _accuracy_rows(df, fname=fname, step_pin=step_pin, condition=condition, edit_kind="write", target=target)
    if measure not in rows.columns:
        raise ValueError(f"{fname}: the rows carry no column '{measure}'")
    colors = {**style.budget_pair_colors(k_values), **(colors or {})}
    wanted = [float(s) for s in strengths]
    rng = np.random.default_rng(0)
    lines: dict[int, dict] = {}
    controls: dict[int, dict] = {}
    seeds = 0
    budgets = [int(k) for k in k_values]
    for index, k in enumerate(budgets):
        sub = tables.pin_k(tables.select(rows, fname=fname, direction="rank", strength=wanted), k)
        _frames.refuse_unpinned(sub, schema=tables.EDIT_SCHEMA, allowed=("strength",), fname=fname)
        present = sorted(float(s) for s in sub["strength"].unique())
        if present != sorted(wanted):
            raise ValueError(f"{fname}: the rows hold write strengths {present}, not {sorted(wanted)}")
        pooled = _frames.pool(sub, group_cols=["strength", "seed"], d_min=d_min, value_col=measure)
        agg = _frames.summarize(pooled, group_cols=["strength"], value_col=measure)
        color = colors[k]
        _frames.draw_series(ax, agg["strength"], agg[f"{measure}_mean"], agg[f"{measure}_se"], color=color, label=f"{k} {style.AXIS_LABELS['K'].lower()}", marker_size=marker_size, band=show_band, band_alpha=band_alpha)
        if show_seeds:
            _frames.draw_seed_points(ax, pooled, x_col="strength", value_col=measure, color=color, rng=rng, jitter=jitter, point_size=point_size)
        lines[k] = {"mean_by_strength": _frames.by_key(agg, "strength", f"{measure}_mean"), "se_by_strength": _frames.by_key(agg, "strength", f"{measure}_se")}
        seeds = max(seeds, _frames.n_seeds(agg))
        if control:
            matched = tables.pin_k(tables.select(rows, fname=fname, direction="random", strength=CONTROL_STRENGTH), k)
            _frames.refuse_unpinned(matched, schema=tables.EDIT_SCHEMA, fname=fname)
            pooled_control = _frames.pool(matched, group_cols=["strength", "seed"], d_min=d_min, value_col=measure)
            summary = _frames.summarize(pooled_control, group_cols=["strength"], value_col=measure)
            mean = float(summary[f"{measure}_mean"].iloc[0])
            se = float(summary[f"{measure}_se"].iloc[0])
            error = 0.0 if np.isnan(se) else se
            if control_ax is not None:
                control_ax.bar([index], [mean], width=CONTROL_BAR_WIDTH, yerr=[error], color=style.OPEN_MARKER_FACE, edgecolor=color, linewidth=style.AXES_LINEWIDTH_PT, error_kw={"ecolor": color, "lw": style.AXES_LINEWIDTH_PT, "capsize": _frames.ERRORBAR_CAPSIZE, "capthick": style.AXES_LINEWIDTH_PT}, zorder=_frames.LINE_ZORDER)
                if show_seeds:
                    _frames.draw_seed_points(control_ax, pooled_control.assign(x=float(index)), x_col="x", value_col=measure, color=color, rng=rng, jitter=jitter, point_size=point_size, over_bars=True)
            else:
                x = max(wanted) + 0.15 + 0.04 * index
                ax.errorbar([x], [mean], yerr=[error], fmt=CONTROL_MARKER, color=color, markerfacecolor=style.OPEN_MARKER_FACE, markeredgecolor=color, markersize=marker_size, elinewidth=style.AXES_LINEWIDTH_PT, capsize=_frames.ERRORBAR_CAPSIZE, capthick=style.AXES_LINEWIDTH_PT, zorder=5)
            controls[k] = {"strength": CONTROL_STRENGTH, "mean": round(mean, 4), "se": None if np.isnan(se) else round(se, 4), "per_seed": _frames.by_key(pooled_control, "seed", measure)}
    ax.set_xticks(wanted)
    ax.set_xticklabels([f"{s:g}" for s in wanted])
    pad = jitter + X_MARGIN * (max(wanted) - min(wanted) + 2.0 * jitter)
    ax.set_xlim(min(wanted) - pad, max(wanted) + (0.3 if control and control_ax is None else pad))
    _frames.finish(ax, x="strength", y=measure, ylim=ylim, chance=show_chance)
    if control and control_ax is None:
        ax.plot([], [], ls="none", marker=CONTROL_MARKER, color=style.REFERENCE_LINE, markerfacecolor=style.OPEN_MARKER_FACE, markersize=marker_size, label=style.WRITE_CONTROL_LABEL.replace("\n", " "))
    if control and control_ax is not None:
        control_ax.set_xticks([])
        margin = (1.0 - CONTROL_BAR_WIDTH) / 2.0
        control_ax.set_xlim(-0.5 - margin, len(budgets) - 0.5 + margin)
        control_ax.set_xlabel(style.WRITE_CONTROL_LABEL)
        control_ax.tick_params(axis="y", left=False, labelleft=False)
        control_ax.spines["left"].set_visible(False)
        control_ax.set_ylim(*ax.get_ylim())
        if show_chance:
            style.chance_line(control_ax)
    if show_legend:
        _frames.legend(ax, loc=legend_loc, kwargs=legend_kwargs)
    return {
        "condition": condition,
        "target": target,
        "measure": measure,
        "step": step,
        "d_min": int(d_min),
        "strengths": wanted,
        "k_values": budgets,
        "lines": lines,
        "control": controls if control else None,
        "n_seeds": seeds,
    }
