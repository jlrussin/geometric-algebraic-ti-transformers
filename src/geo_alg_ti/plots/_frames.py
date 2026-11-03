"""Selection, estimand, and drawing steps shared by every panel.

A panel applies its own pins and hands the rows here. `refuse_unpinned`
rejects any column beyond the table's schema, and any identity column a
panel must hold fixed, that still varies after the pins, so a swept axis left
unpinned is refused without the panel knowing the experiment. `pin_step`
picks one evaluation step (the last, by default). `reduce_steps` applies one
of the reasoning-step estimands of `geo_alg_ti.tables`: a pinned budget, the
mean over a set of budgets, or the best over the sweep. `pool` and
`summarize` wrap the trial-weighted distance pool and the seed summary, so
seeds are the only averaged axis across networks. The drawing helpers keep the band, line,
and per-seed dot idiom identical across the panel modules, and `legend`
places a legend with the panel's defaults under the caller's keyword
arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from geo_alg_ti import tables
from geo_alg_ti.plots import style

# Schema columns a panel must hold at one value unless it draws or loops over them.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "tag", "step", "edit_kind", "direction", "target", "gain", "strength",
    "set_kind", "head", "n_heads", "heads", "draw", "attention_rank",
)
# How a panel with reasoning steps on the x axis reduces over them: the accuracy at each budget,
# or the running best over the budgets up to it within each seed (`tables.running_best_over_k`).
BUDGET_AXIS_STATISTICS: tuple[str, ...] = ("raw", "running_best")
DEFAULT_MEAN_BUDGETS: tuple[int, ...] = tables.DEFAULT_MEAN_BUDGETS
BAND_ALPHA: float = 0.25
POINT_SIZE: float = 3.0
POINT_ALPHA: float = 0.35
JITTER: float = 0.06
MARKER_SIZE: float = 3.0
# Drawing order: the standard-error band lies under the chance line, the per-seed dots over
# it, and the line and its markers on top.
BAND_ZORDER: int = 1
POINT_ZORDER: int = 2
LINE_ZORDER: int = 3
# The cap length of every error bar, in points.
ERRORBAR_CAPSIZE: float = 1.5


def select_distances(df: pd.DataFrame, d_values: Sequence[int], *, fname: str) -> pd.DataFrame:
    """Rows at the requested symbolic distances; raises naming any distance the rows lack."""
    wanted = [int(d) for d in d_values]
    missing = sorted(set(wanted) - {int(d) for d in df["d"].unique()})
    if missing:
        raise ValueError(f"{fname}: the rows hold no symbolic distance {missing}")
    return tables.select(df, fname=fname, d=wanted)


def pin_step(df: pd.DataFrame, step_pin: int | None, *, fname: str) -> tuple[pd.DataFrame, int]:
    """Rows at one evaluation step: `step_pin`, or the last step when it is None."""
    if "step" not in df.columns:
        raise ValueError(f"{fname}: the rows carry no 'step' column to pin")
    step = int(df["step"].max()) if step_pin is None else int(step_pin)
    return tables.select(df, fname=fname, step=step), step


def refuse_unpinned(df: pd.DataFrame, *, schema: Mapping[str, str], allowed: Sequence[str] = (), fname: str) -> None:
    """Raise, naming the column, if a swept axis or an identity column still varies."""
    for name in df.columns:
        if name in allowed:
            continue
        if name in schema and name not in IDENTITY_COLUMNS:
            continue
        remaining = df[name].dropna().unique()
        if len(remaining) > 1:
            values = sorted(remaining, key=style._sort_key)
            raise ValueError(f"{fname} draws one setting at a time; pin '{name}', which still holds {values}")


def reduce_steps(
    df: pd.DataFrame,
    *,
    k_stat: str,
    k_pin: int | None,
    k_values: Sequence[int],
    group_cols: Sequence[str],
    fname: str,
) -> pd.DataFrame:
    """Apply one reasoning-step estimand within each group, which always includes the seed.

    A pinned budget left at None is zero reasoning steps.
    """
    return tables.reduce_over_k(df, k_stat=k_stat, k_pin=k_pin, k_values=k_values, group_cols=group_cols, fname=fname)


def budgets_used(df: pd.DataFrame, k_stat: str, k_pin: int | None, k_values: Sequence[int]) -> list[int]:
    """The reasoning-step values an estimand reads from the rows."""
    present = {int(k) for k in df["K"].unique()}
    if k_stat == "pin":
        return [0 if k_pin is None else int(k_pin)]
    if k_stat == "mean":
        return sorted(present & {int(k) for k in k_values})
    return sorted(present)


def pool(df: pd.DataFrame, *, group_cols: Sequence[str], d_min: int, value_col: str = "value") -> pd.DataFrame:
    """Trial-weighted mean over symbolic distances at least `d_min` within each group."""
    return tables.pool_over_distance(df, group_cols=list(group_cols), value_col=value_col, d_min=int(d_min))


def summarize(df: pd.DataFrame, *, group_cols: Sequence[str], value_col: str = "value") -> pd.DataFrame:
    """Mean and standard error over seeds per group, sorted by the grouping columns."""
    agg = tables.seed_mean_se(df, group_cols=list(group_cols), value_col=value_col)
    return agg.sort_values(list(group_cols)).reset_index(drop=True) if group_cols else agg


def n_seeds(agg: pd.DataFrame) -> int:
    """The seed count behind a summary."""
    return int(agg["n_seeds"].max())


def by_key(agg: pd.DataFrame, key_col: str, value_col: str) -> dict:
    """A JSON-friendly `{key: value}` of one summary column, with NaN written as None."""
    out = {}
    for key, value in zip(agg[key_col], agg[value_col], strict=True):
        out[_plain(key)] = None if pd.isna(value) else round(float(value), 4)
    return out


def _plain(value: object) -> object:
    """A plain Python key: integers stay integers, other numbers become floats."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else float(value)
    return str(value)


def line_family(df: pd.DataFrame, line_col: str | None, cmap: str | None, colors: Mapping[object, str] | None, *, fname: str) -> tuple[list[object], dict[object, str]]:
    """The values of a line family and their colors: `colors` if given, else the column's map."""
    if line_col is None:
        return [None], {None: ""}
    if line_col not in df.columns:
        raise ValueError(f"{fname}: the rows carry no column '{line_col}'")
    values = sorted(df[line_col].dropna().unique(), key=style._sort_key)
    if colors is not None:
        return values, {v: colors[v] for v in values}
    name = cmap if cmap is not None else style.COLUMN_CMAPS.get(line_col)
    if name is None:
        raise ValueError(f"{fname}: no colormap is registered for '{line_col}'; pass cmap or colors")
    return values, dict(zip(values, style.sweep_colors(values, name, column=line_col), strict=True))


def draw_series(
    ax: Axes,
    x: Sequence[float],
    mean: Sequence[float],
    se: Sequence[float] | None,
    *,
    color: str,
    label: str | None,
    marker: str | None = "o",
    marker_size: float = MARKER_SIZE,
    line_width: float | None = None,
    ls: str = "-",
    open_marker: bool = False,
    band: bool = True,
    band_alpha: float = BAND_ALPHA,
    zorder: int = LINE_ZORDER,
) -> None:
    """A line with `marker_size`-point markers and, when the standard error is known, its band.

    The line is `line_width` points wide, or the shared width when None; the band always lies
    at `BAND_ZORDER`, under the chance line and the per-seed dots.
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if band and se is not None:
        spread = np.asarray(se, dtype=float)
        if np.isfinite(spread).any():
            spread = np.nan_to_num(spread)
            ax.fill_between(x, mean - spread, mean + spread, color=color, alpha=band_alpha, lw=0, zorder=BAND_ZORDER)
    ax.plot(
        x,
        mean,
        color=color,
        ls=ls,
        marker=marker,
        markersize=marker_size,
        markerfacecolor=style.OPEN_MARKER_FACE if open_marker else color,
        markeredgecolor=color,
        label=label,
        zorder=zorder,
        **({} if line_width is None else {"lw": line_width}),
    )


def draw_seed_points(
    ax: Axes,
    per_seed: pd.DataFrame,
    *,
    x_col: str,
    value_col: str,
    color: str,
    positions: Mapping[object, float] | None = None,
    open_marker: bool = False,
    marker: str = "o",
    jitter: float = JITTER,
    log_x: bool = False,
    point_size: float = POINT_SIZE,
    rng: np.random.Generator | None = None,
    over_bars: bool = False,
) -> None:
    """One faint dot per seed at each x value, jittered so coincident seeds stay visible.

    The jitter is added to the x value, or, on a log axis (`log_x`), applied as a fraction of
    it, so the spread looks the same at every decade. Dots drawn `over_bars` sit above the bars
    as opaque white dots edged in `color`, so a seed below a bar's top stays visible on any fill.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    for x_value in sorted(per_seed[x_col].unique(), key=style._sort_key):
        values = per_seed.loc[per_seed[x_col] == x_value, value_col].to_numpy(dtype=float)
        center = float(positions[x_value]) if positions is not None else float(x_value)
        offsets = rng.uniform(-jitter, jitter, size=len(values))
        xs = center * (1.0 + offsets) if log_x else np.full(len(values), center) + offsets
        if over_bars:
            ax.scatter(xs, values, s=point_size, marker=marker, facecolors=style.OPEN_MARKER_FACE, edgecolors=color, lw=0.5, zorder=LINE_ZORDER + 1)
        elif open_marker:
            ax.scatter(xs, values, s=point_size, marker=marker, facecolors="none", edgecolors=color, alpha=POINT_ALPHA, lw=0.5, zorder=POINT_ZORDER)
        else:
            ax.scatter(xs, values, s=point_size, marker=marker, color=color, alpha=POINT_ALPHA, lw=0, zorder=POINT_ZORDER)


def legend(ax: Axes, *, loc: str, kwargs: Mapping[str, object] | None = None, **defaults: object) -> None:
    """Place the legend at `loc`; `kwargs` overrides the panel's defaults, `loc` included."""
    ax.legend(**{"loc": loc, **defaults, **dict(kwargs or {})})


def ordinal_axis(ax: Axes, values: Sequence[object], *, order: Sequence[object] | None = None) -> dict[object, float]:
    """Evenly spaced positions for a categorical or unevenly spaced sweep, labeled by value."""
    ordered = list(order) if order is not None else sorted(set(values), key=style._sort_key)
    positions = {value: float(i) for i, value in enumerate(ordered)}
    ax.set_xticks(list(range(len(ordered))))
    ax.set_xticklabels([_format(value) for value in ordered])
    ax.set_xlim(-0.5, len(ordered) - 0.5)
    return positions


def _format(value: object) -> str:
    """A tick label for one sweep value."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def override_axes(ax: Axes, *, xlim: tuple[float, float] | None = None, yticks: Sequence[float] | None = None, ylabel: str | None = None) -> None:
    """Apply a panel's optional x limits, y ticks, and y label after `finish`; None keeps each."""
    if xlim is not None:
        ax.set_xlim(*xlim)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def finish(ax: Axes, *, x: str | None, y: str | None, ylim: tuple[float, float] | None = (-0.02, 1.02), chance: bool = False) -> None:
    """Axis labels from the shared table, the accuracy range, and the chance reference."""
    style.label_axes(ax, x, y)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if chance:
        style.chance_line(ax)
