"""Shared plumbing for the figure scripts: table loading, sizing, saving, and the command line,
plus the content grid layout that both content grid figures, with and without the cue, draw,
the contexts whose reasoning steps to criterion the content figures draw, the settings several
figures pin, and the numbers of accuracy with reasoning written beside a reasoning-step panel's
lines.

Each `fig_<topic>.py` regenerates the transformer panels of one paper figure
from the committed tidy tables under `results/<experiment>/tables/` and writes
PDFs to `figures/out/`. Figures are built at their final physical size in
inches with constrained layout and saved without a tight bounding box, so a
PDF is placed in the manuscript at full scale and never resized. A figure with numbers beside
its lines is laid out once, as its PDF will be, and its layout engine then switched off.
"""

from __future__ import annotations

import argparse
import io
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from geo_alg_ti import config, numbers, plots, tables
from geo_alg_ti.plots import style

RESULTS_ROOT: Path = Path("results")
OUT_DIR: Path = Path("figures") / "out"
PREVIEW_FORMAT: str = "png"
LETTERS: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# The constrained-layout settings a figure is drawn with unless its script passes others: the
# space kept between the panels and the page edge, in inches, and the gap between neighboring
# panels as a fraction of their width or height.
W_PAD_IN: float = 0.042
H_PAD_IN: float = 0.042
WSPACE: float = 0.06
HSPACE: float = 0.06
# The file contract every saved figure meets, whether or not it was built under `figure()`:
# TrueType text in PDF and PostScript output, and the declared page size kept.
SAVE_RC: dict[str, object] = {"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "standard"}
_SCHEMAS = {
    "eval": tables.EVAL_SCHEMA,
    "eval_pairs": tables.PAIR_SCHEMA,
    "ablated_eval_pairs": tables.ABLATED_PAIR_SCHEMA,
    "train": tables.TRAIN_SCHEMA,
    "probe": tables.PROBE_SCHEMA,
    "edit": tables.EDIT_SCHEMA,
    "ablate": tables.ABLATE_SCHEMA,
    "head_scores": tables.HEAD_SCORES_SCHEMA,
}


def load(experiment: str, table: str = "eval", *, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Read one tidy table of one experiment, validated with the experiment's swept columns."""
    if table not in _SCHEMAS:
        raise ValueError(f"no tidy table is named '{table}'")
    swept = config.load_experiment(experiment).swept_columns
    path = Path(results_root) / experiment / "tables" / f"{table}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no '{table}' table at {path}; export the '{experiment}' experiment first")
    return tables.load_table(path, schema=_SCHEMAS[table], swept_columns=swept)


def pins(experiment: str, df: pd.DataFrame, *, fname: str, **fixed: object) -> pd.DataFrame:
    """Pin the experiment's swept axes to the paper's settings; raises if one is left varying."""
    swept = config.load_experiment(experiment).swept_columns
    return tables.apply_swept_pins(df, fixed=fixed, swept_columns=swept, fname=fname)


def blank_figure(
    *,
    width_in: float,
    height_in: float,
    w_pad: float = W_PAD_IN,
    h_pad: float = H_PAD_IN,
    wspace: float = WSPACE,
    hspace: float = HSPACE,
) -> Figure:
    """A figure at exact physical size with constrained layout and no axes yet.

    `w_pad` and `h_pad` are the space between the panels and the page edge in inches; `wspace`
    and `hspace` the gap between neighboring panels as a fraction of their width and height.
    """
    style.apply_style()
    fig = plt.figure(figsize=(width_in, height_in))
    fig.get_layout_engine().set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)
    return fig


def figure(
    nrows: int,
    ncols: int,
    *,
    width_in: float,
    height_in: float,
    w_pad: float = W_PAD_IN,
    h_pad: float = H_PAD_IN,
    wspace: float = WSPACE,
    hspace: float = HSPACE,
    **gridspec: object,
) -> tuple[Figure, np.ndarray]:
    """A `blank_figure` holding an `nrows` x `ncols` grid of axes, returned as a 2-D array."""
    fig = blank_figure(width_in=width_in, height_in=height_in, w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)
    axes = fig.subplots(nrows, ncols, squeeze=False, gridspec_kw=gridspec or None)
    return fig, axes


# The smallest symbolic distance a pool or a distance axis keeps: adjacent pairs need no
# inference, so they are left out.
D_MIN: int = numbers.D_MIN
# The numbers beside a reasoning panel's lines: accuracy at the largest budget of the sweep
# (eight steps), pooled over distances two to six, as `geo_alg_ti.numbers` recomputes them.
ACCURACY_NUMBERS: dict[str, object] = {"k_pin": None, "d_min": D_MIN}
# The setting at which the content figures draw one family of content effects networks, the one
# whose numbers `geo_alg_ti.numbers` recomputes: the frequent cue at one fifth of pretraining
# and p(CoT) 0.9.
FREQUENT_CUE_PROP = numbers.CONTENT_SETTING["frequent_cue_prop"]
P_COT = numbers.CONTENT_SETTING["p_cot"]
# The frequent-cue proportion the components networks were trained at, read from their
# experiment; the figures "Mechanisms underlying the geometric and algebraic strategies" and
# "Additional mechanism results" draw the rank-direction edits at the same proportion.
_COMPONENTS_RUN = config.enumerate_runs(config.load_experiment("components"))[0]
COMPONENTS_FREQUENT_CUE_PROP: float = _COMPONENTS_RUN.config.task.cue_distribution.frequent_cue_prop
# How many of each network's heads, ranked by their attention to the context, are ablated
# together or marked; a random set ablated as their control holds as many.
CONTEXT_HEADS = 2
# The criterion: the first budget reaching this fraction of a seed's best accuracy.
CRITERION_FRACTION = numbers.CRITERION_FRACTION


def place_numbers(fig: Figure, *axes: Axes) -> list[list[dict]]:
    """Place the numbers marked on `axes` once the layout has placed every axes.

    The figure is laid out as its PDF will be and its layout engine switched off, so no axes
    moves again; then each axes' marked numbers move the least until they clear its lines,
    markers, bands, legend and other texts (`plots.placement.place_texts`). Raises when a
    number still overlaps or crowds something, leaves its axes, enters a band or the legend, or
    sits nearer another line than its own (the audit `place_texts` runs with `check`). Returns
    each axes' placement records.
    """
    with matplotlib.rc_context(SAVE_RC), _quiet_font_subsetting():
        fig.savefig(io.BytesIO(), format="pdf")
    fig.set_layout_engine("none")
    return [plots.placement.place_texts(ax, check=True) for ax in axes]


def unreached_handles(*axes: Axes) -> list:
    """The legend handles of `axes` labeled `style.UNREACHED_LABEL`, in order."""
    return [
        handle
        for ax in axes
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True)
        if label == style.UNREACHED_LABEL
    ]


def budget_ticks(df: pd.DataFrame, step: int) -> list[int]:
    """Every `step`-th reasoning-step value in the table, from the smallest: ticks that stay on
    the sweep's budgets, below any unreached cross, which sits above the largest budget."""
    return sorted(int(k) for k in df["K"].unique())[:: int(step)]


def letter_panels(axes: Sequence, letters: Sequence[str] | None = None) -> None:
    """Add a bold letter to each axes, in reading order."""
    flat = [ax for ax in np.ravel(np.asarray(axes, dtype=object)) if ax is not None]
    names = list(letters) if letters is not None else list(LETTERS[: len(flat)])
    for ax, letter in zip(flat, names, strict=True):
        style.panel_letter(ax, letter)


# The content grid layout, drawn by both content grid figures; figure scripts import their helpers
# only from this module.
CONTENT_GRID_ROW_COL = "p_cot"
CONTENT_GRID_COLUMN_COL = "frequent_cue_prop"
# The distances drawn in the reversed-context block.
CONTENT_GRID_STEP_DISTANCES = (2, 3, 4, 5, 6)
# The contexts whose reasoning steps to criterion the content figures draw, by condition name.
CONTENT_STEPS_CONDITIONS = ("frequent_cue_congruent", "frequent_cue_reversed", "frequent_cue_shuffled")
CONTENT_GRID_HEIGHT_IN = style.MAX_HEIGHT_IN
CONTENT_GRID_BLOCK_HEIGHT_RATIOS = (3.15, 2.85, 3.1)
# The steps-to-criterion block: its ticks, every fourth budget (`budget_ticks`); its unreached
# crosses, smaller than the default and a budget and a quarter apart from one condition to the
# next, so crosses at one distance stay apart on the short panels; a range that keeps all three
# rows of crosses in view; and its axis label over two lines, since the label is longer than a
# panel is tall.
CONTENT_GRID_CRITERION_TICK_STEP = 4
CONTENT_GRID_UNREACHED = {"unreached_size": 10.0, "unreached_offset": 1.25}
CONTENT_GRID_CRITERION_YLIM = (-0.6, 12.2)
CONTENT_GRID_CRITERION_YLABEL = "\n".join(style.AXIS_LABELS[tables.K_TO_CRITERION].split(" ", 1))
CONTENT_GRID_ROW_HEADER_OFFSET_PT = 32.0
# Points between a column header and the axes below it.
CONTENT_GRID_COLUMN_HEADER_PAD = 3.0
# A block's letter sits left of its row header.
CONTENT_GRID_BLOCK_LETTER_OFFSET_PT = (-50.0, 2.0)
# Column headers break the setting's name over two lines: the grid's columns are narrow.
CONTENT_GRID_HEADER_FORMAT = {CONTENT_GRID_ROW_COL: "{label} = {value}", CONTENT_GRID_COLUMN_COL: "{label} {value}"}
CONTENT_GRID_LEGEND = {"handlelength": 1.5}
CONTENT_GRID_STEPS_MARKER_SIZE = 2.5


def _content_grid_header(column: str, value: float) -> str:
    """A row or column header naming the swept setting and its value."""
    label = style.AXIS_LABELS[column]
    if column == CONTENT_GRID_COLUMN_COL:
        label = "\n".join(label.split(" ", 1))
    return CONTENT_GRID_HEADER_FORMAT[column].format(label=label, value=plots.content.proportion_label(value))


def _content_grid_axes(block: SubFigure, rows: list[float], columns: list[float]) -> np.ndarray:
    """Equal axes on one grid of a block, with a header left of each row and above each column."""
    axes = block.subplots(len(rows), len(columns), sharex=True, sharey=True, squeeze=False)
    for r, value in enumerate(rows):
        axes[r, 0].annotate(_content_grid_header(CONTENT_GRID_ROW_COL, value), xy=(0.0, 0.5), xycoords="axes fraction", xytext=(-CONTENT_GRID_ROW_HEADER_OFFSET_PT, 0.0), textcoords="offset points", rotation=90, ha="right", va="center", fontweight="bold", annotation_clip=False)
    for c, value in enumerate(columns):
        axes[0, c].set_title(_content_grid_header(CONTENT_GRID_COLUMN_COL, value), fontweight="bold", pad=CONTENT_GRID_COLUMN_HEADER_PAD)
    return axes


def content_grid(experiment: str, *, results_root: Path = RESULTS_ROOT) -> Figure:
    """The content grid of one content effects experiment, drawn from its tidy tables.

    The three blocks are grids over p(CoT) (rows) and the frequent-cue proportion (columns):
    accuracy by distance under the four contexts at zero steps, the shuffled context's distance
    taken in the ordering shown; reversed-context accuracy by reasoning steps; and the
    reasoning steps the congruent, reversed, and shuffled contexts need to reach ninety percent
    of their best accuracy, by distance, a distance whose best accuracy most seeds cannot lift
    above chance marked unreached and breaking its line. A figure-level legend under A names
    the contexts, and one under C names the unreached mark when a panel holds one.
    """
    df = load(experiment, "eval", results_root=results_root)
    rows = sorted(float(v) for v in df[CONTENT_GRID_ROW_COL].unique())
    columns = sorted(float(v) for v in df[CONTENT_GRID_COLUMN_COL].unique())
    fig = blank_figure(width_in=style.FULL_WIDTH_IN, height_in=CONTENT_GRID_HEIGHT_IN)
    blocks = fig.subfigures(3, 1, height_ratios=CONTENT_GRID_BLOCK_HEIGHT_RATIOS)
    by_distance = _content_grid_axes(blocks[0], rows, columns)
    by_steps = _content_grid_axes(blocks[1], rows, columns)
    to_criterion = _content_grid_axes(blocks[2], rows, columns)
    for r, p_cot in enumerate(rows):
        for c, proportion in enumerate(columns):
            sub = pins(experiment, df, fname=experiment, **{CONTENT_GRID_ROW_COL: p_cot, CONTENT_GRID_COLUMN_COL: proportion})
            plots.make_context_accuracy_by_distance(by_distance[r, c], sub, k_pin=0, show_legend=False)
            plots.make_context_accuracy_by_reasoning_steps(by_steps[r, c], sub, contexts=("reversed",), d_values=CONTENT_GRID_STEP_DISTANCES, band_alpha=0.18, marker_size=CONTENT_GRID_STEPS_MARKER_SIZE, show_legend=False)
            plots.make_steps_to_criterion_by_distance(to_criterion[r, c], sub, conditions=CONTENT_STEPS_CONDITIONS, fraction=CRITERION_FRACTION, unreached_below_chance=True, break_at_unreached=True, **CONTENT_GRID_UNREACHED, band_alpha=0.18, marker_size=CONTENT_GRID_STEPS_MARKER_SIZE, show_seeds=False, show_legend=False, ylim=CONTENT_GRID_CRITERION_YLIM)
            to_criterion[r, c].set_ylabel(CONTENT_GRID_CRITERION_YLABEL)
            for ax in (by_distance[r, c], by_steps[r, c], to_criterion[r, c]):
                if r != len(rows) - 1:
                    ax.set_xlabel("")
                if c != 0:
                    ax.set_ylabel("")
            for ax in (by_distance[r, c], by_steps[r, c]):
                ax.set_yticks((0.0, 0.5, 1.0))
            by_steps[r, c].set_xticks((0, 2, 4, 6, 8))
            to_criterion[r, c].set_yticks(budget_ticks(sub, CONTENT_GRID_CRITERION_TICK_STEP))
    handles, labels = by_distance[0, 0].get_legend_handles_labels()
    blocks[0].legend(handles, labels, loc="outside lower center", ncol=len(labels), **CONTENT_GRID_LEGEND)
    unreached = unreached_handles(*to_criterion.ravel())
    if unreached:
        blocks[2].legend(unreached[:1], [style.UNREACHED_LABEL], loc="outside lower center", **CONTENT_GRID_LEGEND)
    for ax, letter in zip((by_distance[0, 0], by_steps[0, 0], to_criterion[0, 0]), "ABC", strict=True):
        style.panel_letter(ax, letter, offset_pt=CONTENT_GRID_BLOCK_LETTER_OFFSET_PT)
    return fig


@contextmanager
def _quiet_font_subsetting() -> Iterator[None]:
    """Hold back the font subsetting reports while a figure is written; they describe the fonts."""
    logger = logging.getLogger("fontTools")
    level = logger.level
    logger.setLevel(max(level, logging.ERROR))
    try:
        yield
    finally:
        logger.setLevel(level)


def save(fig: Figure, name: str, *, out_dir: Path = OUT_DIR, formats: Sequence[str] = ("pdf",)) -> list[Path]:
    """Write the figure in each format plus a PNG preview, close it, and return the paths.

    The files meet `SAVE_RC` whatever parameters the figure was drawn under: text stays
    TrueType and the page keeps its declared physical size. The PDF embeds rasterized artists
    at 600 dpi; the preview is rendered at 200 dpi.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(dict.fromkeys([*formats, PREVIEW_FORMAT]))
    paths = []
    with matplotlib.rc_context(SAVE_RC), _quiet_font_subsetting():
        for fmt in wanted:
            path = out_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=style.PREVIEW_DPI if fmt == PREVIEW_FORMAT else style.RASTER_DPI, bbox_inches=None)
            paths.append(path)
    plt.close(fig)
    return paths


def main(build: Callable[[argparse.Namespace], Figure], *, name: str) -> None:
    """Parse `--results`, `--out`, `--format`, build the figure, and save it."""
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=f"redraw the {name.replace('_', ' ')} figure from the tidy tables")
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT, help="directory holding <experiment>/tables/")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="directory the figure files are written to")
    parser.add_argument("--format", action="append", default=None, help="an output format; repeat for several (pdf)")
    args = parser.parse_args()
    fig = build(args)
    for path in save(fig, name, out_dir=args.out, formats=tuple(args.format or ("pdf",))):
        print(path)
