"""Colors, fonts, sizes, and axis labels shared by every panel.

IWL is red and ICL is blue throughout the paper; the arbitrage and baseline
tests have their own colors; zero reasoning steps is gray and the largest
budget is orange so the two never collide with the IWL and ICL colors. The
four contexts of the content battery keep one color each, and the agreeing and
conflicting halves of the shuffled context take a light and a dark shade of
the shuffled brown. Every gradient
(reasoning steps, symbolic distance within a test, skew alpha, p(CoT),
replay proportion, learning rate, frequent-cue proportion) comes from a named
colormap, never from the red and blue that mark the two tests. Fonts are
real points at final print size: 7 pt body, 6 pt ticks and legends, 9 pt
panel letters. PDFs keep editable text (TrueType, fonttype 42) and are saved
with constrained layout, never with a tight bounding box, so physical sizes
are preserved. Axis labels for the table columns are defined here, so `K` and
`d` never appear on an axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, to_hex

from geo_alg_ti import CHANCE_ACCURACY as CHANCE
from geo_alg_ti import tables

IWL: str = "#d62728"
ICL: str = "#1f77b4"
ARBITRAGE: str = "#9467bd"
BASELINE: str = "#2ca02c"
NO_STEPS: str = "#7f7f7f"
MAX_STEPS: str = "#ff7f0e"
GEOMETRIC: str = "#c0392b"
ALGEBRAIC: str = "#2a78d6"
REFERENCE_LINE: str = "#7f7f7f"
# The congruent, reversed and shuffled contexts of the content battery, and the shuffled
# context's trials whose shown order agrees with the frequent cue's ordering (light) or conflicts
# with it (dark).
CONGRUENT: str = "#3fb780"
REVERSED: str = "#ff7f0e"
SHUFFLED: str = "#875a5a"
SHUFFLED_AGREE: str = "#eac2c7"
SHUFFLED_CONFLICT: str = "#54332d"
# Scalings of the rank direction: intact near black, erased teal, inverted magenta. A control
# takes the color of the manipulation it matches, drawn open: the random direction erased in
# teal, random heads ablated in the purple of the context-reading heads ablated.
EDIT_INTACT: str = "#222222"
EDIT_ERASED: str = "#1b9e77"
EDIT_INVERTED: str = "#e7298a"
CONTEXT_HEADS: str = "#7570b3"
# Finetuning without reasoning, beside the reasoning conditions drawn in the colors above.
NO_REASONING: str = "#a6a6a6"
# The attention heads of the head scatter, light for the first layer to dark for the last.
HEAD_LAYER_SHADES: tuple[str, ...] = ("#c6c6c6", "#9a9a9a", "#6e6e6e", "#424242")
OPEN_MARKER_FACE: str = "#ffffff"

FONT_FAMILY: tuple[str, ...] = ("Helvetica", "Arial", "DejaVu Sans")
FONT_BODY_PT: float = 7.0
FONT_TICK_PT: float = 6.0
FONT_LEGEND_PT: float = 6.0
FONT_TITLE_PT: float = 7.0
FONT_PANEL_LETTER_PT: float = 9.0
AXES_LINEWIDTH_PT: float = 0.6
LINE_WIDTH_PT: float = 1.0
TICK_WIDTH_PT: float = 0.6
TICK_LENGTH_PT: float = 2.5
PANEL_LETTER_OFFSET_PT: tuple[float, float] = (-14.0, 2.0)
RASTER_DPI: int = 600
PREVIEW_DPI: int = 200

# Physical figure sizes: one column, the full page width, and the tallest figure allowed.
COLUMN_WIDTH_IN: float = 3.5
FULL_WIDTH_IN: float = 7.0
MAX_HEIGHT_IN: float = 8.75

STEPS_CMAP: str = "viridis"
SKEW_CMAP: str = "viridis"
ICL_DISTANCE_CMAP: str = "Blues"
BASELINE_DISTANCE_CMAP: str = "Greens"
REVERSED_DISTANCE_CMAP: str = "Oranges"
# The congruent context's distance family runs through the congruent color itself, from near white
# to a dark green of the same hue, so its shades match the context's line; registered with
# matplotlib so it is named like the built-in colormaps here.
CONGRUENT_DISTANCE_CMAP: str = "congruent_greens"
# The neutral colormap of a distance key that stands for several contexts' families at once.
DISTANCE_KEY_CMAP: str = "Greys"
colormaps.register(LinearSegmentedColormap.from_list(CONGRUENT_DISTANCE_CMAP, ("#f2fbf6", CONGRUENT, "#0b3d2c")), force=True)
P_COT_CMAP: str = "plasma"
REPLAY_CMAP: str = "Purples"
LEARNING_RATE_CMAP: str = "cividis"
FREQUENT_CUE_CMAP: str = "YlGnBu"
# The colormap a swept column is drawn with when it becomes a line family.
COLUMN_CMAPS: Mapping[str, str] = {
    "K": STEPS_CMAP,
    "skew_alpha": SKEW_CMAP,
    "p_cot": P_COT_CMAP,
    "replay_prop": REPLAY_CMAP,
    "learning_rate": LEARNING_RATE_CMAP,
    "frequent_cue_prop": FREQUENT_CUE_CMAP,
}
# The slice of its colormap each swept column is sampled over, by its lines and its colorbar
# alike; single-hue maps start above their pale end, the perceptual maps stop short of theirs.
COLUMN_CMAP_SPANS: Mapping[str, tuple[float, float]] = {
    "K": (0.0, 0.92),
    "skew_alpha": (0.0, 0.85),
    "p_cot": (0.0, 0.9),
    "replay_prop": (0.45, 0.95),
    "learning_rate": (0.05, 0.95),
    "frequent_cue_prop": (0.28, 1.0),
}
# The slice every other gradient is sampled over: single-hue maps are pale at zero, so they
# are sampled from 0.35 up; the perceptual maps are pale at one, so they are sampled up to 0.85.
SPAN_PALE_AT_ZERO: tuple[float, float] = (0.35, 1.0)
SPAN_PALE_AT_ONE: tuple[float, float] = (0.0, 0.85)
_PALE_AT_ONE: frozenset[str] = frozenset({"viridis", "plasma", "cividis", "inferno", "magma"})

# The four contexts of the content battery, plus the ICL test when it is drawn beside them and
# the two halves of the shuffled context that `tables.split_shuffled` derives.
CONTEXT_COLORS: Mapping[str, str] = {
    "uninformative": IWL,
    "congruent": CONGRUENT,
    "reversed": REVERSED,
    "shuffled": SHUFFLED,
    "shuffled_agree": SHUFFLED_AGREE,
    "shuffled_conflict": SHUFFLED_CONFLICT,
    "icl_test": ICL,
}
CONTEXT_LABELS: Mapping[str, str] = {
    "uninformative": "Uninformative",
    "congruent": "Congruent",
    "reversed": "Reversed",
    "shuffled": "Shuffled",
    "shuffled_agree": "Shuffled (agree)",
    "shuffled_conflict": "Shuffled (conflict)",
    "icl_test": "ICL test",
}
# The colormap a context's distance family is drawn with, one shade per distance.
CONTEXT_DISTANCE_CMAPS: Mapping[str, str] = {
    "uninformative": "Reds",
    "congruent": CONGRUENT_DISTANCE_CMAP,
    "reversed": REVERSED_DISTANCE_CMAP,
    "shuffled": "copper",
    "shuffled_agree": "pink_r",
    "shuffled_conflict": "copper_r",
    "icl_test": ICL_DISTANCE_CMAP,
}
# A test condition's color and label; a content battery condition on the frequent cue's
# ordering takes those of its context.
CONDITION_COLORS: Mapping[str, str] = {
    "iwl_test": IWL,
    "icl_test": ICL,
    "icl_test_training_cue": ICL,
    "baseline_test": BASELINE,
    "arbitrage_test": ARBITRAGE,
    **{f"frequent_cue_{context}": color for context, color in CONTEXT_COLORS.items() if context != "icl_test"},
}
CONDITION_LABELS: Mapping[str, str] = {
    "iwl_test": "IWL test",
    "icl_test": "ICL test",
    "icl_test_training_cue": "ICL test (training cue)",
    "baseline_test": "Baseline test",
    "arbitrage_test": "Arbitrage test",
    **{f"frequent_cue_{context}": label for context, label in CONTEXT_LABELS.items() if context != "icl_test"},
}
# A symbolic distance whose best accuracy is not above chance for most seeds has no number of
# reasoning steps to criterion; it is marked with a cross in its condition's color above the
# largest budget, each condition raised by the offset (a fraction of the budget spacing) over
# the one before, and a gray cross keys the marks in the legend.
UNREACHED_MARKER: str = "x"
UNREACHED_SIZE: float = 12.0
UNREACHED_OFFSET: float = 0.5
UNREACHED_LABEL: str = "unreached"
# The two response classes of the arbitrage test: which ordering an answer follows. The
# trained ordering is the one finetuning fixed; the context ordering is the one shown.
RESPONSE_CLASS_COLORS: Mapping[str, str] = {
    "prior_consistent_frac": IWL,
    "context_consistent_frac": ICL,
}
RESPONSE_CLASS_LABELS: Mapping[str, str] = {
    "prior_consistent_frac": "Trained ordering",
    "context_consistent_frac": "Context ordering",
}
EDIT_GAIN_COLORS: Mapping[float, str] = {1.0: EDIT_INTACT, 0.0: EDIT_ERASED, -1.0: EDIT_INVERTED}
EDIT_GAIN_LABELS: Mapping[float, str] = {
    1.0: "Intact",
    0.0: "Rank direction erased",
    -1.0: "Rank direction inverted",
}
EDIT_CONTROL_LABEL: str = "Random direction erased"
CONTEXT_HEADS_LABEL: str = "Context-reading heads ablated"
RANDOM_HEADS_LABEL: str = "Random heads ablated"
# The companion strip of the write panel is narrow, so its label takes two lines.
WRITE_CONTROL_LABEL: str = "Random\ndirection"
# The legend title of a distance family; the axis label spells symbolic distance out.
DISTANCE_LEGEND_TITLE: str = "Distance"
# The tick labels of the probe panel's sites: the item embeddings, each block, the output.
EMBEDDING_SITE_LABEL: str = "Embed."
OUTPUT_SITE_LABEL: str = "Output"
SKEW_LABEL_FORMAT: str = "α = {:g}"
AXIS_LABELS: Mapping[str, str] = {
    **tables.COLUMN_LABELS,
    "n_orderings": "Number of orderings",
    "replay_prop": "Replay proportion",
    "learning_rate": "Learning rate",
    "layer": "Layer",
    "strength": "Write strength",
    "chain_algorithm": "Chain algorithm",
    "layout": "Layout",
    "n_layers": "Layers",
    "d_model": "Width",
    "feedforward": "Feedforward",
    "context_consistent_frac": "Agreement with the context",
    "prior_consistent_frac": "Agreement with the trained ordering",
    "written_consistent_frac": "Agreement with written ordering",
    "response_class": "Proportion of responses",
    "probe": "Probe accuracy",
    "gain": "Accuracy gain from reasoning",
    "pair": "Item pair",
    "ordering": "Ordering",
    "frequency": "Training frequency",
}


def apply_style() -> None:
    """Set the shared matplotlib parameters (fonts, fonttype 42, constrained layout)."""
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": list(FONT_FAMILY),
            "font.size": FONT_BODY_PT,
            "axes.labelsize": FONT_BODY_PT,
            "axes.titlesize": FONT_TITLE_PT,
            "xtick.labelsize": FONT_TICK_PT,
            "ytick.labelsize": FONT_TICK_PT,
            "legend.fontsize": FONT_LEGEND_PT,
            "axes.linewidth": AXES_LINEWIDTH_PT,
            "lines.linewidth": LINE_WIDTH_PT,
            "xtick.major.width": TICK_WIDTH_PT,
            "ytick.major.width": TICK_WIDTH_PT,
            "xtick.major.size": TICK_LENGTH_PT,
            "ytick.major.size": TICK_LENGTH_PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            "savefig.bbox": "standard",
        }
    )


def column_span(column: str | None, cmap: str) -> tuple[float, float]:
    """The slice of `cmap` a gradient over `column` samples.

    A swept column drawn with its own colormap takes its span in `COLUMN_CMAP_SPANS`; any
    other gradient takes the default of its colormap family.
    """
    if COLUMN_CMAPS.get(column) == cmap:
        return COLUMN_CMAP_SPANS[column]
    return SPAN_PALE_AT_ONE if cmap in _PALE_AT_ONE else SPAN_PALE_AT_ZERO


def sweep_colors(values: Sequence[object], cmap: str, *, column: str | None = None, by_value: bool = False, span: tuple[float, float] | None = None) -> list[str]:
    """One color per value of a line family, sampled from `cmap` so every line stays visible.

    The colors come from the slice of `cmap` that `column_span` names for `column`. Values are
    placed evenly in sorted order, or, with `by_value`, at their numeric position between the
    smallest and the largest (or between the ends of `span`, which implies `by_value`), so that
    the same number is the same color wherever that family is drawn. A single value takes the
    far end of the slice, however often it is repeated, unless a span places it.
    """
    ordered = list(values)
    low, high = column_span(column, cmap)
    if span is not None:
        numbers = np.asarray([float(v) for v in ordered])
        start, stop = float(span[0]), float(span[1])
        if stop <= start:
            raise ValueError(f"sweep_colors: span {span} must run upward")
        positions = [low + (high - low) * min(max((v - start) / (stop - start), 0.0), 1.0) for v in numbers]
    elif len(set(ordered)) == 1:
        positions = [high] * len(ordered)
    elif by_value:
        numbers = np.asarray([float(v) for v in ordered])
        extent = numbers.max() - numbers.min()
        positions = [low + (high - low) * ((v - numbers.min()) / extent if extent else 1.0) for v in numbers]
    else:
        ranks = {v: i for i, v in enumerate(sorted(set(ordered), key=_sort_key))}
        n = len(ranks)
        positions = [low + (high - low) * ranks[v] / (n - 1) for v in ordered]
    palette = colormaps[cmap]
    return [to_hex(palette(p)) for p in positions]


def sweep_cmap(cmap: str, *, column: str | None = None) -> ListedColormap:
    """The part of `cmap` that `sweep_colors` samples for `column`, for a colorbar keyed to it."""
    low, high = column_span(column, cmap)
    return ListedColormap(colormaps[cmap](np.linspace(low, high, 256)), name=f"{cmap}_sweep")


def _sort_key(value: object) -> tuple[int, object]:
    """Numbers before strings, so a mixed family still orders."""
    return (0, float(value)) if isinstance(value, (int, float, np.integer, np.floating)) else (1, str(value))


def steps_colors(k_values: Sequence[int], cmap: str = STEPS_CMAP) -> list[str]:
    """One color per reasoning-step value from a sequential colormap."""
    return sweep_colors([int(k) for k in k_values], cmap, column="K")


def distance_colors(d_values: Sequence[int], cmap: str = ICL_DISTANCE_CMAP) -> list[str]:
    """One color per symbolic distance from a sequential colormap."""
    return sweep_colors([int(d) for d in d_values], cmap)


def budget_pair_colors(k_values: Sequence[int]) -> dict[int, str]:
    """Colors for a few budgets: zero steps gray, the largest budget orange, others in between."""
    ordered = sorted(int(k) for k in k_values)
    colors = dict(zip(ordered, steps_colors(ordered), strict=True))
    if 0 in colors:
        colors[0] = NO_STEPS
    if ordered and ordered[-1] > 0:
        colors[ordered[-1]] = MAX_STEPS
    return colors


def panel_letter(ax: Axes, letter: str, *, offset_pt: tuple[float, float] = PANEL_LETTER_OFFSET_PT) -> None:
    """Place a bold panel letter at the top left of the axes, `offset_pt` from its corner."""
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=offset_pt,
        textcoords="offset points",
        fontsize=FONT_PANEL_LETTER_PT,
        fontweight="bold",
        ha="left",
        va="bottom",
        annotation_clip=False,
    )


def chance_line(ax: Axes, *, level: float = CHANCE) -> None:
    """Draw the dashed chance reference."""
    ax.axhline(level, color=REFERENCE_LINE, ls="--", lw=AXES_LINEWIDTH_PT, zorder=1)


def label_axes(ax: Axes, x: str | None, y: str | None) -> None:
    """Label the axes from `AXIS_LABELS` by column name; a name not in the table is used as is."""
    if x is not None:
        ax.set_xlabel(AXIS_LABELS.get(x, x))
    if y is not None:
        ax.set_ylabel(AXIS_LABELS.get(y, y))


def site_labels(layers: Sequence[int]) -> list[str]:
    """Tick labels of the probed sites: the item embeddings, then each block, then the output."""
    ordered = [int(layer) for layer in layers]
    if len(ordered) < 3:
        return [str(layer) for layer in ordered]
    return [EMBEDDING_SITE_LABEL, *(str(layer) for layer in ordered[1:-1]), OUTPUT_SITE_LABEL]


# Where a colorbar sits beside its panel: its distance from the panel and its width, as
# fractions of the panel's width.
COLORBAR_PAD: float = 0.02
COLORBAR_FRACTION: float = 0.06


def sweep_colorbar(
    cmap: str,
    *,
    ax: Axes,
    span: tuple[float, float],
    title: str,
    ticks: Sequence[float] | None = None,
    column: str | None = None,
    pad: float = COLORBAR_PAD,
    fraction: float = COLORBAR_FRACTION,
) -> None:
    """A colorbar over `span` on the part of `cmap` that `sweep_colors` samples for `column`.

    The bar is placed beside `ax`, `pad` away and `fraction` wide relative to that panel.
    """
    mappable = ScalarMappable(norm=Normalize(vmin=span[0], vmax=span[1]), cmap=sweep_cmap(cmap, column=column))
    bar = ax.figure.colorbar(mappable, ax=ax, ticks=None if ticks is None else list(ticks), pad=pad, fraction=fraction)
    bar.set_label(title)
    bar.outline.set_linewidth(AXES_LINEWIDTH_PT)


def segmented_colorbar(colors: Sequence[str], labels: Sequence[str], *, ax: Axes, title: str) -> None:
    """A colorbar of equal blocks keying a few levels: one block per color, ticked at the center.

    The bar is placed beside `ax`, `COLORBAR_PAD` away and `COLORBAR_FRACTION` wide relative
    to that panel.
    """
    if len(colors) != len(labels):
        raise ValueError(f"segmented_colorbar: {len(labels)} labels for {len(colors)} colors")
    n = len(colors)
    mappable = ScalarMappable(norm=BoundaryNorm(np.arange(n + 1), n), cmap=ListedColormap(list(colors)))
    bar = ax.figure.colorbar(mappable, ax=ax, ticks=np.arange(n) + 0.5, pad=COLORBAR_PAD, fraction=COLORBAR_FRACTION)
    bar.set_ticklabels(list(labels))
    bar.set_label(title)
    bar.ax.tick_params(length=0)
    bar.outline.set_linewidth(AXES_LINEWIDTH_PT)
