"""The appendix figure "Ablating context-reading heads during finetuning": the ICL test and the
IWL test's trained pairs, pairs with an end item, and interior pairs after finetuning without
reasoning, with reasoning, and with reasoning while each network's context-reading heads or as
many random heads are ablated, at the last finetuning step and over the finetuning steps.
Tables: finetuning_reasoning/eval_pairs, finetuning_reasoning/ablated_eval_pairs.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import D_MIN, blank_figure, letter_panels, load, main
from geo_alg_ti import plots, tables
from geo_alg_ti.plots import style

EXPERIMENT = "finetuning_reasoning"
HEIGHT_IN = 4.7
# The rows: the bars' key, the bars, the lines' key, the lines over finetuning steps.
ROW_HEIGHTS = (0.09, 1.15, 0.09, 1.0)
TITLES = {
    "icl_test": "ICL test",
    "trained": "IWL test, trained pairs",
    "end_item": "IWL test, pairs with an end item",
    "interior": "IWL test, interior pairs",
}
STEP_TICKS = (0, 400, 800, 1200, 1600)
STEP_YLIM = (0.3, 1.02)
STEP_YTICKS = (0.4, 0.6, 0.8, 1.0)
KEY = {"ncol": 4, "handlelength": 1.6, "columnspacing": 1.4, "handletextpad": 0.5, "borderaxespad": 0.0}
SPACING = {"h_pad": 0.02, "hspace": 0.0}


def _key(fig: Figure, slot: object, source: object) -> None:
    """A row that holds only the key of the panel `source`, so the layout keeps its height."""
    ax = fig.add_subplot(slot)
    ax.axis("off")
    handles, labels = source.get_legend_handles_labels()
    ax.legend(handles, labels, loc="center", **KEY)


def build(args: argparse.Namespace) -> Figure:
    """Assemble the figure from the tidy per-pair tables: the bars at the last step above, each
    measure over finetuning steps below."""
    pairs = load(EXPERIMENT, "eval_pairs", results_root=args.results)
    ablated = load(EXPERIMENT, "ablated_eval_pairs", results_root=args.results)
    rows = tables.finetuned_pair_rows(pairs, ablated, fname=EXPERIMENT)
    fig = blank_figure(width_in=style.FULL_WIDTH_IN, height_in=HEIGHT_IN, **SPACING)
    grid = fig.add_gridspec(len(ROW_HEIGHTS), len(plots.learning.PAIR_TYPES), height_ratios=ROW_HEIGHTS)
    bars = fig.add_subplot(grid[1, :])
    plots.make_pair_type_bars(bars, rows, d_min=D_MIN, show_legend=False)
    lines = [fig.add_subplot(grid[3, column]) for column in range(len(plots.learning.PAIR_TYPES))]
    for column, (ax, measure) in enumerate(zip(lines, plots.learning.PAIR_TYPES, strict=True)):
        plots.make_pair_type_by_step(ax, rows, measure=measure, d_min=D_MIN, show_legend=False, ylim=STEP_YLIM)
        ax.set_xticks(STEP_TICKS, [f"{step:,}" for step in STEP_TICKS])
        ax.set_yticks(STEP_YTICKS)
        ax.set_title(TITLES[measure])
        if column:
            ax.set_ylabel("")
    _key(fig, grid[0, :], bars)
    _key(fig, grid[2, :], lines[0])
    letter_panels([bars, lines[0]], "AB")
    return fig


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
