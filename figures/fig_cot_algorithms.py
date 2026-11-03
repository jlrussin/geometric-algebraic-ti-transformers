"""The appendix figure "Alternative chain-of-thought algorithms": accuracy on the ICL test
under a training cue by reasoning steps, one line per distance, for each of the four
chain-generation algorithms, and the reasoning steps to criterion by distance with one line
per algorithm.
Tables: chain_algorithms/eval.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import figure, letter_panels, load, main, pins
from geo_alg_ti import plots
from geo_alg_ti.plots import style

EXPERIMENT = "chain_algorithms"
CONDITION = "icl_test_training_cue"
ALGORITHM_LABELS = {
    "main": "Main algorithm",
    "inferred_relations": "Inferred relations",
    "descend_first": "Descend first",
    "backtrack": "Backtrack at dead ends",
}
# One color per algorithm for the summary, off the red and blue of the two tests.
ALGORITHM_COLORS = {
    "main": "#444444",
    "inferred_relations": "#ff7f0e",
    "descend_first": "#17becf",
    "backtrack": "#bcbd22",
}
SUMMARY_TITLE = "Steps to criterion"
SUMMARY_YLIM = (-0.3, 10.0)
SUMMARY_YTICKS = (0, 2, 4, 6, 8)
LEGEND = {"handlelength": 1.4, "labelspacing": 0.25}
# Points between a panel's title and its axes.
TITLE_PAD = 4.0
# The row gap, as a fraction of a panel's height: the top row's x label clears the letters below.
SPACING = {"hspace": 0.10}


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables.

    The algorithms fill the grid in reading order and the summary takes the slot after them;
    the distance legend appears once, on the first algorithm.
    """
    df = load(EXPERIMENT, "eval", results_root=args.results)
    algorithms = [name for name in ALGORITHM_LABELS if name in set(df["chain_algorithm"])]
    ncols = 3
    nrows = -(-(len(algorithms) + 1) // ncols)
    fig, axes = figure(nrows, ncols, width_in=style.FULL_WIDTH_IN, height_in=2.15 * nrows, **SPACING)
    flat = list(axes.ravel())
    for index, algorithm in enumerate(algorithms):
        sub = pins(EXPERIMENT, df, fname=EXPERIMENT, chain_algorithm=algorithm)
        plots.make_accuracy_by_reasoning_steps(flat[index], sub, condition=CONDITION, band_alpha=0.18, show_legend=(index == 0), legend_loc="lower right", legend_kwargs=LEGEND)
        flat[index].set_title(ALGORITHM_LABELS[algorithm], pad=TITLE_PAD)
    summary = flat[len(algorithms)]
    for index, algorithm in enumerate(algorithms):
        sub = pins(EXPERIMENT, df, fname=EXPERIMENT, chain_algorithm=algorithm)
        plots.make_steps_to_criterion_by_distance(
            summary,
            sub,
            conditions=(CONDITION,),
            colors={CONDITION: ALGORITHM_COLORS[algorithm]},
            labels={CONDITION: ALGORITHM_LABELS[algorithm]},
            show_seeds=False,
            ylim=SUMMARY_YLIM,
            show_legend=(index == len(algorithms) - 1),
            legend_loc="upper left",
            legend_kwargs=LEGEND,
        )
    summary.set_yticks(SUMMARY_YTICKS)
    summary.set_title(SUMMARY_TITLE, pad=TITLE_PAD)
    for ax in flat[len(algorithms) + 1 :]:
        ax.remove()
    letter_panels(flat[: len(algorithms) + 1])
    return fig


if __name__ == "__main__":
    main(build, name="cot_algorithms")
