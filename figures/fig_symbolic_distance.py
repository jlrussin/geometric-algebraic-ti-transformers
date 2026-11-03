"""The figure "Symbolic distance effects (SDE)": accuracy by distance for the IWL test (zero
steps) and the ICL test (mean over one to five steps), and the first budget to criterion, at
every distance, with each test's accuracy at the sweep's largest budget (eight steps), pooled
over distances two to six, written beside its line; then the ICL test's best accuracy at each
budget or below, one line per distance.
Tables: symbolic_distance/eval.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import ACCURACY_NUMBERS, budget_ticks, figure, letter_panels, load, main, pins, place_numbers
from geo_alg_ti import plots
from geo_alg_ti.plots import style

EXPERIMENT = "symbolic_distance"
CONDITIONS = ("icl_test", "iwl_test")
# The gap between the panels, as a fraction of a panel's width.
SPACING = {"wspace": 0.04}
# The three panels side by side span one and a half columns.
WIDTH_IN = 1.5 * style.COLUMN_WIDTH_IN
# The last panel's ticks: every second budget of the sweep (`budget_ticks`).
BUDGET_TICK_STEP = 2


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables; the tests' legend is on the first panel.

    The distances' legend is on the last.
    """
    df = pins(EXPERIMENT, load(EXPERIMENT, "eval", results_root=args.results), fname=EXPERIMENT)
    fig, axes = figure(1, 3, width_in=WIDTH_IN, height_in=1.8, **SPACING)
    plots.make_accuracy_by_distance(axes[0, 0], df, conditions=CONDITIONS, legend_loc="lower left")
    plots.make_steps_to_criterion_by_distance(axes[0, 1], df, conditions=CONDITIONS, show_legend=False, show_seeds=False)
    plots.make_accuracy_numbers(axes[0, 1], df, conditions=CONDITIONS, **ACCURACY_NUMBERS)
    plots.make_accuracy_by_reasoning_steps(
        axes[0, 2],
        df,
        condition="icl_test",
        k_stat="running_best",
        band_alpha=0.18,
        xticks=budget_ticks(df, BUDGET_TICK_STEP),
        ylabel="Accuracy (best so far)",
        legend_loc="lower right",
        legend_kwargs={"ncol": 3, "handlelength": 1.2, "handletextpad": 0.4, "columnspacing": 0.8},
    )
    letter_panels(axes)
    place_numbers(fig, axes[0, 1])
    return fig


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
