"""The appendix figure "Additional symbolic-distance results": per-pair IWL accuracy at
zero steps, per-pair ICL accuracy by reasoning steps, ICL and baseline accuracy by
reasoning steps per distance, and the arbitrage response classes.
Tables: symbolic_distance/eval, symbolic_distance/eval_pairs.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import blank_figure, letter_panels, load, main, pins
from geo_alg_ti import plots
from geo_alg_ti.plots import style

EXPERIMENT = "symbolic_distance"
STEPS = tuple(range(9))
LEGEND = {"handlelength": 1.5}
BY_STEPS = {"band_alpha": 0.18, "show_seeds": True, "point_size": 3.0, "jitter": 0.06}


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables.

    The two per-pair panels each span the full width; the three by-steps panels share the
    last row.
    """
    df = pins(EXPERIMENT, load(EXPERIMENT, "eval", results_root=args.results), fname=EXPERIMENT)
    pairs = pins(EXPERIMENT, load(EXPERIMENT, "eval_pairs", results_root=args.results), fname=EXPERIMENT)
    fig = blank_figure(width_in=style.FULL_WIDTH_IN, height_in=5.6)
    grid = fig.add_gridspec(3, 3)
    iwl_pairs = fig.add_subplot(grid[0, :])
    icl_pairs = fig.add_subplot(grid[1, :])
    icl_steps, baseline_steps, arbitrage = (fig.add_subplot(grid[2, column]) for column in range(3))
    plots.make_pair_accuracy(iwl_pairs, pairs, condition="iwl_test", k_values=(0,), marker_size=3.5, point_size=4.0, jitter=0.05)
    plots.make_pair_accuracy(icl_pairs, pairs, condition="icl_test", k_values=STEPS, marker_size=2.5, line_width=0.8, legend_loc="lower left", legend_kwargs=LEGEND)
    plots.make_accuracy_by_reasoning_steps(icl_steps, df, condition="icl_test", cmap=style.ICL_DISTANCE_CMAP, legend_loc="lower right", legend_kwargs={**LEGEND, "ncol": 3}, **BY_STEPS)
    plots.make_accuracy_by_reasoning_steps(baseline_steps, df, condition="baseline_test", cmap=style.BASELINE_DISTANCE_CMAP, legend_loc="lower right", legend_kwargs={**LEGEND, "ncol": 3}, **BY_STEPS)
    plots.make_response_classes_by_reasoning_steps(arbitrage, df, marker_size=3.5, point_size=4.0, jitter=0.06, legend_loc="center right", legend_kwargs=LEGEND)
    letter_panels([iwl_pairs, icl_pairs, icl_steps, baseline_steps, arbitrage])
    return fig


if __name__ == "__main__":
    main(build, name="basic_tests")
