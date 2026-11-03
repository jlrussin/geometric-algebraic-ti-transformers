"""The transformer panels of the figure "Data-distribution effects", one row: the frequency
schematic, accuracy versus skew alpha, and the ICL-IWL dissociation with its skew colorbar. The
learning curves by skew are the appendix figure "Learning over pretraining by skew"
(`fig_skew_learning.py`).
Tables: skew/eval.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import figure, letter_panels, load, main
from geo_alg_ti import plots
from geo_alg_ti.plots import style

EXPERIMENT = "skew"
SKEW_TICKS = (0, 2, 4, 6, 8, 10)
LEGEND = {"handlelength": 1.2}


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables.

    The dissociation's column is wider to make room for the colorbar beside its panel; the
    figure's legend sits on the accuracy panel.
    """
    df = load(EXPERIMENT, "eval", results_root=args.results)
    # The schematic's colors sit on the same span of the skew colormap as the colorbar.
    skew_span = (float(df["skew_alpha"].min()), float(df["skew_alpha"].max()))
    fig, axes = figure(1, 3, width_in=style.FULL_WIDTH_IN, height_in=1.7, width_ratios=(1.0, 1.0, 1.3))
    scatter = axes[0, 2]
    plots.make_ordering_frequency_schematic(axes[0, 0], color_span=skew_span, legend_kwargs=LEGEND)
    plots.make_accuracy_vs_skew(
        axes[0, 1],
        df,
        ylim=(0.4, 1.02),
        xticks=SKEW_TICKS,
        marker_size=3.5,
        point_size=4.0,
        jitter=0.08,
        legend_loc="lower center",
        legend_kwargs={**LEGEND, "ncol": 2, "columnspacing": 0.8, "bbox_to_anchor": (0.58, 1.0), "frameon": False},
    )
    plots.make_icl_iwl_dissociation(scatter, df, colorbar_ax=scatter)
    letter_panels(axes)
    return fig


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
