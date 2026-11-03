"""The appendix figure "Learning over pretraining by skew", one row: accuracy pooled over
distance against the pretraining step, one line per skew, on the IWL test and on the ICL
test, with the skew colorbar that keys the lines. Tables: skew/eval.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import figure, letter_panels, load, main
from geo_alg_ti import plots
from geo_alg_ti.plots import style

EXPERIMENT = "skew"
STEP_TICKS = (0, 50_000, 100_000)
HEIGHT_IN = 2.0
CONDITIONS = ("iwl_test", "icl_test")


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables.

    Each panel carries its test's colored title; the ICL test's column is wider to make room
    for the colorbar beside its panel.
    """
    df = load(EXPERIMENT, "eval", results_root=args.results)
    fig, axes = figure(1, 2, width_in=style.FULL_WIDTH_IN, height_in=HEIGHT_IN, width_ratios=(1.0, 1.12))
    for ax, condition in zip(axes[0], CONDITIONS, strict=True):
        stat = plots.distribution.DEFAULT_K_STAT[condition]
        plots.make_emergence_by_skew(ax, df, condition=condition, k_stat=stat, xticks=STEP_TICKS, show_legend=False)
        ax.set_title(style.CONDITION_LABELS[condition], color=style.CONDITION_COLORS[condition], fontweight="bold")
    alphas = sorted(float(a) for a in df["skew_alpha"].unique())
    style.sweep_colorbar(style.SKEW_CMAP, ax=axes[0, 1], span=(alphas[0], alphas[-1]), title=style.AXIS_LABELS["skew_alpha"], column="skew_alpha")
    letter_panels(axes)
    return fig


if __name__ == "__main__":
    main(build, name="skew_learning")
