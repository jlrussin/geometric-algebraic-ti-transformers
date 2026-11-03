"""The appendix figure "Additional data-distribution manipulations": ICL and IWL accuracy
versus the number of pretraining orderings, pooled over distances two and above, and IWL
accuracy versus the number of finetuned orderings for the conflicting and disjoint layouts,
pooled over every distance, both layouts starting from the one-ordering run.
Tables: number_of_orderings/eval, conflicting_vs_disjoint/eval.
"""

from __future__ import annotations

import argparse

import pandas as pd
from matplotlib.figure import Figure

from _shared import figure, letter_panels, load, main
from geo_alg_ti import plots
from geo_alg_ti.plots import style

ORDERINGS = "number_of_orderings"
CONFLICT = "conflicting_vs_disjoint"
CONDITIONS = ("icl_test", "iwl_test")
K_STAT = {"iwl_test": "pin", "icl_test": "best"}
YLIM = (0.4, 1.02)
# Both layouts are the IWL test, so both stay in its red: the disjoint layout lighter, dashed,
# with square markers.
LAYOUT_COLORS = {"conflicting": style.IWL, "disjoint": "#ef6548"}
LAYOUT_LABELS = {"conflicting": "Conflicting orderings", "disjoint": "Disjoint orderings"}
LAYOUT_STYLES = {"conflicting": "-", "disjoint": "--"}
LAYOUT_MARKERS = {"conflicting": "o", "disjoint": "s"}
SEED_POINTS = {"show_seeds": True, "point_size": 4.0}
MARKER_SIZE = 3.5


def _share_single_ordering(df: pd.DataFrame) -> pd.DataFrame:
    """The rows at one ordering copied onto the disjoint layout.

    With one ordering the two layouts are the same design, so the experiment runs it under the
    conflicting layout alone; both lines start from that shared point.
    """
    at_one = df["n_orderings"] == 1
    shared = df[at_one & (df["layout"] == "conflicting")].assign(layout="disjoint")
    return pd.concat([df[~(at_one & (df["layout"] == "disjoint"))], shared], ignore_index=True)


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables."""
    orderings = load(ORDERINGS, "eval", results_root=args.results)
    conflict = _share_single_ordering(load(CONFLICT, "eval", results_root=args.results))
    fig, axes = figure(1, 2, width_in=style.FULL_WIDTH_IN, height_in=2.5)
    plots.make_pooled_accuracy_vs_sweep(
        axes[0, 0],
        orderings,
        x_col="n_training_cues",
        conditions=CONDITIONS,
        k_stat=K_STAT,
        log_x=True,
        ylim=YLIM,
        marker_size=MARKER_SIZE,
        jitter=0.05,
        legend_loc="center right",
        **SEED_POINTS,
    )
    plots.make_pooled_accuracy_vs_sweep(
        axes[0, 1],
        conflict,
        x_col="n_orderings",
        conditions=("iwl_test",),
        k_stat="pin",
        k_pin=0,
        d_min=1,
        line_col="layout",
        line_colors=LAYOUT_COLORS,
        line_labels=LAYOUT_LABELS,
        line_styles=LAYOUT_STYLES,
        line_markers=LAYOUT_MARKERS,
        ylim=YLIM,
        marker_size=MARKER_SIZE,
        jitter=0.08,
        legend_loc="lower center",
        legend_kwargs={"ncol": 2, "columnspacing": 1.0, "bbox_to_anchor": (0.5, 1.0), "frameon": False},
        **SEED_POINTS,
    )
    letter_panels(axes)
    return fig


if __name__ == "__main__":
    main(build, name="number_of_orderings_and_conflict")
