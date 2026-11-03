"""The appendix figure "Additional mechanism results": rank decodability by layer keyed by a
frequent-cue proportion colorbar, accuracy after erasing or inverting the rank direction,
agreement with a written ordering, and each attention head's effect when ablated with each
network's context-reading heads marked.
Tables: representation/probe, representation/edit, components/ablate.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import (
    COMPONENTS_FREQUENT_CUE_PROP,
    CONTEXT_HEADS,
    D_MIN,
    blank_figure,
    letter_panels,
    load,
    main,
    pins,
)
from geo_alg_ti import plots
from geo_alg_ti.plots import style

EXPERIMENT = "representation"
HEADS_EXPERIMENT = "components"
HEIGHT_IN = 5.8
# The two rows of panels, and an empty row between them that keeps the top row's x labels apart
# from the keys above the bottom row.
ROW_HEIGHTS = (1.0, 1.0)
ROW_GAP = 0.05
# The top row's slots: the probe panel with its colorbar, and the erase panel. The bottom row's:
# the write panel, its companion strip for the random-direction control, and the head scatter.
TOP_WIDTHS = (1.0, 1.0)
BOTTOM_WIDTHS = (0.82, 0.18, 1.0)
# The panels are wide, so their keys run in two columns.
LEGEND = {"handlelength": 1.5, "ncol": 2, "columnspacing": 1.2}
ABOVE = {"frameon": False, "bbox_to_anchor": (0.5, 1.0), "borderaxespad": 0.2}
MARKER_SIZE = 3.5
SPACING = {"hspace": 0.06, "wspace": 0.04}
# The strip's two-line label drops to sit level with the write panel's x label.
STRIP_LABEL_DROP_PT = 5.0


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables.

    The probe panel keeps every frequent-cue proportion as its line family, keyed by a
    segmented colorbar beside it; the edit panels are pinned to one proportion, and the head
    panel reads the components networks, which are at that proportion. The write panel gets a
    narrow companion strip for the random-direction control that shares its y range.
    """
    probe = load(EXPERIMENT, "probe", results_root=args.results)
    edit = pins(EXPERIMENT, load(EXPERIMENT, "edit", results_root=args.results), fname=EXPERIMENT, frequent_cue_prop=COMPONENTS_FREQUENT_CUE_PROP)
    ablate = load(HEADS_EXPERIMENT, "ablate", results_root=args.results)
    fig = blank_figure(width_in=style.FULL_WIDTH_IN, height_in=HEIGHT_IN, **SPACING)
    rows = fig.add_gridspec(3, 1, height_ratios=(ROW_HEIGHTS[0], ROW_GAP, ROW_HEIGHTS[1]))
    top = rows[0].subgridspec(1, len(TOP_WIDTHS), width_ratios=TOP_WIDTHS)
    bottom = rows[2].subgridspec(1, len(BOTTOM_WIDTHS), width_ratios=BOTTOM_WIDTHS)
    probe_ax = fig.add_subplot(top[0, 0])
    erase_ax = fig.add_subplot(top[0, 1])
    write_ax = fig.add_subplot(bottom[0, 0])
    control_ax = fig.add_subplot(bottom[0, 1], sharey=write_ax)
    heads_ax = fig.add_subplot(bottom[0, 2])
    plots.make_probe_accuracy_by_layer(probe_ax, probe, marker_size=MARKER_SIZE, colorbar_ax=probe_ax)
    # The edit lines start at distance `D_MIN` (two): adjacent pairs need no inference.
    distances = [int(d) for d in sorted(edit["d"].unique()) if d >= D_MIN]
    plots.make_edit_accuracy_by_distance(erase_ax, edit, d_values=distances, marker_size=MARKER_SIZE, legend_loc="lower center", legend_kwargs={**LEGEND, **ABOVE})
    plots.make_write_by_strength(write_ax, edit, control_ax=control_ax, marker_size=MARKER_SIZE, legend_loc="lower center", legend_kwargs={**LEGEND, **ABOVE})
    control_ax.xaxis.labelpad += STRIP_LABEL_DROP_PT
    plots.make_head_ablation_scatter(heads_ax, ablate, k=CONTEXT_HEADS, legend_kwargs={"columnspacing": 1.2})
    letter_panels([probe_ax, erase_ax, write_ax, heads_ax])
    return fig


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
