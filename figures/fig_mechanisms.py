"""The figure "Mechanisms underlying the geometric and algebraic strategies", two transformer
panels. Left: accuracy with the network intact, with the rank direction or a random direction
erased, and with each network's context-reading heads or as many random heads ablated, on the
IWL test at zero steps, the reversed context at zero and eight steps, and the ICL test at eight
steps. Right: the ICL test and the IWL test's trained pairs, pairs with an end item, and
interior pairs after finetuning with reasoning, intact or with the context-reading heads
ablated while finetuning and at test.
Tables: representation/edit, components/ablate, finetuning_reasoning/eval_pairs,
finetuning_reasoning/ablated_eval_pairs.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import COMPONENTS_FREQUENT_CUE_PROP, CONTEXT_HEADS, D_MIN, blank_figure, load, main, pins
from geo_alg_ti import plots, tables
from geo_alg_ti.plots import style

EXPERIMENT = "representation"
HEADS_EXPERIMENT = "components"
REASONING_EXPERIMENT = "finetuning_reasoning"
HEIGHT_IN = 2.6
# The interventions panel keeps its column width; the finetuning panel takes the rest.
WIDTH_RATIOS = (style.COLUMN_WIDTH_IN, style.FULL_WIDTH_IN - style.COLUMN_WIDTH_IN)
# The key sits above the panel, each manipulation over its control.
LEGEND = {"handlelength": 1.2, "columnspacing": 0.8}
# Finetuning with reasoning on every congruent-context trial, intact or with the context-reading
# heads ablated; the bars carry the standard error over networks without the networks' dots.
REASONING_CONDITIONS = (("none", 1.0), ("context_reading", 1.0))
REASONING_LABELS = {("none", 1.0): "Intact", ("context_reading", 1.0): "Ablated"}
REASONING_BARS = {
    "show_seeds": False,
    "bar_width": 0.34,
    "gap": 0.32,
    "measure_labels": {"end_item": "End\nitem"},
    "group_offset_pt": 23.0,
    "split_axis": True,
    "legend_kwargs": {"handlelength": 1.2},
}


def build(args: argparse.Namespace) -> Figure:
    """Assemble the two panels from the tidy tables.

    The edits are pinned to one frequent-cue proportion, the one the components networks were
    trained at; the bars carry the standard error over networks without the networks' dots.
    """
    edit = pins(EXPERIMENT, load(EXPERIMENT, "edit", results_root=args.results), fname=EXPERIMENT, frequent_cue_prop=COMPONENTS_FREQUENT_CUE_PROP)
    ablate = load(HEADS_EXPERIMENT, "ablate", results_root=args.results)
    pairs = load(REASONING_EXPERIMENT, "eval_pairs", results_root=args.results)
    ablated_pairs = load(REASONING_EXPERIMENT, "ablated_eval_pairs", results_root=args.results)
    reasoning = tables.finetuned_pair_rows(pairs, ablated_pairs, fname=REASONING_EXPERIMENT)
    fig = blank_figure(width_in=style.FULL_WIDTH_IN, height_in=HEIGHT_IN)
    interventions, finetuning = fig.subplots(1, 2, width_ratios=WIDTH_RATIOS)
    plots.make_intervention_bars(interventions, edit, ablate, k=CONTEXT_HEADS, d_min=D_MIN, show_seeds=False, legend_kwargs=LEGEND)
    plots.make_pair_type_bars(finetuning, reasoning, conditions=REASONING_CONDITIONS, labels=REASONING_LABELS, d_min=D_MIN, **REASONING_BARS)
    return fig


if __name__ == "__main__":
    main(build, name="mechanisms")
