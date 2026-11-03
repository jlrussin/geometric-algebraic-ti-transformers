"""The appendix figure "Ablation during finetuning across learning rates and proportions of
uninformative context trials": the bars of "Ablating context-reading heads during finetuning"
at the last finetuning step, one panel per finetuning learning rate (rows) and proportion of
fixed-ordering trials with an uninformative context (columns).
Tables: finetuning_reasoning_hyperparameters/eval_pairs,
finetuning_reasoning_hyperparameters/ablated_eval_pairs.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import D_MIN, blank_figure, load, main
from geo_alg_ti import plots, tables
from geo_alg_ti.plots import style

EXPERIMENT = "finetuning_reasoning_hyperparameters"
HEIGHT_IN = 5.0
ROW_TITLE = "Learning rate {}"
COLUMN_TITLE = "Uninformative proportion {}"
# The panels are narrow: the bars are thinner, their seeds left out, and two tick labels take
# two lines, so the IWL test's label row sits lower.
BARS = {
    "show_seeds": False,
    "show_legend": False,
    "bar_width": 0.18,
    "gap": 0.2,
    "measure_labels": {"icl_test": "ICL\ntest", "end_item": "End\nitem"},
    "group_offset_pt": 23.0,
}
YTICKS = (0.0, 0.5, 1.0)
KEY = {"ncol": 4, "handlelength": 1.6, "columnspacing": 1.4, "handletextpad": 0.5}
SPACING = {"h_pad": 0.04, "hspace": 0.04}


def build(args: argparse.Namespace) -> Figure:
    """Assemble the grid from the tidy per-pair tables; the bottom row names the measures, the
    left column the learning rates, and one label at the left the accuracy every panel shows."""
    pairs = load(EXPERIMENT, "eval_pairs", results_root=args.results)
    ablated = load(EXPERIMENT, "ablated_eval_pairs", results_root=args.results)
    rows = tables.finetuned_pair_rows(pairs, ablated, fname=EXPERIMENT)
    rates = sorted(float(r) for r in rows["learning_rate"].unique())
    shares = sorted(float(s) for s in rows["uninformative_prop"].unique())
    fig = blank_figure(width_in=style.FULL_WIDTH_IN, height_in=HEIGHT_IN, **SPACING)
    axes = fig.subplots(len(rates), len(shares), sharex=True, sharey=True, squeeze=False)
    last = len(rates) - 1
    accuracy_label = ""
    for row, rate in enumerate(rates):
        for column, share in enumerate(shares):
            ax = axes[row, column]
            setting = {"learning_rate": rate, "uninformative_prop": share}
            plots.make_pair_type_bars(ax, tables.pick(rows, **setting), d_min=D_MIN, group_label=plots.learning.IWL_GROUP_LABEL if row == last else None, **BARS)
            ax.set_yticks(YTICKS)
            accuracy_label = ax.get_ylabel()
            ax.set_ylabel(ROW_TITLE.format(plots.learning.value_label("learning_rate", rate)) if column == 0 else "")
            if row != last:
                ax.tick_params(labelbottom=False)
            if row == 0:
                ax.set_title(COLUMN_TITLE.format(plots.learning.value_label("uninformative_prop", share)))
    fig.supylabel(accuracy_label, fontsize=style.FONT_BODY_PT)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", **KEY)
    return fig


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
