"""The appendix figure "Replay during finetuning": ICL and IWL accuracy versus finetuning
step, one line per replay proportion, at each finetuning learning rate.
Tables: finetuning_hyperparameters/eval.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import figure, letter_panels, load, main
from geo_alg_ti import plots, tables
from geo_alg_ti.plots import style

EXPERIMENT = "finetuning_hyperparameters"
CONDITIONS = ("icl_test", "iwl_test")
K_STAT = {"iwl_test": "pin", "icl_test": "best"}
# Finetuning without replay is the control: gray and dashed, named in the legend.
NO_REPLAY = 0.0
NO_REPLAY_COLOR = "#444444"
NO_REPLAY_LABEL = "0 (no replay)"
SUBTITLE = "Learning rate {}"
STEP_TICKS = (0, 800, 1600)
STEP_LIM = (0, 1600)
# The row gap, as a fraction of a panel's height: the top row's x label clears the letters below.
SPACING = {"hspace": 0.10}


def build(args: argparse.Namespace) -> Figure:
    """Assemble the grid from the tidy tables: tests down the rows, learning rates across.

    Each panel names its test at the left of its title and its learning rate at the right;
    the replay legend appears once, on the IWL test at the smallest learning rate.
    """
    df = load(EXPERIMENT, "eval", results_root=args.results)
    rates = sorted(float(r) for r in df["learning_rate"].unique())
    replays = sorted(float(r) for r in df["replay_prop"].unique())
    with_replay = [r for r in replays if r != NO_REPLAY]
    colors = dict(zip(with_replay, style.sweep_colors(with_replay, style.REPLAY_CMAP, column="replay_prop", by_value=True), strict=True))
    colors[NO_REPLAY] = NO_REPLAY_COLOR
    fig, axes = figure(len(CONDITIONS), len(rates), width_in=style.FULL_WIDTH_IN, height_in=3.9, **SPACING)
    for row, condition in enumerate(CONDITIONS):
        for column, rate in enumerate(rates):
            ax = axes[row, column]
            sub = tables.pick(df, learning_rate=rate)
            plots.make_accuracy_by_step(
                ax,
                sub,
                condition=condition,
                k_stat=K_STAT[condition],
                line_col="replay_prop",
                colors=colors,
                line_styles={NO_REPLAY: "--"},
                line_labels={NO_REPLAY: NO_REPLAY_LABEL},
                xticks=STEP_TICKS,
                xlim=STEP_LIM,
                show_legend=(condition == "iwl_test" and column == 0),
                legend_loc="upper left",
            )
            ax.set_title(style.CONDITION_LABELS[condition], loc="left", color=style.CONDITION_COLORS[condition], fontweight="bold")
            ax.set_title(SUBTITLE.format(plots.learning.value_label("learning_rate", rate)), loc="right")
    letter_panels(axes)
    return fig


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
