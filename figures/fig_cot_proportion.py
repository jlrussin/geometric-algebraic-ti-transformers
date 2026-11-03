"""The appendix figure "Chain-of-thought supervision improves sample efficiency": ICL
accuracy versus pretraining step, one line per p(CoT), and the accuracy gain from reasoning
over zero steps, keyed by one p(CoT) colorbar. Tables: cot_proportion/eval.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import figure, letter_panels, load, main
from geo_alg_ti import plots
from geo_alg_ti.plots import style

EXPERIMENT = "cot_proportion"
CONDITION = "icl_test"
# p(CoT) is a proportion, so its colors sit on the colormap by value between zero and one.
P_COT_SPAN = (0.0, 1.0)
COLORBAR_TICKS = (0.0, 0.5, 1.0)
STEP_TICKS = (0, 50_000, 100_000)
STEP_LIM = (0, 100_000)
GAIN_YLIM = (-0.03, 0.55)
GAIN_YTICKS = (0.0, 0.25, 0.5)


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables; the gain's wider column carries the colorbar."""
    df = load(EXPERIMENT, "eval", results_root=args.results)
    values = sorted(float(v) for v in df["p_cot"].unique())
    colors = dict(zip(values, style.sweep_colors(values, style.P_COT_CMAP, column="p_cot", span=P_COT_SPAN), strict=True))
    fig, axes = figure(1, 2, width_in=style.FULL_WIDTH_IN, height_in=2.5, width_ratios=(1.0, 1.14))
    accuracy, gain = axes[0]
    plots.make_accuracy_by_step(accuracy, df, condition=CONDITION, line_col="p_cot", colors=colors, xticks=STEP_TICKS, xlim=STEP_LIM, show_legend=False)
    plots.make_reasoning_gain_by_step(gain, df, condition=CONDITION, line_col="p_cot", colors=colors, xticks=STEP_TICKS, xlim=STEP_LIM, ylim=GAIN_YLIM, show_legend=False)
    gain.set_yticks(GAIN_YTICKS)
    style.sweep_colorbar(style.P_COT_CMAP, span=P_COT_SPAN, title=style.AXIS_LABELS["p_cot"], ticks=COLORBAR_TICKS, column="p_cot", ax=gain)
    for ax in (accuracy, gain):
        ax.set_title(style.CONDITION_LABELS[CONDITION], color=style.CONDITION_COLORS[CONDITION], fontweight="bold")
    letter_panels(axes)
    return fig


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
