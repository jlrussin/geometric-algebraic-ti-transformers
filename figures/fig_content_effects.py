"""The figure "Content effects in transitive inference" at frequent-cue proportion 0.2 and
p(CoT) 0.9: accuracy by distance under the four contexts at zero reasoning steps, with the
shuffled context's distance taken in the ordering shown, and the number of reasoning steps the
congruent, reversed, and shuffled contexts need to reach ninety percent of their best accuracy,
by distance, a distance whose best accuracy most seeds cannot lift above chance marked
unreached and breaking its line, with each context's accuracy at the sweep's largest budget
(eight steps), pooled over distances two to six, written beside its line; then the congruent
and reversed contexts' best accuracy at each budget or below, one line per distance.
Tables: content_effects/eval.
"""

from __future__ import annotations

import argparse

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator

from _shared import (
    ACCURACY_NUMBERS,
    CONTENT_STEPS_CONDITIONS,
    CRITERION_FRACTION,
    FREQUENT_CUE_PROP,
    P_COT,
    budget_ticks,
    figure,
    letter_panels,
    load,
    main,
    pins,
    place_numbers,
    unreached_handles,
)
from geo_alg_ti import plots
from geo_alg_ti.plots import placement, style

EXPERIMENT = "content_effects"
# The three panels stack in one column at the width of the figure's transformer column.
WIDTH_IN = 2.3
HEIGHT_IN = 4.9
# The gap between the panels, as a fraction of a panel's height.
SPACING = {"hspace": 0.08}
# The top panel's axis label: its accuracy is at zero reasoning steps.
NO_REASONING_YLABEL = "Accuracy (no reasoning)"
# The criterion panel's ticks: every second budget of the sweep (`budget_ticks`).
CRITERION_TICK_STEP = 2
# The last panel's ticks: every second budget of the sweep (`budget_ticks`).
BUDGET_TICK_STEP = 2


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables."""
    df = load(EXPERIMENT, "eval", results_root=args.results)
    df = pins(EXPERIMENT, df, fname=EXPERIMENT, frequent_cue_prop=FREQUENT_CUE_PROP, p_cot=P_COT)
    fig, axes = figure(3, 1, width_in=WIDTH_IN, height_in=HEIGHT_IN, **SPACING)
    # The legend sits between the shuffled and reversed curves, right of the fall at distance two.
    plots.make_context_accuracy_by_distance(
        axes[0, 0],
        df,
        k_pin=0,
        ylabel=NO_REASONING_YLABEL,
        legend_loc="lower left",
        legend_kwargs={
            "ncol": 2,
            "bbox_to_anchor": (0.255, 0.155),
            "frameon": True,
            "framealpha": 0.5,
            "columnspacing": 0.8,
            "handlelength": 1.2,
            "handletextpad": 0.5,
        },
    )
    # The contexts take the colors of the panel above, so the legend keys the unreached mark only.
    plots.make_steps_to_criterion_by_distance(
        axes[1, 0],
        df,
        conditions=CONTENT_STEPS_CONDITIONS,
        fraction=CRITERION_FRACTION,
        unreached_below_chance=True,
        break_at_unreached=True,
        band_alpha=0.18,
        show_seeds=False,
        show_legend=False,
    )
    axes[1, 0].yaxis.set_major_locator(FixedLocator(budget_ticks(df, CRITERION_TICK_STEP)))
    _unreached_legend(axes[1, 0])
    plots.make_accuracy_numbers(axes[1, 0], df, conditions=CONTENT_STEPS_CONDITIONS, **ACCURACY_NUMBERS)
    # The contexts take the colors of the top panel, so the key names the distances only.
    plots.make_context_accuracy_by_reasoning_steps(
        axes[2, 0],
        df,
        contexts=("congruent", "reversed"),
        k_stat="running_best",
        legend_mode="distance_key",
        band_alpha=0.18,
        xticks=budget_ticks(df, BUDGET_TICK_STEP),
        ylabel="Accuracy (best so far)",
        legend_loc="lower right",
        legend_kwargs={"ncol": 3, "handlelength": 1.2, "handletextpad": 0.4, "columnspacing": 0.8},
    )
    # The two contexts are named in their colors, one above the other under the rising lines and
    # left of the key, and placed with the numbers once the layout is final.
    for context, anchor in (("congruent", (1.8, 0.33)), ("reversed", (1.8, 0.21))):
        name = axes[2, 0].text(*anchor, style.CONTEXT_LABELS[context].lower(), color=style.CONTEXT_COLORS[context], fontsize=style.FONT_LEGEND_PT, ha="left", va="center")
        placement.mark(name, key=f"{context}_name", anchor=anchor)
    letter_panels(axes)
    place_numbers(fig, axes[1, 0], axes[2, 0])
    return fig


def _unreached_legend(ax: Axes) -> None:
    """A legend naming the unreached mark, drawn only when the panel holds one."""
    kept = unreached_handles(ax)
    if kept:
        ax.legend(kept[:1], [style.UNREACHED_LABEL], loc="upper left", handletextpad=0.2)


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
