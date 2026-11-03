"""The appendix figure "Experiments without an explicit cue", one row: accuracy versus skew
alpha on the IWL test, beside the accuracy of answering each pair by the frequency-weighted
majority of the training cues' orderings, and on the ICL test, each for networks pretrained
with the cue and without it; then, for networks pretrained without the cue at frequent-cue
proportion 0.2 and p(CoT) 0.9, accuracy by distance under the four contexts at zero reasoning
steps, the shuffled context's distance taken in the ordering shown, and the running best
accuracy of the congruent and reversed contexts as reasoning steps grow, one line per distance.
Tables: skew/eval, skew_no_cue/eval, content_effects_no_cue/eval.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from _shared import D_MIN, FREQUENT_CUE_PROP, P_COT, figure, letter_panels, load, main, pins
from geo_alg_ti import config, orderings, plots
from geo_alg_ti.plots import style

WITH_CUE = "skew"
WITHOUT_CUE = "skew_no_cue"
CONTENT = "content_effects_no_cue"
MAJORITY_LABEL = "Majority ordering"
LINE_LABELS = {WITH_CUE: "With cue", WITHOUT_CUE: "Without cue"}
# Without the cue, a test's line repeats its color, dashed, with open markers.
LINE_STYLES = {WITH_CUE: "-", WITHOUT_CUE: "--"}
SKEW_TICKS = (0, 2, 4, 6, 8, 10)
SKEW_YLIM = (0.4, 1.02)
HEIGHT_IN = 2.1
LEGEND = {"handlelength": 1.6, "frameon": False}


def majority_ordering_accuracies(df: pd.DataFrame) -> dict[float, float]:
    """The pairwise majority's IWL test accuracy at each skew alpha, averaged over the seeds there.

    Each seed's ordering law and cue frequencies are those of the run without the cue at that
    skew and seed, built from its resolved configuration with the ordering law alone, so drawing
    the figure needs no training code. A law's pairwise relations are computed once, for all
    the skews that share it, and released before the next law's.
    """
    present = {(float(a), int(s)) for a, s in df[["skew_alpha", "seed"]].drop_duplicates().itertuples(index=False)}
    weightings: dict[orderings.OrderingLaw, list[tuple[float, np.ndarray]]] = {}
    for run in config.enumerate_runs(config.load_experiment(WITHOUT_CUE)):
        skew, seed = float(run.axis_values["skew_alpha"]), int(run.axis_values["seed"])
        if (skew, seed) in present:
            task = run.config.task
            law = orderings.OrderingLaw(
                seed=seed,
                pool_size=task.item_pool_size,
                n_training_cues=task.n_training_cues,
                n_reserved_cues=task.n_reserved_cues,
            )
            cues = task.cue_distribution
            weights = orderings.cue_weights(
                orderings.CueDistribution(cues.kind),
                task.n_training_cues,
                skew_alpha=cues.skew_alpha,
                frequent_cue_proportion=cues.frequent_cue_prop,
            )
            weightings.setdefault(law, []).append((skew, weights))
    by_skew: dict[float, list[float]] = {}
    for law, weighted in weightings.items():
        relations = orderings.pair_relations(law)
        for skew, weights in weighted:
            by_skew.setdefault(skew, []).append(orderings.majority_ordering_accuracy(law, weights, relations=relations))
        del relations
    return {skew: float(np.mean(values)) for skew, values in sorted(by_skew.items())}


def accuracy_vs_skew(
    ax: Axes,
    df: pd.DataFrame,
    *,
    condition: str,
    k_stat: str,
    legend_kwargs: Mapping[str, object],
    majority: Mapping[float, float] | None = None,
) -> None:
    """One test's accuracy versus skew alpha with the cue and without it, and `majority` if given.

    Estimand, as on the figure "Data-distribution effects": `k_stat` over reasoning steps (zero
    steps pinned, or the best over the sweep), the trial-weighted pool over distances two and
    up, then the mean and standard error over seeds at the last pretraining step.
    """
    color = style.CONDITION_COLORS[condition]
    plots.make_pooled_accuracy_vs_sweep(
        ax,
        df,
        x_col="skew_alpha",
        conditions=(condition,),
        k_stat=k_stat,
        k_pin=0,
        d_min=D_MIN,
        line_col="experiment",
        line_colors={WITH_CUE: color, WITHOUT_CUE: color},
        line_labels=LINE_LABELS,
        line_styles=LINE_STYLES,
        line_open_markers={WITHOUT_CUE},
        marker_size=3.5,
        point_size=4.0,
        jitter=0.08,
        ylim=SKEW_YLIM,
        show_legend=False,
    )
    if majority is not None:
        ax.plot(list(majority), list(majority.values()), color=style.REFERENCE_LINE, ls=":", label=MAJORITY_LABEL)
    ax.set_xticks(SKEW_TICKS)
    ax.set_title(style.CONDITION_LABELS[condition], color=color, fontweight="bold")
    ax.legend(**{**LEGEND, **legend_kwargs})


def build(args: argparse.Namespace) -> Figure:
    """Assemble the panels from the tidy tables."""
    skew = pd.concat([load(name, "eval", results_root=args.results) for name in (WITH_CUE, WITHOUT_CUE)], ignore_index=True)
    content = load(CONTENT, "eval", results_root=args.results)
    content = pins(CONTENT, content, fname=CONTENT, frequent_cue_prop=FREQUENT_CUE_PROP, p_cot=P_COT)
    majority = majority_ordering_accuracies(skew[skew["experiment"] == WITHOUT_CUE])
    fig, axes = figure(1, 4, width_in=style.FULL_WIDTH_IN, height_in=HEIGHT_IN)
    iwl, icl, by_distance, by_steps = axes[0]
    # Both legends sit just above the chance line, clear of the curves.
    accuracy_vs_skew(iwl, skew, condition="iwl_test", k_stat="pin", majority=majority, legend_kwargs={"loc": "lower right", "bbox_to_anchor": (1.0, 0.17)})
    accuracy_vs_skew(icl, skew, condition="icl_test", k_stat="best", legend_kwargs={"loc": "lower left", "bbox_to_anchor": (0.0, 0.17)})
    plots.make_context_accuracy_by_distance(
        by_distance,
        content,
        k_pin=0,
        show_seeds=True,
        point_size=4.0,
        jitter=0.05,
        # The legend sits along the bottom in two columns, under the reversed line; its light frame
        # leaves the few seeds beneath it visible.
        legend_loc="lower center",
        legend_kwargs={"ncol": 2, "bbox_to_anchor": (0.5, 0.0), "frameon": True, "framealpha": 0.3, "handlelength": 1.2, "handletextpad": 0.4, "columnspacing": 0.8},
    )
    plots.make_context_accuracy_by_reasoning_steps(
        by_steps,
        content,
        contexts=("congruent", "reversed"),
        k_stat="running_best",
        band_alpha=0.18,
        show_seeds=True,
        point_size=4.0,
        jitter=0.05,
        ylabel="Accuracy (best so far)",
        legend_loc="lower right",
    )
    for ax in (by_distance, by_steps):
        ax.set_title("Without cue", fontweight="bold")
    letter_panels(axes)
    return fig


if __name__ == "__main__":
    main(build, name="no_cue")
