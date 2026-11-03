"""The appendix figure "Architecture and learning-rate sweep": accuracy on the ICL test
under a training cue by reasoning steps, one line per distance, for every architecture, on
a grid of depth and width by feedforward and learning rate; each panel prints the
architecture's training loss over the last 5,000 pretraining steps.
Tables: architectures/eval, architectures/train.
"""

from __future__ import annotations

import argparse

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from _shared import figure, load, main, pins
from geo_alg_ti import plots, tables
from geo_alg_ti.plots import style

EXPERIMENT = "architectures"
CONDITION = "icl_test_training_cue"
ARCHITECTURE = ("n_layers", "d_model", "feedforward", "learning_rate")
# The indicator printed on each panel: the mean training loss over the last
# `LOSS_WINDOW_STEPS` pretraining steps within a seed, averaged over seeds. It shows whether
# the architecture learned the pretraining task.
LOSS_WINDOW_STEPS = 5000
LOSS_LABEL = "loss {:.2g}"
# Axes fraction of the lower left corner, below every curve and clear of the legend.
LOSS_POSITION = (0.04, 0.06)
# Rows run deepest and widest first; columns put the feedforward networks before the
# attention-only ones and the larger learning rate first.
FEEDFORWARD_LABELS = {"ffn": "feedforward", "none": "attention-only"}
FEEDFORWARD_ORDER = ("ffn", "none")
# The network of the main figures, named on its own panel.
CANONICAL = {"n_layers": 4, "d_model": 256, "feedforward": "ffn", "learning_rate": 0.001}
CANONICAL_SUFFIX = "(main-text architecture)"
ROW_HEIGHT_IN = 1.45
# Three tight columns, so the legend leaves the lower left of its panel to the loss indicator.
LEGEND = {"ncol": 3, "handlelength": 1.2, "handletextpad": 0.4, "columnspacing": 0.8}
YTICKS = (0.0, 0.5, 1.0)
MARKER_SIZE = 2.5
# Points between a panel's title and its axes.
TITLE_PAD = 3.0
# Tighter page margins and wider gaps between the many small panels: the pads in inches, the
# gaps as fractions of a panel's width and height.
SPACING = {"w_pad": 0.03, "h_pad": 0.03, "wspace": 0.08, "hspace": 0.14}


def _title(depth: int, width: int, feedforward: str, learning_rate: float) -> str:
    """Two lines naming the architecture; a third names the network of the main figures."""
    rate = plots.learning.value_label("learning_rate", learning_rate)
    lines = [f"{depth} {style.AXIS_LABELS['n_layers'].lower()}, {style.AXIS_LABELS['d_model'].lower()} {width}", f"{FEEDFORWARD_LABELS.get(feedforward, feedforward)}, lr {rate}"]
    if {"n_layers": depth, "d_model": width, "feedforward": feedforward, "learning_rate": learning_rate} == CANONICAL:
        lines.append(CANONICAL_SUFFIX)
    return "\n".join(lines)


def training_loss(train: pd.DataFrame) -> dict[tuple[int, int, str, float], float]:
    """Each architecture's mean training loss over the last `LOSS_WINDOW_STEPS` steps.

    The loss is averaged over the logged steps in the window within a seed, then over seeds.
    """
    recent = train[train["step"] >= train["step"].max() - LOSS_WINDOW_STEPS]
    summary = tables.seed_mean_se(recent, group_cols=list(ARCHITECTURE), value_col="loss")
    return {(int(a), int(b), str(c), float(d)): float(v) for a, b, c, d, v in zip(*(summary[col] for col in [*ARCHITECTURE, "loss_mean"]), strict=True)}


def _print_loss(ax: Axes, loss: float) -> None:
    """Print the training-loss indicator in the lower left of a panel."""
    ax.text(*LOSS_POSITION, LOSS_LABEL.format(loss), transform=ax.transAxes, ha="left", va="bottom", fontsize=style.FONT_LEGEND_PT)


def build(args: argparse.Namespace) -> Figure:
    """Assemble the grid from the tidy tables.

    A slot whose architecture the table lacks is left empty; the distance legend appears
    once, on the first panel; the x label sits under the lowest panel of each column and the
    y label beside the leftmost panel of each row.
    """
    df = load(EXPERIMENT, "eval", results_root=args.results)
    losses = training_loss(load(EXPERIMENT, "train", results_root=args.results))
    present = {(int(a), int(b), str(c), float(d)) for a, b, c, d in zip(df["n_layers"], df["d_model"], df["feedforward"], df["learning_rate"], strict=True)}
    rows = sorted({(depth, width) for depth, width, _, _ in present}, reverse=True)
    rates = sorted({rate for *_, rate in present})[::-1]
    columns = [(ff, lr) for ff in FEEDFORWARD_ORDER for lr in rates if any(c == ff and r == lr for _, _, c, r in present)]
    fig, axes = figure(len(rows), len(columns), width_in=style.FULL_WIDTH_IN, height_in=min(style.MAX_HEIGHT_IN, ROW_HEIGHT_IN * len(rows)), **SPACING)
    lowest = {column: max(r for r, (depth, width) in enumerate(rows) if (depth, width, *column) in present) for column in columns}
    leftmost = {row: min(c for c, column in enumerate(columns) if (*row, *column) in present) for row in rows}
    first = True
    for r, (depth, width) in enumerate(rows):
        for c, (feedforward, learning_rate) in enumerate(columns):
            ax = axes[r, c]
            if (depth, width, feedforward, learning_rate) not in present:
                ax.set_visible(False)
                continue
            sub = pins(EXPERIMENT, df, fname=EXPERIMENT, n_layers=depth, d_model=width, feedforward=feedforward, learning_rate=learning_rate)
            plots.make_accuracy_by_reasoning_steps(ax, sub, condition=CONDITION, band_alpha=0.18, marker_size=MARKER_SIZE, show_legend=first, legend_loc="lower right", legend_kwargs=LEGEND)
            first = False
            _print_loss(ax, losses[(depth, width, feedforward, learning_rate)])
            ax.set_title(_title(depth, width, feedforward, learning_rate), pad=TITLE_PAD)
            ax.set_yticks(YTICKS)
            if r != lowest[(feedforward, learning_rate)]:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if c != leftmost[(depth, width)]:
                ax.set_ylabel("")
    return fig


if __name__ == "__main__":
    main(build, name=EXPERIMENT)
