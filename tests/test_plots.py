"""The panel-function contract, on synthetic tables that follow the tidy schemas."""

from __future__ import annotations

import dataclasses
import inspect
import io
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_hex

from geo_alg_ti import battery, cli, config, orderings, streams, tables, training
from geo_alg_ti.plots import _frames, content, distance, distribution, learning, placement, representation, style

SEEDS = (0, 1, 2)
DISTANCES = tuple(range(1, 7))
BUDGETS = tuple(range(0, 9))
N_TRIALS = 20
LAYERS = tuple(range(0, 6))
PRESETS = {
    "finetuned": (
        ("iwl_test", "finetuned", "uninformative"),
        ("icl_test", "novel_unseen_cue", "congruent"),
        ("icl_test_training_cue", "training_distribution", "congruent"),
        ("baseline_test", "finetuned", "congruent"),
        ("arbitrage_test", "finetuned", "reversed"),
    ),
    "pretrained": (
        ("iwl_test", "training_distribution", "uninformative"),
        ("icl_test", "novel_unseen_cue", "congruent"),
        ("icl_test_training_cue", "training_distribution", "congruent"),
    ),
    "content": (
        ("frequent_cue_uninformative", "frequent_cue", "uninformative"),
        ("frequent_cue_congruent", "frequent_cue", "congruent"),
        ("frequent_cue_reversed", "frequent_cue", "reversed"),
        ("frequent_cue_shuffled", "frequent_cue", "shuffled"),
        ("icl_test", "novel_unseen_cue", "congruent"),
    ),
}


def _product(**levels: Sequence[object]) -> pd.DataFrame:
    """One row per combination of the given levels, in the order given."""
    return pd.MultiIndex.from_product([list(v) for v in levels.values()], names=list(levels)).to_frame(index=False)


def _with_swept(df: pd.DataFrame, swept: Mapping[str, Sequence[object]]) -> pd.DataFrame:
    """The rows repeated for every combination of the swept axes."""
    if not swept:
        return df
    combos = _product(**swept)
    return df.merge(combos, how="cross")


def _companions(df: pd.DataFrame, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two consistency fractions the evaluation defines per context and source."""
    context = df["context"].to_numpy()
    source = df["source"].to_numpy()
    with_context = np.where(context == "uninformative", np.nan, values)
    prior = np.where(context == "reversed", 1.0 - values, values)
    prior = np.where(source == "novel_unseen_cue", np.nan, prior)
    return with_context, prior


def eval_frame(
    experiment: str,
    preset: str,
    *,
    swept: Mapping[str, Sequence[object]] | None = None,
    steps: Sequence[int] = (1000, 2000),
    tag: str = "pretraining",
    budgets: Sequence[int] = BUDGETS,
    pairs: bool = False,
) -> pd.DataFrame:
    """A deterministic evaluation (or per-pair) table over every condition of a preset."""
    conditions = pd.DataFrame(PRESETS[preset], columns=["condition", "source", "context"])
    metrics = ("accuracy",) if pairs else ("accuracy", "chain_validity", "chain_length")
    grid = _product(condition=conditions["condition"], metric=metrics, d=DISTANCES, K=budgets, seed=SEEDS, step=steps)
    if pairs:
        ranks = range(1, len(DISTANCES) + 2)
        ordered = [(a, b) for a in ranks for b in ranks if a != b]
        pair_rows = pd.DataFrame(ordered, columns=["rank_a", "rank_b"]).assign(d=lambda p: (p["rank_a"] - p["rank_b"]).abs())
        grid = grid.merge(pair_rows, on="d")
    grid = grid.merge(conditions, on="condition")
    grid = _with_swept(grid, swept or {})
    rng = np.random.default_rng(0)
    values = rng.uniform(0.15, 0.95, size=len(grid))
    is_length = grid["metric"].to_numpy() == "chain_length"
    values = np.where(is_length, np.round(values * 8), values)
    with_context, prior = _companions(grid, values)
    grid["tag"] = tag
    grid["experiment"] = experiment
    grid["value"] = values
    grid["n"] = N_TRIALS
    grid["context_consistent_frac"] = with_context
    grid["prior_consistent_frac"] = prior
    schema = tables.PAIR_SCHEMA if pairs else tables.EVAL_SCHEMA
    columns = [*schema, *(swept or {})]
    frame = grid.loc[:, columns].astype({name: kind for name, kind in schema.items()})
    tables.validate_table(frame, schema=schema, swept_columns=tuple(swept or {}))
    return frame


def probe_frame(experiment: str, *, swept: Mapping[str, Sequence[object]] | None = None, steps: Sequence[int] = (2000,)) -> pd.DataFrame:
    """A deterministic probe table: two probe kinds, three metrics, six layers."""
    kinds = pd.DataFrame([("rank_regression", "r2"), ("rank_regression", "spearman"), ("rank_order", "accuracy")], columns=["kind", "metric"])
    grid = _product(kind=kinds["kind"].unique(), metric=kinds["metric"].unique(), layer=LAYERS, seed=SEEDS, step=steps)
    grid = grid.merge(kinds, on=["kind", "metric"])
    grid = _with_swept(grid, swept or {})
    rng = np.random.default_rng(1)
    grid["experiment"] = experiment
    grid["value"] = rng.uniform(0.3, 0.95, size=len(grid))
    grid["n"] = N_TRIALS
    columns = [*tables.PROBE_SCHEMA, *(swept or {})]
    frame = grid.loc[:, columns].astype({name: kind for name, kind in tables.PROBE_SCHEMA.items()})
    tables.validate_table(frame, schema=tables.PROBE_SCHEMA, swept_columns=tuple(swept or {}))
    return frame


def edit_frame(experiment: str, *, swept: Mapping[str, Sequence[object]] | None = None, steps: Sequence[int] = (2000,)) -> pd.DataFrame:
    """A deterministic edit table: erases at three gains, writes at five strengths, controls.

    Every row carries the write target the write panel pins; erase rows leave it empty.
    """
    erase = pd.DataFrame([("erase", "rank", "", g, np.nan) for g in (1.0, 0.0, -1.0)] + [("erase", "random", "", 0.0, np.nan)], columns=["edit_kind", "direction", "target", "gain", "strength"])
    write = pd.DataFrame(
        [("write", "rank", "unrelated", np.nan, s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)] + [("write", "random", "unrelated", np.nan, 1.0)],
        columns=["edit_kind", "direction", "target", "gain", "strength"],
    )
    edits = pd.concat([erase, write], ignore_index=True)
    base = eval_frame(experiment, "content", steps=steps, budgets=(0, 8))
    base = base[base["metric"] == "accuracy"].drop(columns=list(swept or {}), errors="ignore")
    grid = base.merge(edits, how="cross")
    grid = _with_swept(grid, swept or {})
    rng = np.random.default_rng(2)
    values = rng.uniform(0.15, 0.95, size=len(grid))
    with_context, prior = _companions(grid, values)
    is_write = grid["edit_kind"].to_numpy() == "write"
    grid["value"] = values
    grid["context_consistent_frac"] = with_context
    grid["prior_consistent_frac"] = prior
    grid["written_consistent_frac"] = np.where(is_write, rng.uniform(0.1, 0.9, size=len(grid)), np.nan)
    grid["spearman_vs_familiar"] = np.where(is_write, rng.uniform(-0.2, 0.2, size=len(grid)), np.nan)
    columns = [*tables.EDIT_SCHEMA, *(swept or {})]
    frame = grid.loc[:, columns].astype({name: kind for name, kind in tables.EDIT_SCHEMA.items()})
    tables.validate_table(frame, schema=tables.EDIT_SCHEMA, swept_columns=tuple(swept or {}))
    return frame


HEAD_LABELS = ("L1H1", "L1H2", "L1H3", "L1H4", "L2H1", "L2H2", "L2H3", "L2H4")
HEAD_SET_SIZES = (1, 2)
N_HEAD_DRAWS = 3


def ablate_frame(intact: pd.DataFrame, experiment: str, *, swept: Mapping[str, Sequence[object]] | None = None) -> pd.DataFrame:
    """A deterministic ablate table: the intact rows given, each of eight heads ablated alone, each
    network's top one and two heads, and three draws of each size from outside twice as many.

    The intact rows are an edit table's unedited rows, so the two tables agree on the intact
    networks as the summary bars require; each seed ranks the heads in a different order.
    """
    base = intact.loc[:, [*tables.EVAL_SCHEMA, *(swept or {})]].assign(experiment=experiment)
    unlabeled = {"set_kind": "intact", "head": "", "layer": np.nan, "n_heads": 0, "heads": "", "draw": -1, "attention_rank": -1}
    parts = [base.assign(**unlabeled)]
    rng = np.random.default_rng(4)

    def ablated(rows: pd.DataFrame, **columns: object) -> pd.DataFrame:
        part = rows.assign(**{**unlabeled, **columns})
        part["value"] = rng.uniform(0.1, 0.95, size=len(part))
        return part

    for seed in SEEDS:
        rows = base[base["seed"] == seed]
        order = [HEAD_LABELS[(index + 3 * seed) % len(HEAD_LABELS)] for index in range(len(HEAD_LABELS))]
        for label in HEAD_LABELS:
            parts.append(ablated(rows, set_kind="head", head=label, layer=float(label[1]), n_heads=1, heads=label, attention_rank=order.index(label) + 1))
        for size in HEAD_SET_SIZES:
            parts.append(ablated(rows, set_kind="top", n_heads=size, heads="+".join(order[:size])))
            for draw in range(N_HEAD_DRAWS):
                parts.append(ablated(rows, set_kind="random", n_heads=size, heads="+".join(order[2 * size + draw : 3 * size + draw]), draw=draw))
    columns = [*tables.ABLATE_SCHEMA, *(swept or {})]
    frame = pd.concat(parts, ignore_index=True).loc[:, columns].astype(dict(tables.ABLATE_SCHEMA))
    tables.validate_table(frame, schema=tables.ABLATE_SCHEMA, swept_columns=tuple(swept or {}))
    return frame


def ablated_pairs_frame(pairs: pd.DataFrame, *, swept: Mapping[str, Sequence[object]] | None = None) -> pd.DataFrame:
    """A deterministic per-pair table scored with heads ablated: the rows of `pairs` with their
    own accuracies, each seed's two ablated heads named."""
    frame = pairs.copy()
    values = np.random.default_rng(5).uniform(0.1, 0.95, size=len(frame))
    with_context, prior = _companions(frame, values)
    frame["value"] = values
    frame["context_consistent_frac"] = with_context
    frame["prior_consistent_frac"] = prior
    frame["heads"] = ["+".join(HEAD_LABELS[seed : seed + 2]) for seed in frame["seed"]]
    columns = [*tables.ABLATED_PAIR_SCHEMA, *(swept or {})]
    frame = frame.loc[:, columns].astype(dict(tables.ABLATED_PAIR_SCHEMA))
    tables.validate_table(frame, schema=tables.ABLATED_PAIR_SCHEMA, swept_columns=tuple(swept or {}))
    return frame


def _unedited(edit: pd.DataFrame) -> pd.DataFrame:
    """The rows of an edit table that leave the rank direction intact."""
    return tables.pick(edit, edit_kind="erase", direction="rank", gain=1.0)


def train_frame(experiment: str, *, swept: Mapping[str, Sequence[object]] | None = None, stage: str = "pretraining") -> pd.DataFrame:
    """A deterministic training curve table."""
    grid = _with_swept(_product(seed=SEEDS, step=(100, 200, 300)), swept or {})
    rng = np.random.default_rng(3)
    grid["experiment"] = experiment
    grid["stage"] = stage
    for name in ("loss", "token_cross_entropy", "scalar_squared_error", "answer_accuracy", "chain_token_accuracy"):
        grid[name] = rng.uniform(0.1, 2.0, size=len(grid))
    columns = [*tables.TRAIN_SCHEMA, *(swept or {})]
    frame = grid.loc[:, columns].astype({name: kind for name, kind in tables.TRAIN_SCHEMA.items()})
    tables.validate_table(frame, schema=tables.TRAIN_SCHEMA, swept_columns=tuple(swept or {}))
    return frame


TWO_AXES = {"skew_alpha": (0.0, 1.0), "p_cot": (0.5, 0.9)}
CONTENT_AXES = {"frequent_cue_prop": (0.1, 0.2), "p_cot": (0.5, 0.9)}
PINNED = {"skew_alpha": 0.0, "p_cot": 0.9}
CONTENT_PINNED = {"frequent_cue_prop": 0.2, "p_cot": 0.9}
# The frames of the content-effects networks, whose swept axes are the content ones.
CONTENT_KINDS = ("content", "probe", "edit", "ablate")
# The per-pair frames of the reasoning-during-finetuning networks: the finetuning conditions are
# drawn, the learning rate is pinned.
REASONING_AXES = {"learning_rate": (0.0001, 0.001), "ablated_heads": ("none", "context_reading", "random"), "reasoning_prop": (0.0, 1.0)}
REASONING_PINNED = {"learning_rate": 0.001}
REASONING_KINDS = ("reasoning_pairs", "reasoning_ablated", "reasoning_rows")
REASONING_DRAWN = ("ablated_heads", "reasoning_prop")


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """One swept frame per schema; each swept axis holds two values."""
    edit = edit_frame("representation", swept=CONTENT_AXES)
    reasoning = eval_frame("finetuning_reasoning", "finetuned", swept=REASONING_AXES, tag="finetuning", pairs=True, steps=(800, 1600))
    # Trial counts that differ from pair to pair, so pooling by trials is not a plain mean.
    reasoning["n"] = (N_TRIALS + 3 * reasoning["rank_a"] + reasoning["rank_b"]).astype("int64")
    reasoning_ablated = ablated_pairs_frame(reasoning, swept=REASONING_AXES)
    return {
        "eval": eval_frame("skew", "finetuned", swept=TWO_AXES, tag="finetuning"),
        "pairs": eval_frame("skew", "finetuned", swept=TWO_AXES, tag="finetuning", pairs=True, steps=(2000,), budgets=(0, 8)),
        "content": eval_frame("content_effects", "content", swept=CONTENT_AXES),
        "probe": probe_frame("representation", swept=CONTENT_AXES),
        "edit": edit,
        "ablate": ablate_frame(_unedited(edit), "components", swept=CONTENT_AXES),
        "reasoning_pairs": reasoning,
        "reasoning_ablated": reasoning_ablated,
        "reasoning_rows": tables.finetuned_pair_rows(reasoning, reasoning_ablated),
    }


@dataclasses.dataclass(frozen=True)
class FrameRef:
    """A panel keyword that takes another of the fixture's frames."""

    kind: str


def _keywords(frames: dict[str, pd.DataFrame], kwargs: Mapping[str, object], *, pinned: bool = True) -> dict[str, object]:
    """The panel keywords with each `FrameRef` replaced by its frame, pinned unless asked not to."""
    return {
        name: (_pinned(frames, value.kind, ()) if pinned else frames[value.kind]) if isinstance(value, FrameRef) else value
        for name, value in kwargs.items()
    }


# Every panel: the frame it reads, the keywords it needs beyond (ax, df), and which swept
# columns it draws itself (so they may stay unpinned).
PANELS = [
    (distance.make_accuracy_by_distance, "eval", {}, ()),
    (distance.make_accuracy_by_reasoning_steps, "eval", {}, ()),
    (distance.make_steps_to_criterion_by_distance, "eval", {}, ()),
    (distance.make_accuracy_bars, "eval", {}, ()),
    (distance.make_accuracy_numbers, "eval", {"anchors": {"iwl_test": (2.0, 0.5), "icl_test": (4.0, 0.5)}}, ()),
    (distance.make_pair_accuracy, "pairs", {}, ()),
    (distance.make_response_classes_by_reasoning_steps, "eval", {}, ()),
    (learning.make_accuracy_by_step, "eval", {"condition": "icl_test"}, ()),
    (learning.make_reasoning_gain_by_step, "eval", {"condition": "icl_test"}, ()),
    (learning.make_pooled_accuracy_vs_sweep, "eval", {"x_col": "skew_alpha"}, ("skew_alpha",)),
    (distribution.make_accuracy_vs_skew, "eval", {}, ("skew_alpha",)),
    (distribution.make_icl_iwl_dissociation, "eval", {}, ("skew_alpha",)),
    (distribution.make_emergence_by_skew, "eval", {}, ("skew_alpha",)),
    (content.make_context_accuracy_by_distance, "content", {}, ()),
    (content.make_context_accuracy_by_reasoning_steps, "content", {}, ()),
    (representation.make_probe_accuracy_by_layer, "probe", {}, ("frequent_cue_prop",)),
    (representation.make_edit_accuracy_by_distance, "edit", {}, ()),
    (representation.make_write_by_strength, "edit", {}, ()),
    (representation.make_head_ablation_scatter, "ablate", {}, ()),
    (representation.make_intervention_bars, "edit", {"ablate": FrameRef("ablate")}, ()),
    (learning.make_pair_type_bars, "reasoning_rows", {}, REASONING_DRAWN),
    (learning.make_pair_type_by_step, "reasoning_rows", {}, REASONING_DRAWN),
]
POOLED_PANELS = {
    distance.make_accuracy_bars,
    distance.make_accuracy_numbers,
    distance.make_response_classes_by_reasoning_steps,
    learning.make_pooled_accuracy_vs_sweep,
    distribution.make_accuracy_vs_skew,
    representation.make_head_ablation_scatter,
    representation.make_intervention_bars,
}


# The bar panels, whose bars stand for conditions and so carry no x-axis label.
BAR_PANELS = {distance.make_accuracy_bars, representation.make_intervention_bars, learning.make_pair_type_bars}
# The panels that write onto a panel already drawn, and so label no axis of their own.
ANNOTATION_PANELS = {distance.make_accuracy_numbers}


def _pinned(frames: dict[str, pd.DataFrame], kind: str, keep: Sequence[str]) -> pd.DataFrame:
    """The frame with every swept axis pinned except those in `keep`."""
    pins = CONTENT_PINNED if kind in CONTENT_KINDS else REASONING_PINNED if kind in REASONING_KINDS else PINNED
    return tables.pick(frames[kind], **{name: value for name, value in pins.items() if name not in keep})


def _every_panel():
    """Every `make_*` of the five modules, so a panel missing from PANELS fails the contract."""
    found = set()
    for module in (distance, learning, distribution, content, representation):
        found.update(getattr(module, name) for name in dir(module) if name.startswith("make_"))
    return found


def test_every_panel_is_covered():
    """The table above names every panel function, and no other."""
    listed = {panel for panel, *_ in PANELS} | {distribution.make_ordering_frequency_schematic}
    assert listed == _every_panel()


@pytest.mark.parametrize("panel,kind,kwargs,keep", PANELS, ids=lambda p: getattr(p, "__name__", str(p)))
def test_panels_draw_on_the_given_axes_only(frames, panel, kind, kwargs, keep):
    """No panel creates a figure or an axes or mutates rcParams, and every one reports seeds."""
    df = _pinned(frames, kind, keep)
    fig, ax = plt.subplots()
    try:
        before = plt.get_fignums()
        params = dict(matplotlib.rcParams)
        stats = panel(ax, df, **_keywords(frames, kwargs))
        assert plt.get_fignums() == before
        assert len(fig.axes) == 1
        assert dict(matplotlib.rcParams) == params
        assert isinstance(stats, dict)
        assert stats["n_seeds"] == len(SEEDS)
        assert ax.get_ylabel() or panel in ANNOTATION_PANELS
        assert ax.get_xlabel() or panel in BAR_PANELS | ANNOTATION_PANELS
        assert ax.get_xlabel() not in ("K", "d") and ax.get_ylabel() not in ("K", "d")
    finally:
        plt.close(fig)


def _legend_texts(ax) -> list[str]:
    """The legend entries of an axes, in order."""
    legend = ax.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def test_colorbars_are_drawn_only_beside_a_given_panel(frames):
    """The dissociation and probe panels add one colorbar beside `colorbar_ax`, nothing else."""
    fig, (scatter, probe) = plt.subplots(1, 2)
    try:
        distribution.make_icl_iwl_dissociation(scatter, _pinned(frames, "eval", ("skew_alpha",)), colorbar_ax=scatter)
        assert len(fig.axes) == 3
        assert fig.axes[-1].get_ylabel() == "Skew (α)"
        representation.make_probe_accuracy_by_layer(probe, _pinned(frames, "probe", ("frequent_cue_prop",)), colorbar_ax=probe)
        assert len(fig.axes) == 4
        assert fig.axes[-1].get_ylabel() == "Frequent-cue proportion"
        assert [t.get_text() for t in fig.axes[-1].get_yticklabels()] == ["0.1", "0.2"]
        assert probe.get_legend() is None
    finally:
        plt.close(fig)


def _colorbar_axes(fig, panels) -> list:
    """The axes a figure holds beyond `panels`, in the order they were added."""
    return [ax for ax in fig.axes if ax not in panels]


def test_colorbars_attach_beside_a_panel(frames):
    """The bar sits beside `colorbar_ax` at the panel's pad and fraction, centered on it."""
    fig, (scatter, probe) = plt.subplots(1, 2)
    try:
        stats = distribution.make_icl_iwl_dissociation(scatter, _pinned(frames, "eval", ("skew_alpha",)), colorbar_ax=scatter)
        assert stats["n_seeds"] == len(SEEDS)
        (bar,) = _colorbar_axes(fig, (scatter, probe))
        assert bar.get_ylabel() == "Skew (α)"
        representation.make_probe_accuracy_by_layer(probe, _pinned(frames, "probe", ("frequent_cue_prop",)), colorbar_ax=probe)
        _, probe_bar = _colorbar_axes(fig, (scatter, probe))
        fig.canvas.draw()
        # The panel gives up `pad` and `fraction` of its former width to the gap and the bar,
        # which stays within that width and the panel's height, centered on the panel.
        for panel, placed, pad, fraction in ((scatter, bar, distribution.COLORBAR_PAD, distribution.COLORBAR_FRACTION), (probe, probe_bar, style.COLORBAR_PAD, style.COLORBAR_FRACTION)):
            box, bar_box = panel.get_position(), placed.get_position()
            former = box.width / (1.0 - fraction - pad)
            assert bar_box.x0 > box.x1
            assert (bar_box.x0 - box.x1) / former == pytest.approx(pad, abs=0.005)
            assert bar_box.width / former <= fraction + 0.005
            assert bar_box.height <= box.height + 0.01
            assert (bar_box.y0 + bar_box.y1) / 2 == pytest.approx((box.y0 + box.y1) / 2, abs=0.01)
    finally:
        plt.close(fig)


def test_colorbar_placement_defaults_are_fixed():
    """The distance from its panel and the width a colorbar takes, as fractions of the panel."""
    assert (style.COLORBAR_PAD, style.COLORBAR_FRACTION) == (0.02, 0.06)
    assert (distribution.COLORBAR_PAD, distribution.COLORBAR_FRACTION) == (0.05, 0.15)


def test_labels_follow_the_paper(frames):
    """Axis labels, tick labels, and legend entries carry the paper's wording."""
    fig, ax = plt.subplots()
    try:
        distance.make_pair_accuracy(ax, _pinned(frames, "pairs", ()))
        assert ax.get_xlabel() == "Item pair"
        assert [t.get_text() for t in ax.texts] == [f"Distance {d}" for d in DISTANCES]
        ax.clear()
        distance.make_accuracy_by_reasoning_steps(ax, _pinned(frames, "eval", ()))
        assert ax.get_legend().get_title().get_text() == "Distance"
        ax.clear()
        distance.make_response_classes_by_reasoning_steps(ax, _pinned(frames, "eval", ()))
        assert ax.get_ylabel() == "Proportion of responses"
        assert _legend_texts(ax) == ["Trained ordering", "Context ordering"]
        ax.clear()
        distance.make_steps_to_criterion_by_distance(ax, _pinned(frames, "eval", ()))
        assert ax.get_ylabel() == "Reasoning steps"
        ax.clear()
        distribution.make_accuracy_vs_skew(ax, _pinned(frames, "eval", ("skew_alpha",)))
        assert ax.get_xlabel() == "Skew (α)"
        ax.clear()
        distribution.make_ordering_frequency_schematic(ax)
        assert _legend_texts(ax) == ["α = 0", "α = 2", "α = 5", "α = 10"]
        assert ax.get_legend().get_title().get_text() == ""
        ax.clear()
        representation.make_edit_accuracy_by_distance(ax, _pinned(frames, "edit", ()))
        assert _legend_texts(ax) == ["Intact", "Rank direction inverted", "Rank direction erased", "Random direction erased"]
        ax.clear()
        representation.make_intervention_bars(ax, _pinned(frames, "edit", ()), _pinned(frames, "ablate", ()))
        assert [t.get_text() for t in ax.get_xticklabels()] == ["IWL test\n0 steps", "Reversed\n0 steps", "Reversed\n8 steps", "ICL test\n8 steps"]
        assert ax.get_xticklabels()[0].get_rotation() == 0
        keys = [text for text in _legend_texts(ax) if text.strip()]
        assert keys == ["Intact", "Rank direction erased", "Random direction erased", "Context-reading heads ablated", "Random heads ablated"]
        ax.clear()
        representation.make_head_ablation_scatter(ax, _pinned(frames, "ablate", ()))
        assert ax.get_xlabel() == "Change in IWL test accuracy, 0 steps"
        assert ax.get_ylabel() == "Change in ICL test accuracy, 8 steps"
        assert _legend_texts(ax) == ["Layer 1", "Layer 2", "Context-reading heads"]
        ax.clear()
        representation.make_write_by_strength(ax, _pinned(frames, "edit", ()))
        assert ax.get_ylabel() == "Agreement with written ordering"
        ax.clear()
        representation.make_probe_accuracy_by_layer(ax, _pinned(frames, "probe", ("frequent_cue_prop",)))
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Embed.", "1", "2", "3", "4", "Output"]
        ax.clear()
        learning.make_pooled_accuracy_vs_sweep(ax, _pinned(frames, "eval", ("skew_alpha",)), x_col="skew_alpha", line_col="p_cot", conditions=("iwl_test",), line_labels={0.9: "Frequent"})
        assert _legend_texts(ax) == ["Frequent"]
        assert ax.get_legend().get_title().get_text() == ""
    finally:
        plt.close(fig)


def test_pair_panel_draws_the_budgets_the_rows_hold(frames):
    """Budgets the per-pair rows lack are left out of the lines and reported; none at all raises."""
    df = _pinned(frames, "pairs", ())
    fig, (ax, other) = plt.subplots(1, 2)
    try:
        stats = distance.make_pair_accuracy(ax, df, condition="icl_test", k_values=range(9))
        assert stats["k_values"] == [0, 8]
        assert _legend_texts(ax) == ["0", "8"]
        with pytest.raises(ValueError):
            distance.make_pair_accuracy(ax, df, condition="icl_test", k_values=(3,))
        style.sweep_colorbar(style.P_COT_CMAP, ax=other, span=(0.0, 1.0), title="p(CoT)", ticks=(0.0, 0.5, 1.0))
        assert len(fig.axes) == 3
        assert fig.axes[-1].get_ylabel() == "p(CoT)"
        assert [t.get_text() for t in fig.axes[-1].get_yticklabels()] == ["0.0", "0.5", "1.0"]
    finally:
        plt.close(fig)


def test_steps_panels_draw_the_running_best_within_each_seed(frames):
    """With `k_stat="running_best"`, each line is the mean of each seed's best so far."""
    df = _pinned(frames, "eval", ())
    fig, (ax, other) = plt.subplots(1, 2)
    try:
        stats = distance.make_accuracy_by_reasoning_steps(ax, df, condition="icl_test", k_stat="running_best")
        rows = tables.pick(df, metric="accuracy", condition="icl_test", step=int(df["step"].max()), d=6)
        best = rows.sort_values("K").groupby("seed")["value"].cummax().groupby(rows["K"]).mean()
        assert stats["k_stat"] == "running_best"
        for k in stats["k_values"]:
            assert stats["mean_by_distance_by_k"][6][k] == pytest.approx(best[k], abs=1e-4)
        assert all(np.diff([stats["mean_by_distance_by_k"][6][k] for k in stats["k_values"]]) >= -1e-9)
        ax.clear()
        subset = distance.make_accuracy_by_reasoning_steps(ax, df, condition="icl_test", k_values=(0, 8), k_stat="running_best", xticks=(0, 8), xlim=(-1.0, 9.0), yticks=(0.0, 1.0))
        within = rows[rows["K"].isin((0, 8))].sort_values("K").groupby("seed")["value"].cummax()
        assert subset["mean_by_distance_by_k"][6][8] == pytest.approx(within.groupby(rows["K"]).mean()[8], abs=1e-4)
        assert list(ax.get_xticks()) == [0, 8] and ax.get_xlim() == (-1.0, 9.0) and list(ax.get_yticks()) == [0.0, 1.0]
        assert ax.get_ylabel() == "Accuracy"
        distance.make_accuracy_by_reasoning_steps(other, df, condition="icl_test", ylabel="Accuracy (best so far)")
        assert other.get_ylabel() == "Accuracy (best so far)"
        with pytest.raises(ValueError):
            distance.make_accuracy_by_reasoning_steps(ax, df, k_stat="best")
    finally:
        plt.close(fig)


def test_family_labels_write_small_numbers_in_full():
    """A learning rate of 0.00001 is written out, never in powers of ten."""
    assert learning.value_label("learning_rate", 1e-05) == "0.00001"
    assert learning.value_label("learning_rate", 0.001) == "0.001"
    assert learning.value_label("replay_prop", 0.5) == "0.5"
    assert learning.value_label("n_layers", 2) == "2"
    assert learning.value_label("layout", "disjoint") == "disjoint"


def test_distance_panels_refuse_a_requested_distance_the_rows_lack(frames):
    """A `d_values` entry absent from the rows raises, naming it, rather than being dropped."""
    evals, edits = _pinned(frames, "eval", ()), _pinned(frames, "edit", ())
    content_rows = _pinned(frames, "content", ())
    fig, ax = plt.subplots()
    try:
        distance.make_steps_to_criterion_by_distance(ax, evals, d_values=(2, 3, 4, 5, 6))
        assert [int(t) for t in ax.get_xticks()] == [2, 3, 4, 5, 6]
        calls = (
            lambda: distance.make_accuracy_by_distance(ax, evals, d_values=(2, 9)),
            lambda: distance.make_steps_to_criterion_by_distance(ax, evals, d_values=(2, 9)),
            lambda: content.make_context_accuracy_by_distance(ax, content_rows, d_values=(2, 9)),
            lambda: representation.make_edit_accuracy_by_distance(ax, edits, d_values=(2, 9)),
        )
        for call in calls:
            with pytest.raises(ValueError, match=r"\[9\]"):
                call()
    finally:
        plt.close(fig)


def test_distance_ticks_cover_every_condition_drawn(frames):
    """The ticks are the union of the distances of every condition, not the last one's."""
    df = _pinned(frames, "eval", ())
    df = df[~((df["condition"] == "icl_test") & (df["d"] == 1))]
    fig, ax = plt.subplots()
    try:
        distance.make_accuracy_by_distance(ax, df, conditions=("iwl_test", "icl_test"))
        assert [int(t) for t in ax.get_xticks()] == list(DISTANCES)
    finally:
        plt.close(fig)


def test_context_recovery_draws_two_contexts_with_one_legend_entry_each(frames):
    """Congruent and reversed contexts share the axes in their own shades, keyed by context."""
    df = _pinned(frames, "content", ())
    fig, ax = plt.subplots()
    try:
        stats = content.make_context_accuracy_by_reasoning_steps(ax, df, contexts=("congruent", "reversed"), d_values=(2, 3, 4, 5, 6), k_stat="running_best", show_seeds=True)
        assert _legend_texts(ax) == ["Congruent", "Reversed"]
        assert ax.get_legend().get_title().get_text() == ""
        assert stats["per_context"]["congruent"]["cmap"] == style.CONGRUENT_DISTANCE_CMAP
        assert stats["per_context"]["reversed"]["cmap"] == "Oranges"
        rows = tables.pick(df, metric="accuracy", condition="frequent_cue_reversed", step=int(df["step"].max()), d=6)
        best = rows.sort_values("K").groupby("seed")["value"].cummax().groupby(rows["K"]).mean()
        assert stats["per_context"]["reversed"]["mean_by_distance_by_k"][6][8] == pytest.approx(best[8], abs=1e-4)
        assert stats["per_context"]["reversed"]["mean_by_distance_by_k"][6][0] == pytest.approx(best[0], abs=1e-4)
        ax.clear()
        content.make_context_accuracy_by_reasoning_steps(ax, df)
        assert ax.get_legend().get_title().get_text() == "Distance"
        assert [int(t) for t in ax.get_xticks()] == list(BUDGETS)
    finally:
        plt.close(fig)


def test_context_recovery_distance_key_and_axes(frames):
    """The distance key names the distances in neutral shades; the axes follow the keywords."""
    df = _pinned(frames, "content", ())
    fig, ax = plt.subplots()
    try:
        content.make_context_accuracy_by_reasoning_steps(ax, df, contexts=("congruent", "reversed"), legend_mode="distance_key", xticks=(0, 2, 4), xlim=(-0.5, 8.5), yticks=(0.0, 0.5, 1.0), ylabel="Accuracy (best so far)")
        assert ax.get_ylabel() == "Accuracy (best so far)"
        assert _legend_texts(ax) == [str(d) for d in DISTANCES]
        assert ax.get_legend().get_title().get_text() == "Distance"
        shades = [to_hex(h.get_color()) for h in ax.get_legend().legend_handles]
        assert shades == style.distance_colors(DISTANCES, style.DISTANCE_KEY_CMAP)
        assert list(ax.get_xticks()) == [0, 2, 4] and ax.get_xlim() == (-0.5, 8.5)
        assert list(ax.get_yticks()) == [0.0, 0.5, 1.0]
    finally:
        plt.close(fig)


def test_split_shuffled_halves_draw_as_contexts(frames):
    """Once split, the agreeing and conflicting halves are drawn in their own colors and labels."""
    df = _pinned(frames, "content", ())
    shuffled = (df["condition"] == "frequent_cue_shuffled") & (df["metric"] == "accuracy")
    # Answers that follow the frequent cue's ordering half the time leave the halves equal.
    df.loc[shuffled, "prior_consistent_frac"] = 0.5
    split = tables.split_shuffled(df[shuffled])
    both = pd.concat([df, split], ignore_index=True)
    fig, ax = plt.subplots()
    try:
        contexts = ("congruent", *content.SHUFFLED_SPLIT_CONTEXTS)
        stats = content.make_context_accuracy_by_distance(ax, both, contexts=contexts, k_pin=0)
        assert _legend_texts(ax) == ["Congruent", "Shuffled (agree)", "Shuffled (conflict)"]
        drawn = {line.get_label(): matplotlib.colors.to_hex(line.get_color()) for line in ax.lines if not line.get_label().startswith("_")}
        assert drawn["Shuffled (agree)"] == style.SHUFFLED_AGREE and drawn["Shuffled (conflict)"] == style.SHUFFLED_CONFLICT
        whole = stats["per_context"]["shuffled_agree"]["mean_by_distance"]
        assert whole == stats["per_context"]["shuffled_conflict"]["mean_by_distance"]
        ax.clear()
        with pytest.raises(ValueError, match="no rows match"):
            content.make_context_accuracy_by_distance(ax, df, contexts=content.SHUFFLED_SPLIT_CONTEXTS)
    finally:
        plt.close(fig)


def _floor_frame(frames):
    """Content rows where the shuffled context sits at chance for all seeds at distance six,
    for two of three seeds at distance five, and for one at distance four."""
    df = _pinned(frames, "content", ())
    shuffled = df["condition"] == "frequent_cue_shuffled"
    for d, seeds in ((6, SEEDS), (5, SEEDS[:2]), (4, SEEDS[:1])):
        at_chance = shuffled & (df["d"] == d) & df["seed"].isin(list(seeds)) & (df["metric"] == "accuracy")
        df.loc[at_chance, "value"] = 0.5
    return df


def _crosses(ax) -> dict[str, list[tuple[float, float]]]:
    """The unreached crosses drawn on a panel, by color: (distance, height) per cross."""
    found: dict[str, list[tuple[float, float]]] = {}
    for collection in ax.collections:
        if collection.get_paths() and np.asarray(collection.get_offsets()).size and _is_cross(collection):
            color = matplotlib.colors.to_hex(collection.get_edgecolor()[0])
            found.setdefault(color, []).extend((float(x), float(y)) for x, y in collection.get_offsets())
    return found


def _is_cross(collection) -> bool:
    """Whether a scatter draws the unreached marker."""
    marker = matplotlib.markers.MarkerStyle(style.UNREACHED_MARKER)
    expected = marker.get_path().transformed(marker.get_transform()).vertices
    drawn = collection.get_paths()[0].vertices
    return drawn.shape == expected.shape and np.allclose(drawn, expected)


def test_steps_to_criterion_marks_distances_not_reached_above_chance(frames):
    """A distance most seeds cannot lift above chance is a cross in its condition's color."""
    df = _floor_frame(frames)
    at_chance = (df["condition"] == "frequent_cue_congruent") & (df["d"] == 6) & (df["metric"] == "accuracy")
    df.loc[at_chance, "value"] = 0.5
    conditions = ("frequent_cue_congruent", "frequent_cue_shuffled")
    fig, ax = plt.subplots()
    try:
        stats = distance.make_steps_to_criterion_by_distance(ax, df, conditions=conditions, unreached_below_chance=True)
        shuffled = stats["per_condition"]["frequent_cue_shuffled"]
        assert shuffled["unreached_distances"] == [5, 6] and stats["unreached_distances"] == [5, 6]
        assert shuffled["n_seeds_without_value_by_distance"] == {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3}
        assert shuffled["n_seeds_with_value_by_distance"] == {1: 3, 2: 3, 3: 3, 4: 2, 5: 1, 6: 0}
        assert set(shuffled["mean_by_distance"]) == {1, 2, 3, 4}
        congruent = stats["per_condition"]["frequent_cue_congruent"]
        assert congruent["unreached_distances"] == [6] and set(congruent["mean_by_distance"]) == {1, 2, 3, 4, 5}
        # Distance four averages the two seeds that reach the criterion.
        rows = tables.pick(df, metric="accuracy", condition="frequent_cue_shuffled", step=int(df["step"].max()), d=4)
        reached = tables.first_k_to_criterion(rows[rows["seed"] != SEEDS[0]], group_cols=["d", "seed"])
        assert shuffled["mean_by_distance"][4] == pytest.approx(reached[tables.K_TO_CRITERION].mean(), abs=1e-4)
        # Each condition's crosses take its color, the second condition's raised by the offset,
        # so both conditions show a cross at distance six; all of them lie inside the axes.
        top, raised = max(BUDGETS) + 1, max(BUDGETS) + 1 + style.UNREACHED_OFFSET
        crosses = _crosses(ax)
        assert crosses == {style.CONGRUENT: [(6.0, top)], style.SHUFFLED: [(5.0, raised), (6.0, raised)]}
        fig.canvas.draw()
        low, high = ax.get_ylim()
        assert all(low < y < high for marks in crosses.values() for _, y in marks)
        # One gray cross keys every mark in the legend.
        keys = [line for line in ax.lines if line.get_label() == style.UNREACHED_LABEL]
        assert len(keys) == 1 and keys[0].get_marker() == style.UNREACHED_MARKER and keys[0].get_color() == style.REFERENCE_LINE
        assert _legend_texts(ax).count(style.UNREACHED_LABEL) == 1
        assert [line.get_color() for line in ax.lines if line.get_label() == "Shuffled"] == [style.SHUFFLED]
        assert [int(t) for t in ax.get_xticks()] == list(DISTANCES)
    finally:
        plt.close(fig)


def test_steps_to_criterion_breaks_its_line_at_unreached_distances_when_asked(frames):
    """Breaking leaves no segment across an unreached distance; joining spans it."""
    df = _pinned(frames, "content", ())
    at_chance = (df["condition"] == "frequent_cue_shuffled") & (df["d"] == 3) & (df["metric"] == "accuracy")
    df.loc[at_chance, "value"] = 0.5
    fig, ax = plt.subplots()
    try:
        options = {"conditions": ("frequent_cue_shuffled",), "unreached_below_chance": True, "show_seeds": False}
        distance.make_steps_to_criterion_by_distance(ax, df, break_at_unreached=True, **options)
        runs = [line.get_xdata().tolist() for line in ax.lines if line.get_color() == style.SHUFFLED]
        assert runs == [[1, 2], [4, 5, 6]]
        assert _legend_texts(ax) == ["Shuffled", style.UNREACHED_LABEL]
        ax.clear()
        distance.make_steps_to_criterion_by_distance(ax, df, **options)
        assert [line.get_xdata().tolist() for line in ax.lines if line.get_color() == style.SHUFFLED] == [[1, 2, 4, 5, 6]]
    finally:
        plt.close(fig)


def test_steps_to_criterion_without_the_floor_draws_as_before(frames):
    """Left off, the floor changes nothing: every seed counts and nothing is crossed out."""
    df = _floor_frame(frames)
    conditions = ("frequent_cue_shuffled",)
    fig, (ax, other) = plt.subplots(1, 2)
    try:
        stats = distance.make_steps_to_criterion_by_distance(ax, df, conditions=conditions)
        assert stats["unreached_distances"] == [] and not _crosses(ax)
        assert style.UNREACHED_LABEL not in [line.get_label() for line in ax.lines]
        assert stats["per_condition"]["frequent_cue_shuffled"]["n_seeds_with_value_by_distance"] == {d: 3 for d in DISTANCES}
        rows = tables.pick(df, metric="accuracy", condition="frequent_cue_shuffled", step=int(df["step"].max()))
        reached = tables.first_k_to_criterion(rows, group_cols=["d", "seed"])
        expected = reached.groupby("d")[tables.K_TO_CRITERION].mean()
        for d in DISTANCES:
            assert stats["per_condition"]["frequent_cue_shuffled"]["mean_by_distance"][d] == pytest.approx(expected[d], abs=1e-4)
        distance.make_steps_to_criterion_by_distance(other, df, conditions=conditions, unreached_below_chance=False)
        for mine, theirs in zip(ax.lines, other.lines, strict=True):
            assert np.array_equal(mine.get_xydata(), theirs.get_xydata())
    finally:
        plt.close(fig)


def test_schematic_is_data_free():
    """The frequency schematic takes no table and reports no seeds."""
    fig, ax = plt.subplots()
    try:
        stats = distribution.make_ordering_frequency_schematic(ax)
        assert stats["n_seeds"] == 0
        assert set(stats["frequency_by_skew"]) == {"0", "2", "5", "10"}
        assert stats["frequency_by_skew"]["0"] == pytest.approx([0.1] * 5)
        stats = distribution.make_ordering_frequency_schematic(ax, skew_alphas=(0, 1, 3), n_orderings=100, n_shown=10)
        assert stats["frequency_by_skew"]["0"] == pytest.approx([0.01] * 10)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("panel,kind,kwargs,keep", PANELS, ids=lambda p: getattr(p, "__name__", str(p)))
def test_panels_raise_when_a_swept_axis_is_unpinned(frames, panel, kind, kwargs, keep):
    """A frame with several values of a swept setting and no pin raises, naming the column."""
    df = frames[kind]
    swept = CONTENT_AXES if kind in CONTENT_KINDS else REASONING_AXES if kind in REASONING_KINDS else TWO_AXES
    unpinned = [name for name in swept if name not in keep]
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError) as caught:
            panel(ax, df, **_keywords(frames, kwargs, pinned=False))
        assert any(name in str(caught.value) for name in unpinned)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("panel,kind,kwargs,keep", [row for row in PANELS if row[0] in POOLED_PANELS], ids=lambda p: getattr(p, "__name__", str(p)))
def test_pooled_panels_carry_the_trial_counts(frames, panel, kind, kwargs, keep):
    """A pooled statistic reports the trials behind each seed: every distance two and above."""
    df = _pinned(frames, kind, keep)
    fig, ax = plt.subplots()
    try:
        stats = panel(ax, df, **_keywords(frames, kwargs))
        assert stats["n_trials_per_seed"] == N_TRIALS * sum(1 for d in DISTANCES if d >= 2)
    finally:
        plt.close(fig)


# The panels that draw line series with markers, and so take `marker_size`.
MARKER_PANELS = {
    distance.make_accuracy_by_distance,
    distance.make_accuracy_by_reasoning_steps,
    distance.make_steps_to_criterion_by_distance,
    distance.make_pair_accuracy,
    distance.make_response_classes_by_reasoning_steps,
    learning.make_pooled_accuracy_vs_sweep,
    distribution.make_accuracy_vs_skew,
    content.make_context_accuracy_by_distance,
    content.make_context_accuracy_by_reasoning_steps,
    representation.make_probe_accuracy_by_layer,
    representation.make_edit_accuracy_by_distance,
    representation.make_write_by_strength,
}


def _series_marker_sizes(ax) -> set[float]:
    """The marker sizes of the circle-marked lines on an axes."""
    return {float(line.get_markersize()) for line in ax.lines if line.get_marker() == "o"}


@pytest.mark.parametrize("panel,kind,kwargs,keep", PANELS, ids=lambda p: getattr(p, "__name__", str(p)))
def test_marker_size_reaches_every_marked_line(frames, panel, kind, kwargs, keep):
    """A panel with marked lines takes `marker_size` (the shared size by default); no other does."""
    accepts = "marker_size" in inspect.signature(panel).parameters
    assert accepts == (panel in MARKER_PANELS)
    if not accepts:
        return
    assert inspect.signature(panel).parameters["marker_size"].default == _frames.MARKER_SIZE
    df = _pinned(frames, kind, keep)
    fig, ax = plt.subplots()
    try:
        panel(ax, df, **kwargs)
        assert _series_marker_sizes(ax) == {_frames.MARKER_SIZE}
        ax.clear()
        panel(ax, df, marker_size=5.5, **kwargs)
        assert _series_marker_sizes(ax) == {5.5}
    finally:
        plt.close(fig)


def test_bands_lie_under_the_chance_line_and_the_seed_dots():
    """The standard-error band is drawn at the lowest order; dots above it; the line on top."""
    assert (_frames.BAND_ZORDER, _frames.POINT_ZORDER, _frames.LINE_ZORDER) == (1, 2, 3)
    fig, ax = plt.subplots()
    try:
        _frames.draw_series(ax, [0, 1, 2], [0.5, 0.6, 0.7], [0.1, 0.1, 0.1], color="#000000", label="x")
        style.chance_line(ax)
        (band,) = ax.collections
        assert band.get_zorder() == 1
        assert ax.lines[0].get_zorder() == 3
        assert ax.lines[-1].get_zorder() == 1
        _frames.draw_series(ax, [0, 1, 2], [0.5, 0.6, 0.7], [0.1, 0.1, 0.1], color="#000000", label="y", zorder=5)
        assert ax.collections[-1].get_zorder() == 1
        assert ax.lines[-1].get_zorder() == 5
    finally:
        plt.close(fig)


def test_line_width_reaches_the_line():
    """`line_width` sets the plotted line's width in points; None keeps the shared width."""
    style.apply_style()
    fig, ax = plt.subplots()
    try:
        _frames.draw_series(ax, [0, 1], [0.5, 0.6], None, color="#000000", label="a")
        _frames.draw_series(ax, [0, 1], [0.5, 0.6], None, color="#000000", label="b", line_width=0.8)
        assert [line.get_linewidth() for line in ax.lines] == [style.LINE_WIDTH_PT, 0.8]
    finally:
        plt.close(fig)


def test_pair_panel_takes_its_line_and_error_bar_widths(frames):
    """Several budgets take `line_width`; one budget's bars and caps are `ERRORBAR_LW` wide."""
    assert (distance.ERRORBAR_LW, _frames.ERRORBAR_CAPSIZE) == (0.8, 1.5)
    df = _pinned(frames, "pairs", ())
    fig, ax = plt.subplots()
    try:
        distance.make_pair_accuracy(ax, df, condition="icl_test", k_values=(0, 8), line_width=0.8)
        assert {line.get_linewidth() for line in ax.lines if line.get_marker() == "o"} == {0.8}
        ax.clear()
        distance.make_pair_accuracy(ax, df, condition="iwl_test", k_values=(0,))
        bars = [c for c in ax.containers if isinstance(c, matplotlib.container.ErrorbarContainer)]
        assert bars
        for container in bars:
            _, caps, lines = container
            assert {cap.get_markeredgewidth() for cap in caps} == {0.8}
            assert {line.get_linewidth()[0] for line in lines} == {0.8}
    finally:
        plt.close(fig)


def test_seed_dots_take_their_marker_and_log_jitter():
    """Seed dots take the series' marker; on a log axis the jitter scales with the x value."""
    per_seed = pd.DataFrame({"x": [10.0, 10.0, 1000.0, 1000.0], "value": [0.5, 0.6, 0.7, 0.8]})
    fig, ax = plt.subplots()
    try:
        _frames.draw_seed_points(ax, per_seed, x_col="x", value_col="value", color="#000000", marker="s", jitter=0.05, log_x=True, rng=np.random.default_rng(0))
        small, large = ax.collections
        assert small.get_paths()[0].vertices.shape == large.get_paths()[0].vertices.shape == (5, 2)
        assert np.all(np.abs(small.get_offsets()[:, 0] - 10.0) <= 0.5 + 1e-9)
        assert np.all(np.abs(large.get_offsets()[:, 0] - 1000.0) <= 50.0 + 1e-9)
        assert np.any(np.abs(large.get_offsets()[:, 0] - 1000.0) > 0.5)
        ax.clear()
        _frames.draw_seed_points(ax, per_seed, x_col="x", value_col="value", color="#000000", jitter=0.05, rng=np.random.default_rng(0))
        small, large = ax.collections
        assert np.all(np.abs(large.get_offsets()[:, 0] - 1000.0) <= 0.05 + 1e-9)
        assert small.get_paths()[0].vertices.shape[0] > 5
    finally:
        plt.close(fig)


def test_sweep_seed_dots_follow_their_series_marker(frames):
    """On the sweep panel a series named in `line_markers` gives its marker to its seed dots."""
    df = _pinned(frames, "eval", ("p_cot",))
    fig, ax = plt.subplots()
    try:
        learning.make_pooled_accuracy_vs_sweep(ax, df, x_col="p_cot", conditions=("iwl_test",), log_x=True, jitter=0.05)
        dots = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
        assert dots and all(len(c.get_paths()[0].vertices) > 5 for c in dots)
        xs = np.concatenate([c.get_offsets()[:, 0] for c in dots])
        assert np.all((np.abs(xs - 0.5) <= 0.025 + 1e-9) | (np.abs(xs - 0.9) <= 0.045 + 1e-9))
        assert np.any(np.abs(xs - 0.9) > 0.025)
        ax.clear()
        df = _pinned(frames, "eval", ("p_cot", "skew_alpha"))
        learning.make_pooled_accuracy_vs_sweep(ax, df, x_col="p_cot", conditions=("iwl_test",), line_col="skew_alpha", line_markers={1.0: "s"})
        dots = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
        squares = [c for c in dots if c.get_paths()[0].vertices.shape == (5, 2)]
        assert squares and len(squares) == len(dots) // 2
    finally:
        plt.close(fig)


def test_sweep_draws_open_markers_for_the_named_values_only(frames):
    """A value named in `line_open_markers` draws open markers and seed dots; the others, filled."""
    df = _pinned(frames, "eval", ("p_cot", "skew_alpha"))
    fig, ax = plt.subplots()
    try:
        learning.make_pooled_accuracy_vs_sweep(ax, df, x_col="p_cot", conditions=("iwl_test",), line_col="skew_alpha", line_open_markers={1.0})
        lines = {line.get_label(): line for line in ax.lines if line.get_marker() == "o"}
        assert set(lines) == {"IWL test, 0", "IWL test, 1"}
        filled, hollow = lines["IWL test, 0"], lines["IWL test, 1"]
        assert matplotlib.colors.to_hex(filled.get_markerfacecolor()) == matplotlib.colors.to_hex(filled.get_color())
        assert matplotlib.colors.to_hex(hollow.get_markerfacecolor()) == style.OPEN_MARKER_FACE
        assert matplotlib.colors.to_hex(hollow.get_markeredgecolor()) == matplotlib.colors.to_hex(hollow.get_color())
        dots = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
        open_dots = [c for c in dots if len(c.get_facecolor()) == 0]
        assert open_dots and len(open_dots) == len(dots) // 2
    finally:
        plt.close(fig)


def _pooled_by_seed(rows: pd.DataFrame, condition: str, k: int, group: Sequence[str] = ("seed",)) -> pd.Series:
    """A measure pooled over distances two and above with trial weights, by hand."""
    picked = rows[(rows["metric"] == "accuracy") & (rows["condition"] == condition) & (rows["K"] == k) & (rows["d"] >= 2)]
    weighted = picked.assign(w=picked["value"] * picked["n"]).groupby(list(group))[["w", "n"]].sum()
    return weighted["w"] / weighted["n"]


def test_head_panels_draw_the_changes_the_top_heads_and_the_mean_over_draws(frames):
    """A point is a head's pooled accuracy minus the intact one and the marked points are each
    network's top-ranked heads; the bars read the top set and the mean over the random draws
    within each seed."""
    edit, ablate = _pinned(frames, "edit", ()), _pinned(frames, "ablate", ())
    heads = ablate[ablate["set_kind"] == "head"]
    intact = _pooled_by_seed(ablate[ablate["set_kind"] == "intact"], "icl_test", 8)
    by_head = _pooled_by_seed(heads, "icl_test", 8, group=("seed", "head"))
    fig, ax = plt.subplots()
    try:
        for size in HEAD_SET_SIZES:
            stats = representation.make_head_ablation_scatter(ax, ablate, k=size)
            assert stats["n_heads"] == len(by_head) and stats["k"] == size
            assert sorted(stats["context_heads"]) == list(SEEDS)
            for seed, chosen in stats["context_heads"].items():
                ranked = heads[(heads["seed"] == seed) & (heads["attention_rank"] <= size)].drop_duplicates("head").sort_values("attention_rank")
                assert [one["head"] for one in chosen] == list(ranked["head"])
                for one in chosen:
                    assert one["change_y"] == pytest.approx(by_head[(seed, one["head"])] - intact[seed], abs=1e-4)
            ax.clear()
            stats = representation.make_intervention_bars(ax, edit, ablate, k=size)
            bars = stats["measures"]["ICL test\n8 steps"]["bars"]
            sized = ablate[ablate["n_heads"] == size]
            drawn = _pooled_by_seed(sized[sized["set_kind"] == "random"], "icl_test", 8, group=("seed", "draw")).groupby(level="seed").mean()
            assert bars["Random heads ablated"]["per_seed"] == {int(seed): pytest.approx(value, abs=1e-4) for seed, value in drawn.items()}
            top = _pooled_by_seed(sized[sized["set_kind"] == "top"], "icl_test", 8)
            assert bars["Context-reading heads ablated"]["per_seed"] == {int(seed): pytest.approx(value, abs=1e-4) for seed, value in top.items()}
            assert bars["Intact"]["mean"] == pytest.approx(float(intact.mean()), abs=1e-4)
            ax.clear()
    finally:
        plt.close(fig)


def test_summary_bars_refuse_tables_that_disagree(frames):
    """The bars raise when the intact networks differ, a seed is missing from one table or one
    set, or the tables sit at different steps; the scatter raises unless every network ranks one
    head at each top place."""
    edit, ablate = _pinned(frames, "edit", ()), _pinned(frames, "ablate", ())
    intact = (ablate["set_kind"] == "intact") & (ablate["condition"] == "icl_test") & (ablate["K"] == 8) & (ablate["d"] == 3)
    shifted = ablate.assign(value=ablate["value"].where(~intact, ablate["value"] + 0.5))
    first = ablate["seed"] == SEEDS[0]
    no_top = ablate[~(first & (ablate["set_kind"] == "top"))]
    unranked = ablate.assign(attention_rank=ablate["attention_rank"].where(~(first & (ablate["attention_rank"] == 1)), len(HEAD_LABELS) + 1))
    doubled = ablate.assign(attention_rank=ablate["attention_rank"].where(~(first & (ablate["attention_rank"] == 2)), 1))
    cases = [
        (edit, shifted, "differ"),
        (edit[edit["seed"] != SEEDS[-1]], ablate, "only one"),
        (edit, no_top, "covers seeds"),
        (edit, ablate.assign(step=ablate["step"] + 1), "step"),
    ]
    fig, ax = plt.subplots()
    try:
        for edit_rows, ablate_rows, message in cases:
            with pytest.raises(ValueError, match=message):
                representation.make_intervention_bars(ax, edit_rows, ablate_rows)
            ax.clear()
        for rows in (unranked, doubled):
            with pytest.raises(ValueError, match="one head ranked at each place"):
                representation.make_head_ablation_scatter(ax, rows)
            ax.clear()
    finally:
        plt.close(fig)


def test_every_ablate_set_column_must_be_pinned_or_drawn():
    """The columns that tell one set of ablated heads from another are identity columns, so no
    panel averages over sets without naming them."""
    labels = set(tables.ABLATE_SCHEMA) - set(tables.EVAL_SCHEMA) - {"layer"}
    assert labels <= set(_frames.IDENTITY_COLUMNS)
    ablate = ablate_frame(_unedited(edit_frame("representation")), "components")
    with pytest.raises(ValueError, match="pin 'n_heads'"):
        _frames.refuse_unpinned(tables.pick(ablate, set_kind="top", seed=SEEDS[0]), schema=tables.ABLATE_SCHEMA, allowed=("heads",), fname="test")


def test_bar_seed_dots_sit_over_the_bars(frames):
    """Every seed dot of the bar panels and the control strip is drawn above the bars."""
    fig, (bars, write, strip, accuracy) = plt.subplots(1, 4)
    try:
        representation.make_intervention_bars(bars, _pinned(frames, "edit", ()), _pinned(frames, "ablate", ()))
        representation.make_write_by_strength(write, _pinned(frames, "edit", ()), control_ax=strip)
        distance.make_accuracy_bars(accuracy, _pinned(frames, "eval", ()))
        for ax in (bars, strip, accuracy):
            dots = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
            bar_order = max(patch.get_zorder() for patch in ax.patches)
            assert dots and all(c.get_zorder() > bar_order for c in dots)
    finally:
        plt.close(fig)


def test_edit_error_bars_have_caps_as_wide_as_the_axes_lines(frames):
    """Bar error bars on the summary and the control strip draw their caps at the axes width."""
    assert _frames.ERRORBAR_CAPSIZE == 1.5
    assert representation.CONTROL_BAR_WIDTH == 0.7
    assert inspect.signature(representation.make_intervention_bars).parameters["jitter"].default == _frames.JITTER
    df = _pinned(frames, "edit", ())
    fig, (bars, write, strip) = plt.subplots(1, 3)
    try:
        representation.make_intervention_bars(bars, df, _pinned(frames, "ablate", ()))
        representation.make_write_by_strength(write, df, control_ax=strip, jitter=0.03)
        for ax in (bars, strip):
            containers = [c for c in ax.containers if isinstance(c, matplotlib.container.BarContainer)]
            assert containers
            for container in containers:
                _, caps, lines = container.errorbar
                assert {cap.get_markeredgewidth() for cap in caps} == {style.AXES_LINEWIDTH_PT}
                assert {line.get_linewidth()[0] for line in lines} == {style.AXES_LINEWIDTH_PT}
        assert strip.get_xlim() == (-0.65, 1.65)
        dots = [c for c in strip.collections if isinstance(c, matplotlib.collections.PathCollection)]
        assert len(dots) == 2
        for index, points in enumerate(dots):
            assert np.all(np.abs(points.get_offsets()[:, 0] - index) <= 0.03 + 1e-9)
    finally:
        plt.close(fig)


STEP_PANELS = [
    (learning.make_accuracy_by_step, "eval", {"condition": "icl_test"}, ()),
    (learning.make_reasoning_gain_by_step, "eval", {"condition": "icl_test"}, ()),
    (distribution.make_emergence_by_skew, "eval", {}, ("skew_alpha",)),
    (learning.make_pair_type_by_step, "reasoning_rows", {}, REASONING_DRAWN),
]


@pytest.mark.parametrize("panel,kind,kwargs,keep", STEP_PANELS, ids=lambda p: getattr(p, "__name__", str(p)))
def test_step_panels_take_their_ticks_and_range(frames, panel, kind, kwargs, keep):
    """A step panel follows the drawn steps, widened to outside ticks, unless `xlim` pins it."""
    parameters = inspect.signature(panel).parameters
    assert parameters["xticks"].default is None
    pins_range = "xlim" in parameters
    assert pins_range == (panel is not distribution.make_emergence_by_skew)
    df = _pinned(frames, kind, keep)
    kwargs = _keywords(frames, kwargs)
    steps = sorted(df["step"].unique())
    fig, ax = plt.subplots()
    try:
        panel(ax, df, **kwargs)
        low, high = ax.get_xlim()
        assert low < steps[0] and high > steps[-1]
        assert high - low == pytest.approx((steps[-1] - steps[0]) * 1.1)
        ax.clear()
        panel(ax, df, xticks=(0, 1000, 2000, 5000), **kwargs)
        fig.canvas.draw()
        assert ax.get_xlim() == (0.0, 5000.0)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1000", "2000", "5000"]
        if pins_range:
            assert parameters["xlim"].default is None
            ax.clear()
            panel(ax, df, xticks=(0, 1000, 2000), xlim=(0, 2500), **kwargs)
            assert ax.get_xlim() == (0.0, 2500.0)
            assert [int(t) for t in ax.get_xticks()] == [0, 1000, 2000]
            assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1000", "2000"]
    finally:
        plt.close(fig)


def test_proportion_labels_keep_one_decimal():
    """The frequent-cue proportion is labeled 0.01, 0.5, ..., 1.0, never 1."""
    assert [content.proportion_label(v) for v in (0.01, 0.5, 1.0, 1)] == ["0.01", "0.5", "1.0", "1.0"]


def test_steps_to_criterion_follows_the_drawn_values(frames):
    """`ylim` left at None follows the reached budgets with the axes margins."""
    df = _pinned(frames, "eval", ())
    fig, ax = plt.subplots()
    try:
        distance.make_steps_to_criterion_by_distance(ax, df)
        fig.canvas.draw()
        low, high = ax.get_ylim()
        drawn = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in ax.lines if line.get_marker() == "o"])
        assert low <= np.nanmin(drawn) and np.nanmax(drawn) <= high
        ax.clear()
        distance.make_steps_to_criterion_by_distance(ax, df, ylim=(-0.3, 10.0))
        assert ax.get_ylim() == (-0.3, 10.0)
    finally:
        plt.close(fig)


def test_dissociation_means_take_their_size_and_edge(frames):
    """The per-skew means are larger black-edged markers; their size and edge are keywords."""
    df = _pinned(frames, "eval", ("skew_alpha",))
    fig, ax = plt.subplots()
    try:
        distribution.make_icl_iwl_dissociation(ax, df)
        seeds, means = ax.collections
        assert seeds.get_sizes().tolist() == [8.0]
        assert means.get_sizes().tolist() == [40.0]
        assert means.get_linewidths().tolist() == [0.8]
        assert matplotlib.colors.to_hex(means.get_edgecolors()[0]) == "#000000"
        ax.clear()
        distribution.make_icl_iwl_dissociation(ax, df, mean_size=12.0, mean_edge_width=0.3)
        _, means = ax.collections
        assert means.get_sizes().tolist() == [12.0]
        assert means.get_linewidths().tolist() == [0.3]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("k_pin", [None, 3])
def test_accuracy_bars_pool_distances_at_one_budget(frames, k_pin):
    """Each bar is the seed mean of the trial-weighted pool over distances two and above, at the
    pinned budget or, with none pinned, the largest the table holds."""
    df = _pinned(frames, "eval", ())
    fig, ax = plt.subplots()
    try:
        stats = distance.make_accuracy_bars(ax, df, k_pin=k_pin)
        rows = tables.pick(df, metric="accuracy", step=int(df["step"].max()))
        k = max(BUDGETS) if k_pin is None else k_pin
        assert stats["k_pin"] == k
        heights = [patch.get_height() for patch in ax.patches]
        for index, condition in enumerate(("iwl_test", "icl_test")):
            per_seed = _pooled_by_seed(rows, condition, k)
            summary = stats["per_condition"][condition]
            assert summary["mean"] == pytest.approx(per_seed.mean(), abs=1e-4)
            assert summary["se"] == pytest.approx(per_seed.std(ddof=1) / math.sqrt(len(SEEDS)), abs=1e-4)
            assert heights[index] == pytest.approx(per_seed.mean())
    finally:
        plt.close(fig)


def test_accuracy_bars_draw_one_thin_bar_per_condition_without_tick_labels(frames):
    """One bar per condition, in order and in its color, at the shared width, standing on the axis,
    with dark error bars and caps at the axes line width, no x ticks, and the given axis label."""
    df = _pinned(frames, "content", ())
    conditions = ("frequent_cue_congruent", "frequent_cue_reversed", "frequent_cue_shuffled")
    fig, ax = plt.subplots()
    try:
        distance.make_accuracy_bars(ax, df, conditions=conditions, show_seeds=False, ylabel="Accuracy (reasoning)")
        assert ax.get_ylim() == (0.0, 1.02)
        assert [patch.get_x() + patch.get_width() / 2 for patch in ax.patches] == pytest.approx([0.0, 1.0, 2.0])
        assert {round(patch.get_width(), 6) for patch in ax.patches} == {distance.BAR_WIDTH}
        expected = [matplotlib.colors.to_rgba(style.CONDITION_COLORS[c]) for c in conditions]
        assert [patch.get_facecolor() for patch in ax.patches] == expected
        assert list(ax.get_xticks()) == []
        assert ax.get_xlim() == (-0.5, 2.5)
        assert ax.get_ylabel() == "Accuracy (reasoning)"
        assert not [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
        for container in (c for c in ax.containers if isinstance(c, matplotlib.container.BarContainer)):
            _, caps, lines = container.errorbar
            assert {cap.get_markeredgewidth() for cap in caps} == {style.AXES_LINEWIDTH_PT}
            assert {line.get_linewidth()[0] for line in lines} == {style.AXES_LINEWIDTH_PT}
            assert matplotlib.colors.to_hex(lines[0].get_colors()[0]) == distance.ERROR_COLOR
    finally:
        plt.close(fig)


def test_accuracy_bars_draw_one_seed_without_an_error_bar(frames):
    """With one seed the standard error is undefined: reported as None and drawn as no error bar."""
    df = tables.pick(_pinned(frames, "eval", ()), seed=SEEDS[0])
    fig, ax = plt.subplots()
    try:
        stats = distance.make_accuracy_bars(ax, df)
        assert stats["n_seeds"] == 1
        assert {summary["se"] for summary in stats["per_condition"].values()} == {None}
        containers = [c for c in ax.containers if isinstance(c, matplotlib.container.BarContainer)]
        assert containers and all(container.errorbar is None for container in containers)
    finally:
        plt.close(fig)


def test_accuracy_numbers_write_the_bars_values_beside_their_lines(frames):
    """Each number is its condition's pooled accuracy at the budget, as the bar would show it,
    formatted to two decimals in the condition's color, starting at the end of its line."""
    df = _pinned(frames, "eval", ())
    fig, (bars_ax, ax) = plt.subplots(1, 2)
    try:
        bars = distance.make_accuracy_bars(bars_ax, df, k_pin=8)
        steps = distance.make_steps_to_criterion_by_distance(ax, df, show_seeds=False, show_legend=False)
        stats = distance.make_accuracy_numbers(ax, df, k_pin=8)
        texts = placement.marked_texts(ax)
        assert [t.get_text() for t in texts] == [f"{bars['per_condition'][c]['mean']:.2f}" for c in ("iwl_test", "icl_test")]
        for text, condition in zip(texts, ("iwl_test", "icl_test"), strict=True):
            summary, bar = stats["per_condition"][condition], bars["per_condition"][condition]
            assert (summary["mean"], summary["se"], summary["per_seed"]) == (bar["mean"], bar["se"], bar["per_seed"])
            assert matplotlib.colors.to_hex(text.get_color()) == style.CONDITION_COLORS[condition]
            by_distance = steps["per_condition"][condition]["mean_by_distance"]
            last = max(by_distance)
            assert text.placement_anchor == pytest.approx((float(last), by_distance[last]), abs=1e-4)
        assert stats["n_trials_per_seed"] == bars["n_trials_per_seed"]
    finally:
        plt.close(fig)


def _pair_type_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """The rows of each measure on the seven-item orderings of the synthetic tables: the ICL test
    at distances two and above, and the IWL test's pairs by type."""
    low = np.minimum(df["rank_a"], df["rank_b"])
    high = np.maximum(df["rank_a"], df["rank_b"])
    iwl = df["condition"] == "iwl_test"
    return {
        "icl_test": (df["condition"] == "icl_test") & (df["d"] >= 2),
        "trained": iwl & (high - low == 1),
        "end_item": iwl & (high - low >= 2) & ((low == 1) | (high == 7)),
        "interior": iwl & (high - low >= 2) & (low > 1) & (high < 7),
    }


def _expected_pair_type(rows: pd.DataFrame, measure: str) -> tuple[float, int]:
    """The bar recomputed by hand: within each seed, the measure's pairs pooled by trials at each
    distance and budget, the best budget within each distance, the distances pooled by trials;
    then the mean over seeds. Also the trials behind one seed."""
    sub = rows[_pair_type_masks(rows)[measure]].assign(weighted=lambda r: r["value"] * r["n"])
    by_budget = sub.groupby(["seed", "d", "K"])[["weighted", "n"]].sum()
    by_budget["value"] = by_budget["weighted"] / by_budget["n"]
    best = by_budget.reset_index().groupby(["seed", "d"]).agg(value=("value", "max"), n=("n", "max"))
    per_seed = (best["value"] * best["n"]).groupby(level="seed").sum() / best["n"].groupby(level="seed").sum()
    return float(per_seed.mean()), int(best["n"].groupby(level="seed").sum().max())


def _as_finetuned(pairs: pd.DataFrame, ablated: pd.DataFrame, condition: tuple[str, float]) -> pd.DataFrame:
    """One condition's rows from the table its runs are scored on."""
    heads, proportion = condition
    source = pairs if heads == "none" else ablated
    return tables.pick(source, ablated_heads=heads, reasoning_prop=proportion)


def test_pair_type_bars_recompute_every_measure(frames):
    """Every bar of every condition equals the hand recomputation from its own table, with pair
    accuracies and trial counts that differ from pair to pair."""
    pairs = _pinned(frames, "reasoning_pairs", ())
    ablated = _pinned(frames, "reasoning_ablated", ())
    masks = _pair_type_masks(pairs)
    unordered = {name: len({tuple(sorted(p)) for p in pairs.loc[mask & (pairs["condition"] == "iwl_test"), ["rank_a", "rank_b"]].itertuples(index=False)}) for name, mask in masks.items() if name != "icl_test"}
    assert unordered == {"trained": 6, "end_item": 9, "interior": 6}
    assert pairs["n"].nunique() > 1
    fig, ax = plt.subplots()
    try:
        stats = learning.make_pair_type_bars(ax, tables.finetuned_pair_rows(pairs, ablated))
    finally:
        plt.close(fig)
    assert stats["step"] == 1600 and stats["budgets"] == list(BUDGETS)
    for condition, label in zip(learning.REASONING_CONDITIONS, stats["conditions"], strict=True):
        rows = tables.pick(_as_finetuned(pairs, ablated, condition), step=1600)
        for measure in learning.PAIR_TYPES:
            expected, trials = _expected_pair_type(rows, measure)
            bar = stats["bars"][label][measure]
            assert bar["mean"] == pytest.approx(expected, abs=1e-4), (label, measure)
            assert bar["n_trials_per_seed"] == trials, (label, measure)


def test_pair_type_bars_give_missing_values_no_weight(frames):
    """A pair with a missing accuracy drops out of its pool instead of counting as zero."""
    pairs = _pinned(frames, "reasoning_pairs", ())
    ablated = _pinned(frames, "reasoning_ablated", ())
    holed = pairs.copy()
    drop = (holed["condition"] == "iwl_test") & (holed["rank_a"] == 3) & (holed["rank_b"] == 5)
    holed.loc[drop, "value"] = np.nan
    fig, ax = plt.subplots()
    try:
        stats = learning.make_pair_type_bars(ax, tables.finetuned_pair_rows(holed, ablated))
    finally:
        plt.close(fig)
    rows = tables.pick(holed[~drop], ablated_heads="none", reasoning_prop=1.0, step=1600)
    assert stats["bars"]["Reasoning"]["interior"]["mean"] == pytest.approx(_expected_pair_type(rows, "interior")[0], abs=1e-4)


def test_pair_type_by_step_recomputes_every_step(frames):
    """Each line is the bars' estimand at every evaluation step."""
    pairs = _pinned(frames, "reasoning_pairs", ())
    ablated = _pinned(frames, "reasoning_ablated", ())
    finetuned = tables.finetuned_pair_rows(pairs, ablated)
    fig, ax = plt.subplots()
    try:
        for measure in learning.PAIR_TYPES:
            stats = learning.make_pair_type_by_step(ax, finetuned, measure=measure)
            for condition, label in zip(learning.REASONING_CONDITIONS, stats["conditions"], strict=True):
                rows = _as_finetuned(pairs, ablated, condition)
                for step, value in stats["lines"][label]["mean_by_step"].items():
                    assert value == pytest.approx(_expected_pair_type(tables.pick(rows, step=step), measure)[0], abs=1e-4)
            ax.clear()
    finally:
        plt.close(fig)


def test_pair_type_panels_score_each_run_as_it_was_finetuned(frames):
    """A run that ablated heads while it finetuned is read from the ablated table, every other run
    from the intact one."""
    pairs = _pinned(frames, "reasoning_pairs", ()).assign(value=0.25)
    ablated = _pinned(frames, "reasoning_ablated", ()).assign(value=0.75)
    finetuned = tables.finetuned_pair_rows(pairs, ablated)
    fig, ax = plt.subplots()
    try:
        bars = learning.make_pair_type_bars(ax, finetuned)
        ax.clear()
        lines = learning.make_pair_type_by_step(ax, finetuned)
        for (heads, _), label in zip(learning.REASONING_CONDITIONS, bars["conditions"], strict=True):
            expected = 0.25 if heads == "none" else 0.75
            assert all(bar["mean"] == pytest.approx(expected) for bar in bars["bars"][label].values())
            assert all(value == pytest.approx(expected) for value in lines["lines"][label]["mean_by_step"].values())
    finally:
        plt.close(fig)


def test_pair_type_panels_refuse_tables_they_cannot_score(frames):
    """The intact table in place of the ablated one or a run scored under two head sets raise when
    the rows are built; per-pair rows not built that way, or a condition missing a network, raise
    in both panels, each naming the problem."""
    pairs = _pinned(frames, "reasoning_pairs", ())
    ablated = _pinned(frames, "reasoning_ablated", ())
    two_sets = ablated.assign(heads=np.where(ablated["K"] % 2 == 0, "L1H1+L1H2", "L1H3+L1H4"))
    with pytest.raises(ValueError, match="name no ablated heads"):
        tables.finetuned_pair_rows(pairs, pairs)
    with pytest.raises(ValueError, match="more than one head set"):
        tables.finetuned_pair_rows(pairs, two_sets)
    short = tables.finetuned_pair_rows(pairs, ablated[~((ablated["ablated_heads"] == "random") & (ablated["seed"] == SEEDS[-1]))])
    gap = tables.finetuned_pair_rows(pairs, ablated[~((ablated["ablated_heads"] == "random") & (ablated["seed"] == SEEDS[-1]) & (ablated["step"] == 800))])
    fig, ax = plt.subplots()
    try:
        for panel in (learning.make_pair_type_bars, learning.make_pair_type_by_step):
            with pytest.raises(ValueError, match="pass the rows of finetuned_pair_rows"):
                panel(ax, pairs)
            with pytest.raises(ValueError, match="different seeds"):
                panel(ax, short)
        with pytest.raises(ValueError, match="different seeds"):
            learning.make_pair_type_by_step(ax, gap)
    finally:
        plt.close(fig)


def test_pair_type_panels_label_the_budget_and_place_the_key_above(frames):
    """The accuracy axis names the reasoning-step estimand, one step in the singular, and the
    lines' step axis names finetuning; the bars' key sits above the panel."""
    rows = _pinned(frames, "reasoning_rows", ())
    fig, ax = plt.subplots()
    try:
        learning.make_pair_type_bars(ax, rows)
        assert ax.get_ylabel() == "Accuracy (best of 0–8 steps)"
        legend = ax.get_legend()
        fig.canvas.draw()
        assert legend.get_window_extent().y0 >= ax.get_window_extent().y1 - 1.0
        ax.clear()
        learning.make_pair_type_by_step(ax, rows, k_stat="pin", k_pin=1)
        assert ax.get_ylabel() == "Accuracy (1 step)"
        assert ax.get_xlabel() == "Finetuning step"
        ax.clear()
        learning.make_pair_type_by_step(ax, rows, k_stat="pin", k_pin=8)
        assert ax.get_ylabel() == "Accuracy (8 steps)"
    finally:
        plt.close(fig)


def test_pair_type_bars_can_break_the_axis_after_the_icl_test(frames):
    """With `split_axis` the x axis and the chance line run in two pieces, one under the ICL test
    and one under the IWL test's measures; by default each is one line across the panel. The
    axis pieces stay at the bottom of the axes when the y limits change afterwards."""
    rows = _pinned(frames, "reasoning_rows", ())
    fig, (whole, split) = plt.subplots(1, 2)
    try:
        learning.make_pair_type_bars(whole, rows)
        learning.make_pair_type_bars(split, rows, split_axis=True)
        assert whole.spines["bottom"].get_visible() and not split.spines["bottom"].get_visible()
        centers = split.get_xticks()
        pieces = [tuple(line.get_xdata()) for line in split.lines if line.get_linestyle() == "-" and len(line.get_xdata()) == 2]
        wanted = [(centers[0] - 0.5, centers[0] + 0.5), (centers[1] - 0.5, centers[-1] + 0.5)]
        assert all(piece in pieces for piece in wanted)
        chance = [tuple(line.get_xdata()) for line in split.lines if line.get_linestyle() == "--"]
        assert len(chance) == 2
        split.set_ylim(0.3, 1.0)
        fig.canvas.draw()
        bottom = split.get_window_extent().y0
        for line in (line for line in split.lines if line.get_linestyle() == "-" and tuple(line.get_xdata()) in wanted):
            heights = line.get_transform().transform(line.get_xydata())[:, 1]
            assert heights == pytest.approx([bottom, bottom])
    finally:
        plt.close(fig)


def test_pair_type_panels_refuse_two_conditions_under_one_label(frames):
    """Two conditions given one label would merge their statistics, so both panels raise."""
    rows = _pinned(frames, "reasoning_rows", ())
    shared = {("context_reading", 1.0): "Ablated", ("random", 1.0): "Ablated"}
    fig, ax = plt.subplots()
    try:
        for panel in (learning.make_pair_type_bars, learning.make_pair_type_by_step):
            with pytest.raises(ValueError, match="label of its own"):
                panel(ax, rows, labels=shared)
    finally:
        plt.close(fig)


def test_accuracy_numbers_skip_only_a_condition_unreached_everywhere(frames):
    """A condition the steps panel marks unreached at every distance gets no number; one with no
    line of two or more points for any other reason raises unless it is given an anchor."""
    df = _pinned(frames, "eval", ())
    icl, iwl = style.CONDITION_COLORS["icl_test"], style.CONDITION_COLORS["iwl_test"]
    fig, (ax, other) = plt.subplots(1, 2)
    try:
        unreached = df.copy()
        unreached.loc[unreached["condition"] == "iwl_test", "value"] = tables.CHANCE_ACCURACY
        distance.make_steps_to_criterion_by_distance(ax, unreached, unreached_below_chance=True, show_seeds=False, show_legend=False)
        stats = distance.make_accuracy_numbers(ax, unreached)
        keys = [text.get_gid().removeprefix(placement.GID_PREFIX) for text in placement.marked_texts(ax)]
        assert keys == ["icl_test"] and stats["per_condition"]["iwl_test"]["text"] is None
        other.plot([4.0], [0.8], color=icl, marker="o")
        with pytest.raises(ValueError, match="no line of two or more points"):
            distance.make_accuracy_numbers(other, df)
        stats = distance.make_accuracy_numbers(other, df, anchors={"iwl_test": (2.0, 0.5), "icl_test": (4.0, 0.5)})
        assert all(summary["text"] for summary in stats["per_condition"].values())
        assert distance.line_end(other, icl) == (4.0, 0.8)
        assert distance.line_end(other, icl, min_points=2) is None and distance.line_end(other, iwl) is None
    finally:
        plt.close(fig)


def test_accuracy_numbers_refuse_a_missing_estimand(frames):
    """A condition whose accuracy is missing at every distance has no number to write."""
    df = _pinned(frames, "eval", ())
    df.loc[df["condition"] == "iwl_test", "value"] = np.nan
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="no accuracy at budget"):
            distance.make_accuracy_numbers(ax, df, anchors={"iwl_test": (2.0, 0.5), "icl_test": (4.0, 0.5)})
    finally:
        plt.close(fig)


def _crowded_axes():
    """Two marked lines whose bands overlap along their whole length, and a number at the right
    end of each."""
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    x = np.arange(1, 7, dtype=float)
    for y, color in ((x * 0.8, "#1f77b4"), (x * 0.8 + 0.3, "#d62728")):
        ax.fill_between(x, y - 0.4, y + 0.4, color=color, alpha=0.25, lw=0)
        ax.plot(x, y, color=color, marker="o", markersize=3.0)
        text = ax.text(x[-1], y[-1], "0.90", color=color, fontsize=6.0, ha="left", va="bottom")
        placement.mark(text, key=color, anchor=(x[-1], y[-1]), own_color=color)
    ax.set_xlim(0.7, 6.3)
    ax.set_ylim(-0.5, 6.0)
    return fig, ax


def test_placed_texts_clear_everything_drawn_and_sit_nearest_their_own_lines():
    """After placement no number overlaps a line, a marker, another line's band or another number,
    each stays inside the axes and nearer its own line than the other, and placing again from the
    anchors gives the same positions."""
    fig, ax = _crowded_axes()
    try:
        first = placement.place_texts(ax)
        audit = placement.audit_texts(ax)
        assert placement.problems(audit) == []
        assert all(gap >= placement.CLEAR_PT - 0.01 for gap in audit["clearance_pt"].values())
        assert all(abs(dx) <= placement.SEARCH_PT and abs(dy) <= placement.SEARCH_PT for dx, dy in (r["offset_pt"] for r in first))
        assert placement.place_texts(ax) == first
        assert placement.place_texts(ax, check=True) == first
    finally:
        plt.close(fig)


def test_a_placed_text_may_sit_on_its_own_band_but_not_on_another():
    """A text's own line's band is neither an area to avoid nor an edge to clear; another
    line's band is both."""
    fig, ax = _crowded_axes()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        blue = placement.marked_texts(ax)[0]
        own_points, own_areas, _ = placement._scene(ax, renderer, {blue}, blue.placement_own_color)
        all_points, all_areas, _ = placement._scene(ax, renderer, {blue})
        blue_band, red_band = (c.get_transform().transform(c.get_paths()[0].vertices) for c in ax.collections)
        assert len(own_areas) == len(all_areas) - 1
        assert np.allclose(own_areas[0].vertices, red_band)
        outline: list = []
        placement._sample(np.vstack([blue_band, blue_band[:1]]), 0.0, outline)
        assert len(all_points) - len(own_points) == len(outline)
    finally:
        plt.close(fig)


def test_a_text_may_sit_where_its_line_coincides_with_another():
    """Where two marked lines coincide, a text by the shared stretch is its own line's, so it is
    placed beside it rather than pushed away."""
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    try:
        for color in ("#ff7f0e", "#8c564b"):
            ax.plot([1.0, 3.0, 5.0], [1.0, 3.0, 5.0], color=color, marker="o", markersize=3.0)
        ax.plot([5.0, 6.0], [5.0, 5.6], color="#8c564b", marker="o", markersize=3.0)
        text = ax.text(5.0, 5.0, "0.97", color="#ff7f0e", fontsize=6.0)
        placement.mark(text, key="orange", anchor=(5.0, 5.0), own_color="#ff7f0e")
        ax.set_xlim(0.0, 7.0)
        ax.set_ylim(0.0, 6.5)
        (record,) = placement.place_texts(ax)
        assert placement.problems(placement.audit_texts(ax)) == []
        assert max(abs(v) for v in record["offset_pt"]) <= 15
    finally:
        plt.close(fig)


def test_a_one_point_line_still_owns_its_text():
    """A line kept at a single point still counts as its text's own line."""
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    try:
        ax.plot([2.0], [2.0], color="#1f77b4", marker="o", markersize=3.0)
        ax.plot([1.0, 6.0], [5.0, 5.0], color="#d62728", marker="o", markersize=3.0)
        text = ax.text(2.0, 2.0, "0.90", color="#1f77b4", fontsize=6.0, ha="left", va="bottom")
        placement.mark(text, key="blue", anchor=(2.0, 2.0), own_color="#1f77b4")
        ax.set_xlim(0.0, 7.0)
        ax.set_ylim(0.0, 6.0)
        placement.place_texts(ax)
        assert placement.audit_texts(ax)["misowned"] == []
    finally:
        plt.close(fig)


def test_a_text_with_no_clear_place_takes_the_least_crowded_and_is_reported():
    """Among lines too close together for any position to clear, a text takes the position
    within the search distance with the most clearance, and the audit names it."""
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    try:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        fig.canvas.draw()
        height_pt = ax.get_window_extent().height * 72.0 / fig.dpi
        for y in np.arange(0.0, 1.0001, 8.0 / height_pt):
            ax.axhline(y, color="#7f7f7f", lw=0.6)
        text = ax.text(0.4, 0.5, "0.90", fontsize=6.0, ha="left", va="center")
        placement.mark(text, key="number", anchor=(0.4, 0.5))
        with pytest.raises(ValueError, match="crowded: number"):
            placement.place_texts(ax, search_pt=3.0, check=True)
        (record,) = placement.place_texts(ax, search_pt=3.0)
        audit = placement.audit_texts(ax)
        assert audit["crowded"] == ["number"]
        clearance = {}
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x0, y0 = ax.transData.transform(text.placement_anchor)
                k = fig.dpi / 72.0
                text.set_position(tuple(ax.transData.inverted().transform((x0 + dx * k, y0 + dy * k))))
                clearance[(dx, dy)] = placement.audit_texts(ax)["clearance_pt"]["number"]
        assert clearance[tuple(record["offset_pt"])] == pytest.approx(max(clearance.values()))
    finally:
        plt.close(fig)


def test_numbers_are_placed_once_the_layout_is_final(frames):
    """The figure scripts' helper lays the figure out as saved, switches the layout engine off,
    and places the numbers clear of everything drawn."""
    shared = _shared()
    fig, axes = shared.figure(1, 2, width_in=3.5, height_in=1.8)
    try:
        df = _pinned(frames, "eval", ())
        panel = axes[0, 1]
        distance.make_steps_to_criterion_by_distance(panel, df, show_seeds=False, show_legend=False)
        distance.make_accuracy_numbers(panel, df, **shared.ACCURACY_NUMBERS)
        (records,) = shared.place_numbers(fig, panel)
        assert isinstance(fig.get_layout_engine(), matplotlib.layout_engine.PlaceHolderLayoutEngine)
        assert [r["key"] for r in records] == ["iwl_test", "icl_test"]
        assert placement.problems(placement.audit_texts(panel)) == []
        boxes = [ax.get_position().bounds for ax in fig.axes]
        spots = [t.get_position() for t in placement.marked_texts(panel)]
        with matplotlib.rc_context(shared.SAVE_RC):
            fig.savefig(io.BytesIO(), format="pdf")
        assert [ax.get_position().bounds for ax in fig.axes] == boxes
        assert [t.get_position() for t in placement.marked_texts(panel)] == spots
    finally:
        plt.close(fig)


def test_numbers_that_cannot_be_placed_clear_stop_the_figure():
    """A number that no position clears raises instead of being drawn on top of the lines."""
    shared = _shared()
    fig, axes = shared.figure(1, 1, width_in=1.5, height_in=1.5)
    try:
        ax = axes[0, 0]
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        for y in np.linspace(0.0, 1.0, 40):
            ax.axhline(y, color="#7f7f7f", lw=0.6)
        text = ax.text(0.4, 0.5, "0.90", fontsize=6.0, ha="left", va="center")
        placement.mark(text, key="number", anchor=(0.4, 0.5))
        with pytest.raises(ValueError, match="crowded: number"):
            shared.place_numbers(fig, ax)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("panel,kind", [(distance.make_accuracy_by_distance, "eval"), (content.make_context_accuracy_by_distance, "content")])
def test_accuracy_panels_take_fixed_ticks(frames, panel, kind):
    """The accuracy panels pick their ticks automatically unless `yticks` names them."""
    df = _pinned(frames, kind, ())
    fig, ax = plt.subplots()
    try:
        assert inspect.signature(panel).parameters["yticks"].default is None
        panel(ax, df, yticks=(0.0, 0.5, 1.0))
        assert list(ax.get_yticks()) == [0.0, 0.5, 1.0]
    finally:
        plt.close(fig)


def test_context_accuracy_takes_an_axis_label(frames):
    """The context accuracy panel labels its axis "Accuracy" unless given another label."""
    df = _pinned(frames, "content", ())
    fig, ax = plt.subplots()
    try:
        content.make_context_accuracy_by_distance(ax, df)
        assert ax.get_ylabel() == "Accuracy"
        ax.clear()
        content.make_context_accuracy_by_distance(ax, df, ylabel="Accuracy (no reasoning)")
        assert ax.get_ylabel() == "Accuracy (no reasoning)"
    finally:
        plt.close(fig)


def test_accuracy_by_distance_applies_the_paper_estimands(frames):
    """The IWL test is pinned at zero steps and the ICL test averaged over one to five steps."""
    df = _pinned(frames, "eval", ())
    fig, ax = plt.subplots()
    try:
        stats = distance.make_accuracy_by_distance(ax, df)
        rows = tables.pick(df, metric="accuracy", step=int(df["step"].max()))
        iwl = tables.pick(rows, condition="iwl_test", K=0).groupby("d")["value"].mean()
        icl = tables.pick(rows, condition="icl_test", K=[1, 2, 3, 4, 5]).groupby(["d", "seed"])["value"].mean().groupby("d").mean()
        assert stats["per_condition"]["iwl_test"]["k_values"] == [0]
        assert stats["per_condition"]["icl_test"]["k_values"] == [1, 2, 3, 4, 5]
        for d in DISTANCES:
            assert stats["per_condition"]["iwl_test"]["mean_by_distance"][d] == pytest.approx(iwl[d], abs=1e-4)
            assert stats["per_condition"]["icl_test"]["mean_by_distance"][d] == pytest.approx(icl[d], abs=1e-4)
    finally:
        plt.close(fig)


def test_steps_to_criterion_averages_every_seed(frames):
    """The criterion is relative to each seed's maximum, so every seed reaches it."""
    df = _pinned(frames, "eval", ())
    fig, ax = plt.subplots()
    try:
        stats = distance.make_steps_to_criterion_by_distance(ax, df)
        assert stats["n_seeds"] == len(SEEDS)
        assert set(stats["per_condition"]["icl_test"]["mean_by_distance"]) == set(DISTANCES)
    finally:
        plt.close(fig)


def test_style_constants_are_fixed():
    """IWL red, ICL blue, K=0 gray, K=max orange, 7/6/9 pt fonts."""
    assert (style.IWL, style.ICL) == ("#d62728", "#1f77b4")
    assert (style.NO_STEPS, style.MAX_STEPS) == ("#7f7f7f", "#ff7f0e")
    assert (style.FONT_BODY_PT, style.FONT_TICK_PT, style.FONT_PANEL_LETTER_PT) == (7.0, 6.0, 9.0)
    assert (style.COLUMN_WIDTH_IN, style.FULL_WIDTH_IN, style.MAX_HEIGHT_IN) == (3.5, 7.0, 8.75)
    assert style.CONTEXT_COLORS["reversed"] == "#ff7f0e" and style.CONTEXT_COLORS["shuffled"] == "#875a5a"
    assert style.CONTEXT_COLORS["uninformative"] == style.IWL and style.CONTEXT_COLORS["congruent"] == "#3fb780"
    assert style.CONDITION_COLORS["baseline_test"] == style.BASELINE == "#2ca02c"
    assert style.CONTEXT_DISTANCE_CMAPS["congruent"] == style.CONGRUENT_DISTANCE_CMAP
    assert to_hex(colormaps[style.CONGRUENT_DISTANCE_CMAP].resampled(3)(1)) == style.CONGRUENT
    halves = (style.CONTEXT_COLORS["shuffled_agree"], style.CONTEXT_COLORS["shuffled_conflict"])
    assert halves == ("#eac2c7", "#54332d")
    others = {style.CONTEXT_COLORS[name] for name in ("uninformative", "congruent", "reversed", "shuffled", "icl_test")}
    assert len(set(halves)) == 2 and not set(halves) & others
    assert [style.CONTEXT_LABELS[name] for name in content.SHUFFLED_SPLIT_CONTEXTS] == ["Shuffled (agree)", "Shuffled (conflict)"]
    assert content.CONTEXTS == ("uninformative", "congruent", "reversed", "shuffled")


def test_apply_style_sets_the_print_parameters():
    """Editable text, the point sizes, constrained layout, and never a tight bounding box."""
    style.apply_style()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["font.size"] == 7.0
    assert matplotlib.rcParams["xtick.labelsize"] == 6.0
    assert matplotlib.rcParams["figure.constrained_layout.use"] is True
    assert matplotlib.rcParams["savefig.bbox"] is None or matplotlib.rcParams["savefig.bbox"] == "standard"


def test_gradients_come_from_the_named_colormaps():
    """Every line of a family is a distinct color, and single-value families stay dark."""
    shades = style.steps_colors(range(9))
    assert len(set(shades)) == 9
    assert style.distance_colors([3]) == style.distance_colors([3], style.ICL_DISTANCE_CMAP)
    assert style.sweep_colors([0.0, 5.0], style.SKEW_CMAP, by_value=True) == style.sweep_colors([0.0, 2.5, 5.0], style.SKEW_CMAP, by_value=True)[::2]
    assert style.sweep_colors([2.0], style.SKEW_CMAP, span=(0.0, 5.0)) == style.sweep_colors([0.0, 2.0, 5.0], style.SKEW_CMAP, by_value=True)[1:2]
    assert style.sweep_colors([9.0], style.SKEW_CMAP)[0] == matplotlib.colors.to_hex(matplotlib.colormaps[style.SKEW_CMAP](0.85))
    assert style.sweep_colors([3, 3], style.SKEW_CMAP) == style.sweep_colors([3], style.SKEW_CMAP) * 2
    assert style.site_labels([0, 1, 2, 3, 4, 5]) == ["Embed.", "1", "2", "3", "4", "Output"]
    with pytest.raises(ValueError):
        style.sweep_colors([1.0], style.SKEW_CMAP, span=(5.0, 0.0))


def test_each_swept_column_samples_its_own_slice_of_its_colormap():
    """A column's lines and colorbar share its slice; any other gradient keeps the default."""
    assert set(style.COLUMN_CMAP_SPANS) == set(style.COLUMN_CMAPS)
    assert dict(style.COLUMN_CMAP_SPANS) == {"K": (0.0, 0.92), "skew_alpha": (0.0, 0.85), "p_cot": (0.0, 0.9), "replay_prop": (0.45, 0.95), "learning_rate": (0.05, 0.95), "frequent_cue_prop": (0.28, 1.0)}
    for column, (low, high) in style.COLUMN_CMAP_SPANS.items():
        cmap = style.COLUMN_CMAPS[column]
        assert style.column_span(column, cmap) == (low, high)
        palette = matplotlib.colormaps[cmap]
        assert style.sweep_colors([1.0], cmap, column=column) == [matplotlib.colors.to_hex(palette(high))]
        assert style.sweep_colors([0.0, 1.0], cmap, column=column)[0] == matplotlib.colors.to_hex(palette(low))
        bar = style.sweep_cmap(cmap, column=column)
        assert matplotlib.colors.to_hex(bar(0.0)) == matplotlib.colors.to_hex(palette(low))
        assert matplotlib.colors.to_hex(bar(1.0)) == matplotlib.colors.to_hex(palette(high))
    assert style.column_span("K", style.STEPS_CMAP) == (0.0, 0.92)
    assert style.column_span("K", style.ICL_DISTANCE_CMAP) == (0.35, 1.0)
    assert style.column_span("d", style.ICL_DISTANCE_CMAP) == (0.35, 1.0)
    assert style.column_span(None, style.SKEW_CMAP) == (0.0, 0.85)
    assert style.steps_colors([8])[0] == matplotlib.colors.to_hex(matplotlib.colormaps[style.STEPS_CMAP](0.92))
    assert style.distance_colors([1, 6]) == style.sweep_colors([1, 6], style.ICL_DISTANCE_CMAP)


def _shared():
    """The figure scripts' shared module, loaded the way the `figure` command loads it."""
    return cli._load_figure_module("_shared")


def test_figure_sets_the_layout_pads():
    """A figure takes the shared pads and gaps unless the script passes its own."""
    shared = _shared()
    fig, _ = shared.figure(1, 2, width_in=3.5, height_in=2.0)
    try:
        engine = fig.get_layout_engine().get()
        assert (shared.W_PAD_IN, shared.H_PAD_IN, shared.WSPACE, shared.HSPACE) == (0.042, 0.042, 0.06, 0.06)
        assert (engine["w_pad"], engine["h_pad"], engine["wspace"], engine["hspace"]) == (0.042, 0.042, 0.06, 0.06)
    finally:
        plt.close(fig)
    fig, _ = shared.figure(2, 2, width_in=3.5, height_in=2.0, w_pad=0.03, h_pad=0.01, wspace=0.08, hspace=0.14)
    try:
        engine = fig.get_layout_engine().get()
        assert (engine["w_pad"], engine["h_pad"], engine["wspace"], engine["hspace"]) == (0.03, 0.01, 0.08, 0.14)
    finally:
        plt.close(fig)


def test_save_keeps_the_physical_size(tmp_path):
    """The PDF page is exactly the requested inches, and a PNG preview comes along."""
    shared = _shared()
    fig, axes = shared.figure(1, 2, width_in=3.5, height_in=2.0)
    axes[0, 0].plot([0, 1], [0, 1])
    paths = shared.save(fig, "probe", out_dir=tmp_path)
    assert [path.name for path in paths] == ["probe.pdf", "probe.png"]
    box = re.search(rb"MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", (tmp_path / "probe.pdf").read_bytes())
    assert box is not None
    assert (float(box.group(1)), float(box.group(2))) == (3.5 * 72, 2.0 * 72)
    assert (tmp_path / "probe.png").stat().st_size > 0
    assert not plt.get_fignums()


def _embedded_font_types(pdf: Path) -> set[bytes]:
    """The font subtypes a PDF declares: Type0 for TrueType text, Type3 for outlines."""
    data = pdf.read_bytes()
    return {kind for kind in (b"/Type0", b"/Type1", b"/Type3") if kind in data}


def test_save_embeds_truetype_text_whatever_the_figure_was_drawn_under(tmp_path):
    """A figure drawn under Type 3 parameters still saves with TrueType text through `save`."""
    shared = _shared()
    with matplotlib.rc_context({"pdf.fonttype": 3, "savefig.bbox": "tight"}):
        fig, ax = plt.subplots(figsize=(2.0, 1.5))
        ax.plot([0, 1], [0, 1], label="one")
        ax.set_xlabel("Symbolic distance")
        ax.legend()
    paths = shared.save(fig, "outside", out_dir=tmp_path)
    kinds = _embedded_font_types(paths[0])
    assert b"/Type3" not in kinds and b"/Type0" in kinds
    box = re.search(rb"MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", paths[0].read_bytes())
    assert (float(box.group(1)), float(box.group(2))) == (2.0 * 72, 1.5 * 72)
    fig, axes = shared.figure(1, 1, width_in=2.0, height_in=1.5)
    axes[0, 0].plot([0, 1], [0, 1])
    kinds = _embedded_font_types(shared.save(fig, "inside", out_dir=tmp_path)[0])
    assert b"/Type3" not in kinds and b"/Type0" in kinds


def test_formalization_strip_saves_with_truetype_text(tmp_path):
    """The predictions strip, built under its own parameters, ships TrueType text through `save`."""
    shared = _shared()
    script = cli._load_figure_module("fig_formalization")
    fig = script.build(cli.argparse.Namespace(results=tmp_path, out=tmp_path / "singles"))
    paths = shared.save(fig, "formalization", out_dir=tmp_path)
    kinds = _embedded_font_types(paths[0])
    assert b"/Type3" not in kinds and b"/Type0" in kinds
    for single in sorted((tmp_path / "singles").glob("formalization_*.pdf")):
        assert b"/Type3" not in _embedded_font_types(single)


def _preset_of(experiment: config.Experiment, stage: streams.Stage) -> str:
    """The battery preset an experiment's checkpoints of `stage` are scored on."""
    cfg = config.run_for_index(experiment, 0, config.load_base()).config
    return cfg.evaluation.battery or streams.stage_section(cfg, stage).battery


def _write_tables(root: Path, name: str) -> None:
    """Synthetic tables for one experiment under `<root>/<name>/tables/`, from its declaration."""
    experiment = config.load_experiment(name)
    swept = {axis.name: axis.values for axis in experiment.axes}
    tag = "finetuning" if "finetune" in experiment.stages else "pretraining"
    steps = (200, 1600) if tag == "finetuning" else (50000, 100000)
    out = root / name / "tables"
    if "ablate" in experiment.stages:
        # The intact rows repeat the edit table's unedited rows at the networks' own proportion.
        source = config.load_experiment("representation")
        proportion = {"frequent_cue_prop": config.run_for_index(experiment, 0, config.load_base()).config.task.cue_distribution.frequent_cue_prop}
        edit = edit_frame(source.name, swept={axis.name: axis.values for axis in source.axes}, steps=steps[-1:])
        intact = tables.pick(_unedited(edit), **proportion).drop(columns=list(proportion))
        tables.write_table(ablate_frame(intact, name), out / "ablate.parquet", schema=tables.ABLATE_SCHEMA)
        return
    if "represent" in experiment.stages:
        tables.write_table(probe_frame(name, swept=swept, steps=steps[-1:]), out / "probe.parquet", schema=tables.PROBE_SCHEMA, swept_columns=tuple(swept))
        tables.write_table(edit_frame(name, swept=swept, steps=steps[-1:]), out / "edit.parquet", schema=tables.EDIT_SCHEMA, swept_columns=tuple(swept))
        return
    preset = _preset_of(experiment, streams.Stage(tag))
    assert set(c.name for c in battery.load_preset(preset, contexts_per_pair=1)) == {c[0] for c in PRESETS[preset]}
    tables.write_table(eval_frame(name, preset, swept=swept, steps=steps, tag=tag), out / "eval.parquet", schema=tables.EVAL_SCHEMA, swept_columns=tuple(swept))
    if config.run_for_index(experiment, 0, config.load_base()).config.evaluation.ablated_scoring:
        # Runs scored with heads ablated keep their per-pair rows at every step and budget.
        pairs = eval_frame(name, preset, swept=swept, steps=steps, tag=tag, pairs=True)
        tables.write_table(ablated_pairs_frame(pairs, swept=swept), out / "ablated_eval_pairs.parquet", schema=tables.ABLATED_PAIR_SCHEMA, swept_columns=tuple(swept))
    else:
        pairs = eval_frame(name, preset, swept=swept, steps=steps[-1:], tag=tag, pairs=True, budgets=(0, 8))
    tables.write_table(pairs, out / "eval_pairs.parquet", schema=tables.PAIR_SCHEMA, swept_columns=tuple(swept))
    tables.write_table(train_frame(name, swept=swept, stage=tag), out / "train.parquet", schema=tables.TRAIN_SCHEMA, swept_columns=tuple(swept))


FIGURE_EXPERIMENTS = {
    "symbolic_distance": ("symbolic_distance",),
    "basic_tests": ("symbolic_distance",),
    "content_effects": ("content_effects",),
    "content_grid": ("content_effects",),
    "content_grid_no_cue": ("content_effects_no_cue",),
    "mechanisms": ("representation", "components", "finetuning_reasoning"),
    "cot_algorithms": ("chain_algorithms",),
    "cot_proportion": ("cot_proportion",),
    "finetuning_hyperparameters": ("finetuning_hyperparameters",),
    "number_of_orderings_and_conflict": ("number_of_orderings", "conflicting_vs_disjoint"),
    "representation": ("representation", "components"),
    "skew": ("skew",),
    "skew_learning": ("skew",),
    "no_cue": ("skew", "skew_no_cue", "content_effects_no_cue"),
    "architectures": ("architectures",),
    "finetuning_reasoning": ("finetuning_reasoning",),
    "finetuning_reasoning_hyperparameters": ("finetuning_reasoning_hyperparameters",),
}


def test_every_figure_script_is_listed(repo_root):
    """Every figure script except `fig_formalization.py` builds on synthetic tables."""
    scripts = {path.stem.removeprefix("fig_") for path in (repo_root / "figures").glob("fig_*.py")}
    assert scripts - {"formalization"} == set(FIGURE_EXPERIMENTS)


# Imports every figure script in a fresh interpreter and prints whether torch was loaded.
IMPORT_FIGURES_SCRIPT = """
import importlib.util, pathlib, sys
figures = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(figures))
for path in sorted(figures.glob("fig_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
print("torch" in sys.modules)
"""


def test_figure_scripts_draw_without_the_training_code(repo_root):
    """No figure script imports torch, so the figures draw without the training code."""
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_FIGURES_SCRIPT, str(repo_root / "figures")],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("name", sorted(FIGURE_EXPERIMENTS))
def test_figure_scripts_build_on_synthetic_tables(tmp_path, name):
    """Each script draws a figure no wider than the page and no taller than allowed."""
    for experiment in FIGURE_EXPERIMENTS[name]:
        _write_tables(tmp_path, experiment)
    script = cli._load_figure_module(f"fig_{name}")
    fig = script.build(cli.argparse.Namespace(results=tmp_path, out=tmp_path / "figures"))
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        width, height = fig.get_size_inches()
        assert width <= style.FULL_WIDTH_IN + 1e-6 and height <= style.MAX_HEIGHT_IN + 1e-6
        assert fig.axes
    finally:
        plt.close(fig)


def test_no_cue_figure_draws_the_majority_ordering_accuracy(tmp_path):
    """The dotted IWL line is the majority ordering's accuracy, averaged over the table's seeds."""
    for experiment in FIGURE_EXPERIMENTS["no_cue"]:
        _write_tables(tmp_path, experiment)
    table = pd.read_parquet(tmp_path / "skew_no_cue" / "tables" / "eval.parquet")
    present = {(float(a), int(s)) for a, s in table[["skew_alpha", "seed"]].drop_duplicates().itertuples(index=False)}
    expected: dict[float, list[float]] = {}
    for run in config.enumerate_runs(config.load_experiment("skew_no_cue")):
        skew, seed = float(run.axis_values["skew_alpha"]), int(run.axis_values["seed"])
        if (skew, seed) in present:
            law = training.build_law(run.config, seed)
            expected.setdefault(skew, []).append(orderings.majority_ordering_accuracy(law, streams.training_cue_weights(run.config.task)))
    skews = sorted(float(a) for a in table["skew_alpha"].unique())
    assert sorted(expected) == skews
    assert sum(len(values) for values in expected.values()) == len(present)
    script = cli._load_figure_module("fig_no_cue")
    fig = script.build(cli.argparse.Namespace(results=tmp_path, out=tmp_path / "figures"))
    try:
        (line,) = [line for line in fig.axes[0].lines if line.get_label() == script.MAJORITY_LABEL]
        assert list(line.get_xdata()) == skews
        assert np.allclose(line.get_ydata(), [np.mean(expected[skew]) for skew in skews])
        assert script.MAJORITY_LABEL in [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("unreached", [(1, 2, 3, 4, 5, 6), (1, 3, 5)], ids=["everywhere", "lone_markers"])
def test_content_figure_numbers_a_context_only_where_it_draws_a_line(tmp_path, unreached):
    """A context unreached at every distance has no line on the steps-to-criterion panel and
    gets no number beside it, the other contexts keeping theirs; one drawn only as lone markers
    between unreached distances has no line to anchor a number and stops the figure."""
    _write_tables(tmp_path, "content_effects")
    path = tmp_path / "content_effects" / "tables" / "eval.parquet"
    table = pd.read_parquet(path)
    chance = (table["condition"] == "frequent_cue_shuffled") & table["d"].isin(unreached)
    table.loc[chance, "value"] = tables.CHANCE_ACCURACY
    swept = config.load_experiment("content_effects").swept_columns
    tables.write_table(table, path, schema=tables.EVAL_SCHEMA, swept_columns=swept)
    script = cli._load_figure_module("fig_content_effects")
    args = cli.argparse.Namespace(results=tmp_path, out=tmp_path / "figures")
    if len(unreached) < len(DISTANCES):
        with pytest.raises(ValueError, match="no line of two or more points"):
            script.build(args)
        plt.close("all")
        return
    fig = script.build(args)
    try:
        keys = [text.get_gid().removeprefix(placement.GID_PREFIX) for text in placement.marked_texts(fig.axes[1])]
        assert keys == ["frequent_cue_congruent", "frequent_cue_reversed"]
    finally:
        plt.close(fig)


def test_content_figure_breaks_its_line_and_keeps_its_ticks_at_an_unreached_distance(tmp_path):
    """An unreached distance breaks its context's line, and the step ticks stay on the sweep's
    budgets though its cross sits above them."""
    _write_tables(tmp_path, "content_effects")
    path = tmp_path / "content_effects" / "tables" / "eval.parquet"
    table = pd.read_parquet(path)
    table.loc[(table["condition"] == "frequent_cue_reversed") & (table["d"] == 4), "value"] = tables.CHANCE_ACCURACY
    swept = config.load_experiment("content_effects").swept_columns
    tables.write_table(table, path, schema=tables.EVAL_SCHEMA, swept_columns=swept)
    script = cli._load_figure_module("fig_content_effects")
    fig = script.build(cli.argparse.Namespace(results=tmp_path, out=tmp_path / "figures"))
    try:
        ax = fig.axes[1]
        color = matplotlib.colors.to_hex(style.CONDITION_COLORS["frequent_cue_reversed"])
        runs = [line for line in ax.get_lines() if matplotlib.colors.to_hex(line.get_color()) == color]
        assert sorted(sorted(int(x) for x in line.get_xdata()) for line in runs) == [[1, 2, 3], [5, 6]]
        ticks = sorted(int(k) for k in table["K"].unique())[:: script.CRITERION_TICK_STEP]
        assert ax.get_ylim()[1] > max(ticks)
        assert list(ax.get_yticks()) == ticks
    finally:
        plt.close(fig)


def test_orderings_figure_shares_the_single_ordering_point(tmp_path):
    """Both layout lines of the conflict panel start at one ordering, on the same point."""
    for experiment in FIGURE_EXPERIMENTS["number_of_orderings_and_conflict"]:
        _write_tables(tmp_path, experiment)
    # The declared sweep has no disjoint run at one ordering; the synthetic table must not either.
    path = tmp_path / "conflicting_vs_disjoint" / "tables" / "eval.parquet"
    table = pd.read_parquet(path)
    swept = tuple(axis.name for axis in config.load_experiment("conflicting_vs_disjoint").axes)
    kept = table[~((table["layout"] == "disjoint") & (table["n_orderings"] == 1))]
    tables.write_table(kept, path, schema=tables.EVAL_SCHEMA, swept_columns=swept)
    script = cli._load_figure_module("fig_number_of_orderings_and_conflict")
    fig = script.build(cli.argparse.Namespace(results=tmp_path, out=tmp_path / "figures"))
    try:
        names = set(script.LAYOUT_LABELS.values())
        lines = {line.get_label(): line for line in fig.axes[1].lines if line.get_label() in names}
        assert set(lines) == names
        starts = {name: (float(line.get_xdata()[0]), float(line.get_ydata()[0])) for name, line in lines.items()}
        assert all(x == 1.0 for x, _ in starts.values())
        assert len({y for _, y in starts.values()}) == 1
    finally:
        plt.close(fig)


def test_architectures_figure_prints_the_training_loss_on_every_panel(tmp_path):
    """Every architecture is drawn, and each panel carries its own training-loss indicator."""
    _write_tables(tmp_path, "architectures")
    experiment = config.load_experiment("architectures")
    swept = tuple(axis.name for axis in experiment.axes)
    train = tables.load_table(tmp_path / "architectures" / "tables" / "train.parquet", schema=tables.TRAIN_SCHEMA, swept_columns=swept)
    script = cli._load_figure_module("fig_architectures")
    losses = script.training_loss(train)
    assert len(losses) == math.prod(len(axis.values) for axis in experiment.axes)
    fig = script.build(cli.argparse.Namespace(results=tmp_path, out=tmp_path / "figures"))
    try:
        panels = [ax for ax in fig.axes if ax.get_visible()]
        assert len(panels) == len(losses)
        printed = sorted(text.get_text() for ax in panels for text in ax.texts if text.get_text().startswith("loss "))
        assert printed == sorted(script.LOSS_LABEL.format(value) for value in losses.values())
        # The top-left panel: deepest, widest, feedforward first, larger learning rate first.
        top_left = max(losses, key=lambda key: (key[0], key[1], key[2] == script.FEEDFORWARD_ORDER[0], key[3]))
        first = next(text.get_text() for text in panels[0].texts if text.get_text().startswith("loss "))
        assert first == script.LOSS_LABEL.format(losses[top_left])
    finally:
        plt.close(fig)


def test_figure_command_writes_beside_the_tables(tmp_path):
    """`figure --out <root>` reads `<root>/<experiment>/tables/` and writes `<root>/figures/`."""
    _write_tables(tmp_path, "symbolic_distance")
    assert cli.main(["figure", "--name", "symbolic_distance", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "figures" / "symbolic_distance.pdf").stat().st_size > 0
    assert (tmp_path / "figures" / "symbolic_distance.png").is_file()
    assert cli.figure_directory(None) == Path("figures") / "out"
    assert cli.figure_directory(tmp_path) == tmp_path / "figures"


def test_legend_placement_in_the_keyword_mapping_wins(frames):
    """A `loc` inside `legend_kwargs` overrides the panel's `legend_loc` instead of raising."""
    df = _pinned(frames, "eval", ())
    fig, ax = plt.subplots()
    try:
        distance.make_accuracy_by_distance(ax, df, legend_loc="lower right", legend_kwargs={"loc": "upper left"})
        assert ax.get_legend()._loc == matplotlib.legend.Legend.codes["upper left"]
    finally:
        plt.close(fig)
