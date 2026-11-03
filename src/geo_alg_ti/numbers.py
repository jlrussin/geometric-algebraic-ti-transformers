"""The results the paper prints about the transformers, recomputed from the tidy tables.

``numbers()`` returns one row per printed number: ``id``, ``site`` (where the paper prints it),
``printed``, ``computed``, ``source`` (the value of ``values()`` it comes from) and ``match``
(whether ``computed``, written at the printed precision, equals ``printed``). A number a figure
writes on a panel has that figure as its site. Every value is read from the committed tables
under ``results/<experiment>/tables/`` through the estimands of ``geo_alg_ti.tables``, the ones
the figure scripts draw.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from geo_alg_ti import config, tables

RESULTS_ROOT: Path = config.repo_root() / "results"

SDE_FIGURE = "Symbolic distance effects (SDE) figure, reasoning-steps panel"
CONTENT_FIGURE = "Content effects in transitive inference figure, reasoning-steps panel"
ALGORITHMS = "Extended results, transformers, alternative chain-of-thought algorithms"
ARCHITECTURES_FIGURE = "Architecture and learning-rate sweep figure"
ARCHITECTURES = "Extended results, transformers, architecture robustness"
MECHANISMS = "Extended results, transformers, additional mechanism results"
SITES: tuple[str, ...] = (SDE_FIGURE, CONTENT_FIGURE, ALGORITHMS, ARCHITECTURES_FIGURE, ARCHITECTURES, MECHANISMS)

# The content effects networks the main text reports: frequent-cue proportion 0.2, p(CoT) 0.9.
CONTENT_SETTING: Mapping[str, float] = {"frequent_cue_prop": 0.2, "p_cot": 0.9}
CONTENT_CONTEXTS: tuple[str, ...] = ("congruent", "reversed", "shuffled")
# The smallest distance used: the accuracy beside a reasoning-steps line is pooled over distances
# two and above, and the steps-to-criterion slope is fit over the same distances.
D_MIN: int = 2
# The first budget reaching this fraction of a seed's best accuracy within a distance.
CRITERION_FRACTION: float = 0.9
# The chain-generation algorithms are scored on the ICL test under a training cue.
ALGORITHM_CONDITION: str = "icl_test_training_cue"
ALGORITHM_NAMES: Mapping[str, str] = {
    "main": "main",
    "inferred_relations": "inferred relations",
    "descend_first": "descend first",
    "backtrack": "backtrack at dead ends",
}
# The finetuning conditions of the reasoning-during-finetuning experiments.
NO_REASONING, REASONING, CONTEXT_READING, RANDOM_HEADS = tables.FINETUNING_CONDITIONS
# The grid's setting columns; the finetuning setting was chosen on the grid's networks.
GRID_SETTING: tuple[str, ...] = ("learning_rate", "uninformative_prop")
# Each architecture's loss is its mean training loss over the last 5,000 pretraining steps.
ARCHITECTURE: tuple[str, ...] = ("n_layers", "d_model", "feedforward", "learning_rate")
LOSS_WINDOW_STEPS: int = 5000


def _loss_key(n_layers: int, d_model: int, feedforward: str, learning_rate: float) -> str:
    """The ``values()`` key of one architecture's final training loss."""
    return f"loss_layers{n_layers}_width{d_model}_{feedforward}_lr{learning_rate:g}"


# The architecture figure's panels in reading order: (layers, width, feedforward, learning rate,
# the loss the panel prints).
_LOSSES: tuple[tuple[int, int, str, float, str], ...] = (
    (8, 256, "ffn", 0.001, "0.0017"), (8, 256, "ffn", 0.0001, "0.00044"),
    (8, 256, "none", 0.001, "0.0016"), (8, 256, "none", 0.0001, "0.00076"),
    (8, 128, "ffn", 0.001, "0.0013"), (8, 128, "ffn", 0.0001, "0.00069"),
    (8, 128, "none", 0.001, "0.0012"), (8, 128, "none", 0.0001, "0.0016"),
    (4, 256, "ffn", 0.001, "0.0017"), (4, 256, "ffn", 0.0001, "0.00033"),
    (4, 256, "none", 0.001, "0.0014"), (4, 256, "none", 0.0001, "0.00067"),
    (4, 128, "ffn", 0.001, "0.0024"), (4, 128, "ffn", 0.0001, "0.0016"),
    (4, 128, "none", 0.001, "0.0023"), (4, 128, "none", 0.0001, "0.015"),
    (2, 256, "ffn", 0.001, "0.089"), (2, 256, "ffn", 0.0001, "0.024"),
    (2, 256, "none", 0.001, "0.059"), (2, 256, "none", 0.0001, "0.12"),
    (2, 128, "ffn", 0.001, "0.028"), (2, 128, "ffn", 0.0001, "0.17"),
    (2, 128, "none", 0.001, "0.13"), (2, 128, "none", 0.0001, "0.23"),
)

# (site, printed, source); ids number the rows in this order.
_ROWS: tuple[tuple[str, str, str], ...] = (
    (SDE_FIGURE, "0.88", "icl_test_accuracy_largest_budget"),
    (SDE_FIGURE, "0.72", "iwl_test_accuracy_largest_budget"),
    (CONTENT_FIGURE, "0.99", "congruent_accuracy_largest_budget"),
    (CONTENT_FIGURE, "0.97", "reversed_accuracy_largest_budget"),
    (CONTENT_FIGURE, "0.99", "shuffled_accuracy_largest_budget"),
    (ALGORITHMS, "Three", "algorithms_steps_rise"),
    (ALGORITHMS, "four", "algorithms"),
    (ALGORITHMS, "inferred relations", "algorithms_steps_fall"),
    *((ARCHITECTURES_FIGURE, printed, _loss_key(*arch)) for *arch, printed in _LOSSES),
    (ARCHITECTURES, "two-layer", "highest_loss_layers"),
    (MECHANISMS, "10", "networks"),
    (MECHANISMS, "9", "reasoning_lower_end_item_networks"),
    (MECHANISMS, "9", "reasoning_lower_interior_networks"),
    (MECHANISMS, "8", "context_reading_above_random_end_item_networks"),
    (MECHANISMS, "7", "context_reading_above_random_interior_networks"),
    (MECHANISMS, "100", "context_reading_ahead_from_step"),
    (MECHANISMS, "seven", "unchosen_networks"),
    (MECHANISMS, "seven", "unchosen_icl_test_lower_networks"),
    (MECHANISMS, "six", "unchosen_end_item_higher_networks"),
    (MECHANISMS, "six", "unchosen_interior_higher_networks"),
    (MECHANISMS, "three", "grid_seeds_per_setting"),
    (MECHANISMS, "12", "grid_settings"),
    (MECHANISMS, "12", "grid_icl_test_lower_settings"),
    (MECHANISMS, "11", "grid_trained_higher_settings"),
    (MECHANISMS, "11", "grid_end_item_higher_settings"),
    (MECHANISMS, "10", "grid_interior_higher_settings"),
)
# (id, site, printed, source)
PAPER_NUMBERS: tuple[tuple[str, str, str, str], ...] = tuple(
    (f"tf-{n}", site, printed, source) for n, (site, printed, source) in enumerate(_ROWS, start=1)
)

_SCHEMAS: Mapping[str, Mapping[str, str]] = {
    "eval": tables.EVAL_SCHEMA,
    "train": tables.TRAIN_SCHEMA,
    "eval_pairs": tables.PAIR_SCHEMA,
    "ablated_eval_pairs": tables.ABLATED_PAIR_SCHEMA,
}


def _load(results_root: Path, experiment: str, table: str) -> pd.DataFrame:
    """One tidy table of one experiment, validated with the experiment's swept columns."""
    path = Path(results_root) / experiment / "tables" / f"{table}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no '{table}' table at {path}; export the '{experiment}' experiment first")
    swept = config.load_experiment(experiment).swept_columns
    return tables.load_table(path, schema=_SCHEMAS[table], swept_columns=swept)


def _final_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """The accuracy rows at the last evaluation step, refused unless they hold one tag."""
    rows = df[df["metric"] == "accuracy"]
    rows = rows[rows["step"] == rows["step"].max()]
    if rows["tag"].nunique() != 1:
        raise ValueError(f"the accuracy rows hold the tags {sorted(rows['tag'].unique())}, not one")
    return rows


def values(results_root: Path = RESULTS_ROOT) -> dict[str, object]:
    """Every source value, keyed as ``PAPER_NUMBERS`` names them."""
    out: dict[str, object] = {}
    sde = _load(results_root, "symbolic_distance", "eval")
    swept = config.load_experiment("symbolic_distance").swept_columns
    sde = _final_accuracy(tables.apply_swept_pins(sde, fixed={}, swept_columns=swept, fname="symbolic_distance"))
    for condition in ("icl_test", "iwl_test"):
        out[f"{condition}_accuracy_largest_budget"] = _accuracy_at_largest_budget(sde, condition)
    content = _load(results_root, "content_effects", "eval")
    swept = config.load_experiment("content_effects").swept_columns
    content = tables.apply_swept_pins(content, fixed=CONTENT_SETTING, swept_columns=swept, fname="content_effects")
    content = _final_accuracy(content)
    for context in CONTENT_CONTEXTS:
        accuracy = _accuracy_at_largest_budget(content, f"frequent_cue_{context}")
        out[f"{context}_accuracy_largest_budget"] = accuracy
    out.update(_algorithms(_final_accuracy(_load(results_root, "chain_algorithms", "eval"))))
    out.update(_architectures(_load(results_root, "architectures", "train")))
    out.update(_finetuning_reasoning(results_root))
    return out


def _accuracy_at_largest_budget(rows: pd.DataFrame, condition: str) -> float:
    """One condition's accuracy at the sweep's largest budget, pooled over distances at least
    ``D_MIN`` by trials within each seed, then the mean over seeds."""
    largest = int(rows["K"].max())
    per_seed = tables.accuracy_at_budget(tables.pick(rows, condition=condition), k=largest, d_min=D_MIN)
    return float(tables.seed_mean_se(per_seed, group_cols=["condition"])["value_mean"].iloc[0])


def _algorithms(rows: pd.DataFrame) -> dict[str, object]:
    """For each chain-generation algorithm, the reasoning steps to criterion by distance (the
    first budget reaching ``CRITERION_FRACTION`` of the seed's best accuracy, then the mean over
    seeds) and its least-squares slope over distances at least ``D_MIN``; returns the number of
    algorithms, the number whose steps rise with distance, and the names of those whose steps
    fall."""
    swept = config.load_experiment("chain_algorithms").swept_columns
    slopes = {}
    for algorithm in sorted(rows["chain_algorithm"].unique()):
        fixed = {"chain_algorithm": algorithm, "condition": ALGORITHM_CONDITION}
        sub = tables.apply_swept_pins(rows, fixed=fixed, swept_columns=swept, fname="chain_algorithms")
        reached = tables.first_k_to_criterion(sub, group_cols=["d", "seed"], fraction=CRITERION_FRACTION)
        steps = tables.seed_mean_se(reached, group_cols=["d"], value_col=tables.K_TO_CRITERION)
        steps = steps[steps["d"] >= D_MIN]
        mean = steps[f"{tables.K_TO_CRITERION}_mean"].to_numpy(dtype=float)
        slopes[str(algorithm)] = float(np.polyfit(steps["d"].to_numpy(dtype=float), mean, 1)[0])
    falling = [ALGORITHM_NAMES[name] for name, slope in slopes.items() if slope < 0]
    return {"algorithms": len(slopes), "algorithms_steps_rise": sum(s > 0 for s in slopes.values()),
            "algorithms_steps_fall": ", ".join(falling)}


def _architectures(train: pd.DataFrame) -> dict[str, object]:
    """Each architecture's mean training loss over the last ``LOSS_WINDOW_STEPS`` pretraining
    steps (within a seed, then over seeds), and the layer count whose every architecture ends at
    a higher loss than every architecture with another layer count."""
    recent = train[train["step"] >= train["step"].max() - LOSS_WINDOW_STEPS]
    summary = tables.seed_mean_se(recent, group_cols=list(ARCHITECTURE), value_col="loss")
    out: dict[str, object] = {}
    for row in summary.itertuples(index=False):
        key = _loss_key(int(row.n_layers), int(row.d_model), str(row.feedforward), float(row.learning_rate))
        out[key] = float(row.loss_mean)
    by_depth = summary.groupby("n_layers")["loss_mean"]
    lowest, highest = by_depth.min(), by_depth.max()
    apart = [int(n) for n in lowest.index if lowest[n] > highest.drop(n).max()]
    out["highest_loss_layers"] = ", ".join(f"{_WORDS[n]}-layer" for n in apart) or "none"
    return out


def _condition_key(condition: tuple[str, float]) -> str:
    """The column name of one finetuning condition."""
    return f"{condition[0]}/{condition[1]}"


def _pair_types(results_root: Path, experiment: str, *, group: tuple[str, ...]) -> pd.DataFrame:
    """Each finetuning condition's accuracy on every one of ``tables.PAIR_TYPES`` within each
    step, `group` value and seed (the best budget within each distance, then distances at least
    ``D_MIN`` pooled; distance one for trained pairs), each run scored as it was finetuned: one
    row per step, `group` value and seed, one column per (measure, condition). Raises on a second
    tag, on a swept column outside the condition and `group`, and on a condition that lacks a
    network at some step."""
    rows = tables.finetuned_pair_rows(_load(results_root, experiment, "eval_pairs"), _load(results_root, experiment, "ablated_eval_pairs"), fname=experiment)
    if rows["tag"].nunique() != 1:
        raise ValueError(f"{experiment}: the accuracy rows hold the tags {sorted(rows['tag'].unique())}, not one")
    held = (tables.HEADS_COLUMN, tables.REASONING_COLUMN, *group)
    swept = [name for name in config.load_experiment(experiment).swept_columns if name not in held]
    tables.apply_swept_pins(rows, fixed={}, swept_columns=swept, fname=experiment)
    frames = []
    for condition in tables.FINETUNING_CONDITIONS:
        sub = tables.select(rows, fname=experiment, **{tables.HEADS_COLUMN: condition[0], tables.REASONING_COLUMN: condition[1]})
        for measure in tables.PAIR_TYPES:
            per_seed = tables.pair_type_by_seed(sub, measure, group_cols=("step", *group), k_stat="best", d_min=D_MIN, fname=experiment)
            frames.append(per_seed.assign(measure=measure, condition=_condition_key(condition)))
    table = pd.concat(frames, ignore_index=True).pivot(index=["step", *group, "seed"], columns=["measure", "condition"], values="value")
    if table.isna().to_numpy().any():
        raise ValueError(f"{experiment}: a finetuning condition lacks a network at some step")
    return table


def _gap(frame: pd.DataFrame, measure: str, first: tuple[str, float], second: tuple[str, float]) -> pd.Series:
    """`first` minus `second` on one measure, row by row."""
    return frame[(measure, _condition_key(first))] - frame[(measure, _condition_key(second))]


def _finetuning_reasoning(results_root: Path) -> dict[str, object]:
    """At the last finetuning step: the networks, and those in which reasoning lowers accuracy on
    pairs with an end item and on interior pairs (against no reasoning) and in which ablating the
    context-reading heads beats ablating random heads on them; the networks the setting was not
    chosen on (those outside the grid), and those among them in which ablating the
    context-reading heads (against reasoning intact) lowers the ICL test and raises each
    non-adjacent pair type; the grid's seeds per setting (missing unless every setting has the
    same count), its settings, and the settings at which, in the mean over seeds, ablating the
    context-reading heads (against reasoning intact) lowers the ICL test and raises each IWL pair
    type. Over all evaluation steps, in the mean over networks: the first step from which
    ablating the context-reading heads leads both reasoning intact and random heads on both
    non-adjacent pair types at every later step (missing if it trails at the last step)."""
    main = _pair_types(results_root, "finetuning_reasoning", group=())
    grid = _pair_types(results_root, "finetuning_reasoning_hyperparameters", group=GRID_SETTING)
    last = main.xs(main.index.get_level_values("step").max(), level="step")
    out: dict[str, object] = {"networks": len(last)}
    for measure in ("end_item", "interior"):
        out[f"reasoning_lower_{measure}_networks"] = int((_gap(last, measure, REASONING, NO_REASONING) < 0).sum())
        out[f"context_reading_above_random_{measure}_networks"] = int((_gap(last, measure, CONTEXT_READING, RANDOM_HEADS) > 0).sum())
    unchosen = last[~last.index.isin(grid.index.get_level_values("seed"))]
    out["unchosen_networks"] = len(unchosen)
    out["unchosen_icl_test_lower_networks"] = int((_gap(unchosen, "icl_test", CONTEXT_READING, REASONING) < 0).sum())
    for measure in ("end_item", "interior"):
        out[f"unchosen_{measure}_higher_networks"] = int((_gap(unchosen, measure, CONTEXT_READING, REASONING) > 0).sum())
    means = main.groupby(level="step").mean()
    lead = pd.concat(
        [_gap(means, measure, CONTEXT_READING, other) for measure in ("end_item", "interior") for other in (REASONING, RANDOM_HEADS)],
        axis=1,
    ).min(axis=1)
    behind = lead.index[lead <= 0]
    ahead = lead.index[lead.index > behind.max()] if len(behind) else lead.index
    out["context_reading_ahead_from_step"] = int(ahead.min()) if len(ahead) else float("nan")
    grid_last = grid.xs(grid.index.get_level_values("step").max(), level="step")
    seeds = grid_last.groupby(level=list(GRID_SETTING)).size()
    out["grid_seeds_per_setting"] = int(seeds.iloc[0]) if seeds.nunique() == 1 else float("nan")
    settings = grid_last.groupby(level=list(GRID_SETTING)).mean()
    out["grid_settings"] = len(settings)
    out["grid_icl_test_lower_settings"] = int((_gap(settings, "icl_test", CONTEXT_READING, REASONING) < 0).sum())
    for measure in ("trained", "end_item", "interior"):
        out[f"grid_{measure}_higher_settings"] = int((_gap(settings, measure, CONTEXT_READING, REASONING) > 0).sum())
    return out


# Writing a value as the paper writes it.
_WORDS = ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
_NUMBER = re.compile(r"^([+-]?)(\d[\d,]*)?(?:\.(\d+))?(%?)$")


def format_like(value, printed: str) -> str:
    """``value`` written as ``printed`` writes its number.

    The same decimals (`tables.round_half_up`), a thousands separator, a leading zero and
    a sign where ``printed`` has them; a percent sign where ``printed`` has one (the value
    already in percent); a number word for a word, capitalized where ``printed`` is; ``>= x`` or
    ``< x`` while the value is on that side of x; a text value as it is. A NaN is written "nan".
    """
    if isinstance(value, str):
        return value
    for op, holds in ((">= ", lambda v, b: v >= b), ("< ", lambda v, b: v < b)):
        if printed.startswith(op):
            bound = printed[len(op):]
            return printed if holds(float(value), float(bound)) else format_like(value, bound)
    if math.isnan(float(value)):
        return "nan"
    if printed.lower() in _WORDS:
        whole = float(value).is_integer() and 0 <= value < len(_WORDS)
        if not whole:
            return str(value)
        word = _WORDS[int(value)]
        return word.capitalize() if printed[:1].isupper() else word
    match = _NUMBER.match(printed)
    if match is None:
        raise ValueError(f"no number format in {printed!r}")
    sign, whole, fraction, percent = match.groups()
    rounded = tables.round_half_up(value, len(fraction or ""))
    text = format(abs(rounded), ",f" if whole and "," in whole else "f")
    if whole is None and text.startswith("0."):
        text = text[1:]
    if rounded < 0:
        text = "-" + text
    elif sign == "+" and rounded > 0:
        text = "+" + text
    return text + percent


def numbers(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Every number of ``PAPER_NUMBERS``, recomputed and compared at the printed precision."""
    got = values(results_root)
    rows = []
    for id_, site, printed, source in PAPER_NUMBERS:
        computed = format_like(got[source], printed)
        rows.append({"id": id_, "site": site, "printed": printed, "computed": computed,
                     "source": source, "match": computed == printed})
    return pd.DataFrame(rows)
