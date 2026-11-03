"""Reasoning-step sweep, per-trial scoring, and per-condition aggregation.

Realizes the extended methods' "Evaluation and scoring" paragraph. The
prompt ends after the context; greedy decoding produces up to eight complete
pair emissions; at each budget K the two query-item tokens are appended and
the relation sign is scored against the condition's scoring reference.
Accuracy is sign agreement with that reference. Two companion columns make
every row self-describing:

    context            reference  value            context_consistent  prior_consistent
    uninformative      familiar   agree(familiar)  missing             = value
    congruent          shown      agree(shown)     = value             = value (shown = familiar)
    reversed           shown      agree(shown)     = value             1 - value
    shuffled           shown      agree(shown)     = value             from the familiar sign
    any, novel source  shown      agree(shown)     = value             missing (no familiar)

(the last two columns are the `_frac` companions of the aggregates)

Under a shuffled context `d` is the symbolic distance in the ordering shown and
`rank_a`, `rank_b` are the shown positions of the queried items; under every
other context they are the distance and ranks in the queried ordering. A
shuffled row of the evaluation table holds agreeing and conflicting trials in
equal numbers (a per-pair row does when the contexts per pair are even), so
`tables.split_shuffled` recovers the accuracy of each half from the two
columns above.

Trials are rolled out in chunks that share one prompt length, and one rollout
is scored at every reasoning-step value by adding the query items to the
state after each emission. Every rollout is scored, including malformed
chains; chain validity and length are recorded as separate metrics.
Aggregates keep `seed` as a row axis; only `tables` averages over seeds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from geo_alg_ti.battery import Battery, OrderingSource, ScoringReference, TestCondition
from geo_alg_ti.chains import ChainAlgorithm, canonical_chain_length, validate_chain
from geo_alg_ti.model import ROLLOUT_CHUNK, RelationalTransformer, greedy_chain, score_at_budgets
from geo_alg_ti.sequences import EMISSION_LEN, Target, Vocabulary, encode
from geo_alg_ti.tables import PAIR_SCHEMA
from geo_alg_ti.trials import ContextKind, Relation, Trial

METRICS: tuple[str, ...] = ("accuracy", "chain_validity", "chain_length")
# One record per trial, reasoning-step value, and metric; the column order of the tidy tables.
_TRIAL_COLUMNS: tuple[str, ...] = (
    "tag",
    "condition",
    "source",
    "context",
    "metric",
    "d",
    "K",
    "seed",
    "step",
    "value",
    "rank_a",
    "rank_b",
    "item_a",
    "item_b",
    "predicted_sign",
    "reference_sign",
    "shown_sign",
    "familiar_sign",
    "correct",
    "context_consistent",
    "prior_consistent",
    "chain_len",
    "chain_valid",
)
# The tidy per-pair schema without `experiment`, which is added when the tables are exported.
_AGGREGATE_COLUMNS: tuple[str, ...] = tuple(name for name in PAIR_SCHEMA if name != "experiment")


@dataclass(frozen=True)
class ScoredTrial:
    """One trial at one reasoning-step value, scored; `item_a`, `item_b` are the queried items."""

    condition: str
    source: OrderingSource
    context: ContextKind
    d: int
    K: int
    rank_a: int
    rank_b: int
    item_a: int
    item_b: int
    predicted_sign: int
    reference_sign: int
    shown_sign: int | None
    familiar_sign: int | None
    correct: bool
    context_consistent: bool | None
    prior_consistent: bool | None
    chain_len: int
    chain_valid: bool | None


def score_trial(trial: Trial, condition: TestCondition, predicted_sign: int, *, k: int, chain_len: int, chain_valid: bool | None) -> ScoredTrial:
    """Apply the scoring truth table to one prediction.

    `rank_a`, `rank_b` are the queried ranks in the order asked, or under a shuffled context
    the shown positions of the queried items, so their difference is always `d`.
    """
    context = ContextKind(condition.context)
    # An uninformative context relates neither queried item to the other, so it shows no sign.
    shown = None if context is ContextKind.UNINFORMATIVE or trial.shown_sign == 0 else int(trial.shown_sign)
    familiar = int(trial.familiar_sign) if condition.has_familiar_ordering else None
    reference = familiar if condition.scoring_reference is ScoringReference.FAMILIAR_ORDERING else shown
    if reference is None:
        raise ValueError(f"condition '{condition.name}' shows no ordering to score against")
    predicted = int(predicted_sign)
    shuffled = ContextKind(trial.context_kind) is ContextKind.SHUFFLED
    rank_a, rank_b = trial.shown_positions if shuffled else trial.query_ranks
    return ScoredTrial(
        condition=condition.name,
        source=OrderingSource(condition.source),
        context=context,
        d=int(trial.d),
        K=int(k),
        rank_a=int(rank_a),
        rank_b=int(rank_b),
        item_a=int(trial.query[0]),
        item_b=int(trial.query[1]),
        predicted_sign=predicted,
        reference_sign=int(reference),
        shown_sign=shown,
        familiar_sign=familiar,
        correct=predicted == reference,
        context_consistent=None if shown is None else predicted == shown,
        prior_consistent=None if familiar is None else predicted == familiar,
        chain_len=int(chain_len),
        chain_valid=None if chain_valid is None else bool(chain_valid),
    )


def evaluate_battery(
    model: RelationalTransformer,
    battery: Battery,
    vocab: Vocabulary,
    *,
    step: int,
    tag: str,
    record_trials: bool = True,
    chunk_size: int = ROLLOUT_CHUNK,
    chain_algorithm: ChainAlgorithm = ChainAlgorithm.MAIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every trial at every budget; return per-trial records and per-condition aggregates.

    The per-trial frame is long over `METRICS`: one row per trial, budget, and metric, carrying
    `seed`, `step`, and `tag` alongside the fields of `ScoredTrial`. Trials are rolled out
    `chunk_size` at a time in battery order, whatever their conditions. Chain validity is
    truncated at the canonical length of `chain_algorithm`, the one the network was trained on.
    """
    conditions = {condition.name: condition for condition in battery.conditions}
    budgets = tuple(int(budget) for budget in battery.budgets)
    if not budgets:
        raise ValueError("a battery runs at least one reasoning-step value")
    chunk = max(int(chunk_size), 1)
    frames: list[pd.DataFrame] = []
    with torch.inference_mode():
        for start in range(0, len(battery.trials), chunk):
            entries = battery.trials[start : start + chunk]
            records = _score_chunk(
                model,
                [trial for _, trial in entries],
                [conditions[name] for name, _ in entries],
                vocab,
                budgets=budgets,
                seed=battery.seed,
                step=step,
                tag=tag,
                chain_algorithm=ChainAlgorithm(chain_algorithm),
            )
            frames.append(pd.DataFrame.from_records(records, columns=_TRIAL_COLUMNS))
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_TRIAL_COLUMNS)
    aggregates = aggregate(trials)
    return (trials if record_trials else trials.iloc[:0].copy()), aggregates


def aggregate(
    trials: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("condition", "source", "context", "metric", "d", "K", "seed", "step", "tag"),
) -> pd.DataFrame:
    """Mean `value`, count `n`, and the consistency fractions per group; seed is never averaged.

    `source` and `context` are grouping keys alongside `condition`, which determines them. The
    remaining columns of the tidy schema, `experiment` and the swept axes, are added when the
    tables are exported.
    """
    group = [str(name) for name in group_cols]
    missing = [name for name in group if name not in trials.columns]
    if missing:
        raise ValueError(f"the trial records hold no column {missing}")
    if "seed" in trials.columns and "seed" not in group:
        raise ValueError("seed is a row axis of every aggregate and is never averaged over")
    spec: dict[str, tuple[str, str]] = {
        "value": ("value", "mean"),
        "n": ("value", "count"),
        "context_consistent_frac": ("context_consistent", "mean"),
        "prior_consistent_frac": ("prior_consistent", "mean"),
    }
    out = trials.groupby(group, dropna=False, sort=False).agg(**spec).reset_index()
    out["n"] = out["n"].astype("int64")
    return out.loc[:, _ordered(out.columns)]


def aggregate_by_pair(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-pair accuracy with `rank_a, rank_b` in the queried ordering's own zero-based indexing.

    Under a shuffled context the pair is named by its zero-based shown positions instead.
    """
    accuracy = trials[trials["metric"] == "accuracy"]
    return aggregate(
        accuracy,
        group_cols=("condition", "source", "context", "metric", "d", "K", "seed", "step", "tag", "rank_a", "rank_b"),
    )


def _score_chunk(
    model: RelationalTransformer,
    entries: Sequence[Trial],
    conditions: Sequence[TestCondition],
    vocab: Vocabulary,
    *,
    budgets: Sequence[int],
    seed: int,
    step: int,
    tag: str,
    chain_algorithm: ChainAlgorithm,
) -> list[dict[str, object]]:
    """Roll a chunk of trials out greedily together and score each at every reasoning-step value.

    The records nest trial, then budget, then metric, in the order of `entries`. Chain validity
    is left missing wherever nothing of the chain is read: at zero reasoning steps, under an
    uninformative context, and at every budget when the context states the queried pair itself.
    """
    encoded = [encode(trial, None, vocab, target=Target.PROMPT_ONLY) for trial in entries]
    generated = greedy_chain(model, encoded, max_emissions=max(budgets), budgets=budgets)
    tokens = generated.tokens.cpu().numpy()
    scalars = generated.scalars.cpu().numpy()
    signs = score_at_budgets(generated)
    if signs.shape != (len(entries), len(budgets)):
        raise ValueError(f"signs shaped {signs.shape} came back for {len(entries)} trials at {len(budgets)} budgets")
    records: list[dict[str, object]] = []
    for index, (trial, condition) in enumerate(zip(entries, conditions, strict=True)):
        emissions = decode_emissions(tokens[index], scalars[index], vocab)
        chain_len = chain_length(emissions, trial)
        # Validity compares an emission with the context, which an uninformative one cannot supply.
        uninformative = ContextKind(condition.context) is ContextKind.UNINFORMATIVE
        canonical = None if uninformative else canonical_chain_length(trial, chain_algorithm)
        for budget, sign in zip(budgets, signs[index], strict=True):
            valid = None if canonical is None else _truncated_validity(emissions, trial, min(int(budget), canonical))
            scored = score_trial(trial, condition, int(sign), k=int(budget), chain_len=chain_len, chain_valid=valid)
            records.extend(_records(scored, seed=seed, step=step, tag=tag))
    return records


def decode_emissions(tokens: np.ndarray, scalars: np.ndarray, vocab: Vocabulary) -> tuple[Relation | None, ...]:
    """Decode one rollout's emissions; one missing its relation token decodes to None."""
    emissions: list[Relation | None] = []
    for start in range(0, tokens.size, EMISSION_LEN):
        a, b, relation = int(tokens[start]), int(tokens[start + 1]), int(tokens[start + 2])
        if relation != vocab.rel or not vocab.is_item(a) or not vocab.is_item(b):
            emissions.append(None)
            continue
        sign = int(round(float(scalars[start + 2])))
        emissions.append(Relation(a - vocab.first_item, b - vocab.first_item, sign))
    return tuple(emissions)


def chain_length(emissions: Sequence[Relation | None], trial: Trial) -> int:
    """How many emissions come before the network first states the queried relation.

    The inferred-relations algorithm ends its chain by stating the queried pair, so a canonical
    chain of that algorithm counts one fewer than the same walk under the other algorithms.
    """
    query = {int(trial.query[0]), int(trial.query[1])}
    for index, emission in enumerate(emissions):
        if emission is not None and {emission.a, emission.b} == query:
            return index
    return len(emissions)


def _truncated_validity(emissions: Sequence[Relation | None], trial: Trial, truncate_to: int) -> bool | None:
    """Whether every emission up to the truncation point is available in the context and true."""
    scored = tuple(emissions[: max(int(truncate_to), 0)])
    if not scored:
        return None  # nothing of the chain is read at this reasoning-step value
    if any(emission is None for emission in scored):
        return False
    return validate_chain(emissions, trial, truncate_to=truncate_to).ok


def _records(scored: ScoredTrial, *, seed: int, step: int, tag: str) -> list[dict[str, object]]:
    """One record per metric, each carrying the whole scored trial."""
    shared: dict[str, object] = {
        "tag": tag,
        "condition": scored.condition,
        "source": str(scored.source),
        "context": str(scored.context),
        "d": int(scored.d),
        "K": int(scored.K),
        "seed": int(seed),
        "step": int(step),
        "rank_a": int(scored.rank_a),
        "rank_b": int(scored.rank_b),
        "item_a": int(scored.item_a),
        "item_b": int(scored.item_b),
        "predicted_sign": int(scored.predicted_sign),
        "reference_sign": int(scored.reference_sign),
        "shown_sign": _as_float(scored.shown_sign),
        "familiar_sign": _as_float(scored.familiar_sign),
        "correct": bool(scored.correct),
        "context_consistent": _as_float(scored.context_consistent),
        "prior_consistent": _as_float(scored.prior_consistent),
        "chain_len": int(scored.chain_len),
        "chain_valid": _as_float(scored.chain_valid),
    }
    values = {
        "accuracy": float(scored.correct),
        "chain_validity": _as_float(scored.chain_valid),
        "chain_length": float(scored.chain_len),
    }
    return [{**shared, "metric": metric, "value": values[metric]} for metric in METRICS]


def _as_float(value: bool | int | None) -> float:
    """A measurement a condition leaves undefined, as not-a-number in a column of floats."""
    return float("nan") if value is None else float(value)


def _ordered(columns: Sequence[str]) -> list[str]:
    """The tidy column order, with any further grouping column kept after it."""
    ordered = [name for name in _AGGREGATE_COLUMNS if name in columns]
    return ordered + [name for name in columns if name not in ordered]
