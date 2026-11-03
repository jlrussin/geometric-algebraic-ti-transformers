"""The two-factor test battery: ordering source x context type.

Realizes the extended methods' "IWL and ICL tests" paragraph and the content-effects
test design as one construction. Each declared test condition names where the
queried ordering comes from (the finetuned ordering; the frequent cue's
ordering; an ordering whose cue is drawn per trial from the training cue
distribution; a novel ordering under a reserved, never-trained cue) and what
the context shows (uninformative, congruent, reversed, shuffled). From that
pair it derives its scoring reference: the familiar ordering when the context
is uninformative, the ordering shown in context when it is informative. The
presets under `configs/battery/` name the paper's tests: the IWL test is the
familiar ordering under an uninformative context; the ICL test is a novel
ordering under a reserved cue with its relations in context, recorded
alongside a variant under a training cue; the baseline test is the finetuned
ordering under a congruent context; the arbitrage test is the
finetuned ordering under a reversed context. Conditions are declared sparsely, never as the full
product, and are rebuilt deterministically at job start rather than stored:
all 42 ordered pairs of the seven queryable ranks, a fixed number of
independently sampled contexts per pair, every reasoning-step value. Under a
shuffled context the 42 pairs are read as the shown positions of the queried
items instead, both interior to the shown chain, and the shown order of the
pair agrees with the queried ordering on exactly half the trials at every
shown distance (`shuffled_query`); the ranks of the pair in the queried
ordering are drawn independently of where it is shown.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import yaml

from geo_alg_ti.config import Config, repo_root
from geo_alg_ti.orderings import FREQUENT_CUE, OrderingLaw, sample_cue, seeded_rng
from geo_alg_ti.sequences import Vocabulary
from geo_alg_ti.streams import FinetuningDesign, Stage, stage_section, training_cue_weights
from geo_alg_ti.trials import (
    ITEMS_PER_ORDERING,
    QUERYABLE_RANKS,
    ContextKind,
    ShuffledQuery,
    Trial,
    sample_battery_trial,
)

BATTERY_SALT: int = 0x8A77  # fixed salt, so a battery rebuilds identically in any process


class OrderingSource(StrEnum):
    """Where the queried ordering comes from."""

    FINETUNED = "finetuned"
    FREQUENT_CUE = "frequent_cue"
    TRAINING_DISTRIBUTION = "training_distribution"
    NOVEL_UNSEEN_CUE = "novel_unseen_cue"


class ScoringReference(StrEnum):
    """Which ordering a prediction is scored against."""

    FAMILIAR_ORDERING = "familiar_ordering"
    SHOWN_ORDERING = "shown_ordering"


# The keys one condition of a preset file declares.
_CONDITION_KEYS: frozenset[str] = frozenset({"name", "source", "context", "label"})


@dataclass(frozen=True)
class TestCondition:
    """One declared test: an ordering source, a context type, and the contexts sampled per pair.

    The scoring reference is derived from the context type, never declared.
    """

    name: str
    source: OrderingSource
    context: ContextKind
    contexts_per_pair: int
    label: str = ""

    @property
    def scoring_reference(self) -> ScoringReference:
        """Familiar ordering when the context is uninformative, otherwise the shown ordering."""
        return scoring_reference_for(self.source, self.context)

    @property
    def has_familiar_ordering(self) -> bool:
        """False for a novel ordering under a reserved cue, where no prior exists."""
        return OrderingSource(self.source) is not OrderingSource.NOVEL_UNSEEN_CUE


@dataclass(frozen=True)
class Battery:
    """The built trials of every declared condition, at every reasoning-step value."""

    conditions: tuple[TestCondition, ...]
    trials: tuple[tuple[str, Trial], ...]
    budgets: tuple[int, ...]
    seed: int = 0

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        law: OrderingLaw,
        vocab: Vocabulary,
        *,
        seed: int,
        stage: Stage,
        design: FinetuningDesign | None,
    ) -> "Battery":
        """The battery a stage's checkpoints are scored on, built from the resolved configuration.

        The preset is `evaluation.battery`, or the stage's own when that is null; every condition
        samples `evaluation.contexts_per_pair` contexts per rank pair. The cue weights are the
        cue-frequency distribution pretraining drew from, so the `training_distribution` source
        draws its cues as pretraining did.
        """
        conditions = load_preset(preset_name(cfg, stage), contexts_per_pair=cfg.evaluation.contexts_per_pair)
        return cls.build(
            conditions,
            law,
            vocab,
            seed=seed,
            design=design,
            budgets=cfg.evaluation.reasoning_steps(),
            cue_weights=training_cue_weights(cfg.task),
        )

    @classmethod
    def build(
        cls,
        conditions: Sequence[TestCondition],
        law: OrderingLaw,
        vocab: Vocabulary,
        *,
        seed: int,
        design: FinetuningDesign | None,
        budgets: Sequence[int],
        cue_weights: np.ndarray | None = None,
    ) -> "Battery":
        """Construct every trial deterministically from the seed; identical across processes.

        `cue_weights` are the training cue weights; a `training_distribution` condition draws its
        cue from them and is refused without them, since no default stands for the distribution a
        network was trained on.
        """
        declared = tuple(conditions)
        if not declared:
            raise ValueError("a battery declares at least one test condition")
        steps = tuple(int(budget) for budget in budgets)
        if not steps:
            raise ValueError("a battery runs at least one reasoning-step value")
        pairs = ordered_rank_pairs(QUERYABLE_RANKS)

        sources = {OrderingSource(condition.source) for condition in declared}
        if OrderingSource.FINETUNED in sources and design is None:
            raise ValueError("a finetuned test condition needs the finetuning design")
        trained = design.presented_items if design is not None else frozenset()
        weights = None if cue_weights is None else np.asarray(cue_weights, dtype=float)
        if OrderingSource.TRAINING_DISTRIBUTION in sources:
            if weights is None:
                raise ValueError("a training-distribution test condition needs the training cue weights")
            if weights.shape != (law.n_training_cues,):
                raise ValueError(f"the cue weights cover {weights.size} cues, not {law.n_training_cues}")
        frequent_cue_items: tuple[int, ...] | None = None
        if OrderingSource.FREQUENT_CUE in sources:
            # One set of items per seed, shared by every context the frequent cue is tested under.
            shared = law.sample_items(_rng(seed, _FREQUENT_CUE_KEY), ITEMS_PER_ORDERING, exclude=trained)
            frequent_cue_items = law.ordering(FREQUENT_CUE, shared)

        built: list[tuple[str, Trial]] = []
        for condition in declared:
            key = _condition_key(condition.name)
            is_shuffled = ContextKind(condition.context) is ContextKind.SHUFFLED
            draw = _ordering_sampler(
                condition,
                law,
                design=design,
                frequent_cue_items=frequent_cue_items,
                weights=weights,
                trained=trained,
            )
            for pair_index, rank_pair in enumerate(pairs):
                for context_index in range(int(condition.contexts_per_pair)):
                    rng = _rng(seed, key, pair_index, context_index)
                    cue, items = draw(rng)
                    trial = sample_battery_trial(
                        law,
                        rng,
                        cue=cue,
                        items=items,
                        rank_pair=rank_pair,
                        context_kind=condition.context,
                        exclude=trained,
                        shuffled=shuffled_query(rank_pair, context_index) if is_shuffled else None,
                    )
                    _check_vocabulary(trial, vocab)
                    built.append((condition.name, trial))
        return cls(conditions=declared, trials=tuple(built), budgets=steps, seed=int(seed))

    def digest(self) -> str:
        """Hash of every trial, for the byte-identical rebuild check."""
        running = hashlib.sha256()
        running.update(f"{self.seed}|{','.join(str(budget) for budget in self.budgets)}\n".encode())
        for name, trial in self.trials:
            running.update(_trial_line(name, trial).encode())
        return running.hexdigest()


def preset_name(cfg: Config, stage: Stage) -> str:
    """The preset a stage's checkpoints are scored on: `evaluation.battery`, else the stage's."""
    return cfg.evaluation.battery or stage_section(cfg, stage).battery


def load_preset(name: str, *, contexts_per_pair: int, root: Path | None = None) -> tuple[TestCondition, ...]:
    """Read `configs/battery/<name>.yaml` into test conditions of `contexts_per_pair` contexts each.

    A preset declares each condition's name, ordering source, context type, and label. The number
    of contexts sampled per rank pair is a setting of the run (`evaluation.contexts_per_pair`),
    so it is given here rather than declared in the file.
    """
    if int(contexts_per_pair) < 1:
        raise ValueError(f"contexts_per_pair must be positive, not {contexts_per_pair}")
    root = Path(root) if root is not None else repo_root() / "configs" / "battery"
    with (root / f"{name}.yaml").open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or set(document) != {"name", "conditions"}:
        raise ValueError(f"preset '{name}' declares a name and its conditions, nothing else")
    if document["name"] != name:
        raise ValueError(f"the preset in {name}.yaml names itself '{document['name']}'")
    conditions: list[TestCondition] = []
    for entry in document["conditions"] or ():
        if not isinstance(entry, dict) or set(entry) != _CONDITION_KEYS:
            raise ValueError(f"a condition of preset '{name}' declares exactly {sorted(_CONDITION_KEYS)}")
        condition = TestCondition(
            name=str(entry["name"]),
            source=OrderingSource(entry["source"]),
            context=ContextKind(entry["context"]),
            contexts_per_pair=int(contexts_per_pair),
            label=str(entry["label"]),
        )
        # A condition that cannot be scored is refused here rather than at the first evaluation.
        scoring_reference_for(condition.source, condition.context)
        conditions.append(condition)
    if not conditions:
        raise ValueError(f"preset '{name}' declares no test condition")
    names = [condition.name for condition in conditions]
    if len(set(names)) != len(names):
        raise ValueError(f"preset '{name}' repeats a condition name")
    return tuple(conditions)


def ordered_rank_pairs(queryable_ranks: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """All ordered pairs of distinct queryable ranks (42 for seven ranks)."""
    ranks = sorted(int(rank) for rank in queryable_ranks)
    if len(set(ranks)) != len(ranks):
        raise ValueError("the queryable ranks repeat a rank")
    if len(ranks) < 2:
        raise ValueError("a query needs two distinct queryable ranks")
    return tuple((first, second) for first in ranks for second in ranks if first != second)


def shuffled_query(rank_pair: tuple[int, int], context_index: int) -> ShuffledQuery:
    """One shuffled trial's query: an ordered pair read as shown positions, and its agreement.

    The shown order agrees with the queried ordering on even context indices when the first
    position lies below the second, and on odd ones when it lies above, so the two orders of a
    pair start on opposite sides and alternate, and agree and conflict are balanced exactly at
    every shown distance. With an odd number of contexts per pair, agreement within an ordered
    pair goes with the order asked, so the presets use an even number.
    """
    first, second = int(rank_pair[0]), int(rank_pair[1])
    return ShuffledQuery(positions=(first, second), agree=(int(context_index) + int(first < second)) % 2 == 0)


def scoring_reference_for(source: OrderingSource, context: ContextKind) -> ScoringReference:
    """Uninformative context: the familiar ordering; informative context: the shown ordering.

    A novel ordering under a reserved cue has no familiar ordering, so no reference exists for
    it under an uninformative context.
    """
    source = OrderingSource(source)
    if ContextKind(context) is ContextKind.UNINFORMATIVE:
        if source is OrderingSource.NOVEL_UNSEEN_CUE:
            raise ValueError("a novel ordering under an uninformative context shows no ordering to score against")
        return ScoringReference.FAMILIAR_ORDERING
    return ScoringReference.SHOWN_ORDERING


def _rng(seed: int, *key: int) -> np.random.Generator:
    """The generator of one draw, keyed on the seed and the draw's place in the battery."""
    return seeded_rng(int(seed), BATTERY_SALT, *(int(part) for part in key))


def _condition_key(name: str) -> int:
    """A stable integer for one condition name, so its trials never depend on declaration order."""
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "big")


# The frequent cue's items are drawn once per seed, before any trial, under a key of their own.
_FREQUENT_CUE_KEY: int = _condition_key("frequent_cue_items")


def _ordering_sampler(
    condition: TestCondition,
    law: OrderingLaw,
    *,
    design: FinetuningDesign | None,
    frequent_cue_items: tuple[int, ...] | None,
    weights: np.ndarray | None,
    trained: frozenset[int],
) -> Callable[[np.random.Generator], tuple[int, tuple[int, ...]]]:
    """How a condition draws the cue and the ordering one trial queries, highest rank first."""
    source = OrderingSource(condition.source)
    n_items = ITEMS_PER_ORDERING
    if source is OrderingSource.FINETUNED:
        ordering = design.orderings[0]  # the first fixed ordering is the one the tests query
        if len(ordering.items) + 2 != n_items:
            raise ValueError(f"a displayed ordering of {n_items} items needs {n_items - 2} finetuned items")
        above, below = ordering.buckets[0], ordering.buckets[-1]
        if not above or not below:
            raise ValueError("the finetuned ordering needs a distractor above it and one below it")

        def finetuned(rng: np.random.Generator) -> tuple[int, tuple[int, ...]]:
            # A distractor from the top bucket above the seven and one from the bottom bucket
            # below them, at the places their buckets give, so every queried rank 1..7 is a
            # finetuned item.
            top = int(above[int(rng.integers(len(above)))])
            bottom = int(below[int(rng.integers(len(below)))])
            return ordering.cue, (top, *ordering.items, bottom)

        return finetuned
    if source is OrderingSource.FREQUENT_CUE:
        shared = tuple(frequent_cue_items or ())
        return lambda rng: (FREQUENT_CUE, shared)
    if source is OrderingSource.TRAINING_DISTRIBUTION:

        def from_training_cues(rng: np.random.Generator) -> tuple[int, tuple[int, ...]]:
            cue = sample_cue(weights, rng)
            return cue, law.ordering(cue, law.sample_items(rng, n_items, exclude=trained))

        return from_training_cues
    reserved = law.reserved_cues()

    def novel(rng: np.random.Generator) -> tuple[int, tuple[int, ...]]:
        cue = int(reserved[int(rng.integers(len(reserved)))])
        return cue, law.ordering(cue, law.sample_items(rng, n_items, exclude=trained))

    return novel


def _check_vocabulary(trial: Trial, vocab: Vocabulary) -> None:
    """Raise unless the trial's cue and items all have a token id."""
    vocab.cue_token(trial.cue)
    for item in {*trial.items, *trial.context_items, *trial.distractors}:
        vocab.item_token(item)


def _trial_line(name: str, trial: Trial) -> str:
    """One trial written as text, so a rebuilt battery hashes to the same digest."""
    fields = (
        name,
        str(trial.cue),
        ",".join(str(item) for item in trial.items),
        ",".join(str(item) for item in trial.query),
        ",".join(str(rank) for rank in trial.query_ranks),
        str(trial.d),
        str(trial.context_kind),
        ";".join(f"{rel.a}:{rel.b}:{rel.sign}" for rel in trial.context),
        ",".join(str(item) for item in trial.context_items),
        ",".join(str(item) for item in trial.distractors),
        f"{trial.reference_sign}:{trial.shown_sign}:{trial.familiar_sign}",
    )
    return "|".join(fields) + "\n"
