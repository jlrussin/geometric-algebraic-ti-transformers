"""Cue-indexed orderings: the latent structure behind every training sequence.

Realizes the extended methods' "Pretraining distribution" paragraph: each cue token
indexes a fixed ordering over the item pool through a seeded pseudorandom score
fixed for every cue-item combination. The law is a pure function of
(seed, cue, item), so nothing is materialized: the ordering of any item subset
is recomputed on demand, identically in every process. Training cues are the
first `n_training_cues` ids; the following `n_reserved_cues` ids are never
trained, so an ICL test ordering under a reserved cue is knowable only from
the context. The module also holds the cue-frequency distributions the
experiments sweep: uniform, Zipf with skew alpha, and one frequent cue at a
given proportion of trials; and the IWL test accuracy, under such a
distribution, of answering each pair by the frequency-weighted majority of the
training cues' orderings.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from geo_alg_ti import CHANCE_ACCURACY

ORDERING_SALT: int = 0x0DE7  # fixed salt so the law never depends on call order or process
FREQUENT_CUE: int = 0  # the frequent cue of the content experiments is always cue 0

# Scores under one cue, kept per (seed, cue) and grown with the pool, one draw per process.
_POOL_SCORES: dict[tuple[int, int], np.ndarray] = {}


def rank_lookup(items: Sequence[int]) -> dict[int, int]:
    """Each item's zero-based position in `items`, an ordering given highest rank first."""
    return {int(item): rank for rank, item in enumerate(items)}


def seeded_rng(seed: int, salt: int, *key: int) -> np.random.Generator:
    """The generator of one draw: the seed, a salt naming the kind of draw, then its place."""
    return np.random.default_rng(np.random.SeedSequence([seed, salt, *key]))


def _pool_scores(seed: int, cue: int, pool_size: int) -> np.ndarray:
    """Score of every pool item under one cue, each from its own seeded draw."""
    key = (seed, cue)
    scores = _POOL_SCORES.get(key)
    if scores is None or scores.size < pool_size:
        # Every draw stands alone, so a larger pool extends the scores without moving any of them.
        scores = np.array(
            [seeded_rng(seed, ORDERING_SALT, cue, item).random() for item in range(pool_size)]
        )
        scores.setflags(write=False)
        _POOL_SCORES[key] = scores
    return scores


class CueDistribution(StrEnum):
    """How cues are drawn during pretraining."""

    UNIFORM = "uniform"
    ZIPF = "zipf"
    FREQUENT_CUE = "frequent_cue"


@dataclass(frozen=True)
class OrderingLaw:
    """The cued-ordering law: seeded score per (cue, item), descending sort, item id tiebreak."""

    seed: int
    pool_size: int = 300
    n_training_cues: int = 100
    n_reserved_cues: int = 16

    def score(self, cue: int, items: Sequence[int]) -> np.ndarray:
        """Return each item's U[0, 1] score under `cue`; one seeded draw per (seed, cue, item)."""
        index = np.asarray(list(items), dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= self.pool_size):
            raise ValueError(f"items must lie in 0..{self.pool_size - 1}")
        return _pool_scores(self.seed, int(cue), self.pool_size)[index]

    def ordering(self, cue: int, items: Sequence[int]) -> tuple[int, ...]:
        """Return `items` ordered highest rank first under `cue`."""
        subset = [int(item) for item in items]
        scores = self.score(cue, subset)
        order = sorted(range(len(subset)), key=lambda i: (-float(scores[i]), subset[i]))
        return tuple(subset[i] for i in order)

    def ranks(self, cue: int, items: Sequence[int]) -> dict[int, int]:
        """Return the zero-based rank of every item in `items` under `cue` (rank 0 is the top)."""
        return rank_lookup(self.ordering(cue, items))

    def rank_of(self, cue: int, items: Sequence[int], item: int) -> int:
        """Return the rank of one item within `items` under `cue`."""
        ranks = self.ranks(cue, items)
        if int(item) not in ranks:
            raise ValueError(f"item {item} is not one of the given items")
        return ranks[int(item)]

    def training_cues(self) -> tuple[int, ...]:
        """Return the cue ids that pretraining and finetuning may use."""
        return tuple(range(self.n_training_cues))

    def reserved_cues(self) -> tuple[int, ...]:
        """Return the cue ids that are never trained."""
        return tuple(range(self.n_training_cues, self.n_training_cues + self.n_reserved_cues))

    def is_reserved(self, cue: int) -> bool:
        """Return whether `cue` belongs to the reserved, never-trained block."""
        return self.n_training_cues <= int(cue) < self.n_training_cues + self.n_reserved_cues

    def sample_items(self, rng: np.random.Generator, n: int, *, exclude: Collection[int] = ()) -> tuple[int, ...]:
        """Sample `n` distinct pool items outside `exclude`, without replacement."""
        # An integer population stands for 0..pool_size - 1 and draws as the array of them would.
        population: int | np.ndarray = self.pool_size
        n_available = self.pool_size
        if exclude:
            blocked = np.fromiter(exclude, dtype=np.int64)
            outside = blocked[(blocked < 0) | (blocked >= self.pool_size)]
            if outside.size:
                raise ValueError(f"exclude names items outside the pool of {self.pool_size}: {sorted(set(outside.tolist()))}")
            open_items = np.ones(self.pool_size, dtype=bool)
            open_items[blocked] = False
            population = np.flatnonzero(open_items)
            n_available = int(population.size)
        if n_available < n:
            raise ValueError(f"{n} items were asked for but only {n_available} are available")
        return tuple(int(item) for item in rng.choice(population, size=n, replace=False))


def uniform_cue_weights(n_cues: int) -> np.ndarray:
    """Return equal probability for every training cue."""
    if n_cues < 1:
        raise ValueError("a cue distribution needs at least one cue")
    return np.full(n_cues, 1.0 / n_cues)


def zipf_cue_weights(n_cues: int, skew_alpha: float) -> np.ndarray:
    """Return weights proportional to i^-alpha for cue index i = 1..n; alpha 0 is uniform."""
    if skew_alpha == 0:
        return uniform_cue_weights(n_cues)
    if n_cues < 1:
        raise ValueError("a cue distribution needs at least one cue")
    weights = np.arange(1, n_cues + 1, dtype=float) ** (-float(skew_alpha))
    return weights / weights.sum()


def frequent_cue_weights(n_cues: int, proportion: float) -> np.ndarray:
    """`proportion` on the frequent cue, the remaining mass uniform over the other cues."""
    if n_cues < 1:
        raise ValueError("a cue distribution needs at least one cue")
    if not 0.0 <= proportion <= 1.0:
        raise ValueError("the frequent cue proportion must lie in 0..1")
    if n_cues == 1:
        return np.ones(1)
    weights = np.full(n_cues, (1.0 - proportion) / (n_cues - 1))
    weights[FREQUENT_CUE] = proportion
    return weights


def cue_weights(
    kind: CueDistribution, n_cues: int, *, skew_alpha: float = 0.0, frequent_cue_proportion: float = 0.0
) -> np.ndarray:
    """Dispatch to the cue-frequency distribution named by `kind`."""
    kind = CueDistribution(kind)
    if kind is CueDistribution.UNIFORM:
        return uniform_cue_weights(n_cues)
    if kind is CueDistribution.ZIPF:
        return zipf_cue_weights(n_cues, skew_alpha)
    return frequent_cue_weights(n_cues, frequent_cue_proportion)


def pair_relations(law: OrderingLaw) -> np.ndarray:
    """Each training cue's relation on every pair of pool items, (cue, pair), the pairs in
    `np.triu_indices` order: +1 where the smaller item id ranks higher under the cue, as it does
    on equal scores, and -1 otherwise."""
    scores = np.stack([law.score(cue, range(law.pool_size)) for cue in law.training_cues()])
    first, second = np.triu_indices(law.pool_size, k=1)
    return np.where(scores[:, first] >= scores[:, second], 1.0, -1.0)


def majority_ordering_accuracy(law: OrderingLaw, weights: np.ndarray, *, relations: np.ndarray | None = None) -> float:
    """IWL test accuracy of answering each pair by the frequency-weighted majority of the orderings.

    The IWL test draws its cue from `weights` over the training cues. Each pair of pool items is
    answered with the item that ranks higher under the greater weight of training cues and is
    scored against the ordering of the drawn cue; a pair on which the weight splits evenly scores
    one half. The accuracy is exact over every pair of pool items, each pair counted once. When no
    cue carries most of the weight, these pairwise majorities need not form one ordering. A
    network that answers from the queried items alone scores about this; one that tells the
    orderings apart by other means, such as the items shown in the context, can score higher.
    `relations`, when given, are the law's `pair_relations`, so several weightings of one law
    share them.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (law.n_training_cues,):
        raise ValueError(f"{weights.size} weights were given for {law.n_training_cues} training cues")
    relations = pair_relations(law) if relations is None else np.asarray(relations, dtype=float)
    n_pairs = law.pool_size * (law.pool_size - 1) // 2
    if relations.shape != (law.n_training_cues, n_pairs):
        raise ValueError(f"the relations hold shape {relations.shape}, not {(law.n_training_cues, n_pairs)}")
    majority = np.sign(weights @ relations)
    correct = np.where(majority == 0, CHANCE_ACCURACY, relations == majority)
    return float(weights @ correct.mean(axis=1))


def sample_cue(weights: np.ndarray, rng: np.random.Generator) -> int:
    """Draw one training cue id from `weights`."""
    return int(rng.choice(len(weights), p=np.asarray(weights, dtype=float)))
