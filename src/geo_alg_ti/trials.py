"""One transitive-inference problem: query, adjacent-relation context, distractors.

Realizes the extended methods' "Task and input sequences" and
"Fixed-order learning distribution" paragraphs. A trial is nine ordered items
exposed only through their eight adjacent relations, shuffled in position and
randomly oriented, with the two endpoints present in context but never
queried. Pretraining trials carry no distractors; finetuning trials follow the
human protocol: adjacent queries on a fixed ordering, with two distractors
drawn from rank buckets and shown at their buckets' positions, so the displayed
chain is always nine items long. `with_context` is the single place the four
context types live: uninformative, congruent, reversed, shuffled. A shuffled
context shows the queried ordering's nine items in a new order with both queried
items interior to it; its symbolic distance is the one in the ordering shown,
and whether the shown order of the queried pair agrees with the queried ordering
is set by the test battery, not left to chance.

Sign convention used throughout the package: `Relation(a, b, sign)` with
`sign = +1` meaning that `a` ranks above `b`, so `sign = +1` if and only if
`rank(a) < rank(b)` with rank 0 at the top. Symbolic distance is
`d = |rank(a) - rank(b)|`, taken in the ordering the context shows under a
shuffled context and in the queried ordering under every other.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from geo_alg_ti.orderings import OrderingLaw, rank_lookup, sample_cue

ITEMS_PER_ORDERING: int = 9
QUERYABLE_RANKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
N_MAIN_ITEMS: int = 7
N_DISTRACTORS: int = 2
DISTRACTORS_PER_BUCKET: int = 2
DISTRACTOR_POOL_SIZE: int = (N_MAIN_ITEMS + 1) * DISTRACTORS_PER_BUCKET
# Fresh items a test condition draws outside the finetuning sets: a queried ordering and,
# under an uninformative context, a filler ordering disjoint from it.
FRESH_TEST_ITEMS: int = 2 * ITEMS_PER_ORDERING

BETWEEN_QUERY_WEIGHTS: Mapping[int, float] = {0: 0.10, 1: 0.45, 2: 0.45}


def check_pool(pool_size: int, *, layout: str, n_orderings: int) -> None:
    """Raise unless the item pool holds every finetuning item set plus two fresh orderings.

    The disjoint layout deals one item set, main items and distractor pool, per fixed ordering;
    the conflicting layout deals one set for all of them. A test condition then draws up to two
    whole orderings outside every set: a fresh queried ordering and, under an uninformative
    context, a fresh ordering of filler items disjoint from it.
    """
    n_sets = int(n_orderings) if str(layout) == "disjoint" else 1
    per_set = N_MAIN_ITEMS + DISTRACTOR_POOL_SIZE
    needed = n_sets * per_set + FRESH_TEST_ITEMS
    if int(pool_size) < needed:
        raise ValueError(
            f"item_pool_size {pool_size} is too small: the {layout} layout presents {n_sets} item "
            f"set(s) of {per_set} and needs {needed} counting the {FRESH_TEST_ITEMS} fresh items a "
            "test condition draws; raise task.item_pool_size or, under the disjoint layout, lower "
            "finetuning.n_orderings"
        )


class ContextKind(StrEnum):
    """What the context shows relative to the queried ordering."""

    UNINFORMATIVE = "uninformative"
    CONGRUENT = "congruent"
    REVERSED = "reversed"
    SHUFFLED = "shuffled"


class Relation(NamedTuple):
    """One presented relation: `sign = +1` means `a` ranks above `b`."""

    a: int
    b: int
    sign: int


@dataclass(frozen=True)
class Trial:
    """One problem as shown to the network, with the signs every scoring reference needs.

    `items` is in rank order, highest rank first, so an item's index in it is its rank, and
    `query_ranks` names the queried pair by rank, in the order asked. `displayed` is the queried
    ordering as it is first shown, distractors included: a congruent context shows it, a
    reversed one reverses it, a shuffled one reorders it, and an uninformative one replaces it.
    `context_items` is the ordering the context shows and `context_kind` says how it relates to
    the queried one. The query, its symbolic distance, its shown positions, and the three signs
    follow from those fields; the symbolic distance of a shuffled trial is the shown one.
    """

    cue: int
    items: tuple[int, ...]
    query_ranks: tuple[int, int]
    displayed: tuple[int, ...]
    context: tuple[Relation, ...]
    context_items: tuple[int, ...]
    context_kind: ContextKind
    distractors: tuple[int, ...] = ()

    @property
    def query(self) -> tuple[int, int]:
        """The two queried items, in the order asked."""
        return (self.items[self.query_ranks[0]], self.items[self.query_ranks[1]])

    @property
    def d(self) -> int:
        """The symbolic distance between the queried items.

        Under a shuffled context it is the distance in the ordering the context shows; under
        every other context it is the distance in the queried ordering.
        """
        if self.context_kind is ContextKind.SHUFFLED:
            return symbolic_distance(*self.shown_positions)
        return symbolic_distance(*self.query_ranks)

    @property
    def shown_positions(self) -> tuple[int, int]:
        """The positions of the two queried items in the ordering the context shows, as asked."""
        position = rank_lookup(self.context_items)
        a, b = self.query
        if a not in position or b not in position:
            raise ValueError("the context shows no position for a queried item")
        return (position[a], position[b])

    @property
    def familiar_sign(self) -> int:
        """The queried relation in the queried ordering: +1 when the first item ranks above."""
        return sign_from_ranks(*self.query_ranks)

    @property
    def shown_sign(self) -> int:
        """The queried relation as the context shows it; 0 when the context relates neither item."""
        return _sign_in(self.context_items, self.query)

    @property
    def reference_sign(self) -> int:
        """The sign a prediction is scored against: familiar under an uninformative context."""
        return self.familiar_sign if self.context_kind is ContextKind.UNINFORMATIVE else self.shown_sign


@dataclass(frozen=True)
class ShuffledQuery:
    """Where a shuffled context shows the queried pair, and whether their shown order agrees with
    the queried ordering.

    `positions` are the shown positions of the item asked first and of the item asked second in
    the nine-item shown chain, both interior (each one of `QUERYABLE_RANKS`). `agree` says
    whether the shown order of the two items is their order in the queried ordering (agree) or
    the opposite (conflict).
    """

    positions: tuple[int, int]
    agree: bool


@dataclass(frozen=True)
class FixedOrdering:
    """One finetuned ordering of main items plus its distractor pool dealt into rank buckets.

    `items` is the law ordering of the main items under `cue`. `buckets` holds the distractor
    pool: one bucket above the top item, one in each gap, one below the bottom item. A bucket
    gives its distractors their place in the displayed chain; the cue's law plays no part in it.
    """

    cue: int
    items: tuple[int, ...]
    buckets: tuple[tuple[int, ...], ...]

    @property
    def presented_items(self) -> frozenset[int]:
        """Every item the ordering presents: its main items and its whole distractor pool."""
        return frozenset(self.items).union(*self.buckets)


def adjacent_relations(
    items: Sequence[int], rng: np.random.Generator, *, orient: bool = True, shuffle: bool = True
) -> tuple[Relation, ...]:
    """Complete adjacent chain over `items` (top rank first), randomly oriented and shuffled."""
    shown = [int(item) for item in items]
    pairs = [Relation(a, b, 1) for a, b in zip(shown, shown[1:], strict=False)]
    if orient:
        flip = rng.random(len(pairs)) < 0.5
        pairs = [Relation(p.b, p.a, -p.sign) if f else p for p, f in zip(pairs, flip, strict=True)]
    if shuffle:
        pairs = [pairs[int(i)] for i in rng.choice(len(pairs), size=len(pairs), replace=False)]
    return tuple(pairs)


def sign_from_ranks(rank_a: int, rank_b: int) -> int:
    """Return +1 if rank_a is above rank_b (smaller index), else -1."""
    return 1 if int(rank_a) < int(rank_b) else -1


def symbolic_distance(rank_a: int, rank_b: int) -> int:
    """Return the absolute rank difference."""
    return abs(int(rank_a) - int(rank_b))


def sample_rank_pair(rng: np.random.Generator, *, d: int) -> tuple[int, int]:
    """Sample a pair of queryable ranks at distance `d`, in random orientation."""
    options = [(rank, rank + d) for rank in QUERYABLE_RANKS if rank + d in QUERYABLE_RANKS]
    if not options:
        raise ValueError(f"no queryable pair lies at distance {d}")
    above, below = options[int(rng.integers(len(options)))]
    return (above, below) if rng.random() < 0.5 else (below, above)


def sample_pretraining_trial(law: OrderingLaw, rng: np.random.Generator, *, weights: np.ndarray) -> Trial:
    """Draw a cue and items, sample a distance and rank pair, build the congruent context."""
    cue = sample_cue(weights, rng)
    if law.is_reserved(cue):
        raise ValueError("pretraining uses a training cue, never a reserved one")
    items = law.ordering(cue, law.sample_items(rng, ITEMS_PER_ORDERING))
    max_d = max(QUERYABLE_RANKS) - min(QUERYABLE_RANKS)
    d = int(rng.integers(1, max_d + 1))
    rank_pair = sample_rank_pair(rng, d=d)
    return Trial(
        cue=cue,
        items=items,
        query_ranks=rank_pair,
        displayed=items,
        context=adjacent_relations(items, rng),
        context_items=items,
        context_kind=ContextKind.CONGRUENT,
    )


def build_fixed_ordering(
    law: OrderingLaw, rng: np.random.Generator, *, cue: int, exclude: Collection[int] = ()
) -> FixedOrdering:
    """Draw the main items and a distractor pool outside `exclude`, then arrange them by `cue`."""
    main = law.sample_items(rng, N_MAIN_ITEMS, exclude=exclude)
    distractors = law.sample_items(rng, DISTRACTOR_POOL_SIZE, exclude={*exclude, *main})
    return arrange_fixed_ordering(law, rng, cue=cue, main=main, distractors=distractors)


def arrange_fixed_ordering(
    law: OrderingLaw, rng: np.random.Generator, *, cue: int, main: Sequence[int], distractors: Sequence[int]
) -> FixedOrdering:
    """Order the main items by `cue`'s law and deal the distractor pool into the rank buckets.

    The deal is a uniform random shuffle of the pool sliced into buckets of two, so any distractor
    can land in any bucket. Its bucket, not its score under `cue`, sets where it is shown relative
    to the main items.
    """
    items = law.ordering(cue, main)
    pool = [int(item) for item in distractors]
    per_bucket = DISTRACTORS_PER_BUCKET
    n_buckets = len(items) + 1
    if len(pool) != n_buckets * per_bucket or len(set(pool)) != len(pool):
        raise ValueError(f"a fixed ordering of {len(items)} items deals {n_buckets * per_bucket} distinct distractors")
    if set(pool) & set(items):
        raise ValueError("a distractor is never one of the main items")
    dealt = [pool[int(index)] for index in rng.choice(len(pool), size=len(pool), replace=False)]
    buckets = tuple(tuple(dealt[at : at + per_bucket]) for at in range(0, len(dealt), per_bucket))
    return FixedOrdering(cue=int(cue), items=items, buckets=buckets)


def sample_finetuning_trial(law: OrderingLaw, rng: np.random.Generator, *, ordering: FixedOrdering) -> Trial:
    """Adjacent query on the fixed ordering, two distractors shown at their buckets' positions."""
    if law.is_reserved(ordering.cue):
        raise ValueError("finetuning uses a training cue, never a reserved one")
    above = int(rng.integers(len(ordering.items) - 1))
    rank_pair = (above, above + 1) if rng.random() < 0.5 else (above + 1, above)
    trial = Trial(
        cue=ordering.cue,
        items=ordering.items,
        query_ranks=rank_pair,
        displayed=ordering.items,
        context=(),
        context_items=ordering.items,
        context_kind=ContextKind.CONGRUENT,
    )
    return insert_distractors(trial, rng, ordering=ordering)


def sample_battery_trial(
    law: OrderingLaw,
    rng: np.random.Generator,
    *,
    cue: int,
    items: Sequence[int],
    rank_pair: tuple[int, int],
    context_kind: ContextKind,
    exclude: Collection[int] = (),
    shuffled: ShuffledQuery | None = None,
) -> Trial:
    """Build one test trial on the queried ordering `items`, given highest rank first.

    The ordering arrives as it is to be shown. A novel, training-distribution, or frequent-cue
    ordering is the law ordering of its items. The finetuned ordering is the seven finetuned
    items in law order between a distractor from the top bucket and one from the bottom bucket,
    whose places come from their buckets; the order given is the order queried. A shuffled
    context takes `shuffled`, which places the queried pair in the shown chain and picks its
    ranks itself (see `with_context`), so `rank_pair` serves only the other contexts.
    """
    ordered = tuple(int(item) for item in items)
    trial = Trial(
        cue=int(cue),
        items=ordered,
        query_ranks=(int(rank_pair[0]), int(rank_pair[1])),
        displayed=ordered,
        context=(),
        context_items=ordered,
        context_kind=context_kind,
    )
    return with_context(trial, context_kind, rng, law=law, exclude=exclude, shuffled=shuffled)


def with_context(
    trial: Trial,
    kind: ContextKind,
    rng: np.random.Generator,
    *,
    law: OrderingLaw,
    exclude: Collection[int] = (),
    shuffled: ShuffledQuery | None = None,
) -> Trial:
    """Replace the context with the uninformative, congruent, reversed, or shuffled kind.

    A shuffled context needs `shuffled`, and only a shuffled context takes it. It draws the
    symbolic distance of the queried pair in the queried ordering uniformly from one to six,
    then the pair's upper rank uniformly among the queryable ranks that keep both ranks
    queryable, so that distance is independent of the shown one. The two items at those ranks
    go to the shown positions `shuffled.positions` in the order asked, the item asked first at
    the first position, oriented so that their shown order agrees with the queried ordering
    exactly when `shuffled.agree`; the other seven items fill the remaining positions in a
    uniformly random order. `query_ranks` becomes the two ranks in the order asked, so the
    query, `familiar_sign`, `shown_sign`, and `reference_sign` keep their meaning, and `d` is
    the shown distance. Every other context keeps the query and draws nothing for it.
    """
    kind = ContextKind(kind)
    if shuffled is not None and kind is not ContextKind.SHUFFLED:
        raise ValueError(f"only a shuffled context places the query; a {kind} context takes none")
    displayed = trial.displayed
    query_ranks = trial.query_ranks
    if kind is ContextKind.UNINFORMATIVE:
        blocked = set(trial.items) | set(trial.context_items) | set(trial.distractors) | set(exclude)
        filler = law.sample_items(rng, len(displayed), exclude=blocked)
        shown_items = law.ordering(trial.cue, filler)
    elif kind is ContextKind.CONGRUENT:
        shown_items = tuple(displayed)
    elif kind is ContextKind.REVERSED:
        shown_items = tuple(reversed(displayed))
    else:
        shown_items, query_ranks = _shuffled_chain(trial, shuffled, rng)
    return replace(
        trial,
        query_ranks=query_ranks,
        context=adjacent_relations(shown_items, rng),
        context_items=shown_items,
        context_kind=kind,
    )


def _shuffled_chain(
    trial: Trial, shuffled: ShuffledQuery | None, rng: np.random.Generator
) -> tuple[tuple[int, ...], tuple[int, int]]:
    """The chain a shuffled context shows and the queried ranks in the order asked."""
    if shuffled is None:
        raise ValueError("a shuffled context needs the shown positions of the query and its agreement")
    items = tuple(int(item) for item in trial.items)
    if len(items) != ITEMS_PER_ORDERING or sorted(items) != sorted(int(item) for item in trial.displayed):
        raise ValueError(f"a shuffled context reorders one whole ordering of {ITEMS_PER_ORDERING} items")
    first, second = (int(position) for position in shuffled.positions)
    if first == second or first not in QUERYABLE_RANKS or second not in QUERYABLE_RANKS:
        raise ValueError(f"the shown positions {tuple(shuffled.positions)} are not two distinct interior positions")
    top, bottom = min(QUERYABLE_RANKS), max(QUERYABLE_RANKS)
    distance = int(rng.integers(1, bottom - top + 1))
    upper = int(rng.integers(top, bottom - distance + 1))
    lower = upper + distance
    # The upper rank is asked first when the shown order puts the first position above the
    # second and agrees with the queried ordering, or puts it below and conflicts with it.
    upper_first = bool(shuffled.agree) == (first < second)
    query_ranks = (upper, lower) if upper_first else (lower, upper)
    queried = (items[query_ranks[0]], items[query_ranks[1]])
    rest = [item for item in items if item not in queried]
    order = iter(rest[int(i)] for i in rng.choice(len(rest), size=len(rest), replace=False))
    placed = {first: queried[0], second: queried[1]}
    shown_items = tuple(placed[position] if position in placed else next(order) for position in range(len(items)))
    return shown_items, query_ranks


def insert_distractors(trial: Trial, rng: np.random.Generator, *, ordering: FixedOrdering) -> Trial:
    """Place distractors by bucket, with the declared number of them between the queried items."""
    n_main = len(ordering.items)
    above, below = sorted(trial.query_ranks)
    between_bucket = above + 1
    # A queried end item keeps a distractor beyond it, so a query never sits at an end of the
    # displayed chain; that leaves one distractor fewer free to fall between the query items.
    forced = [bucket for bucket in (0, n_main) if _beyond_a_queried_end(bucket, above, below, n_main)]
    free = N_DISTRACTORS - len(forced)
    if free < 0:
        raise ValueError("there are too few distractors to keep the query away from the ends")
    room = min(free, len(ordering.buckets[between_bucket]))
    counts = [count for count in sorted(BETWEEN_QUERY_WEIGHTS) if count <= room]
    weights = np.array([BETWEEN_QUERY_WEIGHTS[count] for count in counts], dtype=float)
    n_between = int(rng.choice(counts, p=weights / weights.sum()))

    per_bucket = {bucket: 1 for bucket in forced}
    per_bucket[between_bucket] = n_between
    spare = free - n_between
    if spare:
        elsewhere = [bucket for bucket in range(len(ordering.buckets)) if bucket != between_bucket]
        for bucket in rng.choice(elsewhere, size=spare, replace=False):
            per_bucket[int(bucket)] = per_bucket.get(int(bucket), 0) + 1

    shown: list[int] = []
    distractors: list[int] = []
    for bucket in range(len(ordering.buckets)):
        take = per_bucket.get(bucket, 0)
        if take:
            members = ordering.buckets[bucket]
            chosen = sorted(int(i) for i in rng.choice(len(members), size=take, replace=False))
            shown.extend(members[i] for i in chosen)
            distractors.extend(members[i] for i in chosen)
        if bucket < n_main:
            shown.append(ordering.items[bucket])
    return replace(
        trial,
        displayed=tuple(shown),
        context=adjacent_relations(shown, rng),
        context_items=tuple(shown),
        distractors=tuple(distractors),
    )


def _sign_in(shown_items: Sequence[int], query: tuple[int, int]) -> int:
    """The relation the shown order implies between the query items, 0 if one is absent."""
    position = rank_lookup(shown_items)
    a, b = int(query[0]), int(query[1])
    if a not in position or b not in position:
        return 0
    return sign_from_ranks(position[a], position[b])


def _beyond_a_queried_end(bucket: int, above: int, below: int, n_main: int) -> bool:
    """Whether `bucket` lies beyond a queried item at an end of the fixed ordering."""
    return (bucket == 0 and above == 0) or (bucket == n_main and below == n_main - 1)
