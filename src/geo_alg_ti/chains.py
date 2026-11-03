"""Chain-of-thought generation and its validator.

Realizes the extended methods' "Chain-of-thought generation" paragraph and its
algorithm. The main algorithm starts at the first query item; if a context
pair contains both query items it emits nothing and answers from that pair;
otherwise it takes the earliest presented pair containing the start item and
walks one premise per step, emitting each traversed context pair in its
presented orientation, until it reaches the other query item (the answer is
the last relation) or a dead end at an endpoint (the answer is the opposite).
Each successive pair is obtained from the context alone, without consulting
the latent ranks.

The three appendix variants are parameters of the same walk: `inferred_relations`
emits the accumulated relation between the start item and the current
frontier instead of the traversed pair; `descend_first` chooses the first
direction from the latent ranks, toward the lower-ranked neighbor, and
dead-ends when that guess is wrong; `backtrack` returns to the start item
after a dead end and walks the untaken direction until it reaches the other
query item. Every algorithm emits the same `(a, b, sign)` shape, so the
encoder is algorithm-agnostic; the encoder pads every chain to `PAD_TO`
emissions by repeating the answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from geo_alg_ti.trials import Relation, Trial, rank_lookup, sign_from_ranks

PAD_TO: int = 9


class ChainAlgorithm(StrEnum):
    """The chain-generation algorithms compared in the appendix."""

    MAIN = "main"
    INFERRED_RELATIONS = "inferred_relations"
    DESCEND_FIRST = "descend_first"
    BACKTRACK = "backtrack"


@dataclass(frozen=True)
class Chain:
    """A generated chain: intermediate emissions, the answer relation, and how it was reached."""

    emissions: tuple[Relation, ...]
    answer: Relation
    reached_query_item: bool


@dataclass(frozen=True)
class ChainValidity:
    """Per-emission validity and the index of the first invalid emission, if any."""

    per_emission: tuple[bool, ...]
    ok: bool
    first_invalid_index: int | None


def generate_chain(trial: Trial, algorithm: ChainAlgorithm = ChainAlgorithm.MAIN) -> Chain:
    """Run the named algorithm on the presented context."""
    emissions, answer, reached = _run(trial, ChainAlgorithm(algorithm))
    return Chain(emissions=emissions, answer=answer, reached_query_item=reached)


def canonical_chain_length(trial: Trial, algorithm: ChainAlgorithm) -> int:
    """Return the number of emissions the algorithm produces for this trial before the answer."""
    return len(_run(trial, ChainAlgorithm(algorithm))[0])


def emission_validity(emissions: Sequence[Relation], trial: Trial) -> tuple[bool, ...]:
    """Score each emission against the facts available when it is written.

    An emission (a, b, s) is valid when the relation is available and its sign is true. It is
    available when {a, b} is one of the presented relations, or when some third item is already
    related to both a and b, whichever side of each it lies on, so the pair is one step from the
    known facts. The sign is true when s is the relation between a and b in the ordering the
    context shows. The available facts start as the presented relations and grow by every
    preceding emission.
    """
    ranks = rank_lookup(trial.context_items)
    known = {frozenset((rel.a, rel.b)) for rel in trial.context}
    neighbors: dict[int, set[int]] = {}
    for rel in trial.context:
        neighbors.setdefault(rel.a, set()).add(rel.b)
        neighbors.setdefault(rel.b, set()).add(rel.a)

    flags: list[bool] = []
    for a, b, sign in emissions:
        a, b = int(a), int(b)
        pair = frozenset((a, b))
        # A new pair is available only by combining two known facts that share an item.
        derivable = a != b and (
            pair in known
            or any(a in known_of and b in known_of for known_of in neighbors.values())
        )
        truthful = a in ranks and b in ranks and a != b and sign == sign_from_ranks(ranks[a], ranks[b])
        flags.append(bool(derivable and truthful))
        known.add(pair)
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    return tuple(flags)


def validate_chain(emissions: Sequence[Relation], trial: Trial, *, truncate_to: int | None = None) -> ChainValidity:
    """Score a chain's emissions, or just the first `truncate_to` (the truncated measure)."""
    scored = emissions if truncate_to is None else emissions[: max(int(truncate_to), 0)]
    flags = emission_validity(scored, trial)
    first_invalid = next((index for index, ok in enumerate(flags) if not ok), None)
    return ChainValidity(per_emission=flags, ok=first_invalid is None, first_invalid_index=first_invalid)


def _run(trial: Trial, algorithm: ChainAlgorithm) -> tuple[tuple[Relation, ...], Relation, bool]:
    """The emissions, the answer, and whether the walk reached the other query item."""
    start, goal = int(trial.query[0]), int(trial.query[1])
    context = trial.context
    by_item = _relations_by_item(context)

    direct = next((rel for rel in context if {rel.a, rel.b} == {start, goal}), None)
    if direct is not None:
        return (), Relation(start, goal, _sign_from(direct, start)), True

    if start not in by_item:
        raise ValueError("the first query item does not appear in the context")
    first = _first_step(start, by_item[start], context, algorithm, trial.items)
    emissions, reached, direction = _walk(start, first, goal, context, by_item, algorithm)

    if not reached and algorithm is ChainAlgorithm.BACKTRACK:
        # The second leg starts over at the query item and walks the untaken direction.
        untaken = [index for index in by_item[start] if index != first]
        if not untaken:
            raise ValueError("the walk dead-ended and the start item has no second premise")
        back, reached, direction = _walk(start, untaken[0], goal, context, by_item, algorithm)
        emissions = emissions + back

    return emissions, Relation(start, goal, direction if reached else -direction), reached


def _relations_by_item(context: Sequence[Relation]) -> dict[int, list[int]]:
    """Item to the indices of the context relations containing it, in presentation order."""
    by_item: dict[int, list[int]] = {}
    for index, rel in enumerate(context):
        by_item.setdefault(int(rel.a), []).append(index)
        by_item.setdefault(int(rel.b), []).append(index)
    return by_item


def _sign_from(rel: Relation, item: int) -> int:
    """The relation's sign oriented from `item`: +1 when `item` ranks above the other item."""
    return int(rel.sign) if int(rel.a) == int(item) else -int(rel.sign)


def _other(rel: Relation, item: int) -> int:
    """The item the relation pairs with `item`."""
    return int(rel.b) if int(rel.a) == int(item) else int(rel.a)


def _first_step(
    start: int,
    options: Sequence[int],
    context: Sequence[Relation],
    algorithm: ChainAlgorithm,
    items: Sequence[int],
) -> int:
    """Index of the context relation the first step traverses."""
    if algorithm is ChainAlgorithm.DESCEND_FIRST:
        # The descending guess comes from the queried ordering, not from the presented signs.
        ranks = rank_lookup(items)
        downward = [index for index in options if _ranks_below(_other(context[index], start), start, ranks)]
        return downward[0] if downward else options[0]
    return options[0]


def _ranks_below(item: int, start: int, ranks: Mapping[int, int]) -> bool:
    """Whether `item` sits below `start` in the queried ordering."""
    return item in ranks and start in ranks and ranks[item] > ranks[start]


def _walk(
    start: int,
    first: int,
    goal: int,
    context: Sequence[Relation],
    by_item: dict[int, list[int]],
    algorithm: ChainAlgorithm,
) -> tuple[tuple[Relation, ...], bool, int]:
    """Follow the context chain one premise at a time, emitting as the algorithm prescribes."""
    direction = _sign_from(context[first], start)
    emissions: list[Relation] = []
    previous, edge = start, first
    while True:
        current = _other(context[edge], previous)
        if algorithm is ChainAlgorithm.INFERRED_RELATIONS:
            emissions.append(Relation(start, current, direction))
        else:
            emissions.append(context[edge])
        if current == goal:
            return tuple(emissions), True, direction
        onward = [index for index in by_item[current] if _other(context[index], current) != previous]
        if not onward:
            return tuple(emissions), False, direction
        previous, edge = current, onward[0]
