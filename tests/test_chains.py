"""Chain generation for the four algorithms and the validator."""

from __future__ import annotations

import numpy as np
from conftest import verbatim_trial as _trial

from geo_alg_ti import chains, trials

R = trials.Relation
ALGORITHMS = tuple(chains.ChainAlgorithm)

# The extended methods' worked example: i0 > i1 > ... > i8, query (i3, i1), the pair {i2, i3}
# presented before {i3, i4}, so the walk from i3 goes up through i2 to i1.
EXAMPLE_CONTEXT = (R(8, 7, -1), R(2, 3, 1), R(0, 1, 1), R(3, 4, 1), R(2, 1, -1), R(6, 7, 1), R(5, 6, 1), R(5, 4, -1))


def _first(context, *pairs):
    """`context` reordered so the given unordered pairs come first, in the order given."""
    key = {frozenset(p): i for i, p in enumerate(pairs)}
    return tuple(sorted(context, key=lambda r: key.get(frozenset((r.a, r.b)), len(pairs))))


def _grid(n_shuffles=12, items=(14, 3, 27, 8, 41, 0, 19, 33, 11)):
    """Every ordered pair of queryable ranks under many shuffled, randomly oriented contexts."""
    rng = np.random.default_rng(0)
    for x in trials.QUERYABLE_RANKS:
        for z in trials.QUERYABLE_RANKS:
            if x != z:
                for _ in range(n_shuffles):
                    yield _trial(trials.adjacent_relations(items, rng), (items[x], items[z]), items)


def _away(trial):
    """Steps from the first query item to the endpoint that lies away from the second."""
    x, z = trial.query_ranks
    return x if z > x else len(trial.items) - 1 - x


def test_main_algorithm_reproduces_the_methods_worked_example():
    """For the extended methods' example the chain is `i2 i3 +1`, `i2 i1 -1`, answer `i3 i1 -1`."""
    chain = chains.generate_chain(_trial(EXAMPLE_CONTEXT, (3, 1)))
    assert chain.emissions == (R(2, 3, 1), R(2, 1, -1))
    assert chain.answer == R(3, 1, -1)
    assert chain.reached_query_item and len(chain.emissions) == 2
    # The walk went from i3 through i2 to i1: the first emission touches i2, the last touches i1.
    assert 2 in chain.emissions[0][:2] and 1 in chain.emissions[-1][:2]


def test_adjacent_query_emits_no_intermediate_relation():
    """A distance-one query emits nothing and answers from the context pair holding both items."""
    for query, sign in (((3, 4), 1), ((4, 3), -1), ((1, 2), 1), ((7, 6), -1)):
        trial = _trial(EXAMPLE_CONTEXT, query)
        for algorithm in ALGORITHMS:
            chain = chains.generate_chain(trial, algorithm)
            assert chain.emissions == () and chain.answer == R(*query, sign)
            assert chain.reached_query_item
            assert chains.canonical_chain_length(trial, algorithm) == 0


def test_walk_reaching_the_other_query_item_answers_with_the_last_relation():
    """When the walk reaches z the answer is the relation emitted at the final step."""
    down = _first(EXAMPLE_CONTEXT, (2, 3))
    chain = chains.generate_chain(_trial(down, (2, 5)))
    assert chain.emissions == (R(2, 3, 1), R(3, 4, 1), R(5, 4, -1))
    assert chain.reached_query_item and chain.answer == R(2, 5, 1)
    # The final step went from i4 down to i5; that direction, from i2, is the answer.
    up = _first(EXAMPLE_CONTEXT, (5, 4))
    chain = chains.generate_chain(_trial(up, (5, 2)))
    assert chain.emissions == (R(5, 4, -1), R(3, 4, 1), R(2, 3, 1))
    assert chain.reached_query_item and chain.answer == R(5, 2, -1)


def test_dead_end_walk_answers_with_the_opposite_relation():
    """At a dead end the answer is the opposite of the final relation and the flag is false."""
    up = _first(EXAMPLE_CONTEXT, (2, 1))
    chain = chains.generate_chain(_trial(up, (2, 5)))
    assert chain.emissions == (R(2, 1, -1), R(0, 1, 1))
    assert not chain.reached_query_item and chain.answer == R(2, 5, 1)
    down = _first(EXAMPLE_CONTEXT, (6, 7))
    chain = chains.generate_chain(_trial(down, (6, 3)))
    assert chain.emissions == (R(6, 7, 1), R(8, 7, -1))
    assert not chain.reached_query_item and chain.answer == R(6, 3, -1)


def test_exhaustive_main_algorithm_correctness():
    """Every rank pair, both orientations, many shuffles: the answer matches the true sign."""
    for trial in _grid():
        chain = chains.generate_chain(trial)
        assert chain.answer == R(*trial.query, trial.familiar_sign)
        assert len(chain.emissions) == chains.canonical_chain_length(trial, chains.ChainAlgorithm.MAIN)
        if trial.d == 1:
            assert chain.emissions == () and chain.reached_query_item
        elif chain.reached_query_item:
            assert len(chain.emissions) == trial.d
        else:
            assert len(chain.emissions) == _away(trial)
        # Emissions are traversed context pairs, in their presented orientation, one premise each.
        assert all(e in trial.context for e in chain.emissions)
        assert len(set(chain.emissions)) == len(chain.emissions)


def test_every_algorithm_answers_correctly_on_every_pair():
    """All four variants answer correctly over the same exhaustive grid."""
    for trial in _grid(n_shuffles=6):
        x, z = trial.query_ranks
        for algorithm in ALGORITHMS:
            chain = chains.generate_chain(trial, algorithm)
            assert chain.answer == R(*trial.query, trial.familiar_sign)
            n = len(chain.emissions)
            assert n == chains.canonical_chain_length(trial, algorithm)
            if trial.d == 1:
                assert n == 0
            elif algorithm is chains.ChainAlgorithm.DESCEND_FIRST:
                assert chain.reached_query_item == (z > x) and n == (trial.d if z > x else _away(trial))
            elif algorithm is chains.ChainAlgorithm.BACKTRACK:
                assert chain.reached_query_item and n in (trial.d, trial.d + _away(trial))
            else:
                assert n == (trial.d if chain.reached_query_item else _away(trial))


def test_every_algorithm_emits_relation_triples_of_one_shape():
    """Every emission is a `Relation`, so the encoder is algorithm-agnostic."""
    for trial in _grid(n_shuffles=2):
        for algorithm in ALGORITHMS:
            chain = chains.generate_chain(trial, algorithm)
            for e in (*chain.emissions, chain.answer):
                assert type(e) is trials.Relation and len(e) == 3
                assert e.sign in (-1, 1) and e.a in trial.context_items and e.b in trial.context_items
                assert e.a != e.b
            assert (chain.answer.a, chain.answer.b) == trial.query


def test_inferred_relations_emits_facts_anchored_at_the_start_item():
    """Each emission relates the start item to the current frontier."""
    inferred = chains.ChainAlgorithm.INFERRED_RELATIONS
    chain = chains.generate_chain(_trial(EXAMPLE_CONTEXT, (3, 1)), inferred)
    assert chain.emissions == (R(3, 2, -1), R(3, 1, -1)) and chain.answer == R(3, 1, -1)
    up = _first(EXAMPLE_CONTEXT, (2, 1))
    chain = chains.generate_chain(_trial(up, (2, 5)), inferred)
    assert chain.emissions == (R(2, 1, -1), R(2, 0, -1)) and chain.answer == R(2, 5, 1)
    assert not chain.reached_query_item
    for trial in _grid(n_shuffles=3):
        chain = chains.generate_chain(trial, inferred)
        main = chains.generate_chain(trial, chains.ChainAlgorithm.MAIN)
        assert len(chain.emissions) == len(main.emissions)
        rank = {item: r for r, item in enumerate(trial.items)}
        frontier = trial.query[0]
        for e, step in zip(chain.emissions, main.emissions):
            frontier = step.b if step.a == frontier else step.a
            assert e == R(trial.query[0], frontier, trials.sign_from_ranks(rank[e.a], rank[e.b]))


def test_descend_first_uses_the_rank_determined_first_direction():
    """The first step goes to the lower-ranked neighbor of the start item."""
    descend = chains.ChainAlgorithm.DESCEND_FIRST
    # The main walk would go up through {i2, i3}; descending goes down and dead-ends at i8.
    chain = chains.generate_chain(_trial(EXAMPLE_CONTEXT, (3, 1)), descend)
    assert chain.emissions == (R(3, 4, 1), R(5, 4, -1), R(5, 6, 1), R(6, 7, 1), R(8, 7, -1))
    assert not chain.reached_query_item and chain.answer == R(3, 1, -1)
    chain = chains.generate_chain(_trial(EXAMPLE_CONTEXT, (3, 6)), descend)
    assert chain.emissions == (R(3, 4, 1), R(5, 4, -1), R(5, 6, 1))
    assert chain.reached_query_item and chain.answer == R(3, 6, 1)
    for trial in _grid(n_shuffles=3):
        if trial.d >= 2:
            first = chains.generate_chain(trial, descend).emissions[0]
            assert {first.a, first.b} == {trial.query[0], trial.items[trial.query_ranks[0] + 1]}


# The same nine items shown in the opposite order, so presented signs and ranks disagree.
REVERSED_CONTEXT = tuple(R(a, a - 1, 1) for a in range(8, 0, -1))


def _reversed_trial(query, items=tuple(range(9))):
    """A trial whose context shows `items` in the reverse of the queried ordering."""
    rank = {item: r for r, item in enumerate(items)}
    trial = trials.Trial(
        cue=7,
        items=tuple(items),
        query_ranks=(rank[query[0]], rank[query[1]]),
        displayed=tuple(items),
        context=REVERSED_CONTEXT,
        context_items=tuple(reversed(items)),
        context_kind=trials.ContextKind.REVERSED,
    )
    assert trial.shown_sign == trial.reference_sign == -trial.familiar_sign
    return trial


def test_descend_first_takes_its_direction_from_the_ordering_not_the_context():
    """Under a reversed context the first step still goes to the item ranked below the query."""
    descend = chains.ChainAlgorithm.DESCEND_FIRST
    trial = _reversed_trial((3, 1))
    chain = chains.generate_chain(trial, descend)
    # i4 sits below i3 in the queried ordering, though the context presents i4 above it.
    assert chain.emissions == (R(4, 3, 1), R(5, 4, 1), R(6, 5, 1), R(7, 6, 1), R(8, 7, 1))
    assert chains.canonical_chain_length(trial, descend) == 5
    assert not chain.reached_query_item and chain.answer == R(3, 1, trial.shown_sign)
    # Following the presented signs instead would have descended toward i2 and stopped at i1.
    assert [e for e in trial.context if 2 in (e.a, e.b) and 3 in (e.a, e.b)] == [R(3, 2, 1)]


def test_backtrack_resumes_after_a_dead_end():
    """After a dead end the chain returns to the start item and walks the other direction to z."""
    backtrack = chains.ChainAlgorithm.BACKTRACK
    up = _first(EXAMPLE_CONTEXT, (2, 1))
    trial = _trial(up, (2, 5))
    dead_end = chains.generate_chain(trial, chains.ChainAlgorithm.MAIN)
    assert not dead_end.reached_query_item
    chain = chains.generate_chain(trial, backtrack)
    assert chain.emissions == (R(2, 1, -1), R(0, 1, 1), R(2, 3, 1), R(3, 4, 1), R(5, 4, -1))
    assert chain.reached_query_item and chain.answer == R(2, 5, 1)
    # When the first direction already reaches z, backtracking coincides with the main walk.
    down = _first(EXAMPLE_CONTEXT, (2, 3))
    assert chains.generate_chain(_trial(down, (2, 5)), backtrack).emissions == (R(2, 3, 1), R(3, 4, 1), R(5, 4, -1))


def test_chain_length_never_exceeds_the_pad_budget():
    """Emissions plus the answer fit within nine."""
    longest = 0
    for trial in _grid(n_shuffles=4):
        for algorithm in ALGORITHMS:
            chain = chains.generate_chain(trial, algorithm)
            assert len(chain.emissions) + 1 <= chains.PAD_TO
            longest = max(longest, len(chain.emissions))
    # The longest walk, both legs of a backtracking chain from a rank next to an end, is seven.
    assert longest == 7


def test_validator_accepts_canonical_chains_for_every_algorithm():
    """Every emission of a generated chain is valid."""
    for trial in _grid(n_shuffles=3):
        for algorithm in ALGORITHMS:
            chain = chains.generate_chain(trial, algorithm)
            validity = chains.validate_chain(chain.emissions, trial)
            assert validity.ok and validity.first_invalid_index is None
            assert validity.per_emission == (True,) * len(chain.emissions)
            assert chains.emission_validity(chain.emissions, trial) == validity.per_emission


def test_validator_rejects_a_flipped_sign_and_reports_its_index():
    """Flipping one sign makes that emission invalid and `first_invalid_index` points at it."""
    trial = _trial(_first(EXAMPLE_CONTEXT, (2, 1)), (2, 5))
    for algorithm in ALGORITHMS:
        emissions = chains.generate_chain(trial, algorithm).emissions
        for i, e in enumerate(emissions):
            flipped = (*emissions[:i], R(e.a, e.b, -e.sign), *emissions[i + 1 :])
            validity = chains.validate_chain(flipped, trial)
            assert not validity.ok and validity.first_invalid_index == i
            assert validity.per_emission[i] is False
            assert all(validity.per_emission[:i])
    # A true relation that is neither a context edge nor derivable from known facts is invalid.
    stray = chains.validate_chain((R(1, 8, 1),), trial)
    assert not stray.ok and stray.first_invalid_index == 0
    assert chains.validate_chain((R(2, 1, -1), R(1, 8, 1)), trial).first_invalid_index == 1


def test_validator_requires_a_connecting_item_not_merely_a_shared_one():
    """A pair is derivable only when a third item is already related to both of its items."""
    trial = _trial(EXAMPLE_CONTEXT, (3, 1))
    # Both i1 and i5 appear in the context and i1 does rank above i5, but nothing yet connects them.
    assert chains.emission_validity((R(1, 5, 1),), trial) == (False,)
    # Composing {i1, i2} with {i2, i3} gives i1 above i3: i2 is the item they have in common.
    assert chains.emission_validity((R(1, 3, 1),), trial) == (True,)
    # Once i1 and i5 are both related to i3, their own relation follows from those two facts.
    assert chains.emission_validity((R(1, 3, 1), R(3, 5, 1), R(1, 5, 1)), trial) == (True, True, True)


def test_validator_truncated_mode_scores_only_the_first_k_emissions():
    """With `truncate_to`, emissions beyond it are not scored."""
    trial = _trial(_first(EXAMPLE_CONTEXT, (2, 1)), (2, 5))
    emissions = chains.generate_chain(trial, chains.ChainAlgorithm.BACKTRACK).emissions
    assert len(emissions) == 5
    last = emissions[-1]
    damaged = (*emissions[:-1], R(last.a, last.b, -last.sign))
    assert not chains.validate_chain(damaged, trial).ok
    for k in range(5):
        validity = chains.validate_chain(damaged, trial, truncate_to=k)
        assert validity.ok and validity.per_emission == (True,) * k
    assert chains.validate_chain(damaged, trial, truncate_to=5).first_invalid_index == 4
    assert chains.validate_chain(damaged, trial, truncate_to=8).per_emission == (True,) * 4 + (False,)
    assert len(chains.validate_chain(emissions, trial, truncate_to=8).per_emission) == 5


def test_dead_end_and_reached_cases_both_appear_in_a_random_sample():
    """The fixture set is not degenerate: both branches occur."""
    flags = {chains.generate_chain(t).reached_query_item for t in _grid(n_shuffles=3) if t.d >= 2}
    assert flags == {True, False}
    reached = [chains.generate_chain(t).reached_query_item for t in _grid(n_shuffles=12) if t.d >= 2]
    assert 0.4 < np.mean(reached) < 0.6


def test_algorithm_enum_names_match_the_configuration():
    """The enum values are the strings `configs/base.yaml` accepts."""
    assert {a.value for a in chains.ChainAlgorithm} == {"main", "inferred_relations", "descend_first", "backtrack"}
