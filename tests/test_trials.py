"""Trials: contexts, queries, endpoints, distractors, the four context types."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import adjacent_pairs as _adjacent_pairs
from conftest import top_down as _top_down

from geo_alg_ti import orderings, trials


def _pretraining(smoke_law, seed=0):
    rng = np.random.default_rng(seed)
    return trials.sample_pretraining_trial(smoke_law, rng, weights=orderings.uniform_cue_weights(6))


def _finetuning(smoke_law, seed=0, n=1):
    rng = np.random.default_rng(seed)
    ordering = trials.build_fixed_ordering(smoke_law, rng, cue=0)
    return ordering, [trials.sample_finetuning_trial(smoke_law, rng, ordering=ordering) for _ in range(n)]


def _with_context(t, kind, rng, smoke_law, *, positions=(2, 5), agree=True):
    """`with_context`, placing the query at `positions` when the context is shuffled."""
    shuffled = trials.ShuffledQuery(positions, agree) if kind is trials.ContextKind.SHUFFLED else None
    return trials.with_context(t, kind, rng, law=smoke_law, shuffled=shuffled)


def _chain_order(relations):
    """The item order a complete adjacent chain implies, or None if the relations are not one."""
    below = {}
    for a, b, s in relations:
        top, bottom = (a, b) if s == 1 else (b, a)
        if top in below:
            return None
        below[top] = bottom
    tops = set(below) - set(below.values())
    if len(tops) != 1:
        return None
    order = [tops.pop()]
    while order[-1] in below:
        order.append(below[order[-1]])
    return tuple(order) if len(order) == len(relations) + 1 else None


def test_context_specifies_the_ordering_exactly(smoke_law):
    """The eight relations form one complete truthful adjacent chain over the nine items."""
    t = _pretraining(smoke_law)
    ranks = {item: r for r, item in enumerate(t.items)}
    assert len(t.context) == 8
    assert all(abs(ranks[a] - ranks[b]) == 1 and s == trials.sign_from_ranks(ranks[a], ranks[b]) for a, b, s in t.context)


def test_context_position_and_orientation_are_randomized(smoke_law):
    """Across many trials both the presented order and the pair orientation vary."""
    positions, top_first = [], []
    for seed in range(400):
        t = _pretraining(smoke_law, seed)
        top_pair = [i for i, r in enumerate(t.context) if t.items[0] in (r.a, r.b)]
        assert len(top_pair) == 1
        positions.append(top_pair[0])
        top_first.append(t.context[top_pair[0]].a == t.items[0])
    assert set(positions) == set(range(8))
    assert 0.35 < np.mean(top_first) < 0.65


def test_endpoints_appear_in_context_but_are_never_queried(smoke_law):
    """Ranks 0 and 8 occur in the context and never in a query."""
    for seed in range(50):
        t = _pretraining(smoke_law, seed)
        assert set(t.query_ranks) <= set(range(1, 8))
        assert t.items[0] in {x for r in t.context for x in (r.a, r.b)}


def test_distance_one_query_is_a_context_edge(smoke_law):
    """An adjacent query's two items appear together in some context pair."""
    adjacent = [t for t in (_pretraining(smoke_law, seed) for seed in range(120)) if t.d == 1]
    assert len(adjacent) > 5
    for t in adjacent:
        assert sum({r.a, r.b} == set(t.query) for r in t.context) == 1


def test_distance_two_or_more_is_never_a_context_edge(smoke_law):
    """No context pair contains both query items when d >= 2."""
    sampled = [_pretraining(smoke_law, seed) for seed in range(120)]
    assert {t.d for t in sampled} == set(range(1, 7))
    for t in sampled:
        if t.d >= 2:
            assert not any({r.a, r.b} == set(t.query) for r in t.context)


def test_distance_equals_the_rank_difference(smoke_law):
    """`d` is the absolute rank difference of the query items in the queried ordering."""
    t = _pretraining(smoke_law)
    assert t.d == abs(t.query_ranks[0] - t.query_ranks[1])


def test_query_orientation_is_balanced_over_many_samples(smoke_law):
    """The higher-ranked item is first in about half of the queries."""
    higher_first = []
    for seed in range(600):
        t = _pretraining(smoke_law, seed)
        assert t.familiar_sign == trials.sign_from_ranks(*t.query_ranks)
        higher_first.append(t.familiar_sign == 1)
    assert 0.4 < np.mean(higher_first) < 0.6


def test_uninformative_context_excludes_queried_and_distractor_items(smoke_law):
    """Filler items are disjoint from the queried ordering and its distractor pool."""
    ordering, sampled = _finetuning(smoke_law, n=20)
    pool = {item for bucket in ordering.buckets for item in bucket}
    rng = np.random.default_rng(1)
    for t in sampled:
        u = trials.with_context(t, trials.ContextKind.UNINFORMATIVE, rng, law=smoke_law, exclude=pool)
        filler = set(u.context_items)
        assert len(filler) == 9 and filler <= set(range(48))
        assert filler.isdisjoint(set(t.items) | set(t.distractors) | pool)
        assert u.query == t.query and u.cue == t.cue and u.items == t.items
    # Even with no explicit exclusion the queried items never appear in the filler.
    for t in sampled:
        u = trials.with_context(t, trials.ContextKind.UNINFORMATIVE, rng, law=smoke_law)
        assert set(u.context_items).isdisjoint(set(t.items) | set(t.distractors))


def test_derived_fields_follow_from_the_ranks_and_the_context(smoke_law):
    """Query, distance, and the three signs derive from the stored fields; none is stored twice."""
    rng = np.random.default_rng(8)
    _, (f,) = _finetuning(smoke_law)
    for t in (f, _pretraining(smoke_law)):
        assert t.query == (t.items[t.query_ranks[0]], t.items[t.query_ranks[1]])
        assert t.d == abs(t.query_ranks[0] - t.query_ranks[1])
        assert t.familiar_sign == trials.sign_from_ranks(*t.query_ranks)
        assert t.shown_sign == t.reference_sign == t.familiar_sign
        u = trials.with_context(t, trials.ContextKind.UNINFORMATIVE, rng, law=smoke_law)
        assert (u.shown_sign, u.reference_sign, u.familiar_sign) == (0, t.familiar_sign, t.familiar_sign)
        r = trials.with_context(t, trials.ContextKind.REVERSED, rng, law=smoke_law)
        assert (r.shown_sign, r.reference_sign, r.familiar_sign) == (-t.familiar_sign, -t.familiar_sign, t.familiar_sign)
    assert set(dir(trials.Trial)) >= {"query", "d", "familiar_sign", "shown_sign", "reference_sign"}
    assert not {"query", "d", "familiar_sign", "shown_sign", "reference_sign"} & {
        field.name for field in trials.Trial.__dataclass_fields__.values()
    }


def test_displayed_is_the_ordering_as_first_shown_distractors_included(smoke_law):
    """`displayed` keeps the first-shown ordering through every change of context."""
    rng = np.random.default_rng(9)
    _, (f,) = _finetuning(smoke_law)
    assert f.displayed == f.context_items and len(f.displayed) == 9
    assert set(f.distractors) <= set(f.displayed) and set(f.items) <= set(f.displayed)
    p = _pretraining(smoke_law)
    assert p.displayed == p.items == p.context_items
    for t in (f, p):
        for kind in trials.ContextKind:
            if kind is not trials.ContextKind.SHUFFLED or t is p:
                assert _with_context(t, kind, rng, smoke_law).displayed == t.displayed
        r = trials.with_context(t, trials.ContextKind.REVERSED, rng, law=smoke_law)
        assert r.context_items == t.displayed[::-1]
        # A reversed trial shown congruently again presents the whole first-shown ordering.
        back = trials.with_context(r, trials.ContextKind.CONGRUENT, rng, law=smoke_law)
        assert back.context_items == t.displayed and back.shown_sign == t.familiar_sign


def test_uninformative_context_has_the_same_length_as_an_informative_one(smoke_law):
    """Eight relations either way."""
    rng = np.random.default_rng(2)
    _, (f,) = _finetuning(smoke_law)
    p = _pretraining(smoke_law)
    for t in (f, p):
        for kind in trials.ContextKind:
            if kind is trials.ContextKind.SHUFFLED and t is f:
                continue
            c = _with_context(t, kind, rng, smoke_law)
            assert len(c.context) == 8 and len(c.context_items) == 9
            assert c.context_kind == kind


def test_uninformative_context_relations_are_true_under_the_shown_cue(smoke_law):
    """Filler relations are a truthful chain under the shown cue's law; only orientation flips."""
    rng = np.random.default_rng(3)
    for seed in range(30):
        t = _pretraining(smoke_law, seed)
        u = trials.with_context(t, trials.ContextKind.UNINFORMATIVE, rng, law=smoke_law)
        assert u.cue == t.cue
        law_order = smoke_law.ordering(u.cue, u.context_items)
        assert _top_down(u.context) == _adjacent_pairs(law_order)
        assert u.shown_sign == 0
        assert u.familiar_sign == t.familiar_sign and u.reference_sign == t.familiar_sign


def test_congruent_context_equals_the_orderings_own_relations(smoke_law):
    """Set equality with the queried ordering's adjacent relations, up to orientation and order."""
    rng = np.random.default_rng(4)
    for seed in range(30):
        t = _pretraining(smoke_law, seed)
        assert t.items == smoke_law.ordering(t.cue, t.items)
        c = trials.with_context(t, trials.ContextKind.CONGRUENT, rng, law=smoke_law)
        assert _top_down(c.context) == _adjacent_pairs(t.items) == _top_down(t.context)
        assert set(c.context_items) == set(t.items)
        assert c.shown_sign == c.familiar_sign == c.reference_sign == t.familiar_sign
    items = smoke_law.ordering(1, smoke_law.sample_items(rng, 9))
    b = trials.sample_battery_trial(
        smoke_law, rng, cue=1, items=items, rank_pair=(2, 5), context_kind=trials.ContextKind.CONGRUENT
    )
    assert b.items == b.context_items == items
    assert _top_down(b.context) == _adjacent_pairs(items)
    assert b.query_ranks == (2, 5) and b.d == 3 and b.shown_sign == b.reference_sign == 1


def test_reversed_context_flips_the_sign_of_every_pair(smoke_law):
    """Every relation implied by the reversed context is the opposite of the familiar one."""
    rng = np.random.default_rng(5)
    for seed in range(30):
        t = _pretraining(smoke_law, seed)
        r = trials.with_context(t, trials.ContextKind.REVERSED, rng, law=smoke_law)
        assert set(r.context_items) == set(t.items)
        assert _chain_order(r.context) == t.items[::-1]
        familiar = smoke_law.ranks(t.cue, t.items)
        for a, b, s in r.context:
            assert s == -trials.sign_from_ranks(familiar[a], familiar[b])
        assert r.shown_sign == -t.familiar_sign and r.reference_sign == r.shown_sign
        assert r.familiar_sign == t.familiar_sign


def _shuffled(t, rng, smoke_law, positions, agree):
    """`t` under a shuffled context that shows its query at `positions`."""
    return trials.with_context(
        t, trials.ContextKind.SHUFFLED, rng, law=smoke_law, shuffled=trials.ShuffledQuery(positions, agree)
    )


INTERIOR_PAIRS = [(p, q) for p in trials.QUERYABLE_RANKS for q in trials.QUERYABLE_RANKS if p != q]


def test_shuffled_context_shows_the_query_at_the_given_interior_positions(smoke_law):
    """The queried items sit at the shown positions in the order asked; the distance is shown."""
    rng = np.random.default_rng(6)
    t = _pretraining(smoke_law)
    for positions in INTERIOR_PAIRS:
        for agree in (True, False):
            s = _shuffled(t, rng, smoke_law, positions, agree)
            assert s.context_kind is trials.ContextKind.SHUFFLED
            assert _chain_order(s.context) == s.context_items
            assert sorted(s.context_items) == sorted(t.items) and len(set(s.context_items)) == 9
            assert (s.items, s.displayed, s.cue) == (t.items, t.displayed, t.cue)
            assert s.shown_positions == positions
            assert (s.context_items[positions[0]], s.context_items[positions[1]]) == s.query
            assert s.query == (s.items[s.query_ranks[0]], s.items[s.query_ranks[1]])
            assert set(s.query_ranks) <= set(trials.QUERYABLE_RANKS)
            assert s.d == abs(positions[0] - positions[1])
            assert s.shown_sign == trials.sign_from_ranks(*positions)
            assert s.familiar_sign == trials.sign_from_ranks(*s.query_ranks)
            assert s.reference_sign == s.shown_sign
            assert (s.shown_sign == s.familiar_sign) == agree


def test_shuffled_familiar_distance_is_uniform_and_independent_of_the_shown_one(smoke_law):
    """The distance in the queried ordering is uniform over one to six whatever the shown one."""
    t = _pretraining(smoke_law)
    pairs = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        for positions in INTERIOR_PAIRS:
            s = _shuffled(t, rng, smoke_law, positions, seed % 2 == 0)
            pairs.append((s.d, abs(s.query_ranks[0] - s.query_ranks[1]), min(s.query_ranks)))
    shown, familiar, upper = (np.array(column) for column in zip(*pairs, strict=True))
    assert set(zip(shown.tolist(), familiar.tolist(), strict=True)) == {(d, k) for d in range(1, 7) for k in range(1, 7)}
    counts = np.bincount(familiar, minlength=7)[1:]
    assert np.allclose(counts / counts.sum(), 1 / 6, atol=0.02)
    # Given the familiar distance, the upper rank spans the ranks that keep both queryable.
    for k in range(1, 7):
        assert set(upper[familiar == k].tolist()) == set(range(1, 8 - k))
    # At every shown distance the familiar distance averages near 3.5, the uniform mean.
    for d in range(1, 7):
        at_d = familiar[shown == d]
        assert abs(at_d.mean() - 3.5) < 0.35, d


def test_shuffled_context_fills_the_other_positions_at_random(smoke_law):
    """Every item outside the query reaches every free position; orientations are random."""
    t = _pretraining(smoke_law)
    positions = (3, 6)
    reached: dict[int, set[int]] = {}
    top_first = []
    for seed in range(400):
        s = _shuffled(t, np.random.default_rng(seed), smoke_law, positions, True)
        for at, item in enumerate(s.context_items):
            if at not in positions:
                reached.setdefault(item, set()).add(at)
        top_first.extend(relation.sign == 1 for relation in s.context)
    free = set(range(9)) - set(positions)
    assert len(reached) == 9 and all(spots == free for spots in reached.values())
    assert 0.45 < np.mean(top_first) < 0.55


def test_shuffled_context_refuses_what_it_cannot_place(smoke_law):
    """No placement, a placement at an end, one position twice, or a partial ordering is refused."""
    rng = np.random.default_rng(7)
    t = _pretraining(smoke_law)
    with pytest.raises(ValueError, match="shown positions of the query"):
        trials.with_context(t, trials.ContextKind.SHUFFLED, rng, law=smoke_law)
    for positions in ((0, 3), (3, 8), (4, 4)):
        with pytest.raises(ValueError, match="not two distinct interior positions"):
            _shuffled(t, rng, smoke_law, positions, True)
    _, (f,) = _finetuning(smoke_law)
    with pytest.raises(ValueError, match="one whole ordering"):
        _shuffled(f, rng, smoke_law, (2, 5), True)
    with pytest.raises(ValueError, match="only a shuffled context"):
        trials.with_context(t, trials.ContextKind.REVERSED, rng, law=smoke_law, shuffled=trials.ShuffledQuery((2, 5), True))


def test_contexts_other_than_shuffled_draw_nothing_for_the_query(smoke_law):
    """Under every other context the query ranks and their distance are those of the trial."""
    rng = np.random.default_rng(8)
    t = _pretraining(smoke_law)
    for kind in (trials.ContextKind.UNINFORMATIVE, trials.ContextKind.CONGRUENT, trials.ContextKind.REVERSED):
        c = trials.with_context(t, kind, rng, law=smoke_law)
        assert c.query_ranks == t.query_ranks and c.d == abs(t.query_ranks[0] - t.query_ranks[1])


def _between_counts(smoke_law, n):
    """Per trial: whether the query touches an end of the fixed ordering, and the count between."""
    ordering, sampled = _finetuning(smoke_law, n=n)
    main_rank = {item: r for r, item in enumerate(ordering.items)}
    out = []
    for t in sampled:
        shown = {item: position for position, item in enumerate(t.context_items)}
        lo, hi = sorted(shown[q] for q in t.query)
        between = sum(lo < shown[x] < hi for x in t.distractors)
        at_end = min(main_rank[q] for q in t.query) == 0 or max(main_rank[q] for q in t.query) == 6
        out.append((at_end, between))
    return out


def test_distractor_counts_between_the_query_follow_the_declared_weights(smoke_law):
    """Zero, one, or two distractors fall between the query items with probability .10, .45, .45."""
    interior = [b for at_end, b in _between_counts(smoke_law, 3000) if not at_end]
    assert len(interior) > 1500
    freq = np.bincount(interior, minlength=3) / len(interior)
    assert np.allclose(freq, [0.10, 0.45, 0.45], atol=0.04)


def test_distractor_weights_are_truncated_and_renormalized_at_the_ends(smoke_law):
    """A query at the end cannot host two distractors between; the weights renormalize."""
    at_end = [b for is_end, b in _between_counts(smoke_law, 3000) if is_end]
    assert len(at_end) > 600
    assert max(at_end) == 1
    freq = np.bincount(at_end, minlength=2) / len(at_end)
    assert np.allclose(freq, [0.10 / 0.55, 0.45 / 0.55], atol=0.04)


def test_distractors_keep_the_displayed_chain_at_nine_items(smoke_law):
    """Seven main items plus two distractors are shown."""
    rng = np.random.default_rng(0)
    ordering = trials.build_fixed_ordering(smoke_law, rng, cue=0)
    t = trials.sample_finetuning_trial(smoke_law, rng, ordering=ordering)
    assert len(t.context_items) == 9 and len(t.distractors) == 2


def test_finetuning_queries_are_adjacent_in_the_trained_ordering(smoke_law):
    """Every finetuning query has distance one in the trained ordering."""
    rng = np.random.default_rng(0)
    ordering = trials.build_fixed_ordering(smoke_law, rng, cue=0)
    for _ in range(20):
        assert trials.sample_finetuning_trial(smoke_law, rng, ordering=ordering).d == 1


def test_distractor_buckets_cover_every_gap_and_both_ends(smoke_law):
    """Eight buckets of two distinct distractors each, none of them a main item."""
    rng = np.random.default_rng(0)
    ordering = trials.build_fixed_ordering(smoke_law, rng, cue=0)
    assert len(ordering.buckets) == 8 and all(len(b) == 2 for b in ordering.buckets)
    pool = [item for bucket in ordering.buckets for item in bucket]
    assert len(set(pool)) == trials.DISTRACTOR_POOL_SIZE == 16
    assert set(pool).isdisjoint(ordering.items) and len(ordering.items) == 7
    assert ordering.items == smoke_law.ordering(0, ordering.items)


def _bucket_of(ordering, item):
    """The index of the bucket holding a distractor."""
    return next(index for index, bucket in enumerate(ordering.buckets) if item in bucket)


def test_distractors_are_shown_at_their_buckets_positions(smoke_law):
    """A distractor sits between the main items its bucket separates; the main items keep order."""
    ordering, sampled = _finetuning(smoke_law, n=300)
    main = ordering.items
    for t in sampled:
        shown = t.context_items
        assert len(shown) == 9 and len(set(shown)) == 9 and len(t.distractors) == 2
        assert tuple(item for item in shown if item in main) == main
        assert _top_down(t.context) == _adjacent_pairs(shown)
        for distractor in t.distractors:
            bucket = _bucket_of(ordering, distractor)
            at = shown.index(distractor)
            assert bucket == 0 or shown.index(main[bucket - 1]) < at
            assert bucket == 7 or at < shown.index(main[bucket])


def test_bucket_membership_comes_from_the_deal_not_from_the_law(smoke_law):
    """A distractor's bucket comes from a shuffle; the cue's law would mostly put it elsewhere."""
    placed_by_law = []
    for seed in range(30):
        ordering = trials.build_fixed_ordering(smoke_law, np.random.default_rng(seed), cue=seed % 6)
        for bucket, members in enumerate(ordering.buckets):
            for distractor in members:
                gap = smoke_law.rank_of(ordering.cue, (*ordering.items, distractor), distractor)
                placed_by_law.append(gap == bucket)
    assert len(placed_by_law) == 30 * 16
    assert np.mean(placed_by_law) < 0.5


def test_arrange_fixed_ordering_deals_a_full_disjoint_pool(smoke_law):
    """The pool is dealt whole, at random, and never overlaps the main items."""
    rng = np.random.default_rng(0)
    main = smoke_law.sample_items(rng, 7)
    pool = smoke_law.sample_items(rng, 16, exclude=main)
    dealt = [
        trials.arrange_fixed_ordering(smoke_law, np.random.default_rng(seed), cue=2, main=main, distractors=pool)
        for seed in range(3)
    ]
    for ordering in dealt:
        assert ordering.items == smoke_law.ordering(2, main)
        assert sorted(item for bucket in ordering.buckets for item in bucket) == sorted(pool)
    assert dealt[0].buckets != dealt[1].buckets
    again = trials.arrange_fixed_ordering(smoke_law, np.random.default_rng(0), cue=2, main=main, distractors=pool)
    assert again == dealt[0]
    with pytest.raises(ValueError):
        trials.arrange_fixed_ordering(smoke_law, rng, cue=2, main=main, distractors=pool[:15])
    with pytest.raises(ValueError):
        trials.arrange_fixed_ordering(smoke_law, rng, cue=2, main=main, distractors=(*pool[:15], pool[0]))
    with pytest.raises(ValueError):
        trials.arrange_fixed_ordering(smoke_law, rng, cue=2, main=main, distractors=(*pool[:15], main[0]))


def test_pretraining_refuses_a_reserved_cue(smoke_law):
    """Weights reaching into the reserved block are refused, so the holdout cannot be trained on."""
    weights = np.zeros(smoke_law.n_training_cues + smoke_law.n_reserved_cues)
    weights[smoke_law.n_training_cues] = 1.0
    with pytest.raises(ValueError):
        trials.sample_pretraining_trial(smoke_law, np.random.default_rng(0), weights=weights)
    # The same draw restricted to the training cues goes through.
    assert not smoke_law.is_reserved(_pretraining(smoke_law).cue)


def test_pretraining_trials_carry_no_distractors(smoke_law):
    """The extended methods' asymmetry: distractors only in finetuning."""
    assert _pretraining(smoke_law).distractors == ()
