"""The cued-ordering law and the cue-frequency distributions."""

from __future__ import annotations

import itertools
import subprocess
import sys

import numpy as np
import pytest

from geo_alg_ti import orderings

# The nine items `sample_items` drew at seeds 0..49 from the smoke-scale pool of 48, first with
# nothing excluded and then excluding SAMPLE_EXCLUDED, pinned so the draw never moves.
SAMPLE_EXCLUDED = (3, 11, 19, 27, 35, 43, 7, 15, 23)
SAMPLED_PLAIN = (
    (34, 26, 21, 13, 8, 11, 3, 1, 0),
    (18, 44, 40, 6, 1, 20, 11, 37, 31),
    (4, 10, 18, 12, 33, 46, 16, 20, 36),
    (43, 36, 1, 3, 27, 39, 10, 7, 32),
    (47, 38, 29, 44, 41, 21, 3, 37, 43),
    (33, 20, 23, 47, 28, 34, 13, 0, 26),
    (16, 46, 17, 30, 14, 21, 41, 22, 47),
    (38, 43, 10, 25, 37, 34, 45, 2, 28),
    (7, 30, 14, 13, 29, 42, 9, 28, 37),
    (35, 16, 12, 40, 5, 27, 47, 36, 30),
    (47, 23, 37, 7, 8, 11, 31, 39, 34),
    (27, 23, 25, 33, 5, 1, 32, 21, 40),
    (9, 2, 24, 40, 46, 8, 10, 27, 42),
    (34, 40, 8, 3, 44, 36, 12, 43, 35),
    (3, 40, 27, 47, 16, 6, 15, 34, 31),
    (37, 29, 10, 9, 35, 21, 28, 2, 15),
    (16, 0, 34, 24, 33, 23, 21, 18, 4),
    (17, 29, 1, 20, 35, 25, 4, 6, 34),
    (35, 38, 16, 8, 3, 45, 30, 12, 46),
    (2, 21, 16, 17, 39, 40, 12, 23, 14),
    (39, 11, 47, 24, 5, 45, 35, 10, 19),
    (4, 14, 12, 26, 16, 32, 15, 20, 31),
    (8, 30, 27, 3, 0, 46, 12, 15, 41),
    (11, 45, 27, 1, 28, 17, 46, 47, 5),
    (31, 19, 15, 13, 23, 5, 17, 34, 25),
    (7, 9, 0, 35, 17, 24, 6, 10, 20),
    (17, 20, 34, 23, 16, 3, 36, 10, 33),
    (15, 8, 27, 13, 0, 43, 5, 28, 20),
    (38, 6, 44, 34, 36, 1, 9, 35, 26),
    (21, 23, 10, 30, 27, 25, 37, 12, 2),
    (17, 31, 44, 18, 11, 4, 29, 9, 27),
    (20, 22, 25, 2, 17, 37, 21, 30, 3),
    (35, 6, 24, 15, 16, 18, 31, 36, 13),
    (13, 35, 18, 36, 24, 25, 11, 15, 40),
    (37, 45, 30, 0, 47, 2, 10, 4, 5),
    (19, 11, 44, 13, 42, 34, 1, 41, 4),
    (44, 15, 19, 7, 17, 40, 18, 42, 43),
    (28, 13, 33, 6, 11, 3, 42, 41, 26),
    (21, 32, 29, 13, 17, 10, 9, 19, 4),
    (9, 20, 33, 46, 39, 14, 38, 23, 6),
    (42, 44, 29, 2, 0, 19, 47, 22, 1),
    (31, 26, 16, 33, 11, 5, 34, 39, 38),
    (9, 31, 38, 27, 18, 32, 45, 19, 3),
    (25, 16, 20, 12, 1, 22, 26, 39, 0),
    (34, 11, 25, 29, 18, 26, 4, 45, 5),
    (30, 36, 23, 27, 32, 38, 34, 22, 43),
    (37, 29, 9, 12, 3, 24, 20, 11, 6),
    (32, 3, 5, 18, 30, 20, 4, 22, 47),
    (44, 29, 15, 5, 20, 25, 32, 2, 23),
    (29, 23, 43, 14, 25, 35, 1, 2, 17),
)
SAMPLED_EXCLUDING = (
    (33, 26, 21, 13, 8, 12, 2, 1, 0),
    (18, 45, 40, 6, 1, 21, 12, 38, 31),
    (4, 10, 18, 13, 32, 46, 17, 21, 37),
    (8, 36, 1, 2, 29, 40, 10, 6, 32),
    (47, 38, 29, 45, 40, 22, 4, 37, 44),
    (32, 21, 24, 47, 30, 34, 13, 0, 26),
    (44, 18, 17, 31, 14, 40, 41, 22, 47),
    (38, 42, 10, 26, 37, 34, 45, 2, 29),
    (8, 31, 14, 13, 30, 41, 9, 29, 37),
    (34, 17, 12, 39, 5, 28, 32, 37, 31),
    (40, 25, 37, 6, 9, 10, 31, 38, 34),
    (28, 24, 26, 33, 5, 1, 45, 21, 39),
    (9, 2, 24, 40, 46, 8, 10, 29, 41),
    (40, 39, 8, 2, 44, 37, 12, 42, 34),
    (4, 40, 28, 17, 45, 5, 16, 33, 32),
    (36, 30, 10, 9, 34, 22, 29, 1, 16),
    (17, 0, 34, 25, 33, 24, 21, 18, 4),
    (17, 29, 1, 21, 36, 26, 4, 6, 34),
    (34, 38, 16, 9, 4, 17, 31, 13, 46),
    (2, 22, 42, 17, 39, 40, 12, 24, 14),
    (39, 10, 47, 25, 5, 12, 34, 40, 20),
    (4, 14, 12, 26, 16, 31, 45, 21, 32),
    (8, 30, 28, 4, 0, 31, 12, 14, 41),
    (12, 45, 28, 1, 29, 17, 46, 47, 5),
    (31, 20, 14, 13, 25, 5, 17, 34, 26),
    (8, 9, 0, 36, 17, 26, 6, 10, 20),
    (17, 20, 33, 24, 45, 2, 37, 9, 34),
    (16, 9, 45, 13, 0, 42, 5, 29, 21),
    (38, 6, 44, 34, 37, 1, 9, 36, 26),
    (22, 24, 10, 31, 29, 25, 36, 13, 1),
    (17, 31, 44, 18, 12, 4, 30, 9, 29),
    (21, 46, 26, 2, 17, 36, 22, 31, 45),
    (34, 6, 25, 16, 17, 18, 32, 36, 13),
    (14, 34, 18, 37, 25, 26, 12, 16, 40),
    (37, 45, 31, 0, 47, 1, 10, 4, 42),
    (20, 12, 44, 13, 41, 34, 0, 40, 4),
    (44, 14, 20, 6, 18, 40, 46, 17, 42),
    (29, 13, 33, 5, 12, 2, 41, 45, 26),
    (22, 32, 31, 13, 17, 10, 9, 20, 4),
    (9, 21, 33, 46, 38, 14, 39, 24, 5),
    (41, 44, 30, 2, 0, 20, 24, 22, 1),
    (31, 26, 17, 33, 12, 5, 34, 38, 39),
    (9, 31, 38, 28, 18, 33, 4, 20, 2),
    (42, 17, 20, 13, 1, 24, 26, 39, 0),
    (34, 10, 47, 30, 18, 26, 45, 46, 4),
    (31, 36, 24, 28, 33, 38, 34, 22, 42),
    (36, 30, 9, 12, 2, 25, 20, 45, 6),
    (32, 4, 5, 20, 30, 21, 46, 22, 47),
    (45, 30, 16, 5, 21, 26, 33, 1, 24),
    (30, 25, 44, 14, 26, 36, 1, 42, 18),
)


def test_law_is_pure_in_seed_cue_item(smoke_law):
    """Repeated calls in any order give identical scores."""
    items = list(range(48))
    a = smoke_law.score(1, items)
    smoke_law.score(3, items[::-1])
    b = smoke_law.score(1, items)
    assert np.array_equal(a, b)


def test_law_is_stable_across_processes(smoke_law):
    """A subprocess recomputes the same ordering for a fixed (seed, cue, items)."""
    code = "from geo_alg_ti.orderings import OrderingLaw; print(OrderingLaw(0, 48, 6, 2).ordering(1, range(9)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(smoke_law.ordering(1, range(9)))


def test_growing_the_pool_does_not_perturb_existing_scores():
    """Scores of the first 24 items are identical under pool sizes 24 and 300."""
    small = orderings.OrderingLaw(seed=0, pool_size=24, n_training_cues=6, n_reserved_cues=2)
    large = orderings.OrderingLaw(seed=0, pool_size=300)
    assert np.array_equal(small.score(2, range(24)), large.score(2, range(24)))


def test_subset_ordering_equals_restricted_pool_ordering(smoke_law):
    """The ordering of a subset is the pool ordering filtered to that subset."""
    pool = smoke_law.ordering(1, range(48))
    subset = {3, 11, 19, 27, 35, 43, 7, 15, 23}
    assert smoke_law.ordering(1, sorted(subset)) == tuple(i for i in pool if i in subset)


def test_ties_are_broken_by_item_id(monkeypatch):
    """Equal scores resolve deterministically by item id."""
    law = orderings.OrderingLaw(seed=0, pool_size=48, n_training_cues=6, n_reserved_cues=2)
    tied = {10: 0.5, 3: 0.9, 7: 0.5, 1: 0.9, 22: 0.1}
    monkeypatch.setattr(
        orderings.OrderingLaw, "score", lambda self, cue, items: np.array([tied[i] for i in items])
    )
    # Descending score first; within a tie the smaller item id ranks higher.
    assert law.ordering(1, (10, 3, 7, 1, 22)) == (1, 3, 7, 10, 22)
    assert law.ordering(1, (22, 1, 7, 3, 10)) == (1, 3, 7, 10, 22)
    assert law.ranks(1, (10, 3, 7, 1, 22)) == {1: 0, 3: 1, 7: 2, 10: 3, 22: 4}
    assert law.rank_of(1, (10, 3, 7, 1, 22), 7) == 2
    assert orderings.rank_lookup((1, 3, 7, 10, 22)) == law.ranks(1, (10, 3, 7, 1, 22))


def test_training_and_reserved_cues_are_disjoint():
    """Training cues are 0..99 and reserved cues are 100..115 with no overlap."""
    law = orderings.OrderingLaw(seed=0)
    assert law.training_cues() == tuple(range(100))
    assert law.reserved_cues() == tuple(range(100, 116))
    assert law.is_reserved(100) and not law.is_reserved(99)


def test_zipf_weights_at_alpha_zero_equal_uniform():
    """Skew alpha 0 is the uniform distribution."""
    assert np.allclose(orderings.zipf_cue_weights(100, 0.0), orderings.uniform_cue_weights(100))


def test_zipf_weights_are_monotone_and_normalized():
    """Weights decrease with cue index and sum to one."""
    w = orderings.zipf_cue_weights(100, 1.5)
    assert np.all(np.diff(w) < 0) and np.isclose(w.sum(), 1.0)


def test_frequent_cue_weights_place_the_declared_proportion():
    """The frequent cue gets p and every other cue gets (1 - p) / 99."""
    w = orderings.frequent_cue_weights(100, proportion=0.2)
    assert orderings.FREQUENT_CUE == 0
    assert np.isclose(w[0], 0.2) and np.allclose(w[1:], 0.8 / 99)
    assert np.allclose(orderings.cue_weights("frequent_cue", 100, frequent_cue_proportion=0.2), w)


def test_reserved_cue_predicate_flags_every_reserved_cue(smoke_law):
    """`is_reserved` is true for exactly the reserved block."""
    assert [smoke_law.is_reserved(c) for c in range(8)] == [False] * 6 + [True] * 2


def test_sample_items_draws_are_pinned(smoke_law):
    """The items drawn at each seed, with and without exclusions, match the pinned values."""
    for seed, (plain, excluding) in enumerate(zip(SAMPLED_PLAIN, SAMPLED_EXCLUDING, strict=True)):
        assert smoke_law.sample_items(np.random.default_rng(seed), 9) == plain
        drawn = smoke_law.sample_items(np.random.default_rng(seed), 9, exclude=SAMPLE_EXCLUDED)
        assert drawn == excluding
        assert len(set(plain)) == len(set(excluding)) == 9
        assert set(excluding).isdisjoint(SAMPLE_EXCLUDED)
        assert set(plain) | set(excluding) <= set(range(smoke_law.pool_size))
    assert len(SAMPLED_PLAIN) == 50
    with pytest.raises(ValueError):
        smoke_law.sample_items(np.random.default_rng(0), 40, exclude=SAMPLE_EXCLUDED)
    with pytest.raises(ValueError, match="outside the pool"):
        smoke_law.sample_items(np.random.default_rng(0), 9, exclude=[smoke_law.pool_size])


def test_seeded_rng_is_the_seed_sequence_of_its_key():
    """One constructor: seed, salt, then the draw's place, spelled as a `SeedSequence`."""
    direct = np.random.default_rng(np.random.SeedSequence([5, orderings.ORDERING_SALT, 2, 7]))
    assert orderings.seeded_rng(5, orderings.ORDERING_SALT, 2, 7).random() == direct.random()
    assert orderings.seeded_rng(5, 1).random() != orderings.seeded_rng(5, 2).random()
    assert orderings.seeded_rng(5, 1, 3).random() != orderings.seeded_rng(5, 1, 4).random()


def test_majority_ordering_accuracy_is_one_when_every_trial_draws_one_cue(smoke_law):
    """With all the weight on one cue, its ordering answers every pair."""
    weights = np.zeros(smoke_law.n_training_cues)
    weights[2] = 1.0
    assert orderings.majority_ordering_accuracy(smoke_law, weights) == 1.0


def test_majority_ordering_of_two_equal_cues_scores_their_disagreements_one_half(smoke_law):
    """Two cues of equal weight: the pairs they order alike, plus half of the rest."""
    first, second = 1, 4
    weights = np.zeros(smoke_law.n_training_cues)
    weights[[first, second]] = 0.5
    items = range(smoke_law.pool_size)
    a, b = smoke_law.score(first, items), smoke_law.score(second, items)
    pairs = list(itertools.combinations(items, 2))
    alike = np.mean([(a[i] > a[j]) == (b[i] > b[j]) for i, j in pairs])
    assert 0.0 < alike < 1.0
    assert np.isclose(orderings.majority_ordering_accuracy(smoke_law, weights), alike + (1.0 - alike) / 2)


def test_majority_ordering_accuracy_takes_the_laws_relations_once_computed(smoke_law):
    """Relations computed once give each weighting the accuracy it gets alone; others raise."""
    relations = orderings.pair_relations(smoke_law)
    n = smoke_law.pool_size
    assert relations.shape == (smoke_law.n_training_cues, n * (n - 1) // 2)
    for alpha in (0.0, 1.0, 4.0):
        weights = orderings.zipf_cue_weights(smoke_law.n_training_cues, alpha)
        alone = orderings.majority_ordering_accuracy(smoke_law, weights)
        assert orderings.majority_ordering_accuracy(smoke_law, weights, relations=relations) == alone
    with pytest.raises(ValueError, match="relations"):
        orderings.majority_ordering_accuracy(smoke_law, weights, relations=relations[:, :-1])


def test_majority_ordering_accuracy_lies_between_chance_and_one_and_grows_with_skew(smoke_law):
    """Skewing the cue frequencies never lowers the accuracy on this law."""
    alphas = (0.0, 0.5, 1.0, 2.0, 4.0, 10.0)
    accuracies = [orderings.majority_ordering_accuracy(smoke_law, orderings.zipf_cue_weights(smoke_law.n_training_cues, a)) for a in alphas]
    assert all(0.5 <= a <= 1.0 for a in accuracies)
    assert np.all(np.diff(accuracies) >= 0)
    assert accuracies[-1] > accuracies[0]


def test_majority_ordering_accuracy_refuses_weights_of_the_wrong_length(smoke_law):
    """One weight per training cue, no more and no fewer."""
    with pytest.raises(ValueError, match="training cues"):
        orderings.majority_ordering_accuracy(smoke_law, orderings.uniform_cue_weights(smoke_law.n_training_cues + 1))
