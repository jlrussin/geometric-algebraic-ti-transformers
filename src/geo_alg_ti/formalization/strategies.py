"""Closed forms and exhaustive enumeration for the two strategies.

The algebraic strategy walks the presented premises one per reasoning step from a
random starting direction and stops at the target or at an endpoint; the
geometric strategy answers from a built ordering with Gaussian rank noise,
accuracy Phi(d / sigma). The expected walk length satisfies
E[L | d] = (n - 1 + d) / 4, asserted against enumeration in `check_identities`.
The cost model compares building an ordering once against walking on every
query; the diversity panels weigh them under Zipf-distributed orderings; in the
content panels an unfinished walk answers from the built ordering, which the
presented premises show as it is (congruent), reversed, or shuffled, or do not
show at all (uninformative). Numpy only; no random number generator.

Items are the positions 1..n_items of one ordering. The two endpoints are never
queried, so the n_items - 2 interior items give symbolic distances
1..n_items - 3. Queries are uniform over ordered interior pairs and the walk's
direction is a coin flip, so every (ordered pair, direction) case is
enumerated with equal weight. An adjacent pair (distance 1) is read from its
own premise in zero steps when the shortcut is on.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np

from geo_alg_ti import CHANCE_ACCURACY

# The settings the predictions are drawn at; `check_identities` checks the same values.
N_ITEMS: int = 9
# Symbolic distances between interior items.
DISTANCES: tuple[int, ...] = tuple(range(1, N_ITEMS - 2))
# The budgets the algebraic strategy's accuracy is averaged over.
BUDGETS: tuple[int, ...] = (1, 2, 3, 4, 5)
# The reasoning-step budgets the transformers are evaluated at, zero to eight steps: the x axis
# of the panels of accuracy by budget.
SWEEP_BUDGETS: tuple[int, ...] = tuple(range(0, 9))
SIGMA: float = 3.0
MARGIN: float = 0.5
TEMPERATURE: float = 0.15
# The cost model's skew settings: as many orderings as the skew experiment's training cues, and a
# query count and premise cost that put the crossing of the two strategies' costs inside the
# experiment's range of skew alpha, 0 to 10.
PREMISE_COST: float = 20.0
N_QUERIES: int = 500
N_ORDERINGS: int = 100
SKEW_ALPHA_GRID: tuple[float, ...] = tuple(np.linspace(0.0, 10.0, 400).tolist())


class Walks(NamedTuple):
    """Every enumerated walk: distance, length, weight, and whether its answer is correct."""

    d: np.ndarray
    length: np.ndarray
    weight: np.ndarray
    correct: np.ndarray


def iterate_pairs(n_items: int) -> Iterator[tuple[int, int]]:
    """All ordered pairs of distinct interior positions 2..n_items - 1."""
    for start in range(2, n_items):
        for target in range(2, n_items):
            if start != target:
                yield start, target


def walk(n_items: int, start: int, target: int, direction: int) -> tuple[int, bool]:
    """Walk from `start` toward `direction` one premise per step; return (steps, reached target).

    The walk never backtracks. It ends on reaching the target, whose relation
    to the start is then the walked direction, or on reaching an endpoint, in
    which case the target must lie the other way.
    """
    position, steps = start, 0
    while True:
        position += direction
        steps += 1
        if position == target:
            return steps, True
        if position in (1, n_items):
            return steps, False


def enumerate_walks(n_items: int, *, shortcut_adjacent: bool = True) -> Walks:
    """Enumerate every (start, target, direction) with its weight.

    Pairs are uniform and each direction has weight 1/2. With the shortcut an
    adjacent pair is answered from its premise in zero steps and takes one row.
    """
    rows = []
    for start, target in iterate_pairs(n_items):
        d = abs(target - start)
        truth = 1 if target > start else -1
        if shortcut_adjacent and d == 1:
            rows.append((d, 0, 1.0, True))
            continue
        for direction in (1, -1):
            steps, reached = walk(n_items, start, target, direction)
            answer = direction if reached else -direction
            rows.append((d, steps, 0.5, answer == truth))
    table = np.array(rows, dtype=float)
    n_pairs = (n_items - 2) * (n_items - 3)
    return Walks(
        table[:, 0].astype(int),
        table[:, 1].astype(int),
        table[:, 2] / n_pairs,
        table[:, 3].astype(bool),
    )


def expected_steps(n_items: int, d: int) -> float:
    """The closed form (n - 1 + d) / 4."""
    return (n_items - 1 + d) / 4


def steps_by_distance(walks: Walks, d_values: Sequence[int]) -> np.ndarray:
    """E[L | d] from the enumeration."""
    return np.array(
        [
            np.average(walks.length[walks.d == d], weights=walks.weight[walks.d == d])
            for d in d_values
        ]
    )


def completion_probability(
    walks: Walks, d_values: Sequence[int], budgets: Sequence[int]
) -> np.ndarray:
    """P(L <= K | d) for every distance and budget, shaped (distances, budgets)."""
    table = np.empty((len(d_values), len(budgets)))
    for i, d in enumerate(d_values):
        mask = walks.d == d
        weight = walks.weight[mask] / walks.weight[mask].sum()
        length = walks.length[mask]
        table[i] = [weight[length <= budget].sum() for budget in budgets]
    return table


def accuracy_by_reasoning_steps(
    completion: np.ndarray, *, guess_accuracy: float = CHANCE_ACCURACY
) -> np.ndarray:
    """Accuracy at each budget: completed walks are correct, unfinished ones guess."""
    return guess_accuracy + (1.0 - guess_accuracy) * np.asarray(completion, dtype=float)


def distance_distribution(walks: Walks, d_values: Sequence[int]) -> np.ndarray:
    """Probability of each query distance under uniform pairs."""
    return np.array([walks.weight[walks.d == d].sum() for d in d_values])


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via the error function."""
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def geometric_accuracy(d_values: Sequence[int], *, sigma: float = SIGMA) -> np.ndarray:
    """Phi(d / sigma): the geometric strategy's accuracy by distance under Gaussian rank noise."""
    return normal_cdf(np.asarray(d_values, dtype=float) / sigma)


# The contexts whose premises hold the queried items, in the order the content panels draw them.
INFORMATIVE_CONTEXTS: tuple[str, ...] = ("congruent", "reversed", "shuffled")
# The fraction of a shuffled context's queries that agree with the known ordering (every distance).
SHUFFLED_AGREEMENT: float = 0.5


def prior_agreement(context: str, d_values: Sequence[int], *, sigma: float = SIGMA) -> np.ndarray:
    """The probability that the built ordering gives the context's answer, by distance.

    The built ordering is read with the geometric strategy's rank noise, so it
    gives the known ordering's answer with probability Phi(d / sigma). A
    congruent context shows the known ordering; a reversed one shows it
    backwards, so the two agree only when the reading errs; a shuffled one
    agrees with the known ordering on a fraction `SHUFFLED_AGREEMENT` (one half)
    of the queries at every distance, so the two agree with that probability.
    """
    geometric = geometric_accuracy(d_values, sigma=sigma)
    if context == "congruent":
        return geometric
    if context == "reversed":
        return 1.0 - geometric
    if context == "shuffled":
        return np.full_like(geometric, SHUFFLED_AGREEMENT)
    raise ValueError(f"context must be one of {INFORMATIVE_CONTEXTS}, not {context!r}")


def content_accuracy(completion: np.ndarray, agreement: np.ndarray) -> np.ndarray:
    """Accuracy when an unfinished walk answers from the built ordering.

    `completion` is P(L <= K | d), shaped (distances,) or (distances, budgets), and
    `agreement` is `prior_agreement` by distance: a walk that has ended within
    the budget is correct, and an unfinished one gives the built ordering's answer.
    """
    completion = np.asarray(completion, dtype=float)
    agreement = np.asarray(agreement, dtype=float).reshape((-1,) + (1,) * (completion.ndim - 1))
    return completion + (1.0 - completion) * agreement


def orderings_built_by_experience(skew_alpha: float, n_orderings: int, n_queries: int) -> float:
    """Expected number of distinct orderings encountered under Zipf skew in `n_queries` queries.

    Ordering m has frequency proportional to m^-alpha; the geometric strategy
    builds an ordering the first time it is met, so the expected number of
    builds is the sum over orderings of 1 - (1 - p(m))^Q.
    """
    m = np.arange(1, n_orderings + 1, dtype=float)
    p = m**-skew_alpha
    p /= p.sum()
    return float((1.0 - (1.0 - p) ** n_queries).sum())


def cost_per_query(
    *,
    ordering_cost: float,
    n_queries: int,
    n_orderings: int,
    skew_alphas: Sequence[float],
    mean_steps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query cost of the geometric and algebraic strategies across skew values.

    `ordering_cost` is the cost of building one ordering (the per-premise cost
    times its n_items - 1 premises) and one reasoning step costs one unit. The
    geometric strategy pays the build once per encountered ordering plus one
    unit per query; the algebraic strategy pays its mean walk length, which
    for uniform queries equals the mean distance `mean_steps`.
    """
    alphas = np.asarray(skew_alphas, dtype=float)
    builds = np.array([orderings_built_by_experience(a, n_orderings, n_queries) for a in alphas])
    geometric = (ordering_cost * builds + n_queries) / n_queries
    return geometric, np.full_like(alphas, mean_steps)


def logistic(x: np.ndarray) -> np.ndarray:
    """The logistic function."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def strategy_accuracies(
    geometric_cost: np.ndarray,
    algebraic_cost: np.ndarray,
    *,
    margin: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy of each strategy as a soft function of their cost ratio.

    A system capable of both strategies acquires each to the extent that it is
    not much costlier than the other: strategy s is learned to degree
    l_s = logistic((margin - log(C_s / C_other)) / temperature), so it is fully
    learned while cheaper and abandoned once costlier by more than the margin.
    Accuracy on the test that isolates s is 1/2 + 1/2 l_s. Returns the
    geometric (IWL test) and algebraic (ICL test) accuracies.
    """
    ratio = np.log(np.asarray(geometric_cost, dtype=float) / np.asarray(algebraic_cost, dtype=float))
    learned_geometric = logistic((margin - ratio) / temperature)
    learned_algebraic = logistic((margin + ratio) / temperature)
    return CHANCE_ACCURACY + CHANCE_ACCURACY * learned_geometric, CHANCE_ACCURACY + CHANCE_ACCURACY * learned_algebraic


def _require(condition: object, name: str) -> None:
    """Raise `AssertionError(name)` unless the condition holds; never optimized away."""
    if not condition:
        raise AssertionError(name)


def check_identities(
    *,
    n_items: int = N_ITEMS,
    budgets: Sequence[int] = BUDGETS,
    sweep_budgets: Sequence[int] = SWEEP_BUDGETS,
    shortcut_adjacent: bool = True,
    sigma: float = SIGMA,
    premise_cost: float = PREMISE_COST,
    n_queries: int = N_QUERIES,
    n_orderings: int = N_ORDERINGS,
    skew_alphas: Sequence[float] = SKEW_ALPHA_GRID,
    margin: float = MARGIN,
    temperature: float = TEMPERATURE,
) -> None:
    """Check the closed forms against enumeration before any output is written.

    The defaults are the settings the panels draw. `premise_cost` is the cost per
    premise; one ordering costs premise_cost * (n_items - 1). `budgets` are the
    reasoning-step budgets the algebraic accuracy is averaged over; `sweep_budgets`
    are those the panels of accuracy by budget draw, where the completion closed
    form is checked too. Raises
    ValueError on settings the enumeration cannot hold and AssertionError
    naming the identity that failed.
    """
    if n_items < 5:
        raise ValueError(f"n_items must be at least 5 to hold a queryable pair, not {n_items}")
    k_max = max(int(k) for k in budgets)
    if min(int(k) for k in budgets) < 0 or k_max > n_items - 3:
        raise ValueError(f"budgets must lie within 0..{n_items - 3}, not {list(budgets)}")
    if min(int(k) for k in sweep_budgets) < 0:
        raise ValueError(f"sweep budgets must be nonnegative, not {list(sweep_budgets)}")
    n_interior = n_items - 2
    n_pairs = n_interior * (n_interior - 1)
    d_values = np.arange(1, n_items - 2)
    walks = enumerate_walks(n_items, shortcut_adjacent=shortcut_adjacent)
    steps = steps_by_distance(walks, d_values)
    full_budgets = np.arange(0, max(n_items - 2, max(int(k) for k in sweep_budgets) + 1))
    completion = completion_probability(walks, d_values, full_budgets)
    algebraic = accuracy_by_reasoning_steps(completion[:, [int(k) for k in budgets]]).mean(axis=1)
    p_d = distance_distribution(walks, d_values)
    mean_distance = float((d_values * p_d).sum())
    ordering_cost = premise_cost * (n_items - 1)
    alphas = np.asarray(skew_alphas, dtype=float)
    geometric_cost, algebraic_cost = cost_per_query(
        ordering_cost=ordering_cost,
        n_queries=n_queries,
        n_orderings=n_orderings,
        skew_alphas=alphas,
        mean_steps=mean_distance,
    )
    iwl_accuracy, icl_accuracy = strategy_accuracies(
        geometric_cost, algebraic_cost, margin=margin, temperature=temperature
    )

    # The walk: always correct, weights sum to one, one row per case, and
    # 1 <= L <= n_items - 3 steps (0 on shortcut rows).
    _require(walks.correct.all(), "walk answers: every enumerated walk answers correctly")
    _require(math.isclose(float(walks.weight.sum()), 1.0, rel_tol=1e-12), "walk weights sum to one")
    n_rows = 2 * n_pairs - (2 * (n_interior - 1) if shortcut_adjacent else 0)
    _require(len(walks.length) == n_rows, "walk rows: one per (ordered pair, direction) case")
    walked = walks.length[walks.d > 1] if shortcut_adjacent else walks.length
    _require(
        walked.min() == 1 and walked.max() == n_items - 3,
        "walk length: 1 <= L <= n_items - 3",
    )
    if shortcut_adjacent:
        _require((walks.length[walks.d == 1] == 0).all(), "adjacent shortcut: zero steps")

    # Expected steps: E[L | d] = (n_items - 1 + d) / 4 for walked queries.
    for i, d in enumerate(d_values):
        if shortcut_adjacent and d == 1:
            _require(steps[i] == 0.0, "expected steps: adjacent pairs take zero steps")
        else:
            _require(
                math.isclose(steps[i], expected_steps(n_items, int(d)), rel_tol=1e-12),
                f"expected steps: E[L | d] = (n_items - 1 + d) / 4 at d = {d}",
            )

    # Query distribution: p(d) proportional to (n_interior - d), mean distance
    # (n_items - 1) / 3, and without the shortcut the mean walk length equals it.
    _require(
        np.allclose(p_d, 2 * (n_interior - d_values) / n_pairs),
        "distance distribution: p(d) proportional to (n_interior - d)",
    )
    _require(
        math.isclose(mean_distance, (n_items - 1) / 3, rel_tol=1e-12),
        "mean distance: (n_items - 1) / 3",
    )
    full = enumerate_walks(n_items, shortcut_adjacent=False)
    mean_length = float((full.length * full.weight).sum())
    _require(
        math.isclose(mean_length, mean_distance, rel_tol=1e-12),
        "mean walk length without the shortcut equals the mean distance",
    )

    # Completion: P(L <= K | d) = 1/2 [K >= d] + 1/2 min(K, m) / m with
    # m = n_interior - d; nondecreasing in K; every walk ends by K = n_items - 3.
    for i, d in enumerate(d_values):
        m = n_interior - d
        for j, budget in enumerate(full_budgets):
            if shortcut_adjacent and d == 1:
                closed = 1.0
            else:
                closed = 0.5 * (budget >= d) + 0.5 * min(budget, m) / m
            _require(
                math.isclose(completion[i, j], closed, rel_tol=1e-12),
                f"completion probability: closed form at d = {d}, budget = {budget}",
            )
    _require(
        (np.diff(completion, axis=1) >= -1e-12).all(),
        "completion probability: nondecreasing in the budget",
    )
    _require(
        np.allclose(completion[:, n_items - 3], 1.0),
        "completion probability: every walk ends by n_items - 3 steps",
    )

    # Accuracy by distance: the geometric curve rises, the algebraic curve stays
    # in (1/2, 1] and falls with distance, and the two cross.
    geometric = geometric_accuracy(d_values, sigma=sigma)
    _require(
        (np.diff(geometric) > 0).all() and math.isclose(float(normal_cdf(0.0)), 0.5),
        "geometric accuracy: Phi(d / sigma) rises from one half",
    )
    _require(
        (algebraic > CHANCE_ACCURACY).all() and (algebraic <= 1.0 + 1e-12).all(),
        "algebraic accuracy: within (1/2, 1]",
    )
    _require((np.diff(algebraic) <= 1e-12).all(), "algebraic accuracy: nonincreasing in distance")
    _require(
        geometric[0] < algebraic[0] and geometric[-1] > algebraic[-1],
        "accuracy curves: geometric and algebraic cross",
    )

    # Orderings built: the uniform closed form, decreasing in alpha, and bounded
    # by [1, min(n_orderings, n_queries)].
    uniform = n_orderings * (1.0 - (1.0 - 1.0 / n_orderings) ** n_queries)
    built_uniform = orderings_built_by_experience(0.0, n_orderings, n_queries)
    _require(
        math.isclose(built_uniform, uniform, rel_tol=1e-12),
        "orderings built: M (1 - (1 - 1/M)^Q) at alpha = 0",
    )
    builds = [orderings_built_by_experience(a, n_orderings, n_queries) for a in alphas]
    _require(
        all(a >= b - 1e-9 for a, b in zip(builds, builds[1:])),
        "orderings built: decreasing in alpha",
    )
    _require(
        1.0 <= builds[-1] and builds[0] <= min(n_orderings, n_queries),
        "orderings built: within [1, min(M, Q)]",
    )

    # Ordering diversity: the geometric cost falls with alpha and crosses the flat
    # algebraic cost inside the range; IWL-test accuracy rises with alpha, ICL-test
    # accuracy falls, both stay in (1/2, 1], and they cross where the costs cross,
    # at 1/2 + 1/2 logistic(margin / temperature).
    _require((np.diff(geometric_cost) < 0).all(), "geometric cost: decreasing in alpha")
    _require(
        geometric_cost[0] > mean_distance > geometric_cost[-1],
        "costs: the geometric cost crosses the algebraic cost inside the alpha range",
    )
    _require(
        (np.diff(iwl_accuracy) > 0).all() and (np.diff(icl_accuracy) < 0).all(),
        "diversity accuracies: IWL test rises and ICL test falls with alpha",
    )
    for accuracy in (iwl_accuracy, icl_accuracy):
        _require(
            (accuracy > CHANCE_ACCURACY).all() and (accuracy <= 1.0 + 1e-12).all(),
            "diversity accuracies: within (1/2, 1]",
        )
    i_cost = int(np.argmax(geometric_cost <= mean_distance))
    i_accuracy = int(np.argmin(np.abs(iwl_accuracy - icl_accuracy)))
    _require(
        abs(i_cost - i_accuracy) <= 1,
        "diversity crossing: accuracies cross where costs cross",
    )
    both = CHANCE_ACCURACY + CHANCE_ACCURACY * float(logistic(margin / temperature))
    _require(
        abs(iwl_accuracy[i_accuracy] - both) < 0.02 and abs(icl_accuracy[i_accuracy] - both) < 0.02,
        "diversity crossing: accuracy 1/2 + 1/2 logistic(margin / temperature)",
    )

    # Amortization boundary: the costs swap order across c_I* = c_L / (Q (dbar - 1)).
    for q in (2.0, 10.0, 20.0):
        boundary = ordering_cost / (q * (mean_distance - 1.0))
        for step_cost, algebraic_cheaper in ((0.95 * boundary, True), (1.05 * boundary, False)):
            _require(
                (q * step_cost * mean_distance < ordering_cost + step_cost * q) == algebraic_cheaper,
                "amortization boundary: c_I* = c_L / (Q (dbar - 1))",
            )

    # Content accuracy, averaged over the budgets: an unfinished walk answers from the built
    # ordering. A shuffled context's accuracy equals the algebraic strategy's (the built
    # ordering agrees half the time, as a guess does); the congruent and reversed accuracies
    # sum to one plus the completion probability; the reversed accuracy does not rise with
    # distance; congruent >= shuffled >= reversed; and adjacent pairs are answered from their
    # own premise.
    mean_completion = completion[:, [int(k) for k in budgets]].mean(axis=1)
    content = {
        context: content_accuracy(mean_completion, prior_agreement(context, d_values, sigma=sigma))
        for context in INFORMATIVE_CONTEXTS
    }
    _require(
        np.allclose(content["shuffled"], algebraic),
        "content accuracy: shuffled equals the algebraic accuracy",
    )
    _require(
        np.allclose(content["congruent"] + content["reversed"], 1.0 + mean_completion),
        "content accuracy: congruent + reversed = 1 + P(L <= K | d)",
    )
    _require(
        (np.diff(content["reversed"]) <= 1e-12).all(),
        "content accuracy: reversed nonincreasing in distance",
    )
    _require(
        (content["congruent"] >= content["shuffled"] - 1e-12).all()
        and (content["shuffled"] >= content["reversed"] - 1e-12).all(),
        "content accuracy: congruent >= shuffled >= reversed",
    )
    if shortcut_adjacent:
        _require(
            all(math.isclose(accuracy[0], 1.0, rel_tol=1e-12) for accuracy in content.values()),
            "content accuracy: adjacent pairs answered from their own premise",
        )
