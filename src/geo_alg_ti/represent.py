"""Linear rank probes by site and rank-direction edits of the item-token states.

Supports the appendix figure "Additional mechanism results": how strongly the
frequent cue's ranks are linearly decodable from the item representations at
each site of the network, and what happens to behavior when the rank direction
of the item-token states is scaled by a gain (1 leaves it, 0 erases it, -1
inverts it) or overwritten with another ordering at a given strength. Ranks are
pool-wide: every item of the pool has a rank under the frequent cue's ordering,
rank 0 at the top. Item representations are per-occurrence vectors, one row per
item-token occurrence, with an integer rank per row.

The rank direction is fit with a shrunk covariance (the Ledoit-Wolf closed-form
shrinkage of the sample covariance toward a scaled identity, in numpy) and
symmetric whitening: with Sigma^(1/2) and Sigma^(-1/2) from the
eigendecomposition, the whitened basis is the orthonormalized cross-covariance
between the centered vectors and the centered powers of the z-scored rank. An
edit acts through the projector A = Sigma^(-1/2) u u^T Sigma^(1/2) of the
first basis vector u, in the row-vector convention: scaling by a gain g is
x' = x - (1 - g) (x - mu) A; writing at strength s is
x' = x - s (x - mu) A + s t v with the insertion vector v = Sigma^(1/2) u, so
the coordinate r(x) = (x - mu) . q read back by q = Sigma^(-1/2) u becomes
(1 - s) r(x) + s t.

An edit acts on the residual state of every item token after the embedding
(site 0) and after every block (sites 1..n_layers), never on the final
normalized state, each site with its own rank direction fit on the item-token
occurrences of the probe prompts at that site. Within a forward the sites are
edited in order, so the state edited at one site is the one the next block
reads, and every forward of a rollout is edited: the prompt, each emitted
token, and the query items appended at each reasoning-step value. The targets
t at a site come from a line fitted from rank to coordinate on that site's
occurrences, read at each pool item's target rank. Every edit has a matched
random-direction control through the same code path, with u replaced at each
site by a seeded random unit vector of its own and the mean, the whitening,
and, for a write, the targets shared.

A probe is one ridge regression of the z-scored rank on the vectors of one
site, its penalty chosen by exact leave-one-out, scored on held-out items by a
seeded repeated grouped split: the regression metrics (r2, Spearman) of the
held-out predictions and the pairwise ordering accuracy of the per-item mean
predictions. The vectors are read from prompts under the frequent cue whose
query pairs cover every queryable pool item and whose context is
uninformative: other items in the order the cue assigns them.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from geo_alg_ti import evaluate
from geo_alg_ti.battery import Battery
from geo_alg_ti.chains import ChainAlgorithm
from geo_alg_ti.model import RelationalTransformer, batched_hidden_states, item_embedding_sites
from geo_alg_ti.orderings import OrderingLaw, rank_lookup, seeded_rng
from geo_alg_ti.sequences import Encoded, Target, Vocabulary, encode, item_positions
from geo_alg_ti.tables import EDIT_SCHEMA, PROBE_SCHEMA
from geo_alg_ti.trials import ITEMS_PER_ORDERING, ContextKind, Trial, sample_battery_trial

REPRESENT_SALT: int = 0x4E9B  # fixed salt, so every draw here is keyed apart from the streams
N_FIT_PROMPTS: int = 500
ERASE_GAINS: tuple[float, ...] = (3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0)
WRITE_STRENGTHS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
MAX_ABS_SPEARMAN: float = 0.2
RANDOM_DIRECTION_SEED: int = 7
N_BASIS_COMPONENTS: int = 3
N_EDIT_COMPONENTS: int = 1
FOLDS: int = 5
FOLD_REPEATS: int = 3
RIDGE_ALPHAS: np.ndarray = np.logspace(-2, 4, 13)
EIGENVALUE_FLOOR: float = 1e-8
# The tag the edited batteries carry in the edit table.
REPRESENT_TAG: str = "represent"
# The first site: the item embedding table for a probe, the embedding output for an edit.
EMBEDDING_SITE: int = 0

# The draws of this module, keyed apart under the salt: the fit prompts, the item folds, and the
# unrelated write target. The fit prompts use two generators under their key: the query pairs,
# then the displays and contexts.
_PROMPT_KEY: int = 0
_FOLD_KEY: int = 1
_TARGET_KEY: int = 2
_DISPLAY_KEY: int = 1
# The columns each run table holds; `experiment` and the swept axes are added at export.
_PROBE_COLUMNS: tuple[str, ...] = tuple(name for name in PROBE_SCHEMA if name != "experiment")
_EDIT_COLUMNS: tuple[str, ...] = tuple(name for name in EDIT_SCHEMA if name != "experiment")
# The keys one edited battery's trials are aggregated by, as the evaluation aggregates are.
_GROUP: tuple[str, ...] = ("condition", "source", "context", "metric", "d", "K", "seed", "step", "tag")


class Direction(StrEnum):
    """Edit along the fitted rank direction or along a matched random direction."""

    RANK = "rank"
    RANDOM = "random"


class ProbeKind(StrEnum):
    """Regress the rank, or order item pairs by the regression's held-out predictions."""

    RANK_REGRESSION = "rank_regression"
    RANK_ORDER = "rank_order"


class WriteTarget(StrEnum):
    """The ordering a write moves the rank coordinates toward."""

    FAMILIAR = "familiar"
    REVERSED = "reversed"
    UNRELATED = "unrelated"


class EditKind(StrEnum):
    """Scale the rank coordinate by a gain, or write an ordering's coordinates at a strength."""

    ERASE = "erase"
    WRITE = "write"


@dataclass(frozen=True)
class RankDirection:
    """A fitted rank direction at one site.

    `mean` is the occurrence mean; `basis` (width, n_components) the whitened orthonormal basis,
    whose first column is the direction an edit acts on; `whitener` is Sigma^(-1/2) and
    `unwhitener` Sigma^(1/2) of the shrunk covariance, whose floored `eigenvalues` are kept;
    `n_items` is the number of distinct ranks the fit saw, which for pool-wide ranks is the
    number of distinct items.
    """

    mean: np.ndarray
    basis: np.ndarray
    whitener: np.ndarray
    unwhitener: np.ndarray
    eigenvalues: np.ndarray
    n_components: int
    site: int
    n_items: int

    @property
    def direction(self) -> np.ndarray:
        """The whitened rank direction u: the first basis vector, the one component an edit uses."""
        return self.basis[:, 0]

    @property
    def read_back(self) -> np.ndarray:
        """The functional q = Sigma^(-1/2) u; the coordinate of x is (x - mean) . q."""
        return self.whitener @ self.direction

    @property
    def insertion(self) -> np.ndarray:
        """The vector v = Sigma^(1/2) u that adds one unit of coordinate; v . q = 1."""
        return self.unwhitener @ self.direction


@dataclass(frozen=True)
class ItemStates:
    """Per-occurrence representation vectors at one site with their items and ranks."""

    vectors: np.ndarray
    ranks: np.ndarray
    items: np.ndarray
    site: int


@dataclass(frozen=True)
class SiteEdit:
    """The edit of the item-token states at one site.

    In the row-vector convention x' = x - removed (x - mean) A + inserted[item] v, where
    A = q v^T is the projector onto the coordinate along the edit's direction, read back by q
    (`read_back`) and added by v (`insertion`), with v . q = 1: the coordinate (x - mean) . q
    becomes (1 - removed) times itself plus `inserted[item]`. `inserted` holds one coordinate
    per pool item, indexed by item, and is missing on an erase.
    """

    site: int
    mean: torch.Tensor
    read_back: torch.Tensor
    insertion: torch.Tensor
    removed: float
    inserted: torch.Tensor | None

    def to(self, like: torch.Tensor) -> SiteEdit:
        """The same edit with its vectors in the dtype and on the device of `like`."""
        return replace(
            self,
            mean=self.mean.to(like),
            read_back=self.read_back.to(like),
            insertion=self.insertion.to(like),
            inserted=None if self.inserted is None else self.inserted.to(like),
        )

    def apply(self, states: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """The edited states (occurrence, width) of item-token occurrences of the given items."""
        shift = self.removed * ((states - self.mean) @ self.read_back)
        if self.inserted is not None:
            shift = shift - self.inserted[items]
        return states - shift[:, None] * self.insertion


@dataclass(frozen=True)
class EditSpec:
    """One edited battery: the edit at every site, its labels, and the ordering it writes, if any.

    `target_ranks` and `spearman_vs_familiar` are set on a write; an erase carries neither.
    `gain` is missing on a write and `strength` on an erase, as the edit table records them.
    """

    kind: EditKind
    direction: Direction
    target: WriteTarget | None
    gain: float
    strength: float
    edit: tuple[SiteEdit, ...]
    target_ranks: np.ndarray | None
    spearman_vs_familiar: float


def rank_array(ranks: Mapping[int, int] | Sequence[int] | np.ndarray) -> np.ndarray:
    """Pool-wide ranks as an integer array indexed by item.

    A mapping from item to rank covers the items 0..n-1; an array is taken as already indexed.
    """
    if isinstance(ranks, Mapping):
        n = len(ranks)
        if sorted(int(item) for item in ranks) != list(range(n)):
            raise ValueError(f"pool-wide ranks cover every item 0..{n - 1}")
        return np.array([int(ranks[item]) for item in range(n)], dtype=np.int64)
    array = np.asarray(ranks, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("ranks are one rank per item")
    return array


def queryable_items(law: OrderingLaw, cue: int) -> tuple[int, ...]:
    """The pool items a fit prompt can query, in the cue's pool ordering.

    A queried pair always sits inside a nine-item display, so the top and bottom items of the
    pool ordering are never queried; the item embedding site still covers them, since it reads
    every pool item's row.
    """
    order = law.ordering(cue, range(law.pool_size))
    return order[1:-1]


def fit_query_pairs(law: OrderingLaw, cue: int, *, n_prompts: int, seed: int) -> list[tuple[int, int]]:
    """The query pairs of the fit prompts: unordered pairs of queryable items covering every one.

    Every pair of queryable items is taken when `n_prompts` allows. Otherwise `n_prompts` pairs
    are drawn uniformly without replacement and kept in their order among all pairs, and each
    queryable item left out of every drawn pair is appended in one more pair, its partner
    stepped along the queryable items: the k-th such item, in queryable order, is paired with
    the item k + 1 places after it (wrapping around), or with the first queryable item (the
    second, when it is the first itself) if that lands on the item.
    """
    if not 0 <= int(cue) < law.n_training_cues:
        raise ValueError(f"cue {cue} is not a training cue of the law")
    feasible = list(queryable_items(law, cue))
    all_pairs = list(combinations(feasible, 2))
    if int(n_prompts) >= len(all_pairs):
        return all_pairs
    rng = seeded_rng(int(seed), REPRESENT_SALT, _PROMPT_KEY)
    drawn = sorted(int(index) for index in rng.choice(len(all_pairs), size=int(n_prompts), replace=False))
    pairs = [all_pairs[index] for index in drawn]
    covered = {item for pair in pairs for item in pair}
    for k, item in enumerate(item for item in feasible if item not in covered):
        partner = feasible[(feasible.index(item) + 1 + k) % len(feasible)]
        if partner == item:
            partner = feasible[1] if item == feasible[0] else feasible[0]
        pairs.append((item, partner))
    return pairs


def fit_trials(law: OrderingLaw, cue: int, *, n_prompts: int, seed: int) -> list[Trial]:
    """The trials of the probes and the edit fits, one per query pair of `fit_query_pairs`.

    Trials under the frequent cue whose query pairs cover every queryable pool item and whose
    context is uninformative (other items in the order the cue assigns them), so the only rank
    information at an item occurrence is the item's own identity. Each pair is queried, in a
    coin-flipped orientation, inside a nine-item display ordered by the cue's law: the pair,
    one item drawn from those ranked above it and one from those ranked below it (so the pair
    is interior), and five more drawn from the rest of the pool without its top and bottom
    items. The context shows nine filler items drawn outside the display, in the cue's law
    order, as an adjacent chain with random pair orientation and shuffled order. The pairs come
    from one generator and the displays and contexts from a second, advanced in pair order.
    """
    pairs = fit_query_pairs(law, cue, n_prompts=n_prompts, seed=seed)
    order = law.ordering(cue, range(law.pool_size))
    position = rank_lookup(order)
    ends = {order[0], order[-1]}
    rng = seeded_rng(int(seed), REPRESENT_SALT, _PROMPT_KEY, _DISPLAY_KEY)
    trials: list[Trial] = []
    for a, b in pairs:
        hi, lo = sorted((a, b), key=position.__getitem__)
        above = int(rng.choice(order[: position[hi]]))
        below = int(rng.choice(order[position[lo] + 1 :]))
        rest = [item for item in order if item not in ends and item not in (a, b, above, below)]
        extra = [int(item) for item in rng.choice(rest, size=ITEMS_PER_ORDERING - 4, replace=False)]
        items = law.ordering(cue, [a, b, above, below, *extra])
        x, z = (a, b) if rng.random() < 0.5 else (b, a)
        trials.append(
            sample_battery_trial(
                law,
                rng,
                cue=int(cue),
                items=items,
                rank_pair=(items.index(x), items.index(z)),
                context_kind=ContextKind.UNINFORMATIVE,
            )
        )
    return trials


def fit_prompts(law: OrderingLaw, vocab: Vocabulary, cue: int, *, n_prompts: int, seed: int) -> list[Encoded]:
    """The trials of `fit_trials` encoded prompt-only: query, then the uninformative context."""
    return [encode(trial, None, vocab, target=Target.PROMPT_ONLY) for trial in fit_trials(law, cue, n_prompts=n_prompts, seed=seed)]


def collect_item_states(
    model: RelationalTransformer,
    prompts: Sequence[Encoded],
    vocab: Vocabulary,
    *,
    site: int,
    ranks: Mapping[int, int] | np.ndarray,
) -> ItemStates:
    """Item representations at one site: the embedding rows at site 0, else the residual stream.

    Site 0 holds one row per pool item, read from the embedding table without a forward pass,
    since every occurrence of an item enters the network as that same row. Site l of 1..n_layers
    is the output of block l and the last site the final normalized state, read at every
    item-token occurrence of the prompts, query and context alike.
    """
    rank_of = rank_array(ranks)
    if rank_of.size != vocab.pool_size:
        raise ValueError(f"{rank_of.size} ranks for a pool of {vocab.pool_size} items")
    if int(site) == EMBEDDING_SITE:
        items = np.arange(vocab.pool_size, dtype=np.int64)
        rows = model.embedding.weight.detach()[vocab.first_item : vocab.first_item + vocab.pool_size]
        vectors = rows.cpu().numpy().astype(np.float64)
        return ItemStates(vectors=vectors, ranks=rank_of[items], items=items, site=EMBEDDING_SITE)
    return occurrence_states(model, prompts, vocab, sites=(int(site),), ranks=rank_of)[int(site)]


def occurrence_states(
    model: RelationalTransformer,
    prompts: Sequence[Encoded],
    vocab: Vocabulary,
    *,
    sites: Sequence[int],
    ranks: Mapping[int, int] | np.ndarray,
) -> dict[int, ItemStates]:
    """Residual-stream states at every item occurrence of the prompts, one forward for all sites.

    Sites run 0..n_layers + 1 as `hidden_states_at` numbers them; at site 0 the state of an
    occurrence is its item's embedding row, the input to the first block at that position.
    """
    rank_of = rank_array(ranks)
    if rank_of.size != vocab.pool_size:
        raise ValueError(f"{rank_of.size} ranks for a pool of {vocab.pool_size} items")
    wanted = tuple(int(site) for site in sites)
    if not wanted or min(wanted) < 0 or max(wanted) >= model.n_sites:
        raise ValueError(f"sites run 0..{model.n_sites - 1}")
    sequences = list(prompts)
    if not sequences:
        raise ValueError("no prompts to read item states from")
    positions = [item_positions(one.tokens, vocab) for one in sequences]
    tokens = np.stack([one.tokens[where] for one, where in zip(sequences, positions, strict=True)])
    items = (tokens - vocab.first_item).reshape(-1).astype(np.int64)
    states = batched_hidden_states(model, sequences, positions=positions, layers=wanted)
    width = states.shape[-1]
    return {
        site: ItemStates(
            vectors=states[index].reshape(-1, width).astype(np.float64),
            ranks=rank_of[items],
            items=items,
            site=site,
        )
        for index, site in enumerate(wanted)
    }


def shrunk_covariance(x: np.ndarray) -> np.ndarray:
    """The Ledoit-Wolf shrinkage estimate of the covariance of the rows of `x`.

    The sample covariance S is shrunk toward (trace(S) / d) I with the closed-form intensity:
    the ratio of the estimated variance of S's entries, averaged over the rows' outer products,
    to the squared distance of S from the target, each per dimension, capped at one.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("a covariance is estimated from a two-dimensional array of rows")
    n, d = x.shape
    centered = x - x.mean(axis=0)
    sample = centered.T @ centered / n
    mu = float(np.trace(sample)) / d
    squared = centered**2
    delta = float(((sample - mu * np.eye(d)) ** 2).sum()) / d
    beta = (float((squared.T @ squared).sum()) / n - float((sample**2).sum())) / (n * d)
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    return (1.0 - shrinkage) * sample + shrinkage * mu * np.eye(d)


def _symmetric_roots(covariance: np.ndarray, eigenvalue_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sigma^(1/2), Sigma^(-1/2), and the eigenvalues floored at a fraction of the largest."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(eigenvalues.max())
    if not np.isfinite(largest) or largest <= 0:
        raise ValueError("the representation has collapsed: its covariance has no positive eigenvalue")
    eigenvalues = np.clip(eigenvalues, float(eigenvalue_floor) * largest, None)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return root, inverse_root, eigenvalues


def _zscore(values: np.ndarray) -> np.ndarray:
    """Center and scale to unit standard deviation; a constant array has no z-score."""
    values = np.asarray(values, dtype=np.float64)
    spread = float(values.std())
    if spread == 0:
        raise ValueError("every rank is the same, so there is no rank direction to fit")
    return (values - values.mean()) / spread


def _rank_features(ranks: np.ndarray, n_components: int) -> np.ndarray:
    """Column-centered powers 1..n_components of the z-scored rank, (n, n_components)."""
    z = _zscore(ranks)
    features = np.stack([z**power for power in range(1, int(n_components) + 1)], axis=1)
    return features - features.mean(axis=0, keepdims=True)


def fit_rank_direction(
    vectors: np.ndarray,
    ranks: np.ndarray,
    *,
    site: int,
    n_components: int = N_BASIS_COMPONENTS,
    eigenvalue_floor: float = EIGENVALUE_FLOOR,
) -> RankDirection:
    """Fit the rank direction in whitened coordinates from per-occurrence vectors and ranks.

    The basis has `min(n_components, n_items - 1)` columns: the reduced QR of
    Sigma^(-1/2) times the cross-covariance between the centered vectors and the rank features.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    if vectors.ndim != 2 or vectors.shape[0] != ranks.size:
        raise ValueError("one rank per row of vectors")
    if not np.isfinite(vectors).all():
        raise ValueError("the vectors hold a non-finite value")
    n_items = int(np.unique(ranks).size)
    if n_items < 2:
        raise ValueError("a rank direction needs at least two distinct ranks")
    n = vectors.shape[0]
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    unwhitener, whitener, eigenvalues = _symmetric_roots(shrunk_covariance(vectors), eigenvalue_floor)
    n_components = min(int(n_components), n_items - 1)
    if n_components < N_EDIT_COMPONENTS:
        raise ValueError("the basis holds at least the component an edit acts on")
    cross_covariance = centered.T @ _rank_features(ranks, n_components) / n
    basis, _ = np.linalg.qr(whitener @ cross_covariance)
    return RankDirection(
        mean=mean,
        basis=basis,
        whitener=whitener,
        unwhitener=unwhitener,
        eigenvalues=eigenvalues,
        n_components=n_components,
        site=int(site),
        n_items=n_items,
    )


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge coefficients and intercept; the intercept is not penalized, so x and y are centered."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = x.mean(axis=0), float(y.mean())
    centered = x - x_mean
    gram = centered.T @ centered + float(alpha) * np.eye(x.shape[1])
    coefficients = np.linalg.solve(gram, centered.T @ (y - y_mean))
    return coefficients, y_mean - float(x_mean @ coefficients)


def leave_one_out_alpha(x: np.ndarray, y: np.ndarray, alphas: Sequence[float] = RIDGE_ALPHAS) -> float:
    """The penalty with the smallest mean squared leave-one-out error of `ridge_fit`.

    The leave-one-out residuals are exact: each fitted residual divided by one minus its
    leverage, the diagonal of the hat matrix, which the singular value decomposition of the
    centered x gives for every penalty at once; the unpenalized intercept adds 1 / n to it.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    centered = x - x.mean(axis=0)
    y_centered = y - y.mean()
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    projected = left.T @ y_centered
    n = x.shape[0]
    errors = []
    for alpha in alphas:
        shrink = singular**2 / (singular**2 + float(alpha))
        residual = y_centered - left @ (shrink * projected)
        leverage = (left**2) @ shrink + 1.0 / n
        errors.append(float(np.mean((residual / (1.0 - leverage)) ** 2)))
    return float(alphas[int(np.argmin(errors))])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """The rank of each value from 1, ties given the mean of the ranks they span."""
    _, inverse, counts = np.unique(np.asarray(values), return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    return ((starts + ends) / 2.0)[inverse]


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rank correlation: the Pearson correlation of the average ranks of a and b."""
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    scale = float(np.sqrt((ra @ ra) * (rb @ rb)))
    if scale == 0:
        return float("nan")
    return float(ra @ rb / scale)


def _ordering_accuracy(predictions: np.ndarray, items: np.ndarray, z_ranks: np.ndarray) -> float:
    """The fraction of item pairs whose per-item mean predictions are ordered as their ranks.

    Pairs are taken over items sorted by rank; a tie between the two predictions counts one half.
    """
    unique = np.unique(items)
    rank_of = {int(item): float(z_ranks[items == item][0]) for item in unique}
    order = sorted(unique, key=lambda item: rank_of[int(item)])
    means = np.array([predictions[items == item].mean() for item in order])
    difference = means[None, :] - means[:, None]
    upper = difference[np.triu_indices(means.size, k=1)]
    return float((np.sum(upper > 0) + 0.5 * np.sum(upper == 0)) / upper.size)


def _held_out_scores(
    x: np.ndarray, z_ranks: np.ndarray, items: np.ndarray, *, alpha: float, seed: int
) -> dict[str, float]:
    """Held-out-item scores of the ridge at `alpha` under a seeded repeated grouped split.

    Whole items are dealt to folds: an item's fold is its position in a seeded shuffle of the
    item ids modulo the number of folds, redrawn each repeat, and the scores are averaged over
    repeats.
    """
    unique = np.unique(items)
    n_splits = int(min(FOLDS, unique.size))
    if n_splits < 2:
        raise ValueError("a held-out split needs at least two items")
    rng = seeded_rng(int(seed), REPRESENT_SALT, _FOLD_KEY)
    r2s, spearmans, accuracies = [], [], []
    for _ in range(FOLD_REPEATS):
        shuffled = unique[rng.choice(unique.size, size=unique.size, replace=False)]
        fold_of = {int(item): position % n_splits for position, item in enumerate(shuffled)}
        folds = np.array([fold_of[int(item)] for item in items])
        predictions = np.empty_like(z_ranks)
        for fold in range(n_splits):
            train, test = folds != fold, folds == fold
            coefficients, intercept = ridge_fit(x[train], z_ranks[train], alpha)
            predictions[test] = x[test] @ coefficients + intercept
        residual = z_ranks - predictions
        r2s.append(1.0 - float(residual @ residual) / float(((z_ranks - z_ranks.mean()) ** 2).sum()))
        spearmans.append(rank_correlation(predictions, z_ranks))
        accuracies.append(_ordering_accuracy(predictions, items, z_ranks))
    return {"r2": float(np.mean(r2s)), "spearman": float(np.mean(spearmans)), "accuracy": float(np.mean(accuracies))}


def probe_metrics(states: ItemStates, *, seed: int) -> list[dict[str, object]]:
    """Rank decodability at one site: one ridge fit, scored on held-out items, as table rows.

    The rows carry `kind, metric, layer, value, n`: the regression's r2 and Spearman under
    `rank_regression`, and the pairwise ordering accuracy under `rank_order`; `n` is the number
    of occurrences fit on.
    """
    x = np.asarray(states.vectors, dtype=np.float64)
    z_ranks = _zscore(states.ranks)
    items = np.asarray(states.items)
    alpha = leave_one_out_alpha(x, z_ranks)
    scores = _held_out_scores(x, z_ranks, items, alpha=alpha, seed=seed)
    n = int(x.shape[0])
    layer = int(states.site)
    return [
        {"kind": str(ProbeKind.RANK_REGRESSION), "metric": "r2", "layer": layer, "value": scores["r2"], "n": n},
        {"kind": str(ProbeKind.RANK_REGRESSION), "metric": "spearman", "layer": layer, "value": scores["spearman"], "n": n},
        {"kind": str(ProbeKind.RANK_ORDER), "metric": "accuracy", "layer": layer, "value": scores["accuracy"], "n": n},
    ]


def probe_by_layer(
    model: RelationalTransformer,
    prompts: Sequence[Encoded],
    vocab: Vocabulary,
    *,
    ranks: Mapping[int, int] | np.ndarray,
    layers: Iterable[int],
    seed: int,
) -> pd.DataFrame:
    """Rank decodability at every requested site, the residual-stream sites read in one pass."""
    wanted = [int(layer) for layer in layers]
    rank_of = rank_array(ranks)
    rows: list[dict[str, object]] = []
    hidden = [site for site in wanted if site != EMBEDDING_SITE]
    states = occurrence_states(model, prompts, vocab, sites=hidden, ranks=rank_of) if hidden else {}
    for site in wanted:
        if site == EMBEDDING_SITE:
            rows.extend(probe_metrics(collect_item_states(model, prompts, vocab, site=site, ranks=rank_of), seed=seed))
        else:
            rows.extend(probe_metrics(states[site], seed=seed))
    return pd.DataFrame(rows, columns=["kind", "metric", "layer", "value", "n"])


def edit_sites(model: RelationalTransformer) -> tuple[int, ...]:
    """The sites an edit acts on: the embedding output, 0, and each block's output, 1..n_layers."""
    return tuple(range(len(model.blocks) + 1))


def fit_edit_directions(
    model: RelationalTransformer,
    prompts: Sequence[Encoded],
    vocab: Vocabulary,
    *,
    ranks: Mapping[int, int] | np.ndarray,
) -> tuple[tuple[RankDirection, ...], tuple[np.ndarray, ...]]:
    """The rank direction and the write calibration of every edit site, in site order.

    Both are fit at each site on the states of every item-token occurrence of the prompts, read
    in one forward pass; at site 0 the state of an occurrence is its item's embedding row.
    """
    states = occurrence_states(model, prompts, vocab, sites=edit_sites(model), ranks=ranks)
    rank_directions = tuple(fit_rank_direction(one.vectors, one.ranks, site=site) for site, one in states.items())
    calibrations = tuple(
        write_calibration(rank_direction, states[rank_direction.site].vectors, states[rank_direction.site].ranks)
        for rank_direction in rank_directions
    )
    return rank_directions, calibrations


def edit_direction(rank_direction: RankDirection, direction: Direction, seed: int) -> np.ndarray:
    """The unit direction an edit acts on: the fitted one, or a seeded random one of the width.

    The random direction is the first column of the Householder QR factor of a seeded standard
    normal draw: the draw normalized, with its sign flipped when its first entry is positive.
    """
    if Direction(direction) is Direction.RANK:
        return rank_direction.direction
    draw = np.random.default_rng(int(seed)).standard_normal(rank_direction.basis.shape[0])
    return np.linalg.qr(draw[:, None])[0][:, 0]


def direction_seed(seed: int, site: int) -> int:
    """The seed of the random direction at one site: the edit's seed plus the site."""
    return int(seed) + int(site)


def _site_edit(
    rank_direction: RankDirection, direction: Direction, seed: int, *, removed: float, inserted: np.ndarray | None = None
) -> SiteEdit:
    """The edit at the rank direction's site, along the fitted or that site's random direction."""
    unit = edit_direction(rank_direction, direction, direction_seed(seed, rank_direction.site))
    return SiteEdit(
        site=int(rank_direction.site),
        mean=torch.as_tensor(rank_direction.mean, dtype=torch.float64),
        read_back=torch.as_tensor(rank_direction.whitener @ unit, dtype=torch.float64),
        insertion=torch.as_tensor(rank_direction.unwhitener @ unit, dtype=torch.float64),
        removed=float(removed),
        inserted=None if inserted is None else torch.as_tensor(inserted, dtype=torch.float64),
    )


def erase_or_scale(
    rank_directions: Sequence[RankDirection],
    gain: float,
    *,
    direction: Direction = Direction.RANK,
    seed: int = RANDOM_DIRECTION_SEED,
) -> tuple[SiteEdit, ...]:
    """Scale every item token's coordinate along the direction by `gain`, at each fitted site.

    `x' = x - (1 - gain) (x - mean) A`: gain 1 leaves the coordinate, 0 erases it, -1 inverts
    it, and a gain above 1 amplifies it. The random control replaces the fitted direction at each
    site by that site's seeded random unit direction under the same mean and whitening.
    """
    return tuple(_site_edit(rank_direction, direction, seed, removed=1.0 - float(gain)) for rank_direction in rank_directions)


def write_calibration(rank_direction: RankDirection, rows: np.ndarray, ranks: Mapping[int, int] | np.ndarray) -> np.ndarray:
    """The line from rank to coordinate that write targets are read from.

    Fitted on rows given with their ranks, one rank per row (the occurrences the rank
    direction was fit on): the rows' coordinates along the fitted read-back functional are
    regressed on the z-scored ranks. Returns `(intercept, slope, rank mean, rank standard
    deviation)`.
    """
    rows = np.asarray(rows, dtype=np.float64)
    rank_values = rank_array(ranks).astype(np.float64)
    if rows.ndim != 2 or rows.shape[0] != rank_values.size:
        raise ValueError("one rank per row")
    coordinates = (rows - rank_direction.mean) @ rank_direction.read_back
    rank_mean, rank_std = float(rank_values.mean()), float(rank_values.std())
    if rank_std == 0:
        raise ValueError("every rank is the same, so no line from rank to coordinate exists")
    z = (rank_values - rank_mean) / rank_std
    slope = float(coordinates @ z / (z @ z))
    return np.array([float(coordinates.mean()), slope, rank_mean, rank_std])


def target_coordinates(calibration: np.ndarray, target_ranks: Mapping[int, int] | np.ndarray) -> np.ndarray:
    """Per-item target coordinates: the calibration line at the target ranks.

    The target ranks are z-scored with the calibration's own rank mean and standard deviation,
    so every reordering of the same rank values lands on one common scale.
    """
    intercept, slope, rank_mean, rank_std = (float(value) for value in np.asarray(calibration, dtype=np.float64))
    z = (rank_array(target_ranks).astype(np.float64) - rank_mean) / rank_std
    return intercept + slope * z


def write_ordering(
    rank_directions: Sequence[RankDirection],
    target_ranks: Mapping[int, int] | np.ndarray,
    strength: float,
    *,
    calibrations: Sequence[np.ndarray],
    direction: Direction = Direction.RANK,
    seed: int = RANDOM_DIRECTION_SEED,
) -> tuple[SiteEdit, ...]:
    """Move each item token's coordinate toward its target rank's coordinate by `strength`.

    `x' = x - s (x - mean) A + s t v` at each rank direction's site, with that site's
    calibration: the coordinate read back afterwards is `(1 - s) r(x) + s t`, so strength 0
    is the identity and strength 1 writes the target exactly. The targets hold one
    coordinate per item of the target ranks, the calibration line at the item's target rank,
    so an item the fit never saw at a site has a target there too. The random control
    inserts the same targets along each site's seeded random direction.
    """
    if len(calibrations) != len(rank_directions):
        raise ValueError(f"{len(calibrations)} calibrations for {len(rank_directions)} sites")
    s = float(strength)
    return tuple(
        _site_edit(rank_direction, direction, seed, removed=s, inserted=s * target_coordinates(calibration, target_ranks))
        for rank_direction, calibration in zip(rank_directions, calibrations, strict=True)
    )


@contextlib.contextmanager
def edited(model: RelationalTransformer, edit: Sequence[SiteEdit]) -> Iterator[None]:
    """Apply the edit to the item-token states at its sites in every forward of the network.

    Site 0 is edited as the embedding returns and site l as block l returns, so the state edited
    at one site is the one the next block reads, and through the keys and values the blocks
    cache it reaches every later position. Every forward is edited: over a whole
    sequence, over the tokens fed after a cache (each emitted token), and over the query items
    appended at a reasoning-step value. The item positions and the items are read from the token
    ids of each forward.
    """
    modules = (model.embedding, *model.blocks)
    by_site: dict[int, SiteEdit] = {}
    for one in edit:
        if not 0 <= one.site < len(modules) or one.site in by_site:
            raise ValueError(f"an edit acts once at each of the sites 0..{len(modules) - 1}, not at {one.site}")
        if one.inserted is not None and one.inserted.shape != (model.vocab.pool_size,):
            raise ValueError(f"{one.inserted.shape[0]} targets for a pool of {model.vocab.pool_size} items")
        by_site[one.site] = one.to(model.embedding.weight)
    place: dict[str, torch.Tensor] = {}

    def locate(_module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        tokens = (args[0] if args else kwargs["tokens"]).long()
        place["mask"] = item_embedding_sites(tokens, model.vocab)
        place["items"] = tokens[place["mask"]] - model.vocab.first_item

    def at(site: int) -> Callable:
        def hook(_module: torch.nn.Module, _args: tuple, output: torch.Tensor | tuple) -> torch.Tensor | tuple:
            states = output if site == EMBEDDING_SITE else output[0]
            changed = states.clone()
            changed[place["mask"]] = by_site[site].apply(states[place["mask"]], place["items"])
            return changed if site == EMBEDDING_SITE else (changed, *output[1:])

        return hook

    handles = [model.register_forward_pre_hook(locate, with_kwargs=True)]
    try:
        for site in sorted(by_site):
            handles.append(modules[site].register_forward_hook(at(site)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def reversed_ordering(ranks: Mapping[int, int] | np.ndarray) -> np.ndarray:
    """The ranks read bottom-up: rank `n - 1 - r` for every item."""
    rank_values = rank_array(ranks)
    return rank_values.size - 1 - rank_values


def sample_unrelated_ordering(
    ranks: Mapping[int, int] | np.ndarray,
    *,
    seed: int,
    target_seed: int = 0,
    max_abs_spearman: float = MAX_ABS_SPEARMAN,
    max_tries: int = 1000,
) -> np.ndarray:
    """A seeded reordering of the ranks whose rank correlation with them is at most the bound.

    Reorderings are drawn from the generator keyed on the seed and `target_seed` until one is
    accepted, so a given pair of seeds always names the same ordering.
    """
    familiar = rank_array(ranks)
    rng = seeded_rng(int(seed), REPRESENT_SALT, _TARGET_KEY, int(target_seed))
    for _ in range(int(max_tries)):
        candidate = familiar[rng.choice(familiar.size, size=familiar.size, replace=False)]
        if abs(rank_correlation(familiar, candidate)) <= float(max_abs_spearman):
            return candidate
    raise ValueError(f"no reordering within a rank correlation of {max_abs_spearman} in {max_tries} draws")


def write_targets(familiar_ranks: Mapping[int, int] | np.ndarray, *, seed: int) -> dict[WriteTarget, tuple[np.ndarray, float]]:
    """Each write target's ranks and its rank correlation with the familiar ordering.

    The familiar ordering itself (a control that writes what is already there, correlation 1),
    its reverse (correlation -1), and a seeded unrelated ordering with its sampled correlation.
    """
    familiar = rank_array(familiar_ranks)
    unrelated = sample_unrelated_ordering(familiar, seed=seed)
    return {
        WriteTarget.FAMILIAR: (familiar, 1.0),
        WriteTarget.REVERSED: (reversed_ordering(familiar), -1.0),
        WriteTarget.UNRELATED: (unrelated, rank_correlation(familiar, unrelated)),
    }


def edit_specs(
    rank_directions: Sequence[RankDirection],
    calibrations: Sequence[np.ndarray],
    familiar_ranks: Mapping[int, int] | np.ndarray,
    *,
    seed: int,
    gains: Sequence[float] = ERASE_GAINS,
    strengths: Sequence[float] = WRITE_STRENGTHS,
) -> list[EditSpec]:
    """Every edited battery of the representation experiment, in the order they are run.

    Each edit acts at every site of `rank_directions`, with that site's calibration for a
    write. The erase grid is every gain along the rank direction and along the random
    direction. The write grid is every target at every strength along the rank direction,
    plus the unrelated target at every strength along the random direction.
    """
    specs: list[EditSpec] = []
    for gain in gains:
        for direction in Direction:
            specs.append(
                EditSpec(
                    kind=EditKind.ERASE,
                    direction=direction,
                    target=None,
                    gain=float(gain),
                    strength=float("nan"),
                    edit=erase_or_scale(rank_directions, float(gain), direction=direction),
                    target_ranks=None,
                    spearman_vs_familiar=float("nan"),
                )
            )
    targets = write_targets(familiar_ranks, seed=seed)
    written = [(target, Direction.RANK) for target in WriteTarget] + [(WriteTarget.UNRELATED, Direction.RANDOM)]
    for target, direction in written:
        target_ranks, spearman = targets[target]
        for strength in strengths:
            specs.append(
                EditSpec(
                    kind=EditKind.WRITE,
                    direction=direction,
                    target=target,
                    gain=float("nan"),
                    strength=float(strength),
                    edit=write_ordering(
                        rank_directions, target_ranks, float(strength), calibrations=calibrations, direction=direction
                    ),
                    target_ranks=target_ranks,
                    spearman_vs_familiar=float(spearman),
                )
            )
    return specs


def aggregate_edit(trials: pd.DataFrame, spec: EditSpec) -> pd.DataFrame:
    """Aggregate one edited battery's trials and label the rows with the edit.

    Beyond the evaluation aggregates: `written_consistent_frac`, the fraction of trials whose
    predicted sign follows the written ordering, and `spearman_vs_familiar`, the written
    ordering's rank correlation with the familiar one; both missing on an erase.
    """
    aggregates = evaluate.aggregate(trials, group_cols=_GROUP)
    group = list(_GROUP)
    if spec.target_ranks is None:
        aggregates["written_consistent_frac"] = float("nan")
    else:
        target = rank_array(spec.target_ranks)
        expected = np.where(target[trials["item_a"].to_numpy()] < target[trials["item_b"].to_numpy()], 1, -1)
        consistent = trials.assign(_written=(trials["predicted_sign"].to_numpy() == expected).astype(float))
        fractions = consistent.groupby(group, dropna=False, sort=False)["_written"].mean().reset_index()
        aggregates = aggregates.merge(fractions.rename(columns={"_written": "written_consistent_frac"}), on=group, how="left")
    aggregates["edit_kind"] = str(spec.kind)
    aggregates["direction"] = str(spec.direction)
    aggregates["target"] = "" if spec.target is None else str(spec.target)
    aggregates["gain"] = float(spec.gain)
    aggregates["strength"] = float(spec.strength)
    aggregates["spearman_vs_familiar"] = float(spec.spearman_vs_familiar)
    return aggregates.loc[:, list(_EDIT_COLUMNS)]


def edit_table(
    model: RelationalTransformer,
    tests: Battery,
    vocab: Vocabulary,
    specs: Sequence[EditSpec],
    *,
    step: int,
    chain_algorithm: ChainAlgorithm = ChainAlgorithm.MAIN,
) -> pd.DataFrame:
    """The edit table of one checkpoint: the battery scored under every edit, aggregated in order.

    Every battery runs with its edit applied in every forward (see `edited`), at every
    reasoning-step value the battery declares, under the tag `represent`; `chain_algorithm` is
    the one the network was trained on, which sets the length chain validity is truncated at.
    """
    frames = []
    for spec in specs:
        with edited(model, spec.edit):
            trials, _ = evaluate.evaluate_battery(
                model,
                tests,
                vocab,
                step=int(step),
                tag=REPRESENT_TAG,
                chain_algorithm=chain_algorithm,
            )
        frames.append(aggregate_edit(trials, spec))
    if not frames:
        raise ValueError("no edit to score the battery under")
    return pd.concat(frames, ignore_index=True)


def probe_table(probe: pd.DataFrame, *, seed: int, step: int) -> pd.DataFrame:
    """The probe rows as the run table holds them, with the seed and the step added."""
    return probe.assign(seed=int(seed), step=int(step)).loc[:, list(_PROBE_COLUMNS)]
