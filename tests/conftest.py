"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from geo_alg_ti import streams, trials
from geo_alg_ti.orderings import OrderingLaw
from geo_alg_ti.sequences import Vocabulary


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, so tests can read the committed configuration and experiment files."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def smoke_law() -> OrderingLaw:
    """The ordering law at smoke scale."""
    return OrderingLaw(seed=0, pool_size=48, n_training_cues=6, n_reserved_cues=2)


@pytest.fixture(scope="session")
def smoke_design(smoke_law) -> streams.FinetuningDesign:
    """The one fixed ordering seed 0 finetunes on at smoke scale."""
    return design_for(smoke_law)


def vocabulary_for(law: OrderingLaw) -> Vocabulary:
    """The vocabulary a law's pool and cues (training plus reserved) need."""
    return Vocabulary(pool_size=law.pool_size, n_cues=law.n_training_cues + law.n_reserved_cues)


def design_for(
    law: OrderingLaw, *, seed: int = 0, layout: str = "conflicting", n_orderings: int = 1
) -> streams.FinetuningDesign:
    """The fixed ordering(s) one seed finetunes on under a law."""
    return streams.build_finetuning_design(
        law, streams.design_rng(seed), layout=streams.Layout(layout), n_orderings=n_orderings
    )


def verbatim_trial(context, query, items=tuple(range(9)), cue=7) -> trials.Trial:
    """A congruent trial over `items` (top first) whose context is given verbatim."""
    rank = {item: r for r, item in enumerate(items)}
    return trials.Trial(
        cue=cue,
        items=tuple(items),
        query_ranks=(rank[query[0]], rank[query[1]]),
        displayed=tuple(items),
        context=tuple(context),
        context_items=tuple(items),
        context_kind=trials.ContextKind.CONGRUENT,
    )


def top_down(relations) -> set[tuple[int, int]]:
    """Each relation as (higher item, lower item), so orientation and position drop out."""
    return {(a, b) if s == 1 else (b, a) for a, b, s in relations}


def adjacent_pairs(items) -> set[tuple[int, int]]:
    """The adjacent (higher, lower) pairs of an ordering given highest rank first."""
    return set(zip(items[:-1], items[1:], strict=False))
