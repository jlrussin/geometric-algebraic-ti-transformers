"""Pretraining and finetuning streams, replay, and batching.

Realizes the extended methods' "Pretraining distribution" and "Fixed-order learning
distribution" paragraphs. Every example is a pure function of its position
(seed, stage, step, index in batch), so a stream can be resumed, sharded, or
recomputed without state. Pretraining draws a cue from the cue-frequency
distribution and fresh items from the pool on every example; a trial shown
without its chain still fills nine emissions, each the answer, so every
pretraining sequence has one length. Finetuning presents adjacent queries on
the fixed ordering(s), two distractors shown at their buckets' positions, and
answer-only targets; a proportion of trials, one in ten by default as in the
human protocol, shows other items in its context instead of the fixed
ordering, as the IWL test does, and its answer follows the familiar ordering. Replay dilutes rather
than augments: with the replay proportion, a finetuning example is replaced
by a fresh pretraining example carrying a chain target. Reserved cues never
appear in any stream. Batches can be built ahead of the consumer by worker
processes, as host arrays, and are the same whether or not they are.
"""

from __future__ import annotations

import functools
import itertools
import multiprocessing
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum

import numpy as np
import torch

from geo_alg_ti.chains import Chain, ChainAlgorithm, generate_chain
from geo_alg_ti.config import Config, FinetuningConfig, PretrainingConfig, TaskConfig
from geo_alg_ti.orderings import CueDistribution, OrderingLaw, cue_weights, seeded_rng
from geo_alg_ti.sequences import EMISSION_LEN, Encoded, Target, Vocabulary, encode
from geo_alg_ti.trials import (
    DISTRACTOR_POOL_SIZE,
    N_MAIN_ITEMS,
    ContextKind,
    FixedOrdering,
    Relation,
    arrange_fixed_ordering,
    build_fixed_ordering,
    check_pool,
    sample_finetuning_trial,
    sample_pretraining_trial,
    with_context,
)

STREAM_SALT: int = 0x57E3
# The default proportion of finetuning trials whose context shows other items, as in the human
# protocol; their answers are scored against the familiar ordering.
UNINFORMATIVE_CONTEXT_PROP: float = 0.1
# Batches kept in flight per worker process, ahead of the consumer.
PREFETCH_BATCHES: int = 4


class RowKind(IntEnum):
    """What one training row holds: a pretraining trial (also when replayed during finetuning), or
    a fixed-ordering trial whose context is congruent (shows the ordering) or uninformative."""

    PRETRAINING = 0
    CONGRUENT = 1
    UNINFORMATIVE = 2


@dataclass(frozen=True)
class ReasoningPlan:
    """Which rows of a batch write reasoning steps before their scored answer, decided on the host
    when the batch is stacked: their indices, the steps each writes, and the prompt length every
    fixed-ordering row shares (0 in a batch without one)."""

    rows: torch.Tensor
    steps: int
    prompt_len: int

    @property
    def query_starts(self) -> tuple[int, ...]:
        """Where the two query items appended before a fixed-ordering row's answer begin: after no
        reasoning steps, and after `steps` steps when some row writes them; none in a batch
        without a fixed-ordering row."""
        if self.prompt_len == 0:
            return ()
        after_steps = (self.prompt_len + EMISSION_LEN * self.steps,) if self.steps else ()
        return (self.prompt_len, *after_steps)


class Stage(StrEnum):
    """The two training stages."""

    PRETRAINING = "pretraining"
    FINETUNING = "finetuning"


def stage_section(cfg: Config, stage: Stage) -> PretrainingConfig | FinetuningConfig:
    """The stage's own section of the configuration."""
    return cfg.pretraining if Stage(stage) is Stage.PRETRAINING else cfg.finetuning


class Layout(StrEnum):
    """Conflicting: same items, different cues. Disjoint: disjoint items, one cue."""

    CONFLICTING = "conflicting"
    DISJOINT = "disjoint"


@dataclass(frozen=True)
class FinetuningDesign:
    """The fixed ordering(s) a finetuning run presents."""

    layout: Layout
    orderings: tuple[FixedOrdering, ...]

    @property
    def presented_items(self) -> frozenset[int]:
        """Every item any of the orderings presents, distractor pools included."""
        return frozenset().union(*(ordering.presented_items for ordering in self.orderings))


@dataclass
class Batch:
    """One training batch on the target device, right-padded with `<pad>` and loss-masked.

    `answer_index` is each row's answer `<rel>` slot, the one position the answer accuracy reads.
    A batch stacked from a stream also carries, per row, its `RowKind`, the ranks of its queried
    items in the fixed ordering, and the reasoning steps the network writes before answering;
    `reasoning` says, as decided on the host, which rows write those steps and how many.
    """

    tokens: torch.Tensor
    scalars: torch.Tensor
    token_loss: torch.Tensor
    scalar_loss: torch.Tensor
    answer_index: torch.Tensor
    index: tuple[int, ...]
    row_kind: torch.Tensor | None = None
    query_ranks: torch.Tensor | None = None
    reasoning_steps: torch.Tensor | None = None
    reasoning: ReasoningPlan | None = None

    def pin_memory(self) -> Batch:
        """The same batch in page-locked host memory, from which a CUDA copy need not wait."""
        return self._map(lambda tensor: tensor.pin_memory())

    def to(self, device: str) -> Batch:
        """The same batch on `device`; a CUDA copy goes from pinned memory without waiting."""
        on_host = self.tokens.device.type == "cpu"
        if torch.device(device).type == "cuda" and on_host and not self.tokens.is_pinned():
            return self.pin_memory().to(device)
        return self._map(lambda tensor: tensor.to(device, non_blocking=True))

    def _map(self, move: Callable[[torch.Tensor], torch.Tensor]) -> Batch:
        def optional(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return None if tensor is None else move(tensor)

        return Batch(
            tokens=move(self.tokens),
            scalars=move(self.scalars),
            token_loss=move(self.token_loss),
            scalar_loss=move(self.scalar_loss),
            answer_index=move(self.answer_index),
            index=self.index,
            row_kind=optional(self.row_kind),
            query_ranks=optional(self.query_ranks),
            reasoning_steps=optional(self.reasoning_steps),
            reasoning=None if self.reasoning is None else replace(self.reasoning, rows=move(self.reasoning.rows)),
        )


@dataclass
class HostBatch:
    """One batch as host arrays, the form that crosses a process boundary."""

    tokens: np.ndarray
    scalars: np.ndarray
    token_loss: np.ndarray
    scalar_loss: np.ndarray
    answer_index: np.ndarray
    index: tuple[int, ...]
    row_kind: np.ndarray
    query_ranks: np.ndarray
    reasoning_steps: np.ndarray
    reasoning_rows: np.ndarray
    reasoning_plan_steps: int
    prompt_len: int

    def to_batch(self, device: str = "cpu") -> Batch:
        """The same batch as tensors on `device`."""
        return Batch(
            tokens=torch.from_numpy(self.tokens),
            scalars=torch.from_numpy(self.scalars),
            token_loss=torch.from_numpy(self.token_loss),
            scalar_loss=torch.from_numpy(self.scalar_loss),
            answer_index=torch.from_numpy(self.answer_index),
            index=self.index,
            row_kind=torch.from_numpy(self.row_kind),
            query_ranks=torch.from_numpy(self.query_ranks),
            reasoning_steps=torch.from_numpy(self.reasoning_steps),
            reasoning=ReasoningPlan(
                rows=torch.from_numpy(self.reasoning_rows),
                steps=int(self.reasoning_plan_steps),
                prompt_len=int(self.prompt_len),
            ),
        ).to(device)


# Each stage keys its own examples, so the two streams never draw the same problem at one step.
_STAGE_KEYS: dict[Stage, int] = {Stage.PRETRAINING: 0, Stage.FINETUNING: 1}


def example_rng(seed: int, stage: Stage, step: int, index: int) -> np.random.Generator:
    """Return the generator for one example, keyed on its position so streams are stateless."""
    if min(int(step), int(index)) < 0:
        raise ValueError("a position in a stream is never negative")
    return seeded_rng(int(seed), STREAM_SALT, _STAGE_KEYS[Stage(stage)], int(step), int(index))


def design_rng(seed: int) -> np.random.Generator:
    """Return the generator that draws a run's fixed orderings, keyed on its seed alone."""
    return seeded_rng(int(seed), STREAM_SALT)


def build_finetuning_design(law: OrderingLaw, rng: np.random.Generator, *, layout: Layout, n_orderings: int) -> FinetuningDesign:
    """Draw the fixed orderings for the layout."""
    layout = Layout(layout)
    n_orderings = int(n_orderings)
    if n_orderings < 1:
        raise ValueError("a finetuning design presents at least one ordering")
    check_pool(law.pool_size, layout=layout, n_orderings=n_orderings)
    training_cues = law.training_cues()
    if layout is Layout.DISJOINT:
        # Item-disjoint orderings under one cue: each draw excludes everything already presented.
        cue = int(rng.choice(training_cues))
        orderings: list[FixedOrdering] = []
        presented: set[int] = set()
        for _ in range(n_orderings):
            ordering = build_fixed_ordering(law, rng, cue=cue, exclude=presented)
            presented |= ordering.presented_items
            orderings.append(ordering)
        return FinetuningDesign(layout=layout, orderings=tuple(orderings))
    cues = [int(cue) for cue in rng.choice(training_cues, size=n_orderings, replace=False)]
    return FinetuningDesign(layout=layout, orderings=_conflicting_orderings(law, rng, cues=cues))


def pretraining_example(cfg: TaskConfig, law: OrderingLaw, vocab: Vocabulary, *, rng: np.random.Generator) -> Encoded:
    """One pretraining sequence: the chain with probability p(CoT), else nine answer emissions."""
    shows_chain = carries_chain(cfg, rng)
    return _pretraining_encoded(cfg, law, vocab, rng=rng, with_chain=shows_chain)


def carries_chain(cfg: TaskConfig, rng: np.random.Generator) -> bool:
    """Whether the next pretraining trial shows its chain of thought: one draw against p(CoT)."""
    return bool(rng.random() < float(cfg.p_cot))


def finetuning_example(
    cfg: TaskConfig,
    law: OrderingLaw,
    vocab: Vocabulary,
    *,
    design: FinetuningDesign,
    replay_proportion: float,
    uninformative_context_prop: float = UNINFORMATIVE_CONTEXT_PROP,
    reasoning_proportion: float = 0.0,
    reasoning_steps: int = 0,
    rng: np.random.Generator,
) -> Encoded:
    """One finetuning sequence, or a replayed pretraining sequence with the replay proportion.

    With the uninformative-context proportion, the trial's context shows other items instead of
    the fixed ordering, as the IWL test does, and the answer follows the familiar ordering. A
    trial with a congruent context is marked, with the reasoning proportion, to have the network
    write `reasoning_steps` reasoning steps before its answer; that draw comes last, so the trial
    itself does not depend on it.
    """
    if rng.random() < float(replay_proportion):
        # Replay dilutes: the pretraining example takes this row rather than joining it. It shows
        # its chain, as every pretraining trial does at p(CoT) 1, the setting finetuning continues.
        return _pretraining_encoded(cfg, law, vocab, rng=rng, with_chain=True)
    ordering = design.orderings[int(rng.integers(len(design.orderings)))]
    trial = sample_finetuning_trial(law, rng, ordering=ordering)
    proportion = float(uninformative_context_prop)
    kind = RowKind.CONGRUENT
    if proportion > 0 and rng.random() < proportion:
        trial = with_context(trial, ContextKind.UNINFORMATIVE, rng, law=law, exclude=design.presented_items)
        kind = RowKind.UNINFORMATIVE
    steps = 0
    if kind is RowKind.CONGRUENT and float(reasoning_proportion) > 0 and rng.random() < float(reasoning_proportion):
        steps = int(reasoning_steps)
    # The test battery queries the fixed ordering's items within a nine-item ordering that adds a
    # distractor above and below them, so their zero-based ranks there, as in its per-pair table,
    # are one more than their ranks among the main items.
    ranks = (int(trial.query_ranks[0]) + 1, int(trial.query_ranks[1]) + 1)
    encoded = encode(trial, None, vocab, target=Target.ANSWER_ONLY)
    return replace(encoded, row_kind=int(kind), query_ranks=ranks, reasoning_steps=steps)


def pretraining_stream(
    cfg: TaskConfig,
    law: OrderingLaw,
    vocab: Vocabulary,
    *,
    seed: int,
    batch_size: int,
    start_step: int = 0,
    device: str = "cpu",
    workers: int = 0,
) -> Iterator[Batch]:
    """Yield pretraining batches from `start_step` onward, built ahead by `workers` processes."""
    example = functools.partial(pretraining_example, cfg, law, vocab)
    return _stream(
        example,
        vocab,
        stage=Stage.PRETRAINING,
        seed=seed,
        batch_size=batch_size,
        start_step=start_step,
        device=device,
        workers=workers,
    )


def finetuning_stream(
    cfg: TaskConfig,
    law: OrderingLaw,
    vocab: Vocabulary,
    *,
    design: FinetuningDesign,
    seed: int,
    batch_size: int,
    replay_proportion: float,
    uninformative_proportion: float = UNINFORMATIVE_CONTEXT_PROP,
    reasoning_proportion: float = 0.0,
    reasoning_steps: int = 0,
    start_step: int = 0,
    device: str = "cpu",
    workers: int = 0,
) -> Iterator[Batch]:
    """Yield finetuning batches with replay from `start_step` onward, built ahead by `workers`."""
    example = functools.partial(
        finetuning_example,
        cfg,
        law,
        vocab,
        design=design,
        replay_proportion=replay_proportion,
        uninformative_context_prop=uninformative_proportion,
        reasoning_proportion=reasoning_proportion,
        reasoning_steps=reasoning_steps,
    )
    return _stream(
        example,
        vocab,
        stage=Stage.FINETUNING,
        seed=seed,
        batch_size=batch_size,
        start_step=start_step,
        device=device,
        workers=workers,
    )


class _StepBatches:
    """Batch `step` of one stream, computed from its position alone, so every process agrees."""

    def __init__(
        self, example: Callable[..., Encoded], vocab: Vocabulary, *, stage: Stage, seed: int, batch_size: int
    ) -> None:
        self.example = example
        self.vocab = vocab
        self.stage = Stage(stage)
        self.seed = int(seed)
        self.batch_size = int(batch_size)

    def build(self, step: int) -> HostBatch:
        rows = range(self.batch_size)
        examples = [self.example(rng=example_rng(self.seed, self.stage, step, index)) for index in rows]
        return stack(examples, self.vocab, first_index=int(step) * self.batch_size)


def _stream(
    example: Callable[..., Encoded],
    vocab: Vocabulary,
    *,
    stage: Stage,
    seed: int,
    batch_size: int,
    start_step: int,
    device: str,
    workers: int,
) -> Iterator[Batch]:
    """Yield batches of `batch_size` examples from `start_step` on, each keyed on its position.

    With `workers`, that many processes build the batches ahead of the consumer, a bounded
    number at a time, as host arrays; this process turns each into tensors on the device.
    The batches are the same either way.
    """
    batches = _StepBatches(example, vocab, stage=stage, seed=seed, batch_size=batch_size)
    steps = itertools.count(int(start_step))
    if int(workers) <= 0:
        for step in steps:
            yield batches.build(step).to_batch(device)
        return
    # A server started before any device work forks the workers, on every platform alike.
    pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("forkserver"))
    try:
        first = itertools.islice(steps, PREFETCH_BATCHES * int(workers))
        ahead = deque(pool.submit(batches.build, step) for step in first)
        while True:
            stacked = ahead.popleft().result()
            ahead.append(pool.submit(batches.build, next(steps)))
            yield stacked.to_batch(device)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def stack(encoded: Sequence[Encoded], vocab: Vocabulary, *, first_index: int = 0) -> HostBatch:
    """Stack sequences into host arrays, right-padding with `<pad>` outside every loss mask.

    `first_index` is the position of the first row in the stream it came from; the rows are
    numbered on from it.
    """
    if not encoded:
        raise ValueError("a batch holds at least one sequence")
    longest = max(int(example.tokens.size) for example in encoded)
    tokens = np.full((len(encoded), longest), vocab.pad, dtype=np.int64)
    scalars = np.zeros((len(encoded), longest), dtype=np.float32)
    token_loss = np.zeros((len(encoded), longest), dtype=bool)
    scalar_loss = np.zeros((len(encoded), longest), dtype=bool)
    answer_index = np.zeros(len(encoded), dtype=np.int64)
    row_kind = np.array([int(example.row_kind) for example in encoded], dtype=np.int64)
    query_ranks = np.array([example.query_ranks for example in encoded], dtype=np.int64).reshape(len(encoded), 2)
    reasoning_steps = np.array([int(example.reasoning_steps) for example in encoded], dtype=np.int64)
    reasoning_rows = np.flatnonzero(reasoning_steps > 0)
    planned = {int(steps) for steps in reasoning_steps[reasoning_rows]}
    if len(planned) > 1:
        raise ValueError(f"the rows that reason write different numbers of steps: {sorted(planned)}")
    prompts = {int(example.prompt_len) for example in encoded if example.row_kind != RowKind.PRETRAINING}
    if len(prompts) > 1:
        raise ValueError(f"the fixed-ordering rows have prompts of different lengths: {sorted(prompts)}")
    for row, example in enumerate(encoded):
        length = int(example.tokens.size)
        tokens[row, :length] = example.tokens
        scalars[row, :length] = example.scalars
        token_loss[row, :length] = example.token_loss
        scalar_loss[row, :length] = example.scalar_loss
        answer_index[row] = example.answer_index
    return HostBatch(
        tokens=tokens,
        scalars=scalars,
        token_loss=token_loss,
        scalar_loss=scalar_loss,
        answer_index=answer_index,
        index=tuple(range(int(first_index), int(first_index) + len(encoded))),
        row_kind=row_kind,
        query_ranks=query_ranks,
        reasoning_steps=reasoning_steps,
        reasoning_rows=reasoning_rows.astype(np.int64),
        reasoning_plan_steps=planned.pop() if planned else 0,
        prompt_len=prompts.pop() if prompts else 0,
    )


def _pretraining_encoded(
    cfg: TaskConfig, law: OrderingLaw, vocab: Vocabulary, *, rng: np.random.Generator, with_chain: bool
) -> Encoded:
    """One pretraining trial encoded with its chain, or with the answer filling every emission."""
    trial = sample_pretraining_trial(law, rng, weights=training_cue_weights(cfg))
    if with_chain:
        chain = generate_chain(trial, ChainAlgorithm(cfg.chain_algorithm))
    else:
        chain = Chain(emissions=(), answer=Relation(*trial.query, trial.reference_sign), reached_query_item=True)
    return encode(trial, chain, vocab, target=Target.CHAIN)


@functools.lru_cache(maxsize=None)
def training_cue_weights(task: TaskConfig) -> np.ndarray:
    """How often pretraining draws each training cue, held for the process once it is built.

    The test battery draws its `training_distribution` cues from these same weights.
    """
    distribution = task.cue_distribution
    weights = cue_weights(
        CueDistribution(distribution.kind),
        int(task.n_training_cues),
        skew_alpha=float(distribution.skew_alpha),
        frequent_cue_proportion=float(distribution.frequent_cue_prop),
    )
    weights.setflags(write=False)
    return weights


def _conflicting_orderings(
    law: OrderingLaw, rng: np.random.Generator, *, cues: Sequence[int]
) -> tuple[FixedOrdering, ...]:
    """One item set and one distractor pool, ordered and dealt into buckets anew under each cue."""
    main = law.sample_items(rng, N_MAIN_ITEMS)
    distractors = law.sample_items(rng, DISTRACTOR_POOL_SIZE, exclude=main)
    return tuple(
        arrange_fixed_ordering(law, rng, cue=int(cue), main=main, distractors=distractors) for cue in cues
    )
