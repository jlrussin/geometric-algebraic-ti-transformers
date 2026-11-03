"""Pretraining and finetuning streams, replay, layouts, batching."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import design_for, vocabulary_for

from geo_alg_ti import streams
from geo_alg_ti.chains import PAD_TO
from geo_alg_ti.config import CueDistributionConfig, TaskConfig
from geo_alg_ti.orderings import FREQUENT_CUE, OrderingLaw
from geo_alg_ti.sequences import EMISSION_LEN, Target, encode
from geo_alg_ti.trials import (
    DISTRACTOR_POOL_SIZE,
    FRESH_TEST_ITEMS,
    N_MAIN_ITEMS,
    sample_finetuning_trial,
    sample_pretraining_trial,
)

LAW = OrderingLaw(seed=0, pool_size=120, n_training_cues=12, n_reserved_cues=4)
VOCAB = vocabulary_for(LAW)
RESERVED_TOKENS = frozenset(VOCAB.cue_token(cue) for cue in LAW.reserved_cues())
# The item pool and cue counts of the canonical configuration.
CANONICAL = dict(pool_size=300, n_training_cues=100, n_reserved_cues=16)

# Draws both layouts in another interpreter and prints each ordering, one per line.
DESIGN_SCRIPT = """
import sys
from geo_alg_ti import streams
from geo_alg_ti.orderings import OrderingLaw

seed = int(sys.argv[1])
law = OrderingLaw(seed=0, pool_size=120, n_training_cues=12, n_reserved_cues=4)
for layout, n_orderings in (("conflicting", 3), ("disjoint", 2)):
    rng = streams.design_rng(seed)
    design = streams.build_finetuning_design(law, rng, layout=layout, n_orderings=n_orderings)
    for ordering in design.orderings:
        dealt = [item for bucket in ordering.buckets for item in bucket]
        print(ordering.cue, *ordering.items, *dealt)
"""


def _task(p_cot: float = 1.0, *, frequent_cue_prop: float = 0.0) -> TaskConfig:
    """The task section at test scale, with the chain proportion and frequent cue under test."""
    kind = "frequent_cue" if frequent_cue_prop > 0 else "uniform"
    return TaskConfig(
        item_pool_size=LAW.pool_size,
        n_training_cues=LAW.n_training_cues,
        n_reserved_cues=LAW.n_reserved_cues,
        cue_distribution=CueDistributionConfig(kind=kind, skew_alpha=0.0, frequent_cue_prop=frequent_cue_prop),
        p_cot=p_cot,
        chain_algorithm="main",
        show_cue=True,
    )


def _design(layout: str = "conflicting", n_orderings: int = 1, seed: int = 0) -> streams.FinetuningDesign:
    """The fixed ordering(s) one seed finetunes on under the test-scale law."""
    return design_for(LAW, seed=seed, layout=layout, n_orderings=n_orderings)


def _pretraining(step: int, index: int, *, seed: int = 0, p_cot: float = 1.0):
    """One pretraining example at a named position in the stream."""
    rng = streams.example_rng(seed, streams.Stage.PRETRAINING, step, index)
    return streams.pretraining_example(_task(p_cot), LAW, VOCAB, rng=rng)


def _finetuning(step: int, index: int, *, design, replay: float = 0.0, seed: int = 0):
    """One finetuning example at a named position in the stream."""
    rng = streams.example_rng(seed, streams.Stage.FINETUNING, step, index)
    return streams.finetuning_example(_task(), LAW, VOCAB, design=design, replay_proportion=replay, rng=rng)


def _is_chain_target(example) -> bool:
    """Whether an example is laid out as a padded chain target rather than an answer-only one."""
    assert example.n_emissions in (1, PAD_TO)
    return example.n_emissions == PAD_TO


def _query_items(example) -> tuple[int, int]:
    """The two queried items of an encoded example, read from the prompt head."""
    return int(example.tokens[3]) - VOCAB.first_item, int(example.tokens[4]) - VOCAB.first_item


def _context_items(example) -> set[int]:
    """Every item shown in an encoded example's context relations."""
    body = example.tokens[7 : example.prompt_len]
    return {int(token) - VOCAB.first_item for token in body if VOCAB.is_item(int(token))}


def test_example_is_a_function_of_its_position():
    """The same (seed, stage, step, index) gives the same tokens regardless of call order."""
    positions = [(0, 0), (0, 3), (7, 2)]
    forward = [_pretraining(step, index) for step, index in positions]
    backward = [_pretraining(step, index) for step, index in reversed(positions)]
    for example, repeated in zip(forward, reversed(backward), strict=True):
        assert np.array_equal(example.tokens, repeated.tokens)
        assert np.array_equal(example.scalars, repeated.scalars)
        assert np.array_equal(example.token_loss, repeated.token_loss)
    assert not np.array_equal(forward[0].tokens, forward[1].tokens)
    design = _design()
    assert np.array_equal(
        _finetuning(4, 1, design=design).tokens, _finetuning(4, 1, design=design).tokens
    )
    # The two stages key their examples apart, so neither draws the other's problem at a step.
    assert not np.array_equal(_pretraining(0, 0).tokens, _finetuning(0, 0, design=design).tokens)
    assert not np.array_equal(_pretraining(0, 0).tokens, _pretraining(0, 0, seed=1).tokens)


def test_resuming_at_a_step_reproduces_the_same_batches():
    """A stream started at step s yields what the fresh stream yields at step s."""
    kwargs = dict(seed=0, batch_size=4)
    fresh = streams.pretraining_stream(_task(), LAW, VOCAB, **kwargs)
    batches = [next(fresh) for _ in range(4)]
    resumed = next(streams.pretraining_stream(_task(), LAW, VOCAB, start_step=3, **kwargs))
    assert torch.equal(resumed.tokens, batches[3].tokens)
    assert torch.equal(resumed.scalars, batches[3].scalars)
    assert torch.equal(resumed.token_loss, batches[3].token_loss)
    assert resumed.index == batches[3].index == (12, 13, 14, 15)
    tuning = dict(design=_design(), seed=0, batch_size=4, replay_proportion=0.5)
    tuned = streams.finetuning_stream(_task(), LAW, VOCAB, **tuning)
    ahead = [next(tuned) for _ in range(3)]
    later = next(streams.finetuning_stream(_task(), LAW, VOCAB, start_step=2, **tuning))
    assert torch.equal(later.tokens, ahead[2].tokens) and later.index == ahead[2].index
    assert not torch.equal(ahead[0].tokens, ahead[1].tokens)


def test_chain_targets_appear_at_the_declared_proportion():
    """The draw that decides whether a trial shows its chain matches p(CoT) within tolerance."""
    n = 1000

    def carried(step: int, index: int, p_cot: float) -> bool:
        rng = streams.example_rng(0, streams.Stage.PRETRAINING, step, index)
        return streams.carries_chain(_task(p_cot), rng)

    count = sum(carried(step, index, 0.5) for step in range(n // 8) for index in range(8))
    # Four standard errors of a binomial proportion at one half over this many examples.
    assert abs(count / n - 0.5) < 4 * (0.25 / n) ** 0.5
    assert not any(carried(step, 0, 0.0) for step in range(40))
    assert all(carried(step, 0, 1.0) for step in range(40))


def test_a_trial_shown_without_its_chain_fills_every_emission_with_the_answer():
    """At p(CoT) 0 every pretraining target is nine answer emissions, all supervised."""
    for step in range(20):
        example = _pretraining(step, 0, p_cot=0.0)
        prompt = example.prompt_len
        assert example.tokens.size == prompt + PAD_TO * EMISSION_LEN and example.n_emissions == PAD_TO
        query = _query_items(example)
        emissions = example.tokens[prompt:].reshape(PAD_TO, EMISSION_LEN)
        for emission in emissions:
            assert (int(emission[0]) - VOCAB.first_item, int(emission[1]) - VOCAB.first_item) == query
            assert int(emission[2]) == VOCAB.rel and int(emission[3]) == VOCAB.sep
        signs = example.scalars[prompt:].reshape(PAD_TO, EMISSION_LEN)[:, 2]
        assert len(set(signs.tolist())) == 1 and abs(float(signs[0])) == 1.0
        assert bool(example.token_loss[prompt:].all()) and not example.token_loss[:prompt].any()
        assert np.flatnonzero(example.scalar_loss).tolist() == [prompt + 2 + EMISSION_LEN * k for k in range(PAD_TO)]
        # With a chain, the same position is the same trial; its last answer copy carries the sign.
        twin = _pretraining(step, 0, p_cot=1.0)
        assert np.array_equal(twin.tokens[:prompt], example.tokens[:prompt])
        assert float(signs[0]) == float(twin.scalars[prompt + (PAD_TO - 1) * EMISSION_LEN + 2])


def test_pretraining_batches_hold_no_pad_token():
    """Chain and no-chain pretraining rows have one length, so a batch is never padded."""
    stream = streams.pretraining_stream(_task(0.5), LAW, VOCAB, seed=0, batch_size=16)
    all_answer: set[bool] = set()
    for _, batch in zip(range(5), stream, strict=False):
        assert int((batch.tokens == VOCAB.pad).sum()) == 0
        for row in batch.tokens.numpy():
            emissions = row[-PAD_TO * EMISSION_LEN :].reshape(PAD_TO, EMISSION_LEN)
            all_answer.add(bool((emissions == emissions[-1]).all()))
    stream.close()
    # Rows of nine answer copies and rows with a chain both occur, at one length.
    assert all_answer == {True, False}


def test_replay_replaces_rather_than_augments():
    """Batch size is constant and replayed rows carry chain targets."""
    batch_size, steps, replay = 64, 40, 0.1
    design = _design()
    without = streams.finetuning_stream(
        _task(), LAW, VOCAB, design=design, seed=0, batch_size=batch_size, replay_proportion=0.0
    )
    stream = streams.finetuning_stream(
        _task(), LAW, VOCAB, design=design, seed=0, batch_size=batch_size, replay_proportion=replay
    )
    for _ in range(steps):
        batch = next(stream)
        # Dilution, not augmentation: replay leaves the number of rows in a batch untouched.
        assert batch.tokens.shape[0] == batch_size == len(batch.index) == next(without).tokens.shape[0]
    examples = [
        _finetuning(step, index, design=design, replay=replay)
        for step in range(steps)
        for index in range(batch_size)
    ]
    replayed = sum(_is_chain_target(example) for example in examples)
    assert len(examples) == steps * batch_size
    assert abs(replayed / len(examples) - replay) < 4 * (replay * (1 - replay) / len(examples)) ** 0.5
    assert replayed > 0


def test_reserved_cues_never_appear_in_a_training_batch():
    """No stream ever emits a reserved cue token."""
    design = _design()
    pretraining = streams.pretraining_stream(_task(), LAW, VOCAB, seed=0, batch_size=8)
    finetuning = streams.finetuning_stream(
        _task(), LAW, VOCAB, design=design, seed=0, batch_size=8, replay_proportion=0.5
    )
    for _ in range(20):
        for batch in (next(pretraining), next(finetuning)):
            seen = set(batch.tokens.reshape(-1).tolist())
            assert not seen & RESERVED_TOKENS
            cues = set(batch.tokens[:, 1].tolist())
            assert cues <= {VOCAB.cue_token(cue) for cue in LAW.training_cues()}


def test_a_hidden_cue_changes_only_the_cue_position_of_every_training_sequence():
    """Hiding the cue writes the frequent cue's token at index 1 of every pretraining, finetuning,
    and replayed sequence and leaves every other position as the shown cue gives it, so the
    items still follow the orderings of the cues drawn."""
    hidden = replace(VOCAB, show_cue=False)
    design = _design()

    def streams_for(vocab):
        return (
            streams.pretraining_stream(_task(), LAW, vocab, seed=0, batch_size=8),
            streams.finetuning_stream(_task(), LAW, vocab, design=design, seed=0, batch_size=8, replay_proportion=0.5),
        )

    for shown_stream, hidden_stream in zip(streams_for(VOCAB), streams_for(hidden), strict=True):
        shown_cues: set[int] = set()
        for _ in range(10):
            shown, masked = next(shown_stream), next(hidden_stream)
            assert set(masked.tokens[:, 1].tolist()) == {VOCAB.cue_token(FREQUENT_CUE)}
            keep = [column for column in range(shown.tokens.shape[1]) if column != 1]
            assert torch.equal(masked.tokens[:, keep], shown.tokens[:, keep])
            assert torch.equal(masked.scalars, shown.scalars)
            shown_cues |= set(shown.tokens[:, 1].tolist())
        assert len(shown_cues) > 1


def test_hiding_the_cue_changes_nothing_when_every_trial_draws_the_frequent_cue():
    """At frequent-cue proportion 1 every pretraining trial is drawn under the frequent cue, so
    the stream is the same token for token with the cue shown or hidden."""
    task = _task(frequent_cue_prop=1.0)
    shown = streams.pretraining_stream(task, LAW, VOCAB, seed=0, batch_size=8)
    hidden = streams.pretraining_stream(task, LAW, replace(VOCAB, show_cue=False), seed=0, batch_size=8)
    for _ in range(10):
        a, b = next(shown), next(hidden)
        assert torch.equal(a.tokens, b.tokens) and torch.equal(a.scalars, b.scalars)


def test_finetuned_items_are_fixed_across_steps():
    """Every finetuning example queries the same ordering's items."""
    design = _design()
    ordering = design.orderings[0]
    items = {VOCAB.item_token(item) for item in ordering.items}
    displayed = {VOCAB.item_token(item) for item in design.presented_items}
    queried = set()
    for step in range(30):
        for index in range(8):
            example = _finetuning(step, index, design=design)
            assert int(example.tokens[1]) == VOCAB.cue_token(ordering.cue)
            pair = (int(example.tokens[3]), int(example.tokens[4]))
            assert set(pair) <= items
            queried.add(tuple(sorted(pair)))
            shown = {int(token) for token in example.tokens[7 : example.prompt_len] if VOCAB.is_item(token)}
            # The context shows the fixed ordering, or, on the uninformative trials, other items.
            assert shown <= displayed or (len(shown) == 9 and not shown & displayed)
    adjacent = {
        tuple(sorted((VOCAB.item_token(ordering.items[rank]), VOCAB.item_token(ordering.items[rank + 1]))))
        for rank in range(N_MAIN_ITEMS - 1)
    }
    assert queried == adjacent


def test_conflicting_layout_reuses_one_item_set_under_different_cues():
    """Conflicting orderings share their items and differ in cue."""
    design = _design(layout="conflicting", n_orderings=4)
    assert design.layout is streams.Layout.CONFLICTING and len(design.orderings) == 4
    cues = [ordering.cue for ordering in design.orderings]
    assert len(set(cues)) == len(cues)
    shared = {frozenset(ordering.items) for ordering in design.orderings}
    assert len(shared) == 1 and len(next(iter(shared))) == N_MAIN_ITEMS
    sequences_shown = {ordering.items for ordering in design.orderings}
    assert len(sequences_shown) > 1
    for ordering in design.orderings:
        assert len(ordering.buckets) == N_MAIN_ITEMS + 1
        assert all(len(bucket) == 2 for bucket in ordering.buckets)


def test_conflicting_layout_shares_one_distractor_pool_dealt_afresh_per_cue():
    """Every cue of a conflicting design deals the same sixteen distractors into its buckets."""
    design = _design(layout="conflicting", n_orderings=4)
    pools = [sorted(item for bucket in ordering.buckets for item in bucket) for ordering in design.orderings]
    assert len(pools[0]) == len(set(pools[0])) == DISTRACTOR_POOL_SIZE
    assert all(pool == pools[0] for pool in pools)
    assert set(pools[0]).isdisjoint(design.orderings[0].items)
    assert len(design.presented_items) == N_MAIN_ITEMS + DISTRACTOR_POOL_SIZE
    assert len({ordering.buckets for ordering in design.orderings}) > 1


def test_every_distractor_is_shown_at_its_buckets_position_under_every_cue():
    """Ten cues at the canonical pool: under each, every distractor sits where its bucket says."""
    law = OrderingLaw(seed=0, **CANONICAL)
    design = streams.build_finetuning_design(law, streams.design_rng(0), layout="conflicting", n_orderings=10)
    assert len(design.orderings) == 10 and len({ordering.cue for ordering in design.orderings}) == 10
    for ordering in design.orderings:
        main = ordering.items
        assert main == law.ordering(ordering.cue, main)
        for step in range(40):
            rng = streams.example_rng(0, streams.Stage.FINETUNING, step, 0)
            trial = sample_finetuning_trial(law, rng, ordering=ordering)
            shown = trial.context_items
            assert len(shown) == 9 and len(set(shown)) == 9 and len(trial.context) == 8
            assert tuple(item for item in shown if item in main) == main
            top_down = {(a, b) if s == 1 else (b, a) for a, b, s in trial.context}
            assert top_down == set(zip(shown, shown[1:], strict=False))
            for distractor in trial.distractors:
                bucket = next(index for index, members in enumerate(ordering.buckets) if distractor in members)
                at = shown.index(distractor)
                assert bucket == 0 or shown.index(main[bucket - 1]) < at
                assert bucket == N_MAIN_ITEMS or at < shown.index(main[bucket])


def test_design_is_identical_across_processes():
    """The same seed draws the same orderings and buckets in interpreters with other hash seeds."""
    here = []
    for layout, n_orderings in (("conflicting", 3), ("disjoint", 2)):
        for ordering in _design(layout=layout, n_orderings=n_orderings, seed=3).orderings:
            here.append(" ".join(str(x) for x in (ordering.cue, *ordering.items, *itertools.chain(*ordering.buckets))))
    for hash_seed in (1, 2):
        run = subprocess.run(
            [sys.executable, "-c", DESIGN_SCRIPT, "3"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        )
        assert run.stdout.strip().splitlines() == here
    assert _design(seed=4).orderings[0].items != _design(seed=3).orderings[0].items


def test_a_pool_too_small_for_the_layout_fails_loudly():
    """Each item set costs its main items and distractors, the fresh test items come on top, and
    extra cues cost none."""
    per_set = N_MAIN_ITEMS + DISTRACTOR_POOL_SIZE
    floor = per_set + FRESH_TEST_ITEMS
    tight = OrderingLaw(seed=0, pool_size=floor, n_training_cues=12, n_reserved_cues=4)
    design = streams.build_finetuning_design(tight, streams.design_rng(0), layout="conflicting", n_orderings=10)
    assert len(design.orderings) == 10 and len(design.presented_items) == per_set
    short = OrderingLaw(seed=0, pool_size=floor - 1, n_training_cues=12, n_reserved_cues=4)
    with pytest.raises(ValueError, match=rf"item_pool_size {floor - 1} is too small.*needs {floor}"):
        streams.build_finetuning_design(short, streams.design_rng(0), layout="conflicting", n_orderings=1)
    two_floor = 2 * per_set + FRESH_TEST_ITEMS
    two = OrderingLaw(seed=0, pool_size=two_floor - 1, n_training_cues=12, n_reserved_cues=4)
    with pytest.raises(ValueError, match=rf"disjoint layout presents 2 item set.*needs {two_floor}.*n_orderings"):
        streams.build_finetuning_design(two, streams.design_rng(0), layout="disjoint", n_orderings=2)
    one = streams.build_finetuning_design(two, streams.design_rng(0), layout="disjoint", n_orderings=1)
    assert len(one.orderings) == 1


def test_disjoint_layout_uses_item_disjoint_orderings_under_one_cue():
    """Disjoint orderings share one cue and have no item in common."""
    design = _design(layout="disjoint", n_orderings=3)
    assert design.layout is streams.Layout.DISJOINT and len(design.orderings) == 3
    assert len({ordering.cue for ordering in design.orderings}) == 1
    shown = [ordering.presented_items for ordering in design.orderings]
    assert all(len(items) == N_MAIN_ITEMS + DISTRACTOR_POOL_SIZE for items in shown)
    for first, second in itertools.combinations(shown, 2):
        assert not first & second
    assert design.presented_items == frozenset().union(*shown)
    for ordering in design.orderings:
        assert len(ordering.items) == N_MAIN_ITEMS and len(ordering.buckets) == N_MAIN_ITEMS + 1


def test_finetuning_examples_use_answer_only_targets():
    """Except for replayed rows, finetuning targets are answer-only."""
    design = _design()
    for step in range(20):
        example = _finetuning(step, 0, design=design)
        prompt = example.prompt_len
        assert example.tokens.size == prompt + EMISSION_LEN
        # Token loss on the relation and separator, squared error on the relation, none on items.
        assert np.flatnonzero(example.token_loss).tolist() == [prompt + 2, prompt + 3]
        assert np.flatnonzero(example.scalar_loss).tolist() == [prompt + 2]
        assert int(example.tokens[prompt + 2]) == VOCAB.rel
    replayed = [_finetuning(step, 0, design=design, replay=1.0) for step in range(5)]
    for example in replayed:
        assert _is_chain_target(example)
        assert bool(example.token_loss[example.prompt_len :].all())
        assert np.array_equal(
            np.flatnonzero(example.scalar_loss),
            np.flatnonzero(example.tokens == VOCAB.rel)[-PAD_TO:],
        )


def test_uninformative_context_finetuning_trials_show_other_items():
    """At proportion one the context shows other items; by default one trial in ten does."""
    design = _design()
    ordering = design.orderings[0]
    rank = {item: index for index, item in enumerate(ordering.items)}
    for step in range(30):
        rng = streams.example_rng(0, streams.Stage.FINETUNING, step, 0)
        example = streams.finetuning_example(
            _task(), LAW, VOCAB, design=design, replay_proportion=0.0, uninformative_context_prop=1.0, rng=rng
        )
        first, second = _query_items(example)
        shown = _context_items(example)
        assert example.prompt_len == 7 + 8 * EMISSION_LEN and len(shown) == 9
        assert not shown & design.presented_items
        assert abs(rank[first] - rank[second]) == 1
        familiar = 1.0 if rank[first] < rank[second] else -1.0
        assert float(example.scalars[example.prompt_len + 2]) == familiar
    # By default one trial in ten is uninformative, and the default is the named constant.
    n = 400
    uninformative = 0
    for step in range(n // 8):
        for index in range(8):
            rng = streams.example_rng(0, streams.Stage.FINETUNING, step, index)
            plain = _finetuning(step, index, design=design)
            named = streams.finetuning_example(
                _task(), LAW, VOCAB, design=design, replay_proportion=0.0,
                uninformative_context_prop=streams.UNINFORMATIVE_CONTEXT_PROP, rng=rng,
            )
            assert np.array_equal(plain.tokens, named.tokens) and np.array_equal(plain.scalars, named.scalars)
            uninformative += not (_context_items(plain) & design.presented_items)
    # Four standard errors of a binomial proportion at one tenth over this many examples.
    assert streams.UNINFORMATIVE_CONTEXT_PROP == 0.1
    assert abs(uninformative / n - 0.1) < 4 * (0.09 / n) ** 0.5


def test_batch_dtypes_and_shapes_match_the_model_contract():
    """int64 tokens, float32 scalars, bool masks, all of one shape."""
    rng = np.random.default_rng(0)
    trial = sample_pretraining_trial(LAW, rng, weights=np.full(LAW.n_training_cues, 1 / LAW.n_training_cues))
    short = encode(trial, None, VOCAB, target=Target.ANSWER_ONLY)
    long = _pretraining(0, 0, p_cot=1.0)
    batch = streams.stack([short, long], VOCAB).to_batch("cpu")
    assert batch.tokens.dtype is torch.int64 and batch.scalars.dtype is torch.float32
    assert batch.token_loss.dtype is torch.bool and batch.scalar_loss.dtype is torch.bool
    assert batch.answer_index.dtype is torch.int64 and batch.answer_index.shape == (2,)
    assert batch.answer_index.tolist() == [short.answer_index, long.answer_index]
    shape = (2, int(long.tokens.size))
    assert batch.tokens.shape == batch.scalars.shape == batch.token_loss.shape == batch.scalar_loss.shape == shape
    assert batch.index == (0, 1)
    assert streams.stack([short, long], VOCAB, first_index=12).to_batch("cpu").index == (12, 13)
    padded = slice(int(short.tokens.size), shape[1])
    assert bool((batch.tokens[0, padded] == VOCAB.pad).all())
    assert not bool(batch.token_loss[0, padded].any()) and not bool(batch.scalar_loss[0, padded].any())
    assert bool((batch.scalars[0, padded] == 0).all())


def _same_batches(built: streams.Batch, prefetched: streams.Batch) -> None:
    assert built.index == prefetched.index
    for name in ("tokens", "scalars", "token_loss", "scalar_loss", "answer_index"):
        assert torch.equal(getattr(built, name), getattr(prefetched, name))


def test_worker_processes_build_the_same_batches_as_the_serial_stream():
    """Batches built ahead by two worker processes equal those built in line, in order."""
    kwargs = dict(seed=0, batch_size=4, start_step=7)
    serial = streams.pretraining_stream(_task(), LAW, VOCAB, **kwargs)
    ahead = streams.pretraining_stream(_task(), LAW, VOCAB, workers=2, **kwargs)
    try:
        for _ in range(5):
            _same_batches(next(serial), next(ahead))
    finally:
        serial.close()
        ahead.close()


def test_worker_processes_build_the_same_finetuning_batches_with_replay():
    """The finetuning stream, design and replay included, is the same from worker processes."""
    design = design_for(LAW, seed=0)
    kwargs = dict(design=design, seed=0, batch_size=4, replay_proportion=0.3)
    serial = streams.finetuning_stream(_task(), LAW, VOCAB, **kwargs)
    ahead = streams.finetuning_stream(_task(), LAW, VOCAB, workers=2, **kwargs)
    try:
        for _ in range(4):
            _same_batches(next(serial), next(ahead))
    finally:
        serial.close()
        ahead.close()
