"""Sequence layout, the scalar channel, loss masks, and the scored answer index."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import verbatim_trial as _trial

from geo_alg_ti import chains, sequences, trials
from geo_alg_ti.orderings import FREQUENT_CUE

R = trials.Relation
VOCAB = sequences.Vocabulary(pool_size=9, n_cues=8)

# The extended methods' worked example: i0 > i1 > ... > i8 under <cue7>, query (i3, i1), the
# pair {i2, i3} presented before {i3, i4}; the context is shown in the order of the module
# docstring.
EXAMPLE_CONTEXT = (R(8, 7, -1), R(2, 3, 1), R(0, 1, 1), R(3, 4, 1), R(2, 1, -1), R(6, 7, 1), R(5, 6, 1), R(5, 4, -1))
EXAMPLE_CHAIN = chains.Chain(emissions=(R(2, 3, 1), R(2, 1, -1)), answer=R(3, 1, -1), reached_query_item=True)


EXAMPLE_TRIAL = _trial(EXAMPLE_CONTEXT, (3, 1))


def _encode(target=sequences.Target.CHAIN, trial=EXAMPLE_TRIAL, chain=EXAMPLE_CHAIN):
    return sequences.encode(trial, chain if target is sequences.Target.CHAIN else None, VOCAB, target=target)


def _emission_tokens(relation):
    return [VOCAB.item_token(relation.a), VOCAB.item_token(relation.b), VOCAB.rel, VOCAB.sep]


def test_worked_example_encodes_to_the_documented_layout():
    """Every position holds the token the table in the module docstring names."""
    enc = _encode()
    assert len(enc.tokens) == 75
    assert enc.prompt_len == 39 and enc.prompt_len + 4 * enc.n_emissions == 75
    assert enc.answer_index == sequences.scored_answer_index(39, 2) == 47 + 2
    assert enc.tokens[enc.answer_index] == VOCAB.rel
    query_tokens = [VOCAB.item_token(3), VOCAB.item_token(1)]
    tokens = [VOCAB.bos, VOCAB.cue_token(7), VOCAB.qmark, *query_tokens, VOCAB.qrel, VOCAB.sep]
    for relation in (*EXAMPLE_CONTEXT, *EXAMPLE_CHAIN.emissions, *[EXAMPLE_CHAIN.answer] * 7):
        tokens += _emission_tokens(relation)
    assert enc.tokens.tolist() == tokens
    # Eight context, two chain, one answer, and six padding emissions of four tokens each.
    assert enc.tokens[7:39].tolist() == [t for r in EXAMPLE_CONTEXT for t in _emission_tokens(r)]
    assert enc.tokens[39:47].tolist() == [t for r in EXAMPLE_CHAIN.emissions for t in _emission_tokens(r)]
    assert enc.tokens[47:75].tolist() == 7 * _emission_tokens(EXAMPLE_CHAIN.answer)
    scalars = np.zeros(75)
    scalars[9:39:4] = [r.sign for r in EXAMPLE_CONTEXT]
    scalars[41:75:4] = [1, -1] + [-1] * 7
    assert np.array_equal(enc.scalars, scalars)


def test_encode_decode_round_trip():
    """Decoding reproduces the printed sequence of the example."""
    enc = _encode()
    printed = "<bos> <cue7> <qmark> i3 i1 <qrel> <sep>"
    for a, b, s in (*EXAMPLE_CONTEXT, *EXAMPLE_CHAIN.emissions, *[EXAMPLE_CHAIN.answer] * 7):
        printed += f" i{a} i{b} {s:+d} <sep>"
    assert sequences.decode(enc.tokens, enc.scalars, VOCAB) == printed
    assert "i8 i7 -1 <sep> i2 i3 +1 <sep> i0 i1 +1" in printed
    assert printed.endswith("i2 i3 +1 <sep> i2 i1 -1 <sep>" + " i3 i1 -1 <sep>" * 7)


def test_query_items_appear_before_and_after_the_context():
    """The query items occur in the head and again in the answer emission."""
    x, z = VOCAB.item_token(3), VOCAB.item_token(1)
    for target, answer_start in ((sequences.Target.CHAIN, 47), (sequences.Target.ANSWER_ONLY, 39)):
        enc = _encode(target)
        assert enc.tokens[3] == x and enc.tokens[4] == z
        assert enc.tokens[answer_start] == x and enc.tokens[answer_start + 1] == z
        assert enc.tokens[answer_start + 2] == VOCAB.rel == enc.tokens[enc.answer_index]


def test_answer_scalar_follows_the_declared_scoring_reference():
    """Without a chain the answer emission carries the reference the trial is scored against."""
    reversed_context = replace(
        EXAMPLE_TRIAL, context_kind=trials.ContextKind.REVERSED, context_items=EXAMPLE_TRIAL.items[::-1]
    )
    shown = reversed_context.shown_sign
    assert shown == reversed_context.reference_sign == -EXAMPLE_TRIAL.familiar_sign
    enc = sequences.encode(reversed_context, None, VOCAB, target=sequences.Target.ANSWER_ONLY)
    assert enc.scalars[enc.answer_index] == shown != reversed_context.familiar_sign
    # The two references coincide during finetuning, where the encoding is unchanged.
    finetuning = _encode(sequences.Target.ANSWER_ONLY)
    assert EXAMPLE_TRIAL.reference_sign == EXAMPLE_TRIAL.familiar_sign
    assert finetuning.scalars[finetuning.answer_index] == EXAMPLE_TRIAL.reference_sign


def test_query_relation_slot_is_a_sentinel_without_a_scalar():
    """`<qrel>` carries no scalar and no loss."""
    for target in sequences.Target:
        enc = _encode(target)
        assert enc.tokens[5] == VOCAB.qrel
        assert enc.scalars[5] == 0 and not enc.token_loss[5] and not enc.scalar_loss[5]
        assert np.count_nonzero(enc.tokens == VOCAB.qrel) == 1


def test_scalars_are_nonzero_only_at_relation_positions():
    """The scalar channel is zero everywhere except at `<rel>` positions."""
    for target in sequences.Target:
        enc = _encode(target)
        at_rel = enc.tokens == VOCAB.rel
        assert np.array_equal(enc.scalars != 0, at_rel)
        assert set(enc.scalars[at_rel].tolist()) <= {-1.0, 1.0}
        assert at_rel.sum() == 8 + enc.n_emissions


def test_prompt_length_is_the_context_boundary():
    """`prompt_length(8)` is 39: the seven-token head plus eight four-token relations."""
    assert sequences.prompt_length(8) == 39


def test_chain_target_loss_covers_every_position_after_the_prompt():
    """Token loss is on every chain, answer, and padding token and nowhere in the prompt."""
    enc = _encode()
    assert not enc.token_loss[:39].any() and enc.token_loss[39:].all()
    assert not enc.scalar_loss[:39].any()
    assert not np.array_equal(enc.scalar_loss, enc.tokens == VOCAB.rel)
    assert np.array_equal(np.flatnonzero(enc.scalar_loss), np.arange(41, 75, 4))
    prompt_only = _encode(sequences.Target.PROMPT_ONLY)
    assert len(prompt_only.tokens) == 39 and not prompt_only.token_loss.any() and not prompt_only.scalar_loss.any()


def test_scalar_loss_is_a_subset_of_token_loss_at_relation_positions():
    """Loss-bearing relation positions are a subset of loss-bearing token positions."""
    for target in (sequences.Target.CHAIN, sequences.Target.ANSWER_ONLY):
        enc = _encode(target)
        assert enc.scalar_loss.any()
        assert enc.token_loss[enc.scalar_loss].all()
        assert (enc.tokens[enc.scalar_loss] == VOCAB.rel).all()


def test_answer_only_target_supervises_the_relation_and_separator_only():
    """Answer-only targets put token loss on `<rel>` and `<sep>` and scalar loss on `<rel>` only."""
    enc = _encode(sequences.Target.ANSWER_ONLY)
    assert len(enc.tokens) == 43 and enc.n_emissions == 1 and enc.prompt_len == 39
    assert enc.tokens[39:].tolist() == _emission_tokens(R(3, 1, -1)) and enc.scalars[41] == -1
    assert np.flatnonzero(enc.token_loss).tolist() == [41, 42]
    assert np.flatnonzero(enc.scalar_loss).tolist() == [41]
    assert enc.answer_index == 41 == sequences.scored_answer_index(39, 0)


def test_padding_repeats_the_answer_emission():
    """Padding emissions equal the answer emission."""
    enc = _encode()
    answer = slice(47, 51)
    for start in range(51, 75, 4):
        block = slice(start, start + 4)
        assert enc.tokens[block].tolist() == enc.tokens[answer].tolist()
        assert enc.scalars[block].tolist() == enc.scalars[answer].tolist()
    # The encoder owns the padding: any budget from the chain's own length upward is honored.
    for pad_to in (3, sequences.PAD_TO, 12):
        padded = sequences.encode(EXAMPLE_TRIAL, EXAMPLE_CHAIN, VOCAB, pad_to=pad_to)
        assert padded.n_emissions == pad_to and len(padded.tokens) == 39 + 4 * pad_to
        assert padded.answer_index == 47 + 2
    with pytest.raises(ValueError, match="does not fit"):
        sequences.encode(EXAMPLE_TRIAL, EXAMPLE_CHAIN, VOCAB, pad_to=2)
    adjacent = _trial(EXAMPLE_CONTEXT, (4, 3))
    enc = sequences.encode(adjacent, chains.generate_chain(adjacent), VOCAB)
    assert len(enc.tokens) == 75 and enc.answer_index == 41
    for start in range(39, 75, 4):
        assert enc.tokens[start : start + 4].tolist() == _emission_tokens(R(4, 3, -1))
        assert enc.scalars[start + 2] == -1


def test_sequence_length_is_constant_within_a_target_type():
    """Chain targets are 75 tokens and answer-only targets 43 for eight context relations."""
    assert sequences.sequence_length(8, 9, sequences.Target.CHAIN) == 75
    assert sequences.sequence_length(8, 9, sequences.Target.ANSWER_ONLY) == 43


def test_scored_answer_index_agrees_with_generation_for_every_budget():
    """For K = 0..8 the index equals where the appended query's `<rel>` lands after K emissions."""
    for k in range(9):
        assert sequences.scored_answer_index(39, k) == 39 + 4 * k + 2


def test_hidden_state_index_is_the_second_query_item():
    """The state that predicts the scored slot sits at the second appended query item."""
    for k in range(9):
        assert sequences.hidden_state_index(39, k) == sequences.scored_answer_index(39, k) - 1


def test_scoring_length_counts_the_positions_a_scoring_pass_runs_over():
    """At budget 8 the pass holds the 39-token prompt, eight emissions, and the two query items."""
    assert sequences.scoring_length(8, 8) == 39 + 8 * 4 + 2 == 73
    for k in range(9):
        # The last position read is the second query item, so the pass is one position longer.
        assert sequences.scoring_length(8, k) == sequences.hidden_state_index(39, k) + 1
        assert sequences.scoring_length(8, k) == sequences.scored_answer_index(39, k)
    # A padded chain target is the longer sequence at every budget up to the padded length.
    assert sequences.scoring_length(8, 8) < sequences.sequence_length(8, 9, sequences.Target.CHAIN) == 75
    assert sequences.scoring_length(8, 10) == 81 > 75


def test_vocabulary_layout_is_stable():
    """Specials first (pad 0), then cues (training then reserved), then items; ids never move."""
    v = sequences.Vocabulary(pool_size=48, n_cues=8)
    assert v.pad == 0 and v.cue_token(0) == len(sequences.SPECIAL_TOKENS) and v.item_token(0) == v.cue_token(0) + 8


def test_a_hidden_cue_shows_the_frequent_cue_token_under_every_cue():
    """With the cue hidden, index 1 holds the frequent cue's token whatever the trial's cue, and
    every other position is what the shown cue would give."""
    hidden = replace(VOCAB, show_cue=False)
    for cue in range(VOCAB.n_cues):
        trial = replace(EXAMPLE_TRIAL, cue=cue)
        shown = sequences.encode(trial, EXAMPLE_CHAIN, VOCAB)
        masked = sequences.encode(trial, EXAMPLE_CHAIN, hidden)
        assert shown.tokens[1] == VOCAB.cue_token(cue)
        assert masked.tokens[1] == VOCAB.cue_token(FREQUENT_CUE)
        assert np.array_equal(np.delete(masked.tokens, 1), np.delete(shown.tokens, 1))
        assert np.array_equal(masked.scalars, shown.scalars)
    with pytest.raises(ValueError):
        hidden.shown_cue_token(VOCAB.n_cues)


def test_item_positions_find_every_item_occurrence():
    """Every item token, including both copies of the query items, is found."""
    enc = _encode()
    expected = [3, 4] + [i for start in range(7, 75, 4) for i in (start, start + 1)]
    found = sequences.item_positions(enc.tokens, VOCAB)
    assert found.tolist() == expected and len(found) == 36
    assert all(VOCAB.is_item(t) for t in enc.tokens[found])
    assert not any(VOCAB.is_item(t) for t in np.delete(enc.tokens, found))
    assert np.count_nonzero(enc.tokens[found] == VOCAB.item_token(3)) == 11
    assert np.count_nonzero(enc.tokens[found] == VOCAB.item_token(1)) == 11
