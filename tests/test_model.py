"""The transformer, its scalar channel, the loss, greedy generation, and hidden states."""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from geo_alg_ti import model
from geo_alg_ti.chains import generate_chain
from geo_alg_ti.orderings import OrderingLaw, uniform_cue_weights
from geo_alg_ti.sequences import (
    EMISSION_LEN,
    Target,
    Vocabulary,
    encode,
    hidden_state_index,
    item_positions,
    scored_answer_index,
)
from geo_alg_ti.streams import Batch, stack
from geo_alg_ti.trials import sample_pretraining_trial

LAW = OrderingLaw(seed=0, pool_size=48, n_training_cues=6, n_reserved_cues=2)
VOCAB = Vocabulary(pool_size=LAW.pool_size, n_cues=LAW.n_training_cues + LAW.n_reserved_cues)
MAX_LEN = 96
SPEC = model.ModelSpec(vocab_size=VOCAB.size, max_len=MAX_LEN, n_layers=2, d_model=32, d_ff=128)


def _trial(index: int):
    """One pretraining trial, drawn from a generator keyed on `index` so it never moves."""
    rng = np.random.default_rng(index)
    return sample_pretraining_trial(LAW, rng, weights=uniform_cue_weights(LAW.n_training_cues))


def _encoded(index: int = 0, target: Target = Target.CHAIN):
    """One encoded trial and its chain."""
    trial = _trial(index)
    return encode(trial, generate_chain(trial), VOCAB, target=target)


def _network(seed: int = 0) -> model.RelationalTransformer:
    """A freshly initialized network at test scale."""
    torch.manual_seed(seed)
    return model.RelationalTransformer(SPEC, VOCAB)


def _inputs(encoded) -> tuple[torch.Tensor, torch.Tensor]:
    """The token and scalar channels of one encoded sequence, as a batch of one."""
    return (
        torch.as_tensor(encoded.tokens, dtype=torch.long)[None],
        torch.as_tensor(encoded.scalars, dtype=torch.float32)[None],
    )


def _chain_batch(n: int = 8) -> Batch:
    """A batch of chain-target sequences."""
    examples = [_encoded(index) for index in range(n)]
    return stack(examples, VOCAB).to_batch("cpu")


@functools.lru_cache(maxsize=1)
def _trained_network() -> model.RelationalTransformer:
    """A network trained far enough on one batch to write well-formed emissions."""
    net = _network()
    batch = _chain_batch()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, betas=(0.9, 0.95))
    for _ in range(300):
        terms = model.loss_terms(net(batch.tokens, batch.scalars), batch)
        optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    net.eval()
    return net


def test_attention_is_causal():
    """The output at position t has zero gradient with respect to inputs after t."""
    net = _network()
    with torch.no_grad():
        # The scalar channel is the one differentiable input, and it enters through this direction.
        net.relation_direction.normal_(mean=0.0, std=0.5)
    tokens, scalars = _inputs(_encoded())
    scalars = scalars.detach().requires_grad_(True)
    cut = 20
    net(tokens, scalars).logits[0, cut].sum().backward()
    gradient = scalars.grad[0]
    assert torch.count_nonzero(gradient[cut + 1 :]) == 0
    assert torch.count_nonzero(gradient[: cut + 1]) > 0
    # The token channel carries no gradient, so a later token is replaced outright instead.
    later = tokens.clone()
    later[0, cut + 3] = VOCAB.pad
    with torch.no_grad():
        moved = net(later, scalars.detach()).logits[0, : cut + 1]
        assert torch.equal(net(tokens, scalars.detach()).logits[0, : cut + 1], moved)


def test_relation_scalar_enters_only_at_relation_positions():
    """Perturbing a scalar off a relation position changes nothing."""
    net = _network()
    with torch.no_grad():
        # An untrained direction is zero, which would make every scalar inert whatever the layout.
        net.relation_direction.normal_(mean=0.0, std=0.5)
    encoded = _encoded()
    tokens, scalars = _inputs(encoded)
    relation_slots = np.flatnonzero(encoded.tokens == VOCAB.rel)
    other_slots = np.flatnonzero(encoded.tokens != VOCAB.rel)
    assert relation_slots.size and other_slots.size
    with torch.no_grad():
        baseline = net(tokens, scalars)
        off_relation = scalars.clone()
        off_relation[0, torch.as_tensor(other_slots)] = 3.0
        elsewhere = net(tokens, off_relation)
        at_relation = scalars.clone()
        at_relation[0, int(relation_slots[0])] *= -1.0
        flipped = net(tokens, at_relation)
    assert torch.equal(baseline.logits, elsewhere.logits)
    assert torch.equal(baseline.scalar_pred, elsewhere.scalar_pred)
    assert not torch.equal(baseline.logits, flipped.logits)


def test_scalar_head_reads_the_preceding_hidden_state():
    """s_hat_t is a linear readout of the final normalized state at t - 1 by the tied direction."""
    net = _network()
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
        net.relation_bias.fill_(0.25)
    encoded = _encoded()
    tokens, scalars = _inputs(encoded)
    with torch.no_grad():
        out = net(tokens, scalars, return_hidden=True)
        readout = net.final_norm(out.hidden[SPEC.n_layers])[0] @ net.relation_direction + net.relation_bias
        # The same direction carries the sign in: the input at a relation slot is e_<rel> + s r.
        embedded = net.embedding(tokens)[0]
    assert torch.allclose(out.scalar_pred[0], readout, atol=1e-6)
    assert torch.equal(out.hidden[-1], net.final_norm(out.hidden[-2]))
    for slot in np.flatnonzero(encoded.tokens == VOCAB.rel):
        sign = float(encoded.scalars[slot])
        expected = embedded[slot] + sign * net.relation_direction
        assert torch.allclose(out.hidden[0][0, slot], expected, atol=1e-6)
    # The sign of the relation slot after k emissions is read one position earlier.
    for k in range(9):
        assert hidden_state_index(encoded.prompt_len, k) == scored_answer_index(encoded.prompt_len, k) - 1


def test_loss_terms_match_a_manual_computation():
    """Mean cross-entropy plus mean squared error at the relation positions, unweighted."""
    # Six positions over three token ids; the last two positions are supervised, the fourth
    # carries the relation target. The prediction of position t is read from position t - 1.
    tokens = torch.tensor([[1, 2, 0, 1, 1, 0]])
    scalars = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    token_loss = torch.tensor([[False, False, False, False, True, True]])
    scalar_loss = torch.tensor([[False, False, False, False, True, False]])
    logits = torch.full((1, 6, 3), -30.0)
    logits[0, 3] = torch.tensor([0.0, 0.0, 0.0])  # uniform over three ids: cross-entropy ln 3
    logits[0, 4] = torch.tensor([math.log(2.0), 0.0, 0.0])  # id 0 at one half: cross-entropy ln 2
    scalar_pred = torch.full((1, 6), 99.0)
    scalar_pred[0, 3] = 0.25  # against a target of +1: squared error 0.5625
    out = model.ModelOutput(logits=logits, scalar_pred=scalar_pred)
    batch = Batch(
        tokens=tokens,
        scalars=scalars,
        token_loss=token_loss,
        scalar_loss=scalar_loss,
        answer_index=torch.tensor([4]),
        index=(0,),
    )
    terms = model.loss_terms(out, batch)
    assert terms.token_cross_entropy.item() == pytest.approx(math.log(6.0) / 2.0, abs=1e-6)
    assert terms.scalar_squared_error.item() == pytest.approx(0.5625, abs=1e-6)
    assert terms.total.item() == pytest.approx(math.log(6.0) / 2.0 + 0.5625, abs=1e-6)


def test_masked_positions_do_not_contribute_to_the_loss():
    """Changing targets at prompt positions leaves the loss unchanged."""
    encoded = _encoded(target=Target.ANSWER_ONLY)
    batch = stack([encoded], VOCAB).to_batch("cpu")
    net = _network()
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
        out = net(batch.tokens, batch.scalars)
    before = model.loss_terms(out, batch)
    prompt = encoded.prompt_len
    assert not bool(batch.token_loss[0, :prompt].any()) and not bool(batch.scalar_loss[0, :prompt].any())
    batch.tokens[0, 1:prompt] = VOCAB.item_token(0)
    # A context relation slot carries a scalar on the way in and no squared error on the way out.
    batch.scalars[0, 1:prompt] = -batch.scalars[0, 1:prompt]
    after = model.loss_terms(out, batch)
    assert after.token_cross_entropy.item() == before.token_cross_entropy.item()
    assert after.scalar_squared_error.item() == before.scalar_squared_error.item()
    assert after.total.item() == before.total.item()


def test_greedy_generation_is_prefix_consistent():
    """The first K emissions of an eight-emission rollout equal a K-emission rollout."""
    net = _trained_network()
    encoded = _encoded(target=Target.PROMPT_ONLY)
    full = model.greedy_chain(net, [encoded], max_emissions=8)
    assert full.n_emissions == 8 and full.tokens.shape == (1, EMISSION_LEN * 8)
    for budget in range(9):
        short = model.greedy_chain(net, [encoded], max_emissions=budget)
        assert short.n_emissions == budget
        assert torch.equal(short.tokens[0], full.tokens[0, : EMISSION_LEN * budget])
        assert torch.equal(short.scalars[0], full.scalars[0, : EMISSION_LEN * budget])


def test_score_at_budgets_matches_independent_rollouts():
    """For every K = 0..8 the scored sign equals regenerating with that budget and scoring."""
    net = _trained_network()
    encoded = _encoded(target=Target.PROMPT_ONLY)
    prompt = encoded.prompt_len
    budgets = list(range(9))
    generated = model.greedy_chain(net, [encoded], max_emissions=8, budgets=budgets)
    batched = model.score_at_budgets(generated)
    assert batched.shape == (1, len(budgets)) and batched.dtype == np.int64
    for k in budgets:
        alone = model.greedy_chain(net, [encoded], max_emissions=k)
        tokens = torch.cat(
            (
                torch.as_tensor(encoded.tokens[:prompt], dtype=torch.long),
                alone.tokens[0],
                torch.as_tensor(encoded.tokens[3:5], dtype=torch.long),
            )
        )[None]
        scalars = torch.zeros_like(tokens, dtype=torch.float32)
        scalars[0, :prompt] = torch.as_tensor(encoded.scalars[:prompt], dtype=torch.float32)
        scalars[0, prompt : prompt + alone.scalars[0].numel()] = alone.scalars[0]
        with torch.no_grad():
            out = net(tokens, scalars)
        expected = 1 if float(out.scalar_pred[0, hidden_state_index(prompt, k)]) > 0 else -1
        assert int(batched[0, k]) == expected


def test_generated_relation_scalar_is_the_sign_of_the_prediction():
    """The committed scalar is +1 iff s_hat > 0."""
    net = _trained_network()
    seen: set[float] = set()
    for index in range(6):
        encoded = _encoded(index, target=Target.PROMPT_ONLY)
        prompt = encoded.prompt_len
        generated = model.greedy_chain(net, [encoded], max_emissions=8)
        tokens = torch.cat((torch.as_tensor(encoded.tokens[:prompt], dtype=torch.long), generated.tokens[0]))[None]
        scalars = torch.cat((torch.as_tensor(encoded.scalars[:prompt], dtype=torch.float32), generated.scalars[0]))[None]
        with torch.no_grad():
            predicted = net(tokens, scalars).scalar_pred[0]
        for offset, token in enumerate(generated.tokens[0].tolist()):
            committed = float(generated.scalars[0, offset])
            if token != VOCAB.rel:
                assert committed == 0.0
                continue
            assert committed == (1.0 if float(predicted[prompt + offset - 1]) > 0 else -1.0)
            seen.add(committed)
    assert seen == {1.0, -1.0}


def test_tied_embeddings_share_storage():
    """The unembedding uses the embedding matrix."""
    net = _network()
    tokens, scalars = _inputs(_encoded())
    with torch.no_grad():
        out = net(tokens, scalars, return_hidden=True)
        expected = net.final_norm(out.hidden[SPEC.n_layers]) @ net.embedding.weight.T
    assert torch.allclose(out.logits, expected, atol=1e-6)
    assert torch.equal(out.hidden[-1], net.final_norm(out.hidden[-2]))
    shaped_like_the_table = [
        name for name, parameter in net.named_parameters() if tuple(parameter.shape) == (VOCAB.size, SPEC.d_model)
    ]
    assert shaped_like_the_table == ["embedding.weight"]
    with torch.no_grad():
        net.embedding.weight[VOCAB.rel] += 1.0
        moved = net(tokens, scalars).logits
    assert not torch.equal(moved, out.logits)


def test_forward_is_deterministic():
    """Two forward passes on the same input are identical under the determinism flag."""
    net = _network()
    tokens, scalars = _inputs(_encoded())
    with torch.no_grad():
        first = net(tokens, scalars)
        second = net(tokens, scalars)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.scalar_pred, second.scalar_pred)
    twin = _network()
    with torch.no_grad():
        third = twin(tokens, scalars)
    assert torch.equal(first.logits, third.logits)


def _prompts(indices, target: Target = Target.PROMPT_ONLY):
    """Encoded prompts of several trials."""
    return [_encoded(index, target=target) for index in indices]


def _full_sequence(encoded, generated, row: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`[prompt | emitted[:4K] | query]` and its scalars for one rollout row, as a batch of one."""
    prompt = encoded.prompt_len
    emitted = EMISSION_LEN * k
    tokens = torch.cat(
        (
            torch.as_tensor(encoded.tokens[:prompt], dtype=torch.long),
            generated.tokens[row, :emitted],
            torch.as_tensor(encoded.tokens[3:5], dtype=torch.long),
        )
    )[None]
    scalars = torch.zeros_like(tokens, dtype=torch.float32)
    scalars[0, :prompt] = torch.as_tensor(encoded.scalars[:prompt], dtype=torch.float32)
    scalars[0, prompt : prompt + emitted] = generated.scalars[row, :emitted]
    return tokens, scalars


def test_cached_forward_over_a_whole_sequence_equals_the_plain_forward():
    """Asking for the cache changes nothing about the outputs of a whole-sequence forward."""
    net = _network()
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
    tokens, scalars = _inputs(_encoded())
    with torch.no_grad():
        plain = net(tokens, scalars)
        cached = net(tokens, scalars, return_cache=True)
    assert plain.cache is None and cached.cache is not None
    assert torch.allclose(plain.logits, cached.logits, atol=1e-4, rtol=1e-4)
    assert torch.allclose(plain.scalar_pred, cached.scalar_pred, atol=1e-4, rtol=1e-4)


def test_chunked_cached_forwards_equal_the_uncached_forward():
    """Feeding a sequence in pieces over the cache reproduces the whole-sequence outputs."""
    net = _network()
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
    tokens, scalars = _inputs(_encoded())
    length = tokens.shape[1]
    pieces = [3, 2, 4, 1]
    pieces.append(length - sum(pieces))
    assert pieces[-1] > 1
    logits, predictions = [], []
    with torch.no_grad():
        whole = net(tokens, scalars)
        cache = None
        start = 0
        for size in pieces:
            piece = slice(start, start + size)
            out = net(tokens[:, piece], scalars[:, piece], cache=cache, return_cache=True)
            cache = out.cache
            logits.append(out.logits)
            predictions.append(out.scalar_pred)
            start += size
    assert cache.length == length
    assert torch.allclose(torch.cat(logits, dim=1), whole.logits, atol=1e-4, rtol=1e-4)
    assert torch.allclose(torch.cat(predictions, dim=1), whole.scalar_pred, atol=1e-4, rtol=1e-4)


def test_cache_grows_by_the_positions_fed():
    """The cache length follows the positions fed, with one key and value tensor per block."""
    net = _network()
    tokens, scalars = _inputs(_encoded())
    head_dim = SPEC.d_model // SPEC.n_heads
    with torch.no_grad():
        out = net(tokens[:, :10], scalars[:, :10], return_cache=True)
        assert out.cache.length == 10 and out.cache.batch_size == 1
        out = net(tokens[:, 10:13], scalars[:, 10:13], cache=out.cache, return_cache=True)
        assert out.cache.length == 13
        out = net(tokens[:, 13:14], scalars[:, 13:14], cache=out.cache, return_cache=True)
    assert out.cache.length == 14 and len(out.cache.layers) == SPEC.n_layers
    for key, value in out.cache.layers:
        assert key.shape == (1, SPEC.n_heads, 14, head_dim)
        assert value.shape == (1, SPEC.n_heads, 14, head_dim)


def test_a_discarded_branch_leaves_the_cache_unchanged():
    """Reading two tokens off a cache and discarding them alters neither the cache nor the rest."""
    net = _network()
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
    tokens, scalars = _inputs(_encoded())
    with torch.no_grad():
        reference = net(tokens[:, :6], scalars[:, :6])
        out = net(tokens[:, :3], scalars[:, :3], return_cache=True)
        clones = [(key.clone(), value.clone()) for key, value in out.cache.layers]
        branch = net(tokens[:, 3:5], scalars[:, 3:5], cache=out.cache)
        assert branch.cache is None and branch.logits.shape[1] == 2
        continued = net(tokens[:, 3:6], scalars[:, 3:6], cache=out.cache)
    for (key, value), (cloned_key, cloned_value) in zip(out.cache.layers, clones, strict=True):
        assert torch.equal(key, cloned_key) and torch.equal(value, cloned_value)
    assert out.cache.length == 3
    assert torch.allclose(continued.logits, reference.logits[:, 3:6], atol=1e-6)
    assert torch.allclose(continued.scalar_pred, reference.scalar_pred[:, 3:6], atol=1e-6)


def test_budget_predictions_equal_fresh_forwards_over_the_spliced_sequence():
    """The relation read at budget K equals an uncached forward over prompt, K emissions, query."""
    net = _trained_network()
    encoded = _encoded(target=Target.PROMPT_ONLY)
    budgets = list(range(9))
    generated = model.greedy_chain(net, [encoded], max_emissions=8, budgets=budgets)
    assert generated.budget_prediction.shape == (1, 9) and generated.budgets == tuple(budgets)
    signs = model.score_at_budgets(generated)
    guarded = 0
    for k in budgets:
        tokens, scalars = _full_sequence(encoded, generated, 0, k)
        with torch.no_grad():
            fresh = net(tokens, scalars).scalar_pred[0, hidden_state_index(encoded.prompt_len, k)]
        assert torch.allclose(generated.budget_prediction[0, k], fresh, atol=1e-4, rtol=1e-4)
        if abs(float(fresh)) > 1e-3:
            assert int(signs[0, k]) == (1 if float(fresh) > 0 else -1)
        else:
            guarded += 1
    assert guarded == 0


def test_prompts_rolled_out_together_equal_single_rollouts():
    """A group of four prompts gives the same tokens, scalars, and predictions as four alone."""
    net = _trained_network()
    prompts = _prompts(range(4))
    budgets = (0, 3, 8)
    together = model.greedy_chain(net, prompts, max_emissions=8, budgets=budgets)
    assert together.tokens.shape == (4, EMISSION_LEN * 8) and together.budget_prediction.shape == (4, 3)
    for row, encoded in enumerate(prompts):
        alone = model.greedy_chain(net, [encoded], max_emissions=8, budgets=budgets)
        assert torch.equal(together.tokens[row], alone.tokens[0])
        assert torch.equal(together.scalars[row], alone.scalars[0])
        assert torch.allclose(together.budget_prediction[row], alone.budget_prediction[0], atol=1e-4)


def test_unequal_prompt_lengths_raise():
    """Prompts of different lengths cannot share one rollout; the error names both lengths."""
    net = _network()
    short = _encoded(0, target=Target.PROMPT_ONLY)
    long_ = replace(short, prompt_len=short.prompt_len + EMISSION_LEN)
    with pytest.raises(ValueError, match=f"{short.prompt_len}.*{long_.prompt_len}"):
        model.greedy_chain(net, [short, long_], max_emissions=1)


def test_a_budget_beyond_the_rollout_raises():
    """A budget past the emissions rolled out cannot be read."""
    net = _network()
    encoded = _encoded(target=Target.PROMPT_ONLY)
    with pytest.raises(ValueError):
        model.greedy_chain(net, [encoded], max_emissions=2, budgets=(0, 3))
    with pytest.raises(ValueError):
        model.greedy_chain(net, [encoded], max_emissions=2, budgets=(-1,))
    with pytest.raises(ValueError):
        model.score_at_budgets(model.greedy_chain(net, [encoded], max_emissions=2))


def test_hidden_states_are_not_returned_from_a_cached_forward():
    """Hidden states cover whole sequences, so a cached forward refuses to return them."""
    net = _network()
    tokens, scalars = _inputs(_encoded())
    with torch.no_grad():
        out = net(tokens[:, :5], scalars[:, :5], return_cache=True)
        with pytest.raises(ValueError):
            net(tokens, scalars, return_hidden=True, return_cache=True)
        with pytest.raises(ValueError):
            net(tokens[:, 5:8], scalars[:, 5:8], return_hidden=True, cache=out.cache)


def test_hidden_includes_the_final_normalization():
    """The stacked states end with the final normalized state the logits are read from."""
    net = _network()
    tokens, scalars = _inputs(_encoded())
    with torch.no_grad():
        out = net(tokens, scalars, return_hidden=True)
    assert out.hidden.shape[0] == net.n_sites == SPEC.n_layers + 2
    assert torch.equal(out.hidden[-1], net.final_norm(out.hidden[-2]))
    assert torch.allclose(out.logits, out.hidden[-1] @ net.embedding.weight.T, atol=1e-6)


def test_batched_hidden_states_equal_single_sequence_reads():
    """States gathered over four sequences at per-row positions equal four single reads."""
    net = _network()
    sequences = _prompts(range(4), target=Target.CHAIN)
    positions = [item_positions(one.tokens, VOCAB)[:6].tolist() for one in sequences]
    layers = (0, 1, SPEC.n_layers + 1)
    states = model.batched_hidden_states(net, sequences, positions=positions, layers=layers, chunk_size=3)
    assert states.shape == (3, 4, 6, SPEC.d_model)
    for row, one in enumerate(sequences):
        alone = model.hidden_states_at(net, one, positions=positions[row], layers=layers)
        assert np.allclose(states[:, row], alone, atol=1e-4)
    shared = model.batched_hidden_states(net, sequences, positions=positions[0], layers=layers)
    assert np.allclose(shared[:, 0], states[:, 0], atol=1e-4)
    with pytest.raises(ValueError):
        model.batched_hidden_states(net, sequences, positions=[positions[0], positions[1][:3]], layers=layers)


def test_a_prediction_of_exactly_zero_commits_minus_one():
    """With the relation direction and bias at zero every prediction is 0: committed as -1."""
    net = _network()
    net.load_state_dict(_trained_network().state_dict())
    with torch.no_grad():
        net.relation_direction.zero_()
        net.relation_bias.zero_()
    net.eval()
    prompts = _prompts(range(4))
    generated = model.greedy_chain(net, prompts, max_emissions=8, budgets=range(9))
    assert torch.equal(generated.budget_prediction, torch.zeros_like(generated.budget_prediction))
    assert (model.score_at_budgets(generated) == -1).all()
    at_relation = generated.tokens == VOCAB.rel
    assert at_relation.any()
    assert torch.equal(generated.scalars[at_relation], torch.full((int(at_relation.sum()),), -1.0))
    assert torch.equal(generated.scalars[~at_relation], torch.zeros(int((~at_relation).sum())))


def test_cache_validation_rejects_mismatched_caches():
    """A cache of the wrong batch, length, or depth is refused before any attention runs."""
    net = _network()
    tokens, scalars = _inputs(_encoded())
    with torch.no_grad():
        cache = net(tokens[:, :6], scalars[:, :6], return_cache=True).cache
        with pytest.raises(ValueError):
            net(tokens.repeat(2, 1)[:, 6:8], scalars.repeat(2, 1)[:, 6:8], cache=cache)
        with pytest.raises(ValueError):
            net(tokens[:, 6:8], scalars[:, 6:8], cache=model.KVCache(layers=cache.layers, length=5))
        with pytest.raises(ValueError):
            net(tokens[:, 6:8], scalars[:, 6:8], cache=model.KVCache(layers=cache.layers[:1], length=6))
        with pytest.raises(ValueError):
            net(tokens[:, 6:8], scalars[:, 6:8], cache=model.KVCache(layers=(), length=0))


def test_positions_past_the_rotary_table_raise_over_a_cache():
    """Feeding beyond `max_len` through the cache raises rather than wrapping the rotation."""
    net = _network()
    tokens, scalars = _inputs(_encoded())
    tokens = tokens.repeat(1, 3)[:, :MAX_LEN]
    scalars = scalars.repeat(1, 3)[:, :MAX_LEN]
    with torch.no_grad():
        out = net(tokens[:, :-1], scalars[:, :-1], return_cache=True)
        net(tokens[:, -1:], scalars[:, -1:], cache=out.cache)
        with pytest.raises(ValueError):
            net(tokens[:, -2:], scalars[:, -2:], cache=out.cache)


def test_a_multi_token_cached_forward_is_bottom_right_aligned(monkeypatch):
    """Two new tokens over a cache match the whole-sequence forward, and not a top-left mask."""
    net = _network()
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
    tokens, scalars = _inputs(_encoded())
    cut = 9
    with torch.no_grad():
        whole = net(tokens, scalars)
        out = net(tokens[:, :cut], scalars[:, :cut], return_cache=True)
        branch = net(tokens[:, cut : cut + 2], scalars[:, cut : cut + 2], cache=out.cache)
    assert torch.allclose(branch.logits, whole.logits[:, cut : cut + 2], atol=1e-5)
    assert torch.allclose(branch.scalar_pred, whole.scalar_pred[:, cut : cut + 2], atol=1e-5)

    def top_left(n_new, cached_len, *, device):
        keys = torch.arange(cached_len + n_new, device=device)[None, :]
        return keys <= torch.arange(n_new, device=device)[:, None]

    monkeypatch.setattr(model, "allowed_keys", top_left)
    with torch.no_grad():
        wrong = net(tokens[:, cut : cut + 2], scalars[:, cut : cut + 2], cache=out.cache)
    assert not torch.allclose(wrong.logits, whole.logits[:, cut : cut + 2], atol=1e-5)


def test_zero_emission_rollout_reads_the_prompt_alone():
    """`max_emissions=0` gives an empty continuation and the prediction at zero reasoning steps."""
    net = _trained_network()
    encoded = _encoded(target=Target.PROMPT_ONLY)
    generated = model.greedy_chain(net, [encoded], max_emissions=0, budgets=(0,))
    assert generated.tokens.shape == (1, 0) and generated.scalars.shape == (1, 0)
    tokens, scalars = _full_sequence(encoded, generated, 0, 0)
    with torch.no_grad():
        fresh = net(tokens, scalars).scalar_pred[0, hidden_state_index(encoded.prompt_len, 0)]
    assert torch.allclose(generated.budget_prediction[0, 0], fresh, atol=1e-6)


def test_requesting_the_cache_leaves_a_whole_sequence_forward_bit_identical():
    """`return_cache=True` over a whole sequence changes no output bit: one path serves training."""
    net = _network()
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
    tokens, scalars = _inputs(_encoded())
    with torch.no_grad():
        plain = net(tokens, scalars)
        cached = net(tokens, scalars, return_cache=True)
    assert torch.equal(plain.logits, cached.logits)
    assert torch.equal(plain.scalar_pred, cached.scalar_pred)
