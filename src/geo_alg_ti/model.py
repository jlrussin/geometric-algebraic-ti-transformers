"""Decoder-only transformer with a scalar relation channel.

Realizes the extended methods' "Architecture and optimization" paragraph and the
training loss. The canonical network has four layers, eight heads, residual
width 256, feedforward width 1024 with GELU, rotary position embeddings,
pre-layer normalization, no dropout, no additive biases in the transformer
projections, and tied input and output embeddings. A relation is a structural
`<rel>` token plus a scalar s in {-1, +1}: the input at a relation position is
e_<rel> + s * r, and the same learned direction reads the scalar back,
s_hat_t = h_{t-1}^T r + b, from the final normalized hidden state at the
preceding position. The loss is the mean token cross-entropy over the
loss-bearing token positions plus the mean squared error over the loss-bearing
relation positions, unweighted. During greedy generation the model emits
`<rel>` itself; the scalar at that new position is +1 if s_hat > 0 and -1
otherwise. Greedy decoding is prefix-consistent, so one rollout to the largest
budget gives every reasoning-step value: prompts of one length are rolled out
together over a per-block key-value cache, and after each emission the two
query items are appended to the cached state to read the relation at that
budget. Initialization: linear and embedding weights N(0, 0.02), residual
output projections N(0, 0.02 / sqrt(2 L)), the relation direction and its
bias zero.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geo_alg_ti.config import N_HEADS
from geo_alg_ti.sequences import EMISSION_LEN, Encoded, Vocabulary, item_positions
from geo_alg_ti.streams import Batch


INIT_STD: float = 0.02
ROTARY_BASE: float = 10000.0


# How many sequences are rolled out, or read for hidden states, in one pass: about a
# gigabyte of GPU memory at the canonical size, and the same results at any size.
ROLLOUT_CHUNK: int = 1024


@dataclass(frozen=True)
class ModelSpec:
    """Depth, width, heads, and feedforward sublayer; the rest of the architecture is fixed."""

    vocab_size: int
    max_len: int
    n_layers: int = 4
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = N_HEADS
    feedforward: bool = True


@dataclass(frozen=True)
class KVCache:
    """The keys and values every block has seen so far, extended a few tokens at a time.

    `layers` holds one (key, value) pair per block, shaped (batch, head, position, head width),
    with the keys already rotated to their positions. `length` is the number of positions the
    cache covers, which is where the next token fed to the network sits. A cache is never
    modified: a forward that extends it returns a new one.
    """

    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    length: int

    @property
    def batch_size(self) -> int:
        """How many sequences the cache covers."""
        if not self.layers:
            raise ValueError("an empty cache covers no sequences")
        return int(self.layers[0][0].shape[0])


@dataclass
class ModelOutput:
    """Token logits, the scalar prediction per position, optional hidden states, optional cache."""

    logits: torch.Tensor
    scalar_pred: torch.Tensor
    hidden: torch.Tensor | None = None
    cache: KVCache | None = None


@dataclass
class LossTerms:
    """The two terms of the training loss and their unweighted sum."""

    token_cross_entropy: torch.Tensor
    scalar_squared_error: torch.Tensor
    total: torch.Tensor


@dataclass
class Generated:
    """Greedy rollouts of a group of prompts, with the relation read at each requested budget.

    `tokens` and `scalars` hold the continuation after the prompt, one row per prompt, over
    `n_emissions` complete emissions. A scalar is committed only where the network emitted the
    relation token itself, and is the sign of the prediction read at the position before it.
    `budget_prediction[:, j]` is the raw relation prediction read after `budgets[j]` emissions
    with the two query items appended; `score_at_budgets` turns it into signs.
    """

    tokens: torch.Tensor
    scalars: torch.Tensor
    n_emissions: int
    prompt_len: int
    budgets: tuple[int, ...]
    budget_prediction: torch.Tensor | None = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension into each other."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _RotaryPositions(nn.Module):
    """Rotary position embeddings, tabulated once to the longest sequence the network sees."""

    def __init__(self, head_dim: int, max_len: int, base: float = ROTARY_BASE) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"rotary positions need an even head width, not {head_dim}")
        frequencies = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), frequencies)
        table = torch.cat((angles, angles), dim=-1)[None, None]
        self.register_buffer("cosines", table.cos(), persistent=False)
        self.register_buffer("sines", table.sin(), persistent=False)

    def forward(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        """Rotate queries or keys shaped (batch, head, position, width) sitting from `offset` on."""
        length = x.shape[-2]
        tabulated = self.cosines.shape[-2]
        if offset + length > tabulated:
            raise ValueError(f"positions up to {offset + length} exceed the {tabulated} tabulated")
        window = slice(offset, offset + length)
        return x * self.cosines[..., window, :] + _rotate_half(x) * self.sines[..., window, :]


class _Attention(nn.Module):
    """Causal multi-head self-attention with rotary positions and no projection biases."""

    def __init__(self, spec: ModelSpec, rotary: _RotaryPositions) -> None:
        super().__init__()
        if spec.d_model % spec.n_heads:
            raise ValueError(f"width {spec.d_model} does not divide into {spec.n_heads} heads")
        self.n_heads = spec.n_heads
        self.head_dim = spec.d_model // spec.n_heads
        self.rotary = rotary
        self.query = nn.Linear(spec.d_model, spec.d_model, bias=False)
        self.key = nn.Linear(spec.d_model, spec.d_model, bias=False)
        self.value = nn.Linear(spec.d_model, spec.d_model, bias=False)
        self.out = nn.Linear(spec.d_model, spec.d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the width into heads, giving (batch, head, position, head width)."""
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Attend over this position and every earlier one, cached positions included.

        The new positions sit after the `cache`'s, so their queries and keys are rotated from
        there. Without a cache the attention is plainly causal; one new token attends to every
        cached key and itself; several new tokens over a cache use an explicit mask, since the
        built-in causal mask aligns the first new query with the first cached key.
        """
        batch, length, width = x.shape
        cached_len = 0 if cache is None else int(cache[0].shape[-2])
        query = self.rotary(self.split_heads(self.query(x)), offset=cached_len)
        key = self.rotary(self.split_heads(self.key(x)), offset=cached_len)
        value = self.split_heads(self.value(x))
        if cache is not None:
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)
        if cached_len == 0:
            attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        elif length == 1:
            attended = F.scaled_dot_product_attention(query, key, value)
        else:
            allowed = allowed_keys(length, cached_len, device=x.device)
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
        out = self.out(attended.transpose(1, 2).reshape(batch, length, width))
        return out, ((key, value) if return_cache else None)


def allowed_keys(n_new: int, cached_len: int, *, device: torch.device) -> torch.Tensor:
    """Boolean (new position, key) mask: a new position sees the cache and itself, no later key."""
    keys = torch.arange(cached_len + n_new, device=device)[None, :]
    new = torch.arange(n_new, device=device)[:, None]
    return keys <= cached_len + new


class _Block(nn.Module):
    """One pre-normalized block: attention, then the feedforward sublayer when the spec has one."""

    def __init__(self, spec: ModelSpec, rotary: _RotaryPositions) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(spec.d_model)
        self.attention = _Attention(spec, rotary)
        self.feedforward_norm = nn.LayerNorm(spec.d_model) if spec.feedforward else None
        self.feedforward = (
            nn.Sequential(
                nn.Linear(spec.d_model, spec.d_ff, bias=False),
                nn.GELU(),
                nn.Linear(spec.d_ff, spec.d_model, bias=False),
            )
            if spec.feedforward
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Add each sublayer's output to the residual stream; pass the attention cache through."""
        attended, pair = self.attention(self.attention_norm(x), cache=cache, return_cache=return_cache)
        x = x + attended
        if self.feedforward is not None and self.feedforward_norm is not None:
            x = x + self.feedforward(self.feedforward_norm(x))
        return x, pair


class RelationalTransformer(nn.Module):
    """The decoder-only transformer with the scalar relation channel."""

    def __init__(self, spec: ModelSpec, vocab: Vocabulary) -> None:
        super().__init__()
        if spec.vocab_size != vocab.size:
            raise ValueError(f"the spec holds {spec.vocab_size} tokens and the vocabulary {vocab.size}")
        self.spec = spec
        self.vocab = vocab
        self.embedding = nn.Embedding(spec.vocab_size, spec.d_model)
        self.rotary = _RotaryPositions(spec.d_model // spec.n_heads, spec.max_len)
        self.blocks = nn.ModuleList(_Block(spec, self.rotary) for _ in range(spec.n_layers))
        self.final_norm = nn.LayerNorm(spec.d_model)
        # One direction carries a relation's sign into the sequence and reads it back out.
        self.relation_direction = nn.Parameter(torch.zeros(spec.d_model))
        self.relation_bias = nn.Parameter(torch.zeros(()))
        self.apply(self._initialize)
        for block in self.blocks:
            self._scale_residual_output(block.attention.out.weight)
            if block.feedforward is not None:
                self._scale_residual_output(block.feedforward[-1].weight)

    def _initialize(self, module: nn.Module) -> None:
        """Linear and embedding weights start from a narrow normal distribution."""
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)

    def _scale_residual_output(self, weight: torch.Tensor) -> None:
        """A projection writing into the residual stream starts smaller, scaled by the depth."""
        nn.init.normal_(weight, mean=0.0, std=INIT_STD / math.sqrt(2 * self.spec.n_layers))

    @property
    def device(self) -> torch.device:
        """Where the parameters live."""
        return self.embedding.weight.device

    @property
    def n_sites(self) -> int:
        """How many states `hidden` stacks: the input, each block's output, the normalized state."""
        return len(self.blocks) + 2

    def forward(
        self,
        tokens: torch.Tensor,
        scalars: torch.Tensor,
        *,
        return_hidden: bool = False,
        cache: KVCache | None = None,
        return_cache: bool = False,
    ) -> ModelOutput:
        """Run the network over the tokens and their scalars.

        `logits[:, t]` and `scalar_pred[:, t]` are both read from the final normalized state at
        position t and predict position t + 1, so the scalar of the relation slot at
        `scored_answer_index(prompt_len, k)` is read at `hidden_state_index(prompt_len, k)`.
        The scalar channel enters at `<rel>` positions alone; a scalar at any other position
        is ignored. `hidden` stacks the residual stream before the first block, after each
        block, and after the final normalization: `n_layers + 2` states in that order.

        With a `cache`, `tokens` are the positions that follow the cached ones, the outputs
        cover those positions alone, and `return_cache` gives back the cache extended by them.
        Hidden states are returned for whole sequences only, never from a cached forward.
        """
        tokens = tokens.long()
        scalars = scalars.to(self.embedding.weight.dtype)
        if tokens.dim() != 2 or scalars.shape != tokens.shape:
            raise ValueError("tokens and scalars are two-dimensional and of one shape")
        if return_hidden and (cache is not None or return_cache):
            raise ValueError("hidden states cover a whole sequence, not the positions of a cached forward")
        if cache is not None:
            self._check_cache(cache, tokens.shape[0])
        x = self.embedding(tokens)
        signs = torch.where(tokens == self.vocab.rel, scalars, torch.zeros_like(scalars))
        x = x + signs.unsqueeze(-1) * self.relation_direction
        hidden = [x] if return_hidden else []
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            x, pair = block(x, cache=None if cache is None else cache.layers[index], return_cache=return_cache)
            if return_hidden:
                hidden.append(x)
            if pair is not None:
                pairs.append(pair)
        normalized = self.final_norm(x)
        if return_hidden:
            hidden.append(normalized)
        cached_len = 0 if cache is None else cache.length
        return ModelOutput(
            logits=F.linear(normalized, self.embedding.weight),
            scalar_pred=normalized @ self.relation_direction + self.relation_bias,
            hidden=torch.stack(hidden) if return_hidden else None,
            cache=KVCache(layers=tuple(pairs), length=cached_len + tokens.shape[1]) if return_cache else None,
        )

    def _check_cache(self, cache: KVCache, batch: int) -> None:
        """Raise unless the cache holds one pair per block over `length` positions, `batch` rows."""
        if len(cache.layers) != len(self.blocks):
            raise ValueError(f"the cache holds {len(cache.layers)} blocks and the network {len(self.blocks)}")
        for key, value in cache.layers:
            if key.shape[-2] != cache.length or value.shape[-2] != cache.length:
                raise ValueError(f"a cache of length {cache.length} holds keys over {key.shape[-2]} positions")
        if cache.batch_size != batch:
            raise ValueError(f"the cache covers {cache.batch_size} sequences and the tokens {batch}")


def loss_terms(out: ModelOutput, batch: Batch) -> LossTerms:
    """Mean token cross-entropy plus mean squared error at the relation positions.

    Each term is a masked mean: the loss at every position, weighted by its loss mask and
    divided by the number of masked positions, so an empty mask gives zero and no shape depends
    on the data.
    """
    token_mask = batch.token_loss[:, 1:].to(out.logits.dtype)
    per_token = F.cross_entropy(
        out.logits[:, :-1].flatten(0, 1), batch.tokens[:, 1:].flatten().long(), reduction="none"
    ).view_as(token_mask)
    cross_entropy = (per_token * token_mask).sum() / token_mask.sum().clamp(min=1)
    scalar_mask = batch.scalar_loss[:, 1:].to(out.scalar_pred.dtype)
    per_relation = (out.scalar_pred[:, :-1] - batch.scalars[:, 1:]) ** 2
    squared_error = (per_relation * scalar_mask).sum() / scalar_mask.sum().clamp(min=1)
    return LossTerms(
        token_cross_entropy=cross_entropy,
        scalar_squared_error=squared_error,
        total=cross_entropy + squared_error,
    )


@contextlib.contextmanager
def _inference(model: nn.Module) -> Iterator[None]:
    """Evaluation mode without gradients, restoring the training mode afterwards."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def stack_prompts(
    encoded: Sequence[Encoded], vocab: Vocabulary, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack the prompts of encoded sequences of one prompt length into one batch.

    Returns the prompt tokens (batch, prompt length), their scalars, and the two columns of each
    prompt where its query items sit, taken from the item positions of the prompt itself.
    """
    prompts = list(encoded)
    if not prompts:
        raise ValueError("no prompts to stack")
    prompt_len = int(prompts[0].prompt_len)
    for one in prompts:
        if int(one.prompt_len) != prompt_len:
            raise ValueError(f"prompts of {prompt_len} and {int(one.prompt_len)} positions do not stack")
    tokens = np.stack([one.tokens[:prompt_len] for one in prompts])
    scalars = np.stack([one.scalars[:prompt_len] for one in prompts])
    columns = np.stack([item_positions(one.tokens[:prompt_len], vocab)[:2] for one in prompts])
    if columns.shape != (len(prompts), 2):
        raise ValueError("a prompt holds fewer than two item positions")
    return (
        torch.as_tensor(tokens, dtype=torch.long, device=device),
        torch.as_tensor(scalars, dtype=torch.float32, device=device),
        torch.as_tensor(columns, dtype=torch.long, device=device),
    )


def greedy_chain(
    model: RelationalTransformer,
    prompts: Sequence[Encoded],
    *,
    max_emissions: int = 8,
    budgets: Sequence[int] = (),
) -> Generated:
    """Roll every prompt out to `max_emissions` emissions and read the relation at each budget.

    Temperature-zero decoding over a key-value cache: the prompt is fed once, then each emitted
    token extends the cache. At each budget K in `budgets` the two query items are appended to
    the state after K emissions, and the prediction at the second of them is recorded; the state
    itself continues unchanged. A scalar is committed only where the network emitted the relation
    token itself, and is the sign of the prediction read at the position before it.
    """
    tokens, scalars, query_columns = stack_prompts(prompts, model.vocab, device=model.device)
    query = torch.gather(tokens, 1, query_columns)
    return greedy_rollout(model, tokens, scalars, query, max_emissions=max_emissions, budgets=budgets)


def greedy_rollout(
    model: RelationalTransformer,
    tokens: torch.Tensor,
    scalars: torch.Tensor,
    query: torch.Tensor,
    *,
    max_emissions: int = 8,
    budgets: Sequence[int] = (),
) -> Generated:
    """`greedy_chain` over prompts already stacked: tokens and scalars (batch, prompt length) and
    the two query items of each prompt (batch, 2)."""
    n_emissions = int(max_emissions)
    if n_emissions < 0:
        raise ValueError(f"a rollout runs a non-negative number of emissions, not {n_emissions}")
    wanted = tuple(int(budget) for budget in budgets)
    if any(budget < 0 or budget > n_emissions for budget in wanted):
        raise ValueError(f"a budget lies outside 0..{n_emissions}")
    batch, prompt_len = tokens.shape
    query_scalars = torch.zeros_like(query, dtype=torch.float32)
    emitted_tokens: list[torch.Tensor] = []
    emitted_scalars: list[torch.Tensor] = []
    read: dict[int, torch.Tensor] = {}
    with _inference(model):
        out = model(tokens, scalars, return_cache=True)
        for k in range(n_emissions + 1):
            if k in wanted:
                branch = model(query, query_scalars, cache=out.cache)
                read[k] = branch.scalar_pred[:, -1]
            if k == n_emissions:
                break
            for _ in range(EMISSION_LEN):
                token = torch.argmax(out.logits[:, -1], dim=-1)
                committed = torch.where(out.scalar_pred[:, -1] > 0, 1.0, -1.0)
                scalar = torch.where(token == model.vocab.rel, committed, 0.0).to(torch.float32)
                emitted_tokens.append(token[:, None])
                emitted_scalars.append(scalar[:, None])
                out = model(token[:, None], scalar[:, None], cache=out.cache, return_cache=True)
    return Generated(
        tokens=torch.cat(emitted_tokens, dim=1) if emitted_tokens else tokens.new_zeros((batch, 0)),
        scalars=torch.cat(emitted_scalars, dim=1) if emitted_scalars else scalars.new_zeros((batch, 0)),
        n_emissions=n_emissions,
        prompt_len=int(prompt_len),
        budgets=wanted,
        budget_prediction=torch.stack([read[budget] for budget in wanted], dim=1) if wanted else None,
    )


def score_at_budgets(generated: Generated) -> np.ndarray:
    """The sign read at each budget, (rollout, budget): +1 where the prediction is positive."""
    if generated.budget_prediction is None:
        raise ValueError("the rollout read the relation at no budget")
    prediction = generated.budget_prediction.detach().cpu().numpy()
    return np.where(prediction > 0, 1, -1).astype(np.int64)


def item_embedding_sites(tokens: torch.Tensor, vocab: Vocabulary) -> torch.Tensor:
    """Boolean mask of every item-token position in a batch."""
    return (tokens >= vocab.first_item) & (tokens < vocab.size)


def hidden_states_at(
    model: RelationalTransformer, encoded: Encoded, *, positions: Sequence[int], layers: Sequence[int]
) -> np.ndarray:
    """Residual-stream states at the given positions at the given sites.

    Sites run `0..n_layers + 1`: 0 is the embedding input, site l the output of block l, and
    `n_layers + 1` the final normalized state.
    """
    device = model.device
    tokens = torch.as_tensor(encoded.tokens, dtype=torch.long, device=device)[None]
    scalars = torch.as_tensor(encoded.scalars, dtype=torch.float32, device=device)[None]
    with _inference(model):
        out = model(tokens, scalars, return_hidden=True)
    if out.hidden is None:
        raise ValueError("the network returned no hidden states")
    wanted_layers = torch.as_tensor([int(layer) for layer in layers], dtype=torch.long, device=device)
    wanted_positions = torch.as_tensor([int(position) for position in positions], dtype=torch.long, device=device)
    return out.hidden[wanted_layers][:, 0][:, wanted_positions].cpu().numpy()


def batched_hidden_states(
    model: RelationalTransformer,
    encoded: Sequence[Encoded],
    *,
    positions: Sequence[int] | Sequence[Sequence[int]],
    layers: Sequence[int],
    chunk_size: int = ROLLOUT_CHUNK,
) -> np.ndarray:
    """Residual-stream states of many sequences of one length, (site, sequence, position, width).

    `positions` is one sequence of positions read from every sequence, or one per sequence with
    the same count in each. Sites are as in `hidden_states_at`. Sequences are run in chunks of
    `chunk_size`, one forward per chunk.
    """
    sequences = list(encoded)
    if not sequences:
        raise ValueError("no sequences to read states from")
    length = int(sequences[0].tokens.size)
    for one in sequences:
        if int(one.tokens.size) != length:
            raise ValueError(f"sequences of {length} and {int(one.tokens.size)} positions do not stack")
    per_row = _positions_per_row(positions, len(sequences))
    device = model.device
    wanted_layers = torch.as_tensor([int(layer) for layer in layers], dtype=torch.long, device=device)
    chunk = max(int(chunk_size), 1)
    chunks: list[np.ndarray] = []
    with _inference(model):
        for start in range(0, len(sequences), chunk):
            rows = sequences[start : start + chunk]
            tokens = torch.as_tensor(np.stack([one.tokens for one in rows]), dtype=torch.long, device=device)
            scalars = torch.as_tensor(np.stack([one.scalars for one in rows]), dtype=torch.float32, device=device)
            out = model(tokens, scalars, return_hidden=True)
            if out.hidden is None:
                raise ValueError("the network returned no hidden states")
            states = out.hidden[wanted_layers]
            index = torch.as_tensor(per_row[start : start + chunk], dtype=torch.long, device=device)
            index = index[None, :, :, None].expand(states.shape[0], -1, -1, states.shape[-1])
            chunks.append(torch.gather(states, 2, index).cpu().numpy())
    return np.concatenate(chunks, axis=1)


def _positions_per_row(positions: Sequence[int] | Sequence[Sequence[int]], n_rows: int) -> np.ndarray:
    """One row of positions per sequence: the shared positions repeated, or the given rows."""
    given = list(positions)
    if not given:
        raise ValueError("no positions to read")
    if all(isinstance(position, int | np.integer) for position in given):
        return np.tile(np.asarray(given, dtype=np.int64)[None, :], (n_rows, 1))
    if len(given) != n_rows:
        raise ValueError(f"{len(given)} rows of positions for {n_rows} sequences")
    rows = [np.asarray([int(position) for position in row], dtype=np.int64) for row in given]
    counts = {row.size for row in rows}
    if len(counts) != 1:
        raise ValueError(f"the rows hold different numbers of positions: {sorted(counts)}")
    return np.stack(rows)
