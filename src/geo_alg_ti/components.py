"""Mean ablation of attention heads, and the heads that read the context while a network reasons.

Supports the components experiment: which attention heads read the context while the network
writes its reasoning steps, and what the network loses without them. A head's output is its
slice of the input to its layer's output projection, before the projection mixes the heads.
Layers and heads are numbered from one in labels and tables (`L4H3` is the third head of the
fourth layer) and from zero in code.

A greedy rollout runs two kinds of forward pass: over the sequence itself (the
prompt, then each emitted token, extending the key-value cache) and, at each
reasoning-step value, over the two query items appended to the state without
extending it. Each pass's kind and first position are read from its arguments,
so a replacement can be keyed by both. A replacement in a sequence pass reaches
every later position through the keys and values of the layers above it, so
replacing a head in every pass of a rollout equals replacing it over the whole
sequence; a replacement in a query pass changes only that read.

Ablation replaces a head's output with its mean at the same position and kind
of pass, over greedy rollouts of pretraining-distribution prompts.

Heads are ranked by where they attend, never by what their ablation does. On every trial of
the test conditions whose answer needs the context, the intact network writes its chain to
the largest number of reasoning steps, and one forward over the prompt and that chain gives
each head's attention. Every reasoning step written before the network first states the
queried relation reads from its own four positions onto the context relations that mention
either of its items, four positions each; a step that is malformed, or whose items no context
relation mentions, reads nothing and is left out. A head's score is its attention mass onto
those relations, averaged over the step's positions and then pooled over every step of every
trial, and the heads of a network are ranked from the highest score. The test battery is scored with
the network intact, with each head ablated alone, with the k highest-ranked heads ablated
together, and, as their control, with k heads drawn at random from the heads outside the top
2k.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import torch

from geo_alg_ti import evaluate
from geo_alg_ti.battery import Battery
from geo_alg_ti.chains import PAD_TO, ChainAlgorithm
from geo_alg_ti.model import ROLLOUT_CHUNK, RelationalTransformer, allowed_keys, greedy_chain, stack_prompts
from geo_alg_ti.orderings import OrderingLaw, seeded_rng
from geo_alg_ti.sequences import EMISSION_LEN, PROMPT_HEAD_LEN, Target, Vocabulary, encode
from geo_alg_ti.tables import ABLATE_SCHEMA, HEAD_SCORES_SCHEMA
from geo_alg_ti.trials import Relation, Trial, sample_pretraining_trial

COMPONENTS_SALT: int = 0x3C51  # fixed salt, so every draw here is keyed apart from the streams
N_REFERENCE_TRIALS: int = 1024
# The sizes of the sets of highest-ranked heads ablated together, and the random draws of each
# size that serve as their control.
SET_SIZES: tuple[int, ...] = (1, 2, 4)
N_RANDOM_DRAWS: int = 20
ABLATE_TAG: str = "ablate"

# The draws of this module, keyed apart under the salt.
_REFERENCE_KEY: int = 0
_RANDOM_SET_KEY: int = 1
_ABLATE_COLUMNS: tuple[str, ...] = tuple(name for name in ABLATE_SCHEMA if name != "experiment")
_HEAD_SCORE_COLUMNS: tuple[str, ...] = tuple(name for name in HEAD_SCORES_SCHEMA if name != "experiment")


class Forward(StrEnum):
    """A rollout's passes over the sequence, or over the query items appended at a budget."""

    SEQUENCE = "sequence"
    QUERY = "query"


class SetKind(StrEnum):
    """How the network was scored: intact, one head ablated, the highest-ranked heads ablated,
    or a random set of heads ablated as their control."""

    INTACT = "intact"
    HEAD = "head"
    TOP = "top"
    RANDOM = "random"


@dataclass(frozen=True)
class Head:
    """One attention head; zero-based."""

    layer: int
    head: int

    @property
    def label(self) -> str:
        """The table label, numbered from one: `L4H3`."""
        return f"L{self.layer + 1}H{self.head + 1}"


def heads(model: RelationalTransformer) -> tuple[Head, ...]:
    """Every attention head, layer by layer."""
    return tuple(Head(layer, head) for layer in range(len(model.blocks)) for head in range(model.spec.n_heads))


# The test conditions of a battery preset whose answer needs the context; the heads are ranked on
# the chains the network writes on their trials.
CHAIN_CONDITIONS: Mapping[str, tuple[str, ...]] = {
    "content": ("icl_test", "frequent_cue_reversed"),
    "pretrained": ("icl_test", "icl_test_training_cue"),
}


def chain_conditions(preset: str) -> tuple[str, ...]:
    """The conditions of a battery preset whose chains the heads are ranked on."""
    if preset not in CHAIN_CONDITIONS:
        raise ValueError(f"the '{preset}' battery has no conditions to rank heads on; use {sorted(CHAIN_CONDITIONS)}")
    return CHAIN_CONDITIONS[preset]


# A replacement callback: (layer, kind of pass, offset of its first position, head outputs) to a
# replacement, or None to leave them.
_Replace = Callable[[int, Forward, int, torch.Tensor], torch.Tensor | None]


@contextlib.contextmanager
def _hooked(model: RelationalTransformer, *, head_outputs: _Replace) -> Iterator[None]:
    """Route every layer's head outputs through the callback."""
    place: dict[str, object] = {}

    def locate(_module: torch.nn.Module, _args: tuple, kwargs: dict) -> None:
        cache = kwargs.get("cache")
        place["stream"] = Forward.SEQUENCE if cache is None or kwargs.get("return_cache") else Forward.QUERY
        place["offset"] = 0 if cache is None else int(cache.length)

    def on_heads(layer: int) -> Callable:
        def hook(_module: torch.nn.Module, args: tuple) -> torch.Tensor | None:
            return head_outputs(layer, place["stream"], place["offset"], args[0])

        return hook

    handles = [model.register_forward_pre_hook(locate, with_kwargs=True)]
    try:
        for layer, block in enumerate(model.blocks):
            handles.append(block.attention.out.register_forward_pre_hook(on_heads(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def recorded_attention(
    model: RelationalTransformer, *, reduce: Callable[[torch.Tensor], torch.Tensor] | None = None
) -> Iterator[dict[int, torch.Tensor]]:
    """Each layer's attention probabilities in the latest forward, (batch, head, query, key), or
    what `reduce` makes of them, so that only the reduction of each layer is kept.

    The network attends through a fused kernel that returns no probabilities, so a hook on each
    layer's attention recomputes them from the normalized input the layer receives: queries
    and keys from its projections, rotated to their positions after any cached ones, scaled by
    one over the square root of the head width, masked causally, and normalized by a softmax.
    The network's outputs are unchanged.
    """
    recorded: dict[int, torch.Tensor] = {}

    def on_attention(layer: int) -> Callable:
        def hook(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
            x = args[0]
            cache = kwargs.get("cache")
            cached_len = 0 if cache is None else int(cache[0].shape[-2])
            query = module.rotary(module.split_heads(module.query(x)), offset=cached_len)
            key = module.rotary(module.split_heads(module.key(x)), offset=cached_len)
            if cache is not None:
                key = torch.cat((cache[0], key), dim=-2)
            scores = query @ key.transpose(-2, -1) / math.sqrt(module.head_dim)
            allowed = allowed_keys(x.shape[1], cached_len, device=x.device)
            probabilities = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
            recorded[layer] = probabilities if reduce is None else reduce(probabilities)

        return hook

    handles = [
        block.attention.register_forward_pre_hook(on_attention(layer), with_kwargs=True)
        for layer, block in enumerate(model.blocks)
    ]
    try:
        yield recorded
    finally:
        for handle in handles:
            handle.remove()


def reading_positions(
    context: Sequence[Relation], steps: Sequence[Relation | None], *, prompt_len: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each reading step's own positions and the positions of the context relations it reads.

    Step e of the chain sits at `prompt_len + 4 e` onward and context relation j at
    `PROMPT_HEAD_LEN + 4 j` onward, four positions each. A step reads the context relations that
    mention either of its items; a malformed step (None), or one whose items no context relation
    mentions, reads nothing and is left out.
    """
    span = np.arange(EMISSION_LEN)
    read: list[tuple[np.ndarray, np.ndarray]] = []
    for index, step in enumerate(steps):
        if step is None:
            continue
        items = {int(step.a), int(step.b)}
        mentioned = [j for j, relation in enumerate(context) if items & {int(relation.a), int(relation.b)}]
        if not mentioned:
            continue
        queries = prompt_len + EMISSION_LEN * index + span
        keys = (PROMPT_HEAD_LEN + EMISSION_LEN * np.asarray(mentioned)[:, None] + span).ravel()
        read.append((queries, keys))
    return read


def reading_weights(positions: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]], length: int) -> torch.Tensor:
    """The (sequence, query, key) weights of a batch's reading steps.

    Every pair of a step's position and a position it reads weighs one over the step's length,
    so contracting the weights with a head's attention sums, over the steps, each step's
    attention mass onto what it reads averaged over the step's positions.
    """
    weights = np.zeros((len(positions), int(length), int(length)), dtype=np.float32)
    for row, steps in enumerate(positions):
        for queries, keys in steps:
            weights[row, queries[:, None], keys[None, :]] = 1.0 / EMISSION_LEN
    return torch.as_tensor(weights)


def layer_reading_mass(probabilities: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """One layer's attention mass per head onto what the steps read, summed over the steps."""
    return torch.einsum("bhqk,bqk->h", probabilities, weights.to(probabilities.dtype))


@dataclass(frozen=True)
class HeadScores:
    """Each head's mean attention mass from the reasoning steps onto the context relations they
    read, pooled over the scored steps of every trial. `n_steps` counts those scored steps;
    `n_trials` counts every trial of the conditions scored, including trials whose chain has no
    step to score."""

    scores: Mapping[Head, float]
    n_steps: int
    n_trials: int

    @property
    def ranked(self) -> tuple[Head, ...]:
        """Every head from the highest score down; a tie goes to the earlier head by layer."""
        return tuple(sorted(self.scores, key=lambda one: (-self.scores[one], one.layer, one.head)))

    @property
    def ranks(self) -> dict[Head, int]:
        """Each head's attention rank, one for the highest score."""
        return {one: place for place, one in enumerate(self.ranked, start=1)}


def head_scores(
    model: RelationalTransformer,
    tests: Battery,
    vocab: Vocabulary,
    conditions: Sequence[str],
    *,
    chunk_size: int = ROLLOUT_CHUNK,
) -> HeadScores:
    """Score every head on the chains the intact network writes on the named conditions' trials.

    Each trial is rolled out greedily to the battery's largest budget; its reasoning steps are
    its emissions up to, not including, the first that states the queried pair, and one forward
    over the prompt and the whole rollout gives the attention they are scored on. A head's score
    is its attention mass pooled over every scored step of every trial, so a trial with more
    steps weighs more.
    """
    wanted = set(conditions)
    trials = [trial for name, trial in tests.trials if name in wanted]
    if not trials:
        raise ValueError(f"the battery holds no trial of {sorted(wanted)}")
    k_max = max(int(budget) for budget in tests.budgets)
    totals = torch.zeros((len(model.blocks), model.spec.n_heads), dtype=torch.float64)
    n_steps = 0
    chunk = max(int(chunk_size), 1)
    for start in range(0, len(trials), chunk):
        group = trials[start : start + chunk]
        prompts = [encode(trial, None, vocab, target=Target.PROMPT_ONLY) for trial in group]
        generated = greedy_chain(model, prompts, max_emissions=k_max)
        emitted_tokens = generated.tokens.cpu().numpy()
        emitted_scalars = generated.scalars.cpu().numpy()
        positions = []
        for row, trial in enumerate(group):
            emissions = evaluate.decode_emissions(emitted_tokens[row], emitted_scalars[row], vocab)
            steps = emissions[: evaluate.chain_length(emissions, trial)]
            positions.append(reading_positions(trial.context, steps, prompt_len=generated.prompt_len))
        n_steps += sum(len(steps) for steps in positions)
        prompt_tokens, prompt_scalars, _ = stack_prompts(prompts, vocab, device=model.device)
        tokens = torch.cat((prompt_tokens, generated.tokens), dim=1)
        scalars = torch.cat((prompt_scalars, generated.scalars), dim=1)
        weights = reading_weights(positions, tokens.shape[1]).to(model.device)
        # Each layer's attention is contracted with the weights as it is computed, not kept.
        contract = functools.partial(layer_reading_mass, weights=weights)
        with torch.inference_mode(), recorded_attention(model, reduce=contract) as mass:
            model(tokens, scalars)
            totals += torch.stack([mass[layer] for layer in sorted(mass)]).double().cpu()
    if n_steps == 0:
        raise ValueError("the network wrote no reasoning step that reads the context")
    means = (totals / n_steps).numpy()
    return HeadScores(
        scores={one: float(means[one.layer, one.head]) for one in heads(model)},
        n_steps=int(n_steps),
        n_trials=len(trials),
    )


def head_scores_table(scores: HeadScores, *, seed: int, step: int) -> pd.DataFrame:
    """The head-score table of one checkpoint: one row per head, layer by layer."""
    ranks = scores.ranks
    rows = [
        {
            "head": one.label,
            "layer": one.layer + 1,
            "seed": int(seed),
            "step": int(step),
            "score": float(value),
            "attention_rank": ranks[one],
            "n_steps": scores.n_steps,
            "n_trials": scores.n_trials,
        }
        for one, value in sorted(scores.scores.items(), key=lambda item: (item[0].layer, item[0].head))
    ]
    return pd.DataFrame(rows, columns=list(_HEAD_SCORE_COLUMNS))


def reading_heads(
    model: RelationalTransformer,
    tests: Battery,
    vocab: Vocabulary,
    law: OrderingLaw,
    *,
    conditions: Sequence[str],
    cue_weights: np.ndarray,
    seed: int,
) -> tuple[HeadScores, Means]:
    """A network's heads scored by their attention to the context on its chains on the named
    conditions of `tests`, and every head's means over rollouts of the pretraining trials
    drawn from `cue_weights`, to every reasoning-step value of `tests`."""
    scores = head_scores(model, tests, vocab, conditions)
    reference = reference_trials(law, cue_weights, seed=seed)
    return scores, reference_means(model, reference, vocab, budgets=tests.budgets)


def reference_trials(law: OrderingLaw, weights: np.ndarray, *, seed: int) -> list[Trial]:
    """The pretraining-distribution trials whose rollouts give the ablation means."""
    rng = seeded_rng(seed, COMPONENTS_SALT, _REFERENCE_KEY)
    return [sample_pretraining_trial(law, rng, weights=weights) for _ in range(N_REFERENCE_TRIALS)]


@dataclass(frozen=True)
class Means:
    """Mean head outputs by kind of pass: (position, layer, width) tables.

    Positions no reference rollout reached are missing; reading one raises.
    """

    head_outputs: Mapping[Forward, torch.Tensor]
    reached: Mapping[Forward, np.ndarray]

    def read(self, stream: Forward, layer: int, offset: int, length: int) -> torch.Tensor:
        """The means of one layer over `length` positions from `offset`, (position, width)."""
        reached = self.reached[stream][offset : offset + length]
        if reached.size != length or not reached.all():
            raise ValueError(f"no reference rollout reached {stream} positions {offset}..{offset + length - 1}")
        return self.head_outputs[stream][offset : offset + length, layer]


def reference_means(model: RelationalTransformer, trials: Sequence[Trial], vocab: Vocabulary, *, budgets: Sequence[int]) -> Means:
    """Each head's mean output by kind of pass and position over greedy rollouts.

    The rollouts run to the padded chain length or to the largest budget in `budgets`, whichever
    is longer, so every position a scored rollout reads has a mean.
    """
    shape = (model.spec.max_len, len(model.blocks), model.spec.d_model)
    sums = {stream: torch.zeros(shape, dtype=torch.float64, device=model.device) for stream in Forward}
    counts = {stream: torch.zeros(shape[0], dtype=torch.float64, device=model.device) for stream in Forward}

    def add(layer: int, stream: Forward, offset: int, x: torch.Tensor) -> None:
        sums[stream][offset : offset + x.shape[1], layer] += x.sum(dim=0).double()
        if layer == 0:
            counts[stream][offset : offset + x.shape[1]] += x.shape[0]

    prompts = [encode(trial, None, vocab, target=Target.PROMPT_ONLY) for trial in trials]
    emissions = max([PAD_TO, *(int(budget) for budget in budgets)])
    with _hooked(model, head_outputs=add):
        for start in range(0, len(prompts), ROLLOUT_CHUNK):
            greedy_chain(model, prompts[start : start + ROLLOUT_CHUNK], max_emissions=emissions, budgets=budgets)
    means = {}
    for stream, total in sums.items():
        seen = counts[stream][:, None, None]
        means[stream] = torch.where(seen > 0, total / seen.clamp(min=1), torch.nan).float()
    reached = {stream: (counts[stream] > 0).cpu().numpy() for stream in Forward}
    return Means(head_outputs=means, reached=reached)


def _head_masks(model: RelationalTransformer, chosen: Sequence[Head]) -> dict[int, torch.Tensor]:
    """Per layer, the columns of the output projection's input that the chosen heads write."""
    width = model.spec.d_model // model.spec.n_heads
    masks: dict[int, torch.Tensor] = {}
    for one in chosen:
        mask = masks.setdefault(one.layer, torch.zeros(model.spec.d_model, dtype=torch.bool, device=model.device))
        mask[one.head * width : (one.head + 1) * width] = True
    return masks


@contextlib.contextmanager
def ablated(model: RelationalTransformer, chosen: Sequence[Head], means: Means) -> Iterator[None]:
    """Replace the chosen heads' outputs with their reference means in every forward."""
    masks = _head_masks(model, chosen)

    def replace_heads(layer: int, stream: Forward, offset: int, x: torch.Tensor) -> torch.Tensor | None:
        mask = masks.get(layer)
        if mask is None:
            return None
        mean = means.read(stream, layer, offset, x.shape[1]).to(x.dtype)
        return torch.where(mask, mean, x)

    with _hooked(model, head_outputs=replace_heads):
        yield


class RowAblation:
    """The chosen heads' replacements in whole-sequence forwards, built once for a run.

    Every position takes the heads' sequence means, except the two query items appended before
    an answer, which take the means of a pass over those items, the way a scored rollout reads
    its answer. The table for each position of those query items is built the first time it is
    asked for and kept.
    """

    def __init__(self, model: RelationalTransformer, chosen: Sequence[Head], means: Means) -> None:
        self.heads = tuple(chosen)
        self.masks = _head_masks(model, chosen)
        reached = means.reached[Forward.SEQUENCE]
        self.limit = int(reached.size if reached.all() else np.argmin(reached))
        self._means = means
        self._sequence = {layer: means.read(Forward.SEQUENCE, layer, 0, self.limit) for layer in self.masks}
        self._with_query: dict[tuple[int, int], torch.Tensor] = {}

    def sequence(self, layer: int) -> torch.Tensor:
        """The sequence means of one layer over every position reached, (position, width)."""
        return self._sequence[layer]

    def with_query(self, layer: int, start: int) -> torch.Tensor:
        """The sequence means of one layer with the query-pass means at `start` and `start + 1`."""
        key = (int(layer), int(start))
        if key not in self._with_query:
            table = self._sequence[layer].clone()
            table[start : start + 2] = self._means.read(Forward.QUERY, layer, start, 2)
            self._with_query[key] = table
        return self._with_query[key]


@contextlib.contextmanager
def ablated_rows(
    model: RelationalTransformer,
    ablation: RowAblation,
    *,
    rows: torch.Tensor,
    query_at: torch.Tensor,
    starts: Sequence[int],
) -> Iterator[None]:
    """Replace the heads of `ablation` on the marked rows of a whole-sequence forward, as a rollout
    scored at the same reasoning-step value would.

    `rows` marks the rows replaced, (batch,); `query_at` holds each row's position of the first
    of the two query items appended before its answer, (batch,), one of `starts`. A forward reads
    nothing back from the device. Only a forward over whole sequences, with no key-value cache,
    is accepted.
    """
    starts = tuple(int(start) for start in starts)
    marked = rows.to(torch.bool)[:, None, None]
    at_start = tuple(marked & (query_at == start)[:, None, None] for start in starts)

    def replace_heads(layer: int, stream: Forward, offset: int, x: torch.Tensor) -> torch.Tensor | None:
        mask = ablation.masks.get(layer)
        if mask is None:
            return None
        if stream is not Forward.SEQUENCE or offset != 0:
            raise ValueError("rows are ablated in a forward over whole sequences only")
        length = x.shape[1]
        if length > ablation.limit:
            raise ValueError(f"no reference rollout reached sequence positions {ablation.limit}..{length - 1}")
        replacement = ablation.sequence(layer)[:length].to(x.dtype).expand_as(x)
        for at, start in zip(at_start, starts, strict=True):
            replacement = torch.where(at, ablation.with_query(layer, start)[:length].to(x.dtype), replacement)
        return torch.where(marked & mask, replacement, x)

    with _hooked(model, head_outputs=replace_heads):
        yield


def random_set(ranked: Sequence[Head], size: int, *, seed: int, draw: int) -> tuple[Head, ...]:
    """`size` heads drawn without replacement from outside the top `2 size`, keyed by the seed,
    the size, and the draw, listed by attention rank."""
    outside = tuple(ranked)[2 * int(size) :]
    picked = seeded_rng(seed, COMPONENTS_SALT, _RANDOM_SET_KEY, int(size), int(draw)).choice(
        len(outside), size=int(size), replace=False
    )
    return tuple(outside[index] for index in sorted(picked))


@dataclass(frozen=True)
class AblationSet:
    """Heads ablated together, listed by attention rank, how they were chosen, and the random
    draw they are (-1 for a set that was not drawn)."""

    kind: SetKind
    heads: tuple[Head, ...] = ()
    draw: int = -1


def check_set_sizes(n_heads: int) -> None:
    """Raise unless a network of `n_heads` heads leaves, for every k in `SET_SIZES`, the k heads
    a random set draws from outside its top 2k: at least three times the largest k."""
    largest = max(SET_SIZES)
    if n_heads < 3 * largest:
        outside = max(n_heads - 2 * largest, 0)
        raise ValueError(
            f"{outside} of the network's {n_heads} heads lie outside the top {2 * largest}; "
            f"a random set of {largest} needs {largest}, so the ablations need at least {3 * largest} heads"
        )


def ablation_sets(ranked: Sequence[Head], *, seed: int) -> tuple[AblationSet, ...]:
    """Every set a network is scored under: intact, each head alone, the top k, and random k.

    `ranked` lists every head from the highest score down. The single heads come layer by
    layer; then, for each k in `SET_SIZES`, the k highest-ranked heads; then, for each k,
    `N_RANDOM_DRAWS` sets of k heads drawn without replacement from the heads outside the top
    2k, each draw keyed by the seed, k, and its index. Raises, as `check_set_sizes` does, for
    a network with too few heads.
    """
    ranked = tuple(ranked)
    check_set_sizes(len(ranked))
    sets = [AblationSet(SetKind.INTACT)]
    sets.extend(AblationSet(SetKind.HEAD, (one,)) for one in sorted(ranked, key=lambda one: (one.layer, one.head)))
    sets.extend(AblationSet(SetKind.TOP, ranked[:size]) for size in SET_SIZES)
    for size in SET_SIZES:
        for draw in range(N_RANDOM_DRAWS):
            sets.append(AblationSet(SetKind.RANDOM, random_set(ranked, size, seed=seed, draw=draw), draw))
    return tuple(sets)


def _columns(chosen: AblationSet, ranks: Mapping[Head, int]) -> dict[str, object]:
    """The ablate table's label columns for one set; a single head also names itself."""
    single = chosen.heads[0] if chosen.kind is SetKind.HEAD else None
    return {
        "set_kind": str(chosen.kind),
        "head": "" if single is None else single.label,
        "layer": float("nan") if single is None else float(single.layer + 1),
        "n_heads": len(chosen.heads),
        "heads": "+".join(one.label for one in chosen.heads),
        "draw": int(chosen.draw),
        "attention_rank": -1 if single is None else int(ranks[single]),
    }


def ablate_table(
    model: RelationalTransformer,
    tests: Battery,
    vocab: Vocabulary,
    means: Means,
    scores: HeadScores,
    *,
    step: int,
    chain_algorithm: ChainAlgorithm,
) -> pd.DataFrame:
    """The ablate table of one checkpoint: the test battery under every set of `ablation_sets`.

    The random draws are keyed by the battery's seed. A set of heads is scored once and its
    rows are reused wherever it recurs: the highest-ranked head, scored alone and as the top set
    of one; a head ranked third or lower, scored alone and as a random draw of one (drawn from
    outside the top two); and a set drawn more than once among the random draws.
    """
    algorithm = ChainAlgorithm(chain_algorithm)
    ranks = scores.ranks
    scored: dict[frozenset[Head], pd.DataFrame] = {}
    frames = []
    for chosen in ablation_sets(scores.ranked, seed=tests.seed):
        key = frozenset(chosen.heads)
        if key not in scored:
            with ablated(model, chosen.heads, means):
                _, scored[key] = evaluate.evaluate_battery(
                    model, tests, vocab, step=int(step), tag=ABLATE_TAG, record_trials=False, chain_algorithm=algorithm
                )
        frames.append(scored[key].assign(**_columns(chosen, ranks)).loc[:, list(_ABLATE_COLUMNS)])
    return pd.concat(frames, ignore_index=True)
