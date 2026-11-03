"""Token layout, the scalar relation channel, loss masks, and the scored answer index.

Realizes the extended methods' "Task and input sequences",
"Chain-of-thought generation", and "Architecture and optimization" paragraphs.
The extended methods' worked example, for the ordering i0 > i1 > ... > i8 under cue
<cue7> with query (i3, i1) and a two-step chain, encodes as follows (what each
position holds, with its loss masks; "." means no scalar):

    idx  tokens                       holds                scalar  token loss  scalar loss
      0  <bos>                        start                  .        -           -
      1  <cue7>                       cue                    .        -           -
      2  <qmark>                      query opener           .        -           -
      3  i3                           query item x           .        -           -
      4  i1                           query item z           .        -           -
      5  <qrel>                       unresolved relation    .        -           -
      6  <sep>                        separator              .        -           -
      7-10   i8 i7 <rel> <sep>        context relation 1    -1        -           -
      ...    (eight context relations, positions shuffled, orientations random)
     35-38   i5 i4 <rel> <sep>        context relation 8    -1        -           -
    ---- the prompt ends at index 39; generation and scoring start here ----
     39-42   i2 i3 <rel> <sep>        chain emission 1      +1       yes         yes
     43-46   i2 i1 <rel> <sep>        chain emission 2      -1       yes         yes
     47-50   i3 i1 <rel> <sep>        answer                -1       yes         yes
     51-54   i3 i1 <rel> <sep>        padding (answer repeated to nine emissions)
     ...

Every emission is exactly four tokens: item, item, `<rel>`, `<sep>`. The
relation sign rides as a scalar s in {-1, +1} on the `<rel>` position: the
input there is e_<rel> + s * r and the output is s_hat_t = h_{t-1}^T r + b,
read from the final normalized hidden state at the position before the `<rel>`
slot. The query's `<qrel>` is an unresolved-relation sentinel and carries no
scalar. Chain targets place token loss on every position after the prompt and
scalar loss at every `<rel>` there. Answer-only targets (finetuning) are the
prompt plus one answer emission with token loss on `<rel>` and `<sep>` and
scalar loss on `<rel>`; the answer items carry no loss. At evaluation the
prompt ends after the context; after K generated emissions the two query
items are appended and the scalar sign at the following `<rel>` slot is
scored, so `scored_answer_index(prompt_len, k) = prompt_len + 4 k + 2` and the
hidden state read is at `prompt_len + 4 k + 1`.

Vocabulary layout: specials first (`<pad>` = 0, `<bos>`, `<qmark>`, `<qrel>`,
`<sep>`, `<rel>`), then the cues (training cues, then reserved cues), then the
items. With the cue hidden (`task.show_cue: false`), index 1 holds the token of
the frequent cue (cue 0) in every sequence, whichever cue the trial was drawn
under, so no sequence says which ordering its items follow; the layout and every
index are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np

from geo_alg_ti.chains import PAD_TO, Chain
from geo_alg_ti.orderings import FREQUENT_CUE
from geo_alg_ti.trials import Relation, Trial

EMISSION_LEN: int = 4
PROMPT_HEAD_LEN: int = 7
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<qmark>", "<qrel>", "<sep>", "<rel>")
# The ids of the special tokens, fixed by their place in `SPECIAL_TOKENS`.
_PAD, _BOS, _QMARK, _QREL, _SEP, _REL = range(len(SPECIAL_TOKENS))


class Target(StrEnum):
    """What the encoded sequence supervises."""

    CHAIN = "chain"
    ANSWER_ONLY = "answer_only"
    PROMPT_ONLY = "prompt_only"


@dataclass(frozen=True)
class Vocabulary:
    """Token ids for specials, cues, and items; the layout never moves once fixed.

    `show_cue` false puts one shared token at the cue position of every sequence.
    """

    pool_size: int
    n_cues: int
    show_cue: bool = True

    @cached_property
    def size(self) -> int:
        """Total number of token ids."""
        return len(SPECIAL_TOKENS) + self.n_cues + self.pool_size

    @property
    def first_cue(self) -> int:
        """Token id of the first cue; the specials occupy everything below it."""
        return len(SPECIAL_TOKENS)

    @cached_property
    def first_item(self) -> int:
        """Token id of the first pool item; the cues occupy everything below it."""
        return len(SPECIAL_TOKENS) + self.n_cues

    @property
    def pad(self) -> int:
        """Id of `<pad>`."""
        return _PAD

    @property
    def bos(self) -> int:
        """Id of `<bos>`."""
        return _BOS

    @property
    def qmark(self) -> int:
        """Id of `<qmark>`."""
        return _QMARK

    @property
    def qrel(self) -> int:
        """Id of `<qrel>`."""
        return _QREL

    @property
    def sep(self) -> int:
        """Id of `<sep>`."""
        return _SEP

    @property
    def rel(self) -> int:
        """Id of `<rel>`."""
        return _REL

    def cue_token(self, cue: int) -> int:
        """Token id of cue `cue`."""
        if not 0 <= cue < self.n_cues:
            raise ValueError(f"cue {cue} is outside 0..{self.n_cues - 1}")
        return self.first_cue + cue

    def shown_cue_token(self, cue: int) -> int:
        """Token id at the cue position of a trial under `cue`: its own, or cue 0's when hidden."""
        token = self.cue_token(cue)
        return token if self.show_cue else self.cue_token(FREQUENT_CUE)

    def item_token(self, item: int) -> int:
        """Token id of pool item `item`."""
        if not 0 <= item < self.pool_size:
            raise ValueError(f"item {item} is outside 0..{self.pool_size - 1}")
        return self.first_item + item

    def is_item(self, token: int) -> bool:
        """Whether `token` is an item token."""
        return self.first_item <= int(token) < self.size

    def name(self, token: int) -> str:
        """Printable name of a token id."""
        token = int(token)
        if 0 <= token < self.first_cue:
            return SPECIAL_TOKENS[token]
        if token < self.first_item:
            return f"<cue{token - self.first_cue}>"
        if token < self.size:
            return f"i{token - self.first_item}"
        raise ValueError(f"token {token} is outside 0..{self.size - 1}")


@dataclass(frozen=True)
class Encoded:
    """One encoded sequence with its scalar channel, loss masks, and indices.

    A finetuning sequence also says what kind of trial it is (`streams.RowKind`), the ranks of
    its queried items in the ordering the test battery queries them in (-1 on a replayed
    pretraining trial), and how many reasoning steps the network writes before its scored answer.
    """

    tokens: np.ndarray
    scalars: np.ndarray
    token_loss: np.ndarray
    scalar_loss: np.ndarray
    prompt_len: int
    answer_index: int
    n_emissions: int
    row_kind: int = 0
    query_ranks: tuple[int, int] = (-1, -1)
    reasoning_steps: int = 0


def encode(trial: Trial, chain: Chain | None, vocab: Vocabulary, *, target: Target = Target.CHAIN, pad_to: int = PAD_TO) -> Encoded:
    """Lay out one trial (and its chain, for chain targets) as tokens, scalars, and loss masks.

    `pad_to` applies to chain targets only: they carry the chain, the answer, and the answer
    repeated until `pad_to` emissions follow the context; a chain with no emissions lays out the
    answer `pad_to` times, which is how a pretraining trial shown without its chain is encoded. An
    answer-only target is the prompt plus one answer emission, and a prompt-only target stops at
    the context. Without a chain, the answer relation carries the trial's scoring reference, the
    one sign every scored quantity agrees with.
    """
    target = Target(target)
    if target is Target.CHAIN and chain is None:
        raise ValueError("a chain target carries a generated chain")

    item = vocab.item_token
    tokens: list[int] = [_BOS, vocab.shown_cue_token(trial.cue), _QMARK, item(trial.query[0]), item(trial.query[1])]
    tokens += [_QREL, _SEP]
    scalars: list[float] = [0.0] * len(tokens)

    def emit(relation: Relation) -> None:
        tokens.extend([item(relation.a), item(relation.b), _REL, _SEP])
        scalars.extend([0.0, 0.0, float(relation.sign), 0.0])

    for relation in trial.context:
        emit(relation)
    prompt_len = len(tokens)

    answer = chain.answer if chain is not None else Relation(*trial.query, trial.reference_sign)
    n_before_answer = len(chain.emissions) if target is Target.CHAIN else 0
    if target is Target.CHAIN:
        for relation in chain.emissions:
            emit(relation)
    if target is not Target.PROMPT_ONLY:
        emit(answer)
    if target is Target.CHAIN:
        n_padding = pad_to - n_before_answer - 1
        if n_padding < 0:
            raise ValueError(f"a chain of {n_before_answer + 1} emissions does not fit in {pad_to}")
        for _ in range(n_padding):
            emit(answer)

    token_ids = np.array(tokens, dtype=np.int64)
    token_loss = np.zeros(token_ids.size, dtype=bool)
    scalar_loss = np.zeros(token_ids.size, dtype=bool)
    if target is Target.CHAIN:
        # Everything written after the prompt is supervised, with the scalar at each relation.
        token_loss[prompt_len:] = True
        scalar_loss[prompt_len:] = token_ids[prompt_len:] == _REL
    elif target is Target.ANSWER_ONLY:
        relation_at = scored_answer_index(prompt_len, 0)
        token_loss[relation_at : relation_at + 2] = True
        scalar_loss[relation_at] = True
    return Encoded(
        tokens=token_ids,
        scalars=np.array(scalars, dtype=np.float32),
        token_loss=token_loss,
        scalar_loss=scalar_loss,
        prompt_len=prompt_len,
        answer_index=scored_answer_index(prompt_len, n_before_answer),
        n_emissions=(token_ids.size - prompt_len) // EMISSION_LEN,
    )


def decode(tokens: np.ndarray, scalars: np.ndarray, vocab: Vocabulary) -> str:
    """Render a sequence as text (`+1`/`-1` for the scalar at `<rel>`)."""
    token_ids = np.asarray(tokens).reshape(-1)
    values = np.asarray(scalars).reshape(-1)
    if token_ids.size != values.size:
        raise ValueError("the token and scalar channels have different lengths")
    parts = [
        f"{int(value):+d}" if int(token) == vocab.rel else vocab.name(int(token))
        for token, value in zip(token_ids, values, strict=True)
    ]
    return " ".join(parts)


def prompt_length(n_context: int) -> int:
    """Length of the prompt: the seven-token head plus four tokens per context relation."""
    return PROMPT_HEAD_LEN + n_context * EMISSION_LEN


def scored_answer_index(prompt_len: int, k: int) -> int:
    """Index of the answer `<rel>` slot after `k` emissions and the appended query items."""
    return prompt_len + EMISSION_LEN * k + 2


def hidden_state_index(prompt_len: int, k: int) -> int:
    """Index of the state that predicts the scored `<rel>` slot: the second query item."""
    return scored_answer_index(prompt_len, k) - 1


def scoring_length(n_context: int, k: int) -> int:
    """Positions a scoring pass runs over: the prompt, `k` emissions, and the two query items."""
    return hidden_state_index(prompt_length(n_context), k) + 1


def sequence_length(n_context: int, pad_to: int, target: Target) -> int:
    """Total length of an encoded sequence for a target type."""
    prompt = prompt_length(n_context)
    if target == Target.PROMPT_ONLY:
        return prompt
    if target == Target.ANSWER_ONLY:
        return prompt + EMISSION_LEN
    return prompt + pad_to * EMISSION_LEN


def item_positions(tokens: np.ndarray, vocab: Vocabulary) -> np.ndarray:
    """Indices of every item-token occurrence, including both copies of the query items."""
    tokens = np.asarray(tokens)
    return np.flatnonzero((tokens >= vocab.first_item) & (tokens < vocab.size))
