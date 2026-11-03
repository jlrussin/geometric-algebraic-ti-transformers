"""Reasoning before the scored answer while finetuning, and heads ablated while finetuning."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import design_for
from geo_alg_ti import components, config, model, streams, training
from geo_alg_ti.components import Head, HeadScores
from geo_alg_ti.orderings import OrderingLaw, uniform_cue_weights
from geo_alg_ti.sequences import EMISSION_LEN, Vocabulary, scored_answer_index
from geo_alg_ti.streams import RowKind, Stage
from geo_alg_ti.trials import sample_pretraining_trial

LAW = OrderingLaw(seed=0, pool_size=48, n_training_cues=6, n_reserved_cues=2)
VOCAB = Vocabulary(pool_size=LAW.pool_size, n_cues=LAW.n_training_cues + LAW.n_reserved_cues)
SPEC = model.ModelSpec(vocab_size=VOCAB.size, max_len=75, n_layers=2, d_model=32, d_ff=128)
DESIGN = design_for(LAW)
TASK = config.load_config(None, ("task.item_pool_size=48", "task.n_training_cues=6", "task.n_reserved_cues=2")).task
PROMPT_LEN = 39
STEPS = 2


def _network() -> model.RelationalTransformer:
    """A freshly initialized network whose relation direction is large enough to matter."""
    torch.manual_seed(0)
    net = model.RelationalTransformer(SPEC, VOCAB)
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
    return net


@pytest.fixture(scope="module")
def net():
    return _network()


@pytest.fixture(scope="module")
def ablation(net):
    rng = np.random.default_rng(0)
    trials = [sample_pretraining_trial(LAW, rng, weights=uniform_cue_weights(LAW.n_training_cues)) for _ in range(32)]
    means = components.reference_means(net, trials, VOCAB, budgets=range(STEPS + 1))
    scores = HeadScores(scores={}, n_steps=0, n_trials=0)
    heads = (Head(0, 1), Head(1, 3))
    rows = components.RowAblation(net, heads, means)
    return training.HeadAblation(heads=heads, means=means, scores=scores, rows=rows, during_training=True)


def _batch(*, replay: float = 0.0, uninformative: float = 0.0, reasoning: float = 1.0, steps: int = STEPS, n: int = 8):
    """A finetuning batch of `n` rows, each drawn at its own position of the stream."""
    rows = [
        streams.finetuning_example(
            TASK,
            LAW,
            VOCAB,
            design=DESIGN,
            replay_proportion=replay,
            uninformative_context_prop=uninformative,
            reasoning_proportion=reasoning,
            reasoning_steps=steps,
            rng=streams.example_rng(0, Stage.FINETUNING, 0, index),
        )
        for index in range(n)
    ]
    return streams.stack(rows, VOCAB).to_batch()


def _answer_prediction(net, batch, scope=None):
    """The prediction each row's answer is scored on, from one forward over the whole batch."""
    with torch.no_grad(), (scope if scope is not None else torch.enable_grad()):
        out = net(batch.tokens, batch.scalars)
    return out.scalar_pred.gather(1, batch.answer_index[:, None] - 1).flatten()


def _battery_read(net, batch, budget):
    """The prediction a scored rollout reads after `budget` steps, from each row's prompt."""
    query = batch.tokens[:, PROMPT_LEN : PROMPT_LEN + 2]
    generated = model.greedy_rollout(
        net, batch.tokens[:, :PROMPT_LEN], batch.scalars[:, :PROMPT_LEN], query, max_emissions=budget, budgets=(budget,)
    )
    return generated.budget_prediction[:, 0]


def test_the_reasoning_draw_comes_after_the_trial():
    """Marking rows to reason changes no token; only rows with a congruent context are marked."""
    marked = _batch(replay=0.3, uninformative=0.3, reasoning=1.0, n=32)
    plain = _batch(replay=0.3, uninformative=0.3, reasoning=0.0, n=32)
    assert torch.equal(marked.tokens, plain.tokens) and torch.equal(marked.scalars, plain.scalars)
    kinds = marked.row_kind
    assert {int(kind) for kind in kinds} == {int(kind) for kind in RowKind}
    congruent = kinds == int(RowKind.CONGRUENT)
    assert bool((marked.reasoning_steps[congruent] == STEPS).all())
    assert bool((marked.reasoning_steps[~congruent] == 0).all())
    assert bool((plain.reasoning_steps == 0).all())
    assert marked.reasoning.steps == STEPS and marked.reasoning.prompt_len == PROMPT_LEN
    assert torch.equal(marked.reasoning.rows, torch.nonzero(congruent).flatten())
    assert plain.reasoning.steps == 0 and plain.reasoning.rows.numel() == 0


def test_fixed_ordering_rows_name_adjacent_queryable_ranks():
    """A fixed-ordering row queries two adjacent ranks counted from one; a replayed row none."""
    batch = _batch(replay=0.3, n=32)
    fixed = batch.row_kind != int(RowKind.PRETRAINING)
    ranks = batch.query_ranks[fixed]
    assert bool(((ranks >= 1) & (ranks <= 7)).all())
    assert bool(((ranks[:, 0] - ranks[:, 1]).abs() == 1).all())
    assert bool((batch.query_ranks[~fixed] == -1).all())


def test_a_batch_without_reasoning_is_left_as_it_is(net):
    """No row marked to reason: the batch itself comes back."""
    batch = _batch(reasoning=0.0)
    assert training.with_reasoning(net, batch) is batch


def test_a_reasoning_row_is_scored_where_the_battery_reads_it(net):
    """The rewritten row's answer prediction equals a scored rollout's read after as many steps."""
    batch = _batch()
    rewritten = training.with_reasoning(net, batch)
    answer_at = scored_answer_index(PROMPT_LEN, STEPS)
    assert bool((rewritten.answer_index == answer_at).all())
    assert rewritten.tokens.shape[1] == PROMPT_LEN + EMISSION_LEN * (STEPS + 1)
    assert torch.equal(rewritten.token_loss.nonzero()[:, 1].unique(), torch.tensor([answer_at, answer_at + 1]))
    assert torch.equal(rewritten.scalar_loss.nonzero()[:, 1].unique(), torch.tensor([answer_at]))
    assert torch.equal(rewritten.scalars[:, answer_at], batch.scalars[:, PROMPT_LEN + 2])
    expected = _battery_read(net, batch, STEPS)
    torch.testing.assert_close(_answer_prediction(net, rewritten), expected, atol=1e-5, rtol=1e-5)


def test_ablated_rows_match_an_ablated_rollout(net, ablation):
    """With the heads ablated, a whole-sequence forward reads what an ablated rollout reads,
    after the reasoning steps and at once."""
    for reasoning in (1.0, 0.0):
        batch = _batch(reasoning=reasoning)
        rewritten = training.with_reasoning(net, batch, ablation=ablation)
        with components.ablated(net, ablation.heads, ablation.means):
            expected = _battery_read(net, batch, STEPS if reasoning else 0)
        rows = torch.ones(batch.tokens.shape[0], dtype=torch.bool)
        starts = rewritten.reasoning.query_starts
        assert starts == ((PROMPT_LEN, PROMPT_LEN + EMISSION_LEN * STEPS) if reasoning else (PROMPT_LEN,))
        scope = components.ablated_rows(
            net, ablation.rows, rows=rows, query_at=rewritten.answer_index - 2, starts=starts
        )
        torch.testing.assert_close(_answer_prediction(net, rewritten, scope), expected, atol=1e-5, rtol=1e-5)


def test_ablated_rows_leave_unmarked_rows_intact(net, ablation):
    """A row not marked keeps the intact network's outputs; a marked one does not."""
    batch = _batch(reasoning=0.0)
    rows = torch.zeros(batch.tokens.shape[0], dtype=torch.bool)
    rows[0] = True
    intact = _answer_prediction(net, batch)
    scope = components.ablated_rows(
        net, ablation.rows, rows=rows, query_at=batch.answer_index - 2, starts=batch.reasoning.query_starts
    )
    ablated = _answer_prediction(net, batch, scope)
    torch.testing.assert_close(ablated[1:], intact[1:])
    assert not torch.isclose(ablated[0], intact[0])


def test_ablated_rows_refuse_a_cached_forward(net, ablation):
    """Rows are ablated over whole sequences; a forward over a cache is refused."""
    batch = _batch(reasoning=0.0)
    rows = torch.ones(batch.tokens.shape[0], dtype=torch.bool)
    with torch.no_grad():
        prefix = net(batch.tokens[:, :PROMPT_LEN], batch.scalars[:, :PROMPT_LEN], return_cache=True)
        with components.ablated_rows(
            net, ablation.rows, rows=rows, query_at=batch.answer_index - 2, starts=(PROMPT_LEN,)
        ):
            with pytest.raises(ValueError, match="whole sequences"):
                net(batch.tokens[:, PROMPT_LEN:], batch.scalars[:, PROMPT_LEN:], cache=prefix.cache)


def test_reasoning_rows_and_replayed_rows_share_one_batch(net):
    """Rows of every kind stack: replayed chains keep their targets, the rest are padded."""
    batch = _batch(replay=0.4, uninformative=0.2, n=24)
    rewritten = training.with_reasoning(net, batch)
    replayed = batch.row_kind == int(RowKind.PRETRAINING)
    assert bool(replayed.any())
    width = batch.tokens.shape[1]
    assert torch.equal(rewritten.tokens[replayed, :width], batch.tokens[replayed])
    assert torch.equal(rewritten.token_loss[replayed, :width], batch.token_loss[replayed])
    answer_at = scored_answer_index(PROMPT_LEN, STEPS)
    reasoning = batch.reasoning_steps > 0
    assert bool((rewritten.tokens[reasoning, answer_at + 2 :] == VOCAB.pad).all())
    uninformative = batch.row_kind == int(RowKind.UNINFORMATIVE)
    assert torch.equal(rewritten.answer_index[uninformative], batch.answer_index[uninformative])


def test_reasoning_steps_must_fit_the_longest_sequence():
    """Reasoning longer than the test sweep, or leaving no room for the answer, is refused, unless
    no trial reasons."""
    with pytest.raises(ValueError, match="reasoning_steps"):
        config.load_config(None, ("finetuning.reasoning_prop=0.5", "evaluation.max_reasoning_steps=2"))
    with pytest.raises(ValueError, match="no room for the answer"):
        config.load_config(
            None, ("finetuning.reasoning_prop=0.5", "finetuning.reasoning_steps=9", "evaluation.max_reasoning_steps=9")
        )
    config.load_config(None, ("evaluation.max_reasoning_steps=2",))
    config.load_config(None, ("finetuning.reasoning_prop=0.5", "evaluation.max_reasoning_steps=9"))


def test_rows_that_reason_write_one_number_of_steps():
    """A batch whose rows would write different numbers of steps is refused when stacked."""
    rows = [
        streams.finetuning_example(
            TASK, LAW, VOCAB, design=DESIGN, replay_proportion=0.0, uninformative_context_prop=0.0,
            reasoning_proportion=1.0, reasoning_steps=steps, rng=streams.example_rng(0, Stage.FINETUNING, 0, index),
        )
        for index, steps in enumerate((2, 3))
    ]
    with pytest.raises(ValueError, match="different numbers of steps"):
        streams.stack(rows, VOCAB)


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory, repo_root):
    """A tiny pretrained network that writes enough well-formed steps to rank heads on."""
    out = tmp_path_factory.mktemp("pretrained")
    cfg = config.load_config(
        repo_root / "configs" / "smoke.yaml", ("pretraining.steps=400", "pretraining.eval_every=400", "logging.log_every=400")
    )
    summary = training.run_stage(cfg, Stage.PRETRAINING, seed=0, init_from=None, out_dir=out, device="cpu")
    return summary.checkpoint


@pytest.mark.parametrize("heads", ["context_reading", "random"])
def test_finetuning_writes_the_ablated_scores_and_the_answers_by_pair(tmp_path, repo_root, pretrained, heads):
    """A run that reasons and ablates writes its head scores, its ablated scores at every
    evaluation, and the answers of its fixed-ordering trials by pair; a random set lies outside
    the top four."""
    cfg = config.load_config(
        repo_root / "configs" / "smoke.yaml",
        (
            "finetuning.steps=5",
            "finetuning.eval_every=2",
            "finetuning.reasoning_prop=1.0",
            "finetuning.reasoning_steps=2",
            f"finetuning.ablated_heads={heads}",
            "evaluation.ablated_scoring=true",
            "logging.log_every=2",
        ),
    )
    out = training.run_stage(cfg, Stage.FINETUNING, seed=0, init_from=pretrained, out_dir=tmp_path / heads, device="cpu").out_dir
    for name in ("ablated_eval", "ablated_eval_pairs", "eval_pairs"):
        assert {path.name for path in (out / name).glob("step_*.parquet")} == {
            f"step_{step:08d}.parquet" for step in (0, 2, 4, 5)
        }
    log = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    label = next(entry["heads"] for entry in log if entry["event"] == "ablated_heads")
    assert [entry["step"] for entry in log if entry["event"] == "ablated_evaluation"] == [0, 2, 4, 5]
    ablated = pd.read_parquet(out / "ablated_eval" / "step_00000005.parquet")
    assert set(ablated["heads"]) == {label}
    ranks = pd.read_parquet(out / "head_scores" / "step_00000000.parquet").set_index("head")["attention_rank"]
    chosen = [ranks[one] for one in label.split("+")]
    assert sorted(chosen) == [1, 2] if heads == "context_reading" else min(chosen) > 4
    answers = pd.read_parquet(out / "train_pairs.parquet")
    assert set(answers["context"]) <= {"congruent", "uninformative"}
    assert set(answers.loc[answers["context"] == "congruent", "reasoning_steps"]) == {2}
    assert set(answers.loc[answers["context"] == "uninformative", "reasoning_steps"]) <= {0}
    # Two updates to a window, the last one closed by the final update.
    assert set(answers["step"]) == {2, 4, 5}
    assert (answers["n"] > 0).all() and answers["value"].between(0, 1).all()


def test_heads_are_ranked_on_the_pretraining_preset(repo_root, pretrained):
    """The ablation ranks heads on the pretraining stage's preset even when the run is scored on
    another one."""
    cfg = config.load_config(
        repo_root / "configs" / "smoke.yaml",
        ("finetuning.ablated_heads=context_reading", "evaluation.battery=finetuned", "evaluation.ablated_scoring=true"),
    )
    vocab = training.build_vocabulary(cfg)
    net = training.build_model(cfg, vocab)
    net.load_state_dict(training.load_checkpoint(pretrained).model_state)
    law = training.build_law(cfg, 0)
    ablation = training.build_head_ablation(cfg, net, law, vocab, seed=0, budgets=cfg.evaluation.reasoning_steps())
    assert ablation is not None and len(ablation.heads) == 2 and ablation.heads == ablation.scores.ranked[:2]


@pytest.mark.parametrize(("heads", "size"), [("random", 6), ("context_reading", 17), ("none", 17)])
def test_a_set_larger_than_the_network_allows_is_refused(repo_root, heads, size):
    """A random set needs three times its size in heads, a top set (also the one scored ablated)
    its size; the configuration refuses either before any run starts."""
    overrides = (f"finetuning.ablated_heads={heads}", f"finetuning.n_ablated_heads={size}", "evaluation.ablated_scoring=true")
    with pytest.raises(ValueError, match="the network has 16"):
        config.load_config(repo_root / "configs" / "smoke.yaml", overrides)
    config.load_config(repo_root / "configs" / "smoke.yaml", overrides[:1])


def test_answers_by_pair_are_recorded_on_one_fixed_ordering(repo_root):
    """Reasoning, ablated heads, or ablated scoring with several fixed orderings is refused."""
    settings = (
        ("finetuning.reasoning_prop=0.5", "finetuning.reasoning_steps=2"),
        ("finetuning.ablated_heads=random",),
        ("evaluation.ablated_scoring=true",),
    )
    for setting in settings:
        with pytest.raises(ValueError, match="one fixed ordering"):
            config.load_config(repo_root / "configs" / "smoke.yaml", (*setting, "finetuning.n_orderings=2"))
    config.load_config(repo_root / "configs" / "smoke.yaml", ("finetuning.n_orderings=2",))


def test_a_batch_of_replayed_trials_alone_takes_no_ablation(net, ablation):
    """With no fixed-ordering row there is no query position to place, and the update runs."""
    batch = _batch(replay=1.0)
    assert batch.reasoning.query_starts == () and bool((batch.row_kind == int(RowKind.PRETRAINING)).all())
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
    out, _ = training.train_step(net, optimizer, batch, ablation=ablation)
    with torch.no_grad():
        intact = net(batch.tokens, batch.scalars)
    torch.testing.assert_close(out.scalar_pred, intact.scalar_pred)


def test_a_run_that_neither_reasons_nor_ablates_writes_pairs_at_its_end(tmp_path, repo_root, pretrained):
    """No reasoning, no ablated heads, no ablated scoring: the per-pair table at the final
    evaluation only, and no answers by pair."""
    cfg = config.load_config(repo_root / "configs" / "smoke.yaml", ("finetuning.steps=2", "finetuning.eval_every=1"))
    out = training.run_stage(cfg, Stage.FINETUNING, seed=0, init_from=pretrained, out_dir=tmp_path, device="cpu").out_dir
    assert {path.name for path in (out / "eval_pairs").glob("step_*.parquet")} == {"step_00000002.parquet"}
    assert not (out / "train_pairs.parquet").exists() and not (out / "ablated_eval").exists()


def test_the_uninformative_proportion_is_a_setting(repo_root):
    """The base configuration keeps one fixed-ordering trial in ten uninformative; at zero, every
    fixed-ordering trial of the stream shows the ordering."""
    cfg = config.load_config(None, ())
    assert cfg.finetuning.uninformative_prop == streams.UNINFORMATIVE_CONTEXT_PROP
    design = design_for(LAW)
    stream = streams.finetuning_stream(
        TASK, LAW, VOCAB, design=design, seed=0, batch_size=64, replay_proportion=0.1, uninformative_proportion=0.0
    )
    try:
        kinds = torch.cat([next(stream).row_kind for _ in range(8)])
    finally:
        stream.close()
    assert not bool((kinds == int(RowKind.UNINFORMATIVE)).any()) and bool((kinds == int(RowKind.CONGRUENT)).any())
