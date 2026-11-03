"""Head ablation, heads ranked by their attention to the context, and the ablate stage."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from geo_alg_ti import battery, cli, components, config, evaluate, model, tables, training
from geo_alg_ti.chains import PAD_TO, generate_chain
from geo_alg_ti.components import AblationSet, Forward, Head, HeadScores, SetKind
from geo_alg_ti.orderings import OrderingLaw, uniform_cue_weights
from geo_alg_ti.sequences import EMISSION_LEN, PROMPT_HEAD_LEN, Target, Vocabulary, encode
from geo_alg_ti.streams import Stage
from geo_alg_ti.trials import Relation, sample_pretraining_trial

LAW = OrderingLaw(seed=0, pool_size=48, n_training_cues=6, n_reserved_cues=2)
VOCAB = Vocabulary(pool_size=LAW.pool_size, n_cues=LAW.n_training_cues + LAW.n_reserved_cues)
SPEC = model.ModelSpec(vocab_size=VOCAB.size, max_len=75, n_layers=2, d_model=32, d_ff=128)
WEIGHTS = uniform_cue_weights(LAW.n_training_cues)
PROMPT_LEN = 39
BUDGETS = (0, 1, 2)


def _network() -> model.RelationalTransformer:
    """A freshly initialized network whose relation direction is large enough to matter."""
    torch.manual_seed(0)
    net = model.RelationalTransformer(SPEC, VOCAB)
    with torch.no_grad():
        net.relation_direction.normal_(mean=0.0, std=0.5)
    return net


def _trials(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [sample_pretraining_trial(LAW, rng, weights=WEIGHTS) for _ in range(n)]


def _prompts(trials):
    return [encode(trial, None, VOCAB, target=Target.PROMPT_ONLY) for trial in trials]


def _content_battery() -> battery.Battery:
    conditions = battery.load_preset("content", contexts_per_pair=1)
    return battery.Battery.build(conditions, LAW, VOCAB, seed=0, design=None, budgets=BUDGETS, cue_weights=WEIGHTS)


@pytest.fixture(scope="module")
def net():
    return _network()


@pytest.fixture(scope="module")
def means(net):
    return components.reference_means(net, _trials(32), VOCAB, budgets=BUDGETS)


def test_passes_are_located_by_kind_and_offset(net):
    """A rollout's passes are the prompt, its emissions, and one query pass per budget, in order."""
    seen = []

    def log(layer, stream, offset, x):
        if layer == 0:
            seen.append((str(stream), offset, x.shape[1]))

    with components._hooked(net, head_outputs=log):
        model.greedy_chain(net, _prompts(_trials(2)), max_emissions=2, budgets=BUDGETS)
        net(torch.zeros(1, 75, dtype=torch.long), torch.zeros(1, 75))
    expected = [("sequence", 0, PROMPT_LEN)]
    for k in BUDGETS:
        expected.append(("query", PROMPT_LEN + EMISSION_LEN * k, 2))
        if k < max(BUDGETS):
            expected += [("sequence", PROMPT_LEN + EMISSION_LEN * k + t, 1) for t in range(EMISSION_LEN)]
    assert seen == [*expected, ("sequence", 0, 75)]


def test_hooks_are_removed_on_exit(net):
    """No hook outlives its context, even one left by an exception."""
    with pytest.raises(RuntimeError):
        with components._hooked(net, head_outputs=lambda *args: None):
            raise RuntimeError
    assert not net._forward_pre_hooks
    assert all(not block.attention.out._forward_pre_hooks for block in net.blocks)


def test_ablating_nothing_leaves_the_battery_unchanged(net, means):
    """No ablated head scores every trial record exactly as the plain network does."""
    tests = _content_battery()
    clean, _ = evaluate.evaluate_battery(net, tests, VOCAB, step=0, tag="t")
    with components.ablated(net, (), means):
        hooked, _ = evaluate.evaluate_battery(net, tests, VOCAB, step=0, tag="t")
    pd.testing.assert_frame_equal(clean, hooked, check_exact=True)


def test_ablating_with_a_trials_own_outputs_changes_nothing(net):
    """Means over one rollout, replacing every head, reproduce that rollout exactly."""
    # The wrong kind of pass or position would replace an output with another place's.
    trial = _trials(1, seed=3)[0]
    own = components.reference_means(net, [trial], VOCAB, budgets=BUDGETS)
    prompts = _prompts([trial])
    clean = model.greedy_chain(net, prompts, max_emissions=max(BUDGETS), budgets=BUDGETS)
    with components.ablated(net, components.heads(net), own):
        replaced = model.greedy_chain(net, prompts, max_emissions=max(BUDGETS), budgets=BUDGETS)
    assert torch.equal(clean.tokens, replaced.tokens) and torch.equal(clean.scalars, replaced.scalars)
    assert torch.equal(clean.budget_prediction, replaced.budget_prediction)


def test_reference_rollouts_reach_a_budget_beyond_the_padded_chain():
    """A budget past the padded chain length still gets the means of its query pass."""
    budget = PAD_TO + 1
    spec = dataclasses.replace(SPEC, max_len=PROMPT_LEN + EMISSION_LEN * budget + 2)
    torch.manual_seed(0)
    longer = model.RelationalTransformer(spec, VOCAB)
    means = components.reference_means(longer, _trials(2), VOCAB, budgets=(0, budget))
    offset = PROMPT_LEN + EMISSION_LEN * budget
    assert means.read(Forward.QUERY, 0, offset, 2).shape == (2, SPEC.d_model)
    assert means.reached[Forward.SEQUENCE][: offset].all()


def test_ablating_a_head_replaces_only_its_columns(net, means):
    """At its own layer, a head's ablation rewrites its slice of the projection input only."""
    head = Head(layer=1, head=5)
    width = SPEC.d_model // SPEC.n_heads
    columns = slice(head.head * width, (head.head + 1) * width)
    trial = _trials(1)[0]
    encoded = encode(trial, generate_chain(trial), VOCAB)
    sequence = torch.as_tensor(encoded.tokens)[None]
    scalars = torch.as_tensor(encoded.scalars, dtype=torch.float32)[None]
    seen = {}

    def observe(_module, args):
        seen.setdefault("inputs", []).append(args[0].detach().clone())

    with torch.no_grad():
        handle = net.blocks[1].attention.out.register_forward_pre_hook(observe)
        net(sequence, scalars)
        handle.remove()
        with components.ablated(net, (head,), means):
            handle = net.blocks[1].attention.out.register_forward_pre_hook(observe)
            net(sequence, scalars)
            handle.remove()
    clean, ablated = seen["inputs"]
    keep = torch.ones(SPEC.d_model, dtype=torch.bool)
    keep[columns] = False
    assert torch.equal(clean[..., keep], ablated[..., keep])
    assert torch.equal(ablated[0, :, columns], means.head_outputs[Forward.SEQUENCE][:75, 1, columns])


def test_means_cover_the_sequence_and_the_budget_reads_only(means):
    """Sequence means exist at every position; query means exactly where a budget is read."""
    sequence = means.head_outputs[Forward.SEQUENCE]
    assert not torch.isnan(sequence).any() and sequence.shape == (75, SPEC.n_layers, SPEC.d_model)
    query = means.head_outputs[Forward.QUERY]
    read = {PROMPT_LEN + EMISSION_LEN * k + t for k in BUDGETS for t in (0, 1)}
    for position in range(75):
        assert bool(torch.isnan(query[position]).all()) is (position not in read)
    with pytest.raises(ValueError, match="no reference rollout"):
        means.read(Forward.QUERY, 0, PROMPT_LEN + 2, 2)


def test_heads_are_labeled_from_one_layer_by_layer(net):
    """Labels number layers and heads from one; every head appears once, layer by layer."""
    assert Head(3, 2).label == "L4H3"
    found = components.heads(net)
    assert len(found) == SPEC.n_layers * SPEC.n_heads == len(set(found))
    assert [one.label for one in found[:2]] == ["L1H1", "L1H2"]


def test_chain_conditions_name_conditions_of_their_preset():
    """The conditions heads are ranked on exist in their preset; another preset raises."""
    for preset, names in components.CHAIN_CONDITIONS.items():
        assert set(names) <= {condition.name for condition in battery.load_preset(preset, contexts_per_pair=1)}
    with pytest.raises(ValueError, match="no conditions"):
        components.chain_conditions("finetuned")


def _attention_inputs(net, run):
    """Each layer's normalized attention input and output-projection input during `run()`."""
    seen = {"x": {}, "heads": {}}
    handles = []
    for layer, block in enumerate(net.blocks):
        handles.append(block.attention.register_forward_pre_hook(lambda _m, args, layer=layer: seen["x"].setdefault(layer, args[0].detach().clone())))
        handles.append(block.attention.out.register_forward_pre_hook(lambda _m, args, layer=layer: seen["heads"].setdefault(layer, args[0].detach().clone())))
    try:
        run()
    finally:
        for handle in handles:
            handle.remove()
    return seen


@pytest.mark.parametrize("cached", [False, True], ids=["whole", "cached"])
def test_recorded_attention_reproduces_each_heads_output(net, cached):
    """The recomputed probabilities times each head's values give that head's output, with or
    without a key-value cache, and recording them leaves the network's outputs unchanged."""
    trial = _trials(1, seed=5)[0]
    encoded = encode(trial, generate_chain(trial), VOCAB)
    tokens = torch.as_tensor(encoded.tokens)[None]
    scalars = torch.as_tensor(encoded.scalars, dtype=torch.float32)[None]
    split = PROMPT_LEN + 3
    with torch.no_grad():
        cache = net(tokens[:, :split], scalars[:, :split], return_cache=True).cache if cached else None
        new_tokens, new_scalars = (tokens[:, split:], scalars[:, split:]) if cached else (tokens, scalars)
        plain = net(new_tokens, new_scalars, cache=cache)
        with components.recorded_attention(net) as attention:
            seen = _attention_inputs(net, lambda: net(new_tokens, new_scalars, cache=cache))
            recorded = net(new_tokens, new_scalars, cache=cache)
    assert torch.equal(plain.logits, recorded.logits) and torch.equal(plain.scalar_pred, recorded.scalar_pred)
    offset = split if cached else 0
    for layer, block in enumerate(net.blocks):
        probabilities = attention[layer]
        assert probabilities.shape == (1, SPEC.n_heads, new_tokens.shape[1], offset + new_tokens.shape[1])
        assert torch.allclose(probabilities.sum(-1), torch.ones(()), atol=1e-6)
        assert not probabilities[0, :, 0, offset + 1 :].any()
        values = block.attention.split_heads(block.attention.value(seen["x"][layer]))
        if cached:
            values = torch.cat((cache.layers[layer][1], values), dim=-2)
        heads = (probabilities @ values).transpose(1, 2).reshape(1, new_tokens.shape[1], SPEC.d_model)
        assert torch.allclose(heads, seen["heads"][layer], atol=1e-6)
    assert all(not block.attention._forward_pre_hooks for block in net.blocks)


def test_attention_contracted_in_its_hook_gives_the_same_reading_mass(net):
    """Contracting each layer's attention as it is recorded equals contracting the kept maps."""
    trial = _trials(1, seed=5)[0]
    chain = generate_chain(trial)
    encoded = encode(trial, chain, VOCAB)
    tokens = torch.as_tensor(encoded.tokens)[None]
    scalars = torch.as_tensor(encoded.scalars, dtype=torch.float32)[None]
    read = components.reading_positions(trial.context, list(chain.emissions), prompt_len=PROMPT_LEN)
    weights = components.reading_weights([read], tokens.shape[1])
    with torch.inference_mode():
        with components.recorded_attention(net) as attention:
            net(tokens, scalars)
        whole = torch.stack([components.layer_reading_mass(attention[layer], weights) for layer in sorted(attention)]).double()
        contract = functools.partial(components.layer_reading_mass, weights=weights)
        with components.recorded_attention(net, reduce=contract) as mass:
            net(tokens, scalars)
    assert read and torch.equal(torch.stack([mass[layer] for layer in sorted(mass)]).double(), whole)


def test_a_step_reads_the_context_relations_mentioning_its_items():
    """Step e sits at 39 + 4e..42 + 4e and relation j at 7 + 4j..10 + 4j with eight relations; a
    step reads every relation naming either of its items, a malformed or unanchored step none."""
    context = [Relation(10 + j, 11 + j, 1) for j in range(8)]
    steps = [Relation(12, 13, 1), None, Relation(30, 31, 1), Relation(17, 16, -1)]
    read = components.reading_positions(context, steps, prompt_len=PROMPT_LEN)
    assert PROMPT_LEN == PROMPT_HEAD_LEN + len(context) * EMISSION_LEN == 39
    assert [list(queries) for queries, _ in read] == [[39, 40, 41, 42], [51, 52, 53, 54]]
    # Items 12 and 13 sit in relations 1, 2, 3; items 16 and 17 in relations 5, 6, 7.
    assert [sorted(keys) for _, keys in read] == [list(range(11, 23)), list(range(27, 39))]


def test_planted_attention_ranks_the_heads_that_read_the_context_first():
    """On a hand-built sequence, the head that puts all its attention on the relations a step reads
    ranks first, one that puts half there second, and the rest tie in layer order."""
    context = [Relation(10 + j, 11 + j, 1) for j in range(8)]
    steps = [Relation(12, 13, 1), Relation(17, 16, -1)]
    read = components.reading_positions(context, steps, prompt_len=PROMPT_LEN)
    length = PROMPT_LEN + len(steps) * EMISSION_LEN
    weights = components.reading_weights([read], length)
    attention = {layer: torch.zeros(1, SPEC.n_heads, length, length) for layer in range(SPEC.n_layers)}
    for layer in attention:
        attention[layer][..., 0] = 1.0  # every other head attends to the first position
    for (queries, keys) in read:
        for planted, (layer, head) in ((1.0, (1, 3)), (0.5, (0, 6))):
            attention[layer][0, head, queries] = 0.0
            attention[layer][0, head, queries[:, None], keys[None, :]] = planted / keys.size
            attention[layer][0, head, queries, 0] = 1.0 - planted
    totals = torch.stack([components.layer_reading_mass(attention[layer], weights) for layer in sorted(attention)])
    every = [Head(layer, head) for layer in range(SPEC.n_layers) for head in range(SPEC.n_heads)]
    scores = HeadScores(scores={one: float(totals[one.layer, one.head]) / len(read) for one in every}, n_steps=len(read), n_trials=1)
    assert scores.scores[Head(1, 3)] == pytest.approx(1.0) and scores.scores[Head(0, 6)] == pytest.approx(0.5)
    assert scores.ranked[:3] == (Head(1, 3), Head(0, 6), Head(0, 0))
    assert scores.ranks[Head(1, 3)] == 1 and sorted(scores.ranks.values()) == list(range(1, len(every) + 1))


def _ranked(n_layers: int = 4, n_heads: int = 8) -> tuple[Head, ...]:
    """Every head of a network in an arbitrary but fixed order."""
    every = [Head(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    order = np.random.default_rng(7).choice(len(every), size=len(every), replace=False)
    return tuple(every[index] for index in order)


def test_ablation_sets_are_deterministic_and_keep_the_controls_apart():
    """Intact, every head alone, the top k, and the draws of k from outside the top 2k, keyed by
    the seed alone; no set repeats a head and no draw shares one with the top 2k."""
    ranked = _ranked()
    sets = components.ablation_sets(ranked, seed=3)
    assert sets == components.ablation_sets(ranked, seed=3)
    assert sets != components.ablation_sets(ranked, seed=4)
    kinds = [one.kind for one in sets]
    sizes, draws = components.SET_SIZES, components.N_RANDOM_DRAWS
    assert kinds == [SetKind.INTACT, *[SetKind.HEAD] * len(ranked), *[SetKind.TOP] * len(sizes), *[SetKind.RANDOM] * (len(sizes) * draws)]
    assert sets[0] == AblationSet(SetKind.INTACT)
    assert [one.heads[0] for one in sets if one.kind is SetKind.HEAD] == sorted(ranked, key=lambda one: (one.layer, one.head))
    assert [one.heads for one in sets if one.kind is SetKind.TOP] == [ranked[:size] for size in sizes]
    rank = {one: place for place, one in enumerate(ranked, start=1)}
    for size in sizes:
        drawn = [one for one in sets if one.kind is SetKind.RANDOM and len(one.heads) == size]
        assert [one.draw for one in drawn] == list(range(draws))
        for one in drawn:
            assert len(set(one.heads)) == size and all(rank[head] > 2 * size for head in one.heads)
            assert not set(one.heads) & set(ranked[:size])
            assert [rank[head] for head in one.heads] == sorted(rank[head] for head in one.heads)
        assert len({one.heads for one in drawn}) > 1
    assert all(one.draw == -1 for one in sets if one.kind is not SetKind.RANDOM)


def test_too_few_heads_outside_the_top_raise():
    """A network too small to draw a random set of every size outside the top 2k is refused; one
    with exactly three times the largest size is not."""
    enough = 3 * max(components.SET_SIZES)
    components.check_set_sizes(enough)
    assert len(components.ablation_sets(_ranked(n_layers=1, n_heads=enough), seed=0)) > 0
    with pytest.raises(ValueError, match="outside the top"):
        components.check_set_sizes(enough - 1)
    with pytest.raises(ValueError, match="outside the top"):
        components.ablation_sets(_ranked(n_layers=1, n_heads=enough - 1), seed=0)


def _ablate_argv(repo_root, out, run, experiment, overrides):
    argv = ["--config", str(repo_root / "configs" / "smoke.yaml")]
    for override in overrides:
        argv += ["--set", override]
    return ["ablate", *argv, "--experiment", experiment.name, "--index", str(run.index), "--out", str(out), "--device", "cpu"]


@contextlib.contextmanager
def _quick() -> Iterator[None]:
    """Fewer reference rollouts and random draws, so the stage runs in seconds."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(components, "N_REFERENCE_TRIALS", 64)
        patch.setattr(components, "N_RANDOM_DRAWS", 3)
        yield


@pytest.fixture(scope="module")
def staged(tmp_path_factory, repo_root):
    """A tiny content checkpoint ablated through the subcommand, then exported.

    The network is pretrained long enough to write well-formed reasoning steps to rank heads on.
    """
    out = tmp_path_factory.mktemp("components")
    overrides = ("pretraining.steps=400", "pretraining.eval_every=400", "logging.log_every=400")
    base = config.load_config(repo_root / "configs" / "smoke.yaml", overrides)
    experiment = config.load_experiment("components")
    run = config.enumerate_runs(experiment, base)[0]
    upstream = next(
        one
        for one in config.enumerate_runs(config.load_experiment(run.depends_on_experiment), base)
        if one.run_id == run.depends_on_run_id
    )
    training.run_stage(
        upstream.config,
        Stage.PRETRAINING,
        seed=0,
        init_from=None,
        out_dir=config.run_directory(out, upstream.experiment, upstream.run_id),
        device="cpu",
    )
    argv = _ablate_argv(repo_root, out, run, experiment, overrides)
    with _quick():
        assert cli.main(argv) == 0
    written = tables.export_experiment(experiment.name, results_root=out, base=base)
    return out, run, argv, written


def test_the_ablate_stage_writes_a_complete_table(staged):
    """Every set at every condition, distance, and budget, labeled by its kind, heads, and draw."""
    _, run, _, written = staged
    assert set(written) == {"ablate", "head_scores"}
    ablate = tables.load_table(written["ablate"], schema=tables.ABLATE_SCHEMA)
    n_heads = run.config.model.n_layers * model.N_HEADS
    sizes, draws = components.SET_SIZES, 3
    per_set = ablate.groupby(["set_kind", "heads", "draw"]).size()
    assert per_set.nunique() == 1
    counts = per_set.groupby(level="set_kind").size().to_dict()
    assert counts["intact"] == 1 and counts["head"] == n_heads and counts["top"] == len(sizes)
    assert counts["random"] == len(ablate.loc[ablate["set_kind"] == "random", ["heads", "draw"]].drop_duplicates())
    assert ablate.loc[ablate["set_kind"] == "random", ["n_heads", "draw"]].drop_duplicates().shape[0] == len(sizes) * draws
    single = ablate[ablate["set_kind"] == "head"]
    assert (single["head"] == single["heads"]).all() and (single["n_heads"] == 1).all()
    assert set(single["layer"]) == set(range(1, run.config.model.n_layers + 1))
    assert sorted(single["attention_rank"].unique()) == list(range(1, n_heads + 1))
    others = ablate[ablate["set_kind"] != "head"]
    assert others["layer"].isna().all() and (others["head"] == "").all() and (others["attention_rank"] == -1).all()
    intact = ablate[ablate["set_kind"] == "intact"]
    assert (intact["n_heads"] == 0).all() and (intact["heads"] == "").all()
    assert ((ablate["draw"] >= 0) == (ablate["set_kind"] == "random")).all()
    assert set(ablate["K"]) == set(run.config.evaluation.reasoning_steps())


def test_the_sets_follow_the_head_scores(staged):
    """Each head's rank is its place by score; the top sets are the highest-ranked heads, every
    random set lies outside the top 2k, and a set scored twice scores alike."""
    _, run, _, written = staged
    scores = tables.load_table(written["head_scores"], schema=tables.HEAD_SCORES_SCHEMA)
    ablate = tables.load_table(written["ablate"], schema=tables.ABLATE_SCHEMA)
    n_heads = run.config.model.n_layers * model.N_HEADS
    assert len(scores) == n_heads and sorted(scores["attention_rank"]) == list(range(1, n_heads + 1))
    by_rank = scores.sort_values("attention_rank")
    assert (by_rank["score"].diff().dropna() <= 0).all() and scores["score"].between(0.0, 1.0).all()
    assert (scores["layer"] == scores["head"].str[1].astype(int)).all()
    assert scores["n_steps"].nunique() == 1 and int(scores["n_steps"].iloc[0]) > 0
    tests = cli._scored_network(config.load_experiment("components"), run, out=staged[0], device="cpu").tests
    chained = [name for name, _ in tests.trials if name in components.chain_conditions("content")]
    assert (scores["n_trials"] == len(chained)).all()
    rank = dict(zip(scores["head"], scores["attention_rank"], strict=True))
    order = list(by_rank["head"])
    single = ablate[ablate["set_kind"] == "head"]
    assert all(rank[head] == place for head, place in zip(single["head"], single["attention_rank"], strict=True))
    for size in components.SET_SIZES:
        top = ablate.loc[(ablate["set_kind"] == "top") & (ablate["n_heads"] == size), "heads"].unique()
        assert list(top) == ["+".join(order[:size])]
        drawn = ablate.loc[(ablate["set_kind"] == "random") & (ablate["n_heads"] == size), "heads"].unique()
        assert all(rank[head] > 2 * size for heads in drawn for head in heads.split("+"))
    keys = ["condition", "metric", "d", "K"]
    first = ablate[(ablate["set_kind"] == "top") & (ablate["n_heads"] == 1)].set_index(keys)["value"]
    alone = single[single["head"] == order[0]].set_index(keys)["value"]
    assert np.array_equal(first.sort_index().to_numpy(), alone.sort_index().to_numpy(), equal_nan=True)


def test_intact_rows_equal_the_plain_evaluation(staged):
    """The intact rows are the test battery's aggregates without any hook."""
    out, run, _, written = staged
    ablate = tables.load_table(written["ablate"], schema=tables.ABLATE_SCHEMA)
    network = cli._scored_network(config.load_experiment("components"), run, out=out, device="cpu")
    _, plain = evaluate.evaluate_battery(network.model, network.tests, network.vocab, step=0, tag="t")
    intact = ablate[ablate["set_kind"] == "intact"]
    merged = intact.merge(plain, on=["condition", "metric", "d", "K"], suffixes=("", "_plain"))
    assert len(merged) == len(plain) == len(intact)
    assert np.array_equal(merged["value"].to_numpy(), merged["value_plain"].to_numpy(), equal_nan=True)
    assert (merged["n"] == merged["n_plain"]).all()


def test_the_ablate_tables_are_in_the_manifest_and_deterministic(staged):
    """The manifest lists both tables; a rerun writes byte-identical run tables."""
    out, run, argv, _ = staged
    manifest = yaml.safe_load((config.tables_directory(out, run.experiment) / "MANIFEST.yaml").read_text())
    assert {entry["table"] for entry in manifest["tables"]} == {"ablate", "head_scores"}
    run_dir = config.run_directory(out, run.experiment, run.run_id)
    before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("step_*.parquet")}
    with _quick():
        assert cli.main(argv) == 0
    after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("step_*.parquet")}
    assert before and before == after


def test_the_ablate_stage_refuses_a_small_network_before_any_work(tmp_path, repo_root, capsys):
    """A network with too few heads for every set is refused from its configuration: nothing is
    loaded, no head is scored, no reference mean taken, and no run directory made."""
    overrides = ("model.n_layers=1",)
    base = config.load_config(repo_root / "configs" / "smoke.yaml", overrides)
    experiment = config.load_experiment("components")
    run = config.enumerate_runs(experiment, base)[0]
    upstream = next(
        one
        for one in config.enumerate_runs(config.load_experiment(run.depends_on_experiment), base)
        if one.run_id == run.depends_on_run_id
    )
    cfg = upstream.config
    net = training.build_model(cfg, training.build_vocabulary(cfg))
    assert len(components.heads(net)) < 3 * max(components.SET_SIZES)
    training.save_checkpoint(
        config.checkpoint_path(config.run_directory(tmp_path, upstream.experiment, upstream.run_id), training.CHECKPOINT_NAME),
        model=net,
        optimizer=training.build_optimizer(net, cfg, Stage.PRETRAINING),
        cfg=cfg,
        step=cfg.pretraining.steps,
        stage=Stage.PRETRAINING,
    )

    def refused(*args, **kwargs):
        raise AssertionError("a network with too few heads reached the scoring")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("head_scores", "reference_trials", "reference_means", "ablate_table"):
            patch.setattr(components, name, refused)
        patch.setattr(tables, "write_run_table", refused)
        patch.setattr(cli, "_scored_network", refused)
        assert cli.main(_ablate_argv(repo_root, tmp_path, run, experiment, overrides)) == 1
    assert "outside the top" in capsys.readouterr().err
    assert not config.run_directory(tmp_path, run.experiment, run.run_id).exists()
