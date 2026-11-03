"""The two-factor test battery: construction, holdout, scoring references, presets."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import pytest
import yaml
from conftest import adjacent_pairs as _adjacent_pairs
from conftest import design_for as _design
from conftest import top_down as _top_down
from conftest import vocabulary_for as _vocab

from geo_alg_ti import battery, config, streams
from geo_alg_ti.battery import load_preset
from geo_alg_ti.orderings import FREQUENT_CUE, OrderingLaw, cue_weights, uniform_cue_weights, zipf_cue_weights
from geo_alg_ti.trials import QUERYABLE_RANKS, ContextKind

BUDGETS = (0, 1, 2)
CONTEXTS_PER_PAIR = 10
# The distances of the 42 ordered pairs of seven ranks: 7 - d unordered pairs at each d, twice.
PAIRS_AT_DISTANCE = {d: 2 * (7 - d) for d in range(1, 7)}

# Rebuilds the two presets in another interpreter and prints their digests, one per line.
REBUILD_SCRIPT = """
import sys
from geo_alg_ti import battery, streams
from geo_alg_ti.orderings import OrderingLaw, uniform_cue_weights
from geo_alg_ti.sequences import Vocabulary

seed = int(sys.argv[1])
law = OrderingLaw(seed=seed, pool_size=48, n_training_cues=6, n_reserved_cues=2)
vocab = Vocabulary(pool_size=48, n_cues=8)
design = streams.build_finetuning_design(
    law, streams.design_rng(seed), layout=streams.Layout.CONFLICTING, n_orderings=1
)
for preset, chosen in (("finetuned", design), ("content", None)):
    built = battery.Battery.build(
        battery.load_preset(preset, contexts_per_pair=10), law, vocab, seed=seed, design=chosen,
        budgets=(0, 1, 2), cue_weights=uniform_cue_weights(6),
    )
    print(built.digest())
"""


def _build(law, conditions, *, design, seed=0, **kwargs):
    """Build the conditions (or a preset) under uniform cue weights unless others are given."""
    if isinstance(conditions, str):
        conditions = load_preset(conditions, contexts_per_pair=CONTEXTS_PER_PAIR)
    kwargs.setdefault("cue_weights", uniform_cue_weights(law.n_training_cues))
    return battery.Battery.build(
        conditions,
        law,
        _vocab(law),
        seed=seed,
        design=design,
        budgets=BUDGETS,
        **kwargs,
    )


def _by_condition(built):
    grouped = defaultdict(list)
    for name, trial in built.trials:
        grouped[name].append(trial)
    return grouped


def _sign_in(items, a, b):
    """The sign the order of `items` implies between a and b: +1 when a comes first."""
    return 1 if items.index(a) < items.index(b) else -1


def _shows_the_finetuned_ordering(trial, ordering):
    """The seven finetuned items in law order at ranks 1..7, one end-bucket distractor per end."""
    return (
        len(trial.items) == 9
        and trial.items[1:-1] == ordering.items
        and trial.items[0] in ordering.buckets[0]
        and trial.items[-1] in ordering.buckets[-1]
    )


def _pair_of(trial):
    """The ordered pair a battery trial was built from: shown positions if shuffled, else ranks."""
    return trial.shown_positions if trial.context_kind is ContextKind.SHUFFLED else trial.query_ranks


def _digest_in_new_process(seed, hash_seed):
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    run = subprocess.run(
        [sys.executable, "-c", REBUILD_SCRIPT, str(seed)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return run.stdout.split()


def test_ordered_rank_pairs_are_the_forty_two_pairs():
    """Seven queryable ranks give 42 ordered pairs."""
    pairs = battery.ordered_rank_pairs((1, 2, 3, 4, 5, 6, 7))
    assert len(pairs) == 42 and len(set(pairs)) == 42


def test_trial_count_is_pairs_times_contexts_per_pair(smoke_law, smoke_design):
    """420 trials per condition at ten contexts per pair."""
    conditions = load_preset("finetuned", contexts_per_pair=CONTEXTS_PER_PAIR)
    built = _build(smoke_law, conditions, design=smoke_design)
    per_condition = _by_condition(built)
    assert built.conditions == tuple(conditions)
    assert built.budgets == BUDGETS
    assert set(per_condition) == {condition.name for condition in conditions}
    for condition in conditions:
        assert condition.contexts_per_pair == 10
        assert len(per_condition[condition.name]) == 42 * 10 == 420
    assert len(built.trials) == 420 * len(conditions)

    three = battery.TestCondition(
        name="iwl_test",
        source=battery.OrderingSource.FINETUNED,
        context=ContextKind.UNINFORMATIVE,
        contexts_per_pair=3,
    )
    assert len(_build(smoke_law, (three,), design=smoke_design).trials) == 42 * 3 == 126


def test_rebuild_is_byte_identical(smoke_law, smoke_design):
    """The same seed gives the same digest in a second process."""
    finetuned = _build(smoke_law, "finetuned", design=smoke_design)
    content = _build(smoke_law, "content", design=None)
    here = [finetuned.digest(), content.digest()]
    assert all(isinstance(digest, str) and digest for digest in here)
    assert here[0] != here[1]
    # Two fresh interpreters with different string-hash seeds reproduce both digests exactly.
    assert _digest_in_new_process(0, 1) == here
    assert _digest_in_new_process(0, 2) == here
    # The digest depends on the seed, so the agreement above is not a constant.
    assert _build(smoke_law, "finetuned", design=_design(smoke_law), seed=1).digest() != here[0]


def test_every_condition_covers_every_distance_one_through_six(smoke_law, smoke_design):
    """No distance is missing from any condition."""
    assert sum(PAIRS_AT_DISTANCE.values()) == 42
    finetuned = _build(smoke_law, "finetuned", design=smoke_design)
    content = _build(smoke_law, "content", design=None)
    seen = set()
    for built in (finetuned, content):
        for name, its_trials in _by_condition(built).items():
            seen.add(name)
            expected = {d: n_pairs * 10 for d, n_pairs in PAIRS_AT_DISTANCE.items()}
            assert Counter(trial.d for trial in its_trials) == expected, name
            # A shuffled condition reads each pair as the shown positions of the queried items.
            pairs = Counter(_pair_of(trial) for trial in its_trials)
            assert set(pairs) == set(battery.ordered_rank_pairs(QUERYABLE_RANKS)), name
            assert set(pairs.values()) == {10}, name
            for trial in its_trials:
                assert trial.d == abs(_pair_of(trial)[0] - _pair_of(trial)[1])
                assert trial.query == (trial.items[trial.query_ranks[0]], trial.items[trial.query_ranks[1]])
    assert {"iwl_test", "icl_test", "baseline_test", "arbitrage_test", "frequent_cue_shuffled"} <= seen


def test_uninformative_condition_contains_no_queried_or_distractor_item(smoke_law, smoke_design):
    """Filler items are disjoint from the familiar ordering and its distractors."""
    design = smoke_design
    trained = design.presented_items
    finetuned = _by_condition(_build(smoke_law, "finetuned", design=design))
    content = _by_condition(_build(smoke_law, "content", design=None))
    pretrained = _by_condition(_build(smoke_law, "pretrained", design=None))
    frequent_cue_items = {item for trial in content["frequent_cue_congruent"] for item in trial.items}
    checks = (
        (finetuned["iwl_test"], trained),
        (content["frequent_cue_uninformative"], frequent_cue_items),
        (pretrained["iwl_test"], set()),
    )
    for its_trials, familiar_items in checks:
        assert len(its_trials) == 420
        for trial in its_trials:
            assert trial.context_kind is ContextKind.UNINFORMATIVE
            shown = set(trial.context_items)
            assert len(trial.context) == 8 and len(shown) == 9
            assert shown == {item for relation in trial.context for item in relation[:2]}
            assert shown.isdisjoint(familiar_items)
            assert shown.isdisjoint(trial.items)
            assert shown.isdisjoint(trial.query)
            assert shown.isdisjoint(trial.distractors)
            assert trial.shown_sign == 0
            assert trial.reference_sign == trial.familiar_sign != 0
            # The filler is a complete, truthful adjacent chain under the shown cue's law.
            order = smoke_law.ordering(trial.cue, trial.context_items)
            assert _top_down(trial.context) == _adjacent_pairs(order)


def test_reversed_condition_implies_the_opposite_sign_on_every_pair(smoke_law, smoke_design):
    """Every pair's shown sign is the opposite of its familiar sign."""
    design = smoke_design
    finetuned = _by_condition(_build(smoke_law, "finetuned", design=design))
    content = _by_condition(_build(smoke_law, "content", design=None))
    for its_trials in (finetuned["arbitrage_test"], content["frequent_cue_reversed"]):
        assert len(its_trials) == 420
        for trial in its_trials:
            assert trial.context_kind is ContextKind.REVERSED
            if its_trials is finetuned["arbitrage_test"]:
                assert _shows_the_finetuned_ordering(trial, design.orderings[0])
            else:
                assert trial.items == smoke_law.ordering(trial.cue, trial.items)
            assert set(trial.context_items) == set(trial.items) and len(trial.context) == 8
            assert trial.shown_sign == -trial.familiar_sign != 0
            assert trial.reference_sign == trial.shown_sign
            for a, b, sign in trial.context:
                assert sign == -_sign_in(trial.items, a, b)
            backwards = tuple(reversed(trial.items))
            assert _top_down(trial.context) == _adjacent_pairs(backwards)


def test_finetuned_source_queries_the_seven_finetuned_items_at_ranks_one_to_seven(smoke_law):
    """Under every seed, the queryable ranks of a finetuned condition name the finetuned items."""
    finetuned_only = tuple(
        condition
        for condition in load_preset("finetuned", contexts_per_pair=2)
        if condition.source is battery.OrderingSource.FINETUNED
    )
    assert {condition.name for condition in finetuned_only} == {"iwl_test", "baseline_test", "arbitrage_test"}
    laws = [smoke_law] + [
        OrderingLaw(seed=seed, pool_size=300, n_training_cues=100, n_reserved_cues=16) for seed in range(20)
    ]
    for law in laws:
        for seed in range(3) if law is smoke_law else (law.seed,):
            design = _design(law, seed=seed)
            ordering = design.orderings[0]
            built = _by_condition(_build(law, finetuned_only, design=design, seed=seed))
            for name, its_trials in built.items():
                assert len(its_trials) == 42 * 2
                for trial in its_trials:
                    assert _shows_the_finetuned_ordering(trial, ordering), (name, law.seed, seed)
                    assert trial.cue == ordering.cue
                    assert set(trial.query) <= set(ordering.items)
                    first, second = trial.query_ranks
                    assert trial.query == (trial.items[first], trial.items[second])
                    assert trial.familiar_sign == _sign_in(ordering.items, *trial.query)
                    if name == "baseline_test":
                        assert _top_down(trial.context) == _adjacent_pairs(trial.items)
                    elif name == "arbitrage_test":
                        assert _top_down(trial.context) == _adjacent_pairs(tuple(reversed(trial.items)))
                    else:
                        assert set(trial.context_items).isdisjoint(design.presented_items)


def test_novel_source_uses_only_reserved_cues_and_unseen_items(smoke_law, smoke_design):
    """The ICL test draws a reserved cue and items outside every trained ordering."""
    design = smoke_design
    trained = design.presented_items
    assert len(trained) == 7 + 8 * 2
    ordering = design.orderings[0]
    assert trained == ordering.presented_items == {*ordering.items, *(i for b in ordering.buckets for i in b)}
    finetuned = _by_condition(_build(smoke_law, "finetuned", design=design))
    novel = finetuned["icl_test"]
    assert len(novel) == 420
    for trial in novel:
        assert smoke_law.is_reserved(trial.cue) and trial.cue in smoke_law.reserved_cues()
        assert len(set(trial.items)) == 9 and set(trial.items).isdisjoint(trained)
        assert trial.items == smoke_law.ordering(trial.cue, trial.items)
        assert trial.context_kind is ContextKind.CONGRUENT
        assert set(trial.context_items) == set(trial.items) and len(trial.context) == 8
        assert _top_down(trial.context) == _adjacent_pairs(trial.items)
        assert trial.shown_sign == trial.familiar_sign == trial.reference_sign != 0
        assert set(trial.query_ranks) <= set(QUERYABLE_RANKS)
        assert not trial.distractors
    # Every other condition of the battery stays on training cues.
    for name, its_trials in finetuned.items():
        if name != "icl_test":
            assert not any(smoke_law.is_reserved(trial.cue) for trial in its_trials), name
    # Without a finetuned ordering the ICL test still draws only reserved cues.
    pretrained = _by_condition(_build(smoke_law, "pretrained", design=None))
    assert all(smoke_law.is_reserved(trial.cue) for trial in pretrained["icl_test"])


def test_training_distribution_source_draws_a_cue_per_trial(smoke_law):
    """Each trial of the sampled IWL test draws its own training cue."""
    pretrained = _by_condition(_build(smoke_law, "pretrained", design=None))
    sampled = pretrained["iwl_test"]
    assert len(sampled) == 420
    cues = [trial.cue for trial in sampled]
    assert set(cues) == set(smoke_law.training_cues())
    by_pair = defaultdict(set)
    for trial in sampled:
        by_pair[trial.query_ranks].add(trial.cue)
        assert trial.context_kind is ContextKind.UNINFORMATIVE
        assert len(set(trial.items)) == 9
        assert trial.items == smoke_law.ordering(trial.cue, trial.items)
    # The ten contexts of one rank pair are not all under one cue.
    assert all(len(drawn) > 1 for drawn in by_pair.values())

    # The draw follows the training cue weights: all the mass on one cue pins every trial to it.
    one_hot = np.zeros(smoke_law.n_training_cues)
    one_hot[4] = 1.0
    pinned = _by_condition(_build(smoke_law, "pretrained", design=None, cue_weights=one_hot))
    assert {trial.cue for trial in pinned["iwl_test"]} == {4}
    assert {trial.cue for trial in pinned["icl_test_training_cue"]} == {4}
    assert all(smoke_law.is_reserved(trial.cue) for trial in pinned["icl_test"])
    # A skewed distribution makes the first cue dominate.
    skewed = _by_condition(
        _build(smoke_law, "pretrained", design=None, cue_weights=zipf_cue_weights(6, 3.0))
    )
    counts = Counter(trial.cue for trial in skewed["iwl_test"])
    assert counts.most_common(1)[0][0] == 0 and counts[0] > 0.7 * 420


def test_training_distribution_source_needs_the_cue_weights(smoke_law, smoke_design):
    """A condition drawing cues from the training distribution refuses to build without them."""
    for preset in ("pretrained", "finetuned"):
        with pytest.raises(ValueError, match="cue weights"):
            _build(smoke_law, preset, design=smoke_design, cue_weights=None)
    # Weights over another number of cues would draw cue ids the network never trained on.
    with pytest.raises(ValueError, match="cues"):
        _build(smoke_law, "pretrained", design=None, cue_weights=uniform_cue_weights(5))
    # A preset without that source needs no weights.
    assert _build(smoke_law, "content", design=None, cue_weights=None).trials


def test_from_config_draws_cues_and_contexts_as_configured(smoke_law, smoke_design, repo_root):
    """The configured battery draws cues as pretraining did and samples the configured contexts."""
    smoke = repo_root / "configs" / "smoke.yaml"
    frequent = config.load_config(
        smoke,
        (
            "task.cue_distribution.kind=frequent_cue",
            "task.cue_distribution.frequent_cue_prop=1.0",
            "evaluation.contexts_per_pair=3",
        ),
    )
    vocab = _vocab(smoke_law)
    built = battery.Battery.from_config(
        frequent, smoke_law, vocab, seed=0, stage=streams.Stage.PRETRAINING, design=None
    )
    assert built.conditions == load_preset("pretrained", contexts_per_pair=3)
    assert all(condition.contexts_per_pair == 3 for condition in built.conditions)
    assert built.budgets == frequent.evaluation.reasoning_steps()
    per_condition = _by_condition(built)
    assert len(per_condition["iwl_test"]) == 42 * 3
    # All the mass on the frequent cue pins every sampled IWL trial to it.
    assert {trial.cue for trial in per_condition["iwl_test"]} == {FREQUENT_CUE}

    skewed = config.load_config(
        smoke, ("task.cue_distribution.kind=zipf", "task.cue_distribution.skew_alpha=3.0")
    )
    built = battery.Battery.from_config(
        skewed, smoke_law, vocab, seed=0, stage=streams.Stage.PRETRAINING, design=None
    )
    sampled = _by_condition(built)["iwl_test"]
    assert len(sampled) == 42 * skewed.evaluation.contexts_per_pair
    counts = Counter(trial.cue for trial in sampled)
    assert counts.most_common(1)[0][0] == 0 and counts[0] > 0.7 * len(sampled)
    # The same trials come from building the preset by hand with the skewed weights.
    by_hand = battery.Battery.build(
        load_preset("pretrained", contexts_per_pair=skewed.evaluation.contexts_per_pair),
        smoke_law,
        vocab,
        seed=0,
        design=None,
        budgets=skewed.evaluation.reasoning_steps(),
        cue_weights=cue_weights("zipf", smoke_law.n_training_cues, skew_alpha=3.0),
    )
    assert built.digest() == by_hand.digest()

    # The finetuning stage scores its own preset, and `evaluation.battery` overrides either.
    design = smoke_design
    finetuned = battery.Battery.from_config(
        skewed, smoke_law, vocab, seed=0, stage=streams.Stage.FINETUNING, design=design
    )
    assert finetuned.conditions == load_preset("finetuned", contexts_per_pair=2)
    overridden = config.load_config(smoke, ("evaluation.battery=content",))
    content = battery.Battery.from_config(
        overridden, smoke_law, vocab, seed=0, stage=streams.Stage.FINETUNING, design=design
    )
    assert content.conditions == load_preset("content", contexts_per_pair=2)


def test_scoring_reference_is_familiar_when_the_context_is_uninformative():
    """Uninformative context scores against the familiar ordering, which a novel ordering lacks."""
    for source in battery.OrderingSource:
        if source is battery.OrderingSource.NOVEL_UNSEEN_CUE:
            with pytest.raises(ValueError, match="no ordering to score against"):
                battery.scoring_reference_for(source, "uninformative")
        else:
            assert battery.scoring_reference_for(source, "uninformative") == battery.ScoringReference.FAMILIAR_ORDERING


def test_preset_loader_refuses_a_condition_that_cannot_be_scored(tmp_path):
    """A novel ordering under an uninformative context is refused when the preset is read."""
    root = tmp_path / "battery"
    root.mkdir()
    condition = {"name": "novel_blind", "source": battery.OrderingSource.NOVEL_UNSEEN_CUE.value,
                 "context": "uninformative", "label": "novel ordering, uninformative context"}
    (root / "blind.yaml").write_text(yaml.safe_dump({"name": "blind", "conditions": [condition]}), encoding="utf-8")
    with pytest.raises(ValueError, match="no ordering to score against"):
        battery.load_preset("blind", contexts_per_pair=2, root=root)


def test_scoring_reference_is_shown_when_the_context_is_informative():
    """Congruent, reversed, and shuffled contexts score against the shown ordering."""
    for context in ("congruent", "reversed", "shuffled"):
        assert battery.scoring_reference_for(battery.OrderingSource.FINETUNED, context) == battery.ScoringReference.SHOWN_ORDERING


def test_presets_declare_every_condition_a_figure_needs():
    """Finetuned preset: IWL, ICL, baseline, arbitrage. Content preset: four contexts and ICL."""
    Source = battery.OrderingSource
    finetuned = {condition.name: condition for condition in load_preset("finetuned", contexts_per_pair=10)}
    paper_tests = {
        "iwl_test": (Source.FINETUNED, ContextKind.UNINFORMATIVE, "IWL test"),
        "icl_test": (Source.NOVEL_UNSEEN_CUE, ContextKind.CONGRUENT, "ICL test"),
        "baseline_test": (Source.FINETUNED, ContextKind.CONGRUENT, "baseline test"),
        "arbitrage_test": (Source.FINETUNED, ContextKind.REVERSED, "arbitrage test"),
    }
    for name, (source, context, label) in paper_tests.items():
        condition = finetuned[name]
        assert (condition.source, condition.context, condition.label) == (source, context, label)
    content = {condition.name: condition for condition in load_preset("content", contexts_per_pair=10)}
    for context in ContextKind:
        condition = content[f"frequent_cue_{context}"]
        assert (condition.source, condition.context, condition.label) == (Source.FREQUENT_CUE, context, str(context))
    assert content["icl_test"].source is Source.NOVEL_UNSEEN_CUE
    # A network that was only pretrained has no finetuned ordering, so its IWL test draws a
    # training cue per trial; the ICL test is the same under every preset.
    pretrained = {condition.name: condition for condition in load_preset("pretrained", contexts_per_pair=10)}
    assert (pretrained["iwl_test"].source, pretrained["iwl_test"].context) == (
        Source.TRAINING_DISTRIBUTION,
        ContextKind.UNINFORMATIVE,
    )
    assert pretrained["icl_test"] == finetuned["icl_test"] == content["icl_test"]
    assert pretrained["icl_test_training_cue"] == finetuned["icl_test_training_cue"]


def test_condition_names_are_unique_and_snake_case():
    """Names are unique and lowercase with underscores."""
    snake_case = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")
    for preset in ("finetuned", "pretrained", "content"):
        conditions = load_preset(preset, contexts_per_pair=10)
        names = [condition.name for condition in conditions]
        assert len(set(names)) == len(names), preset
        assert all(snake_case.fullmatch(name) for name in names), preset
        for condition in conditions:
            assert condition.scoring_reference == battery.scoring_reference_for(
                condition.source, condition.context
            )


def test_presets_load_from_the_committed_files(repo_root):
    """All three presets load from `configs/battery/` with a source, context, and label each."""
    root = repo_root / "configs" / "battery"
    assert {path.stem for path in root.glob("*.yaml")} == {"pretrained", "finetuned", "content"}
    for name in ("pretrained", "finetuned", "content"):
        conditions = load_preset(name, root=root, contexts_per_pair=4)
        assert conditions
        for condition in conditions:
            assert isinstance(condition.source, battery.OrderingSource)
            assert isinstance(condition.context, ContextKind)
            assert condition.label and condition.contexts_per_pair == 4


def test_scoring_reference_follows_context():
    """Uninformative scores against the familiar ordering; informative against the shown one."""
    for name in ("pretrained", "finetuned", "content"):
        for condition in load_preset(name, contexts_per_pair=1):
            expected = "familiar_ordering" if condition.context == "uninformative" else "shown_ordering"
            assert condition.scoring_reference == expected


def test_preset_loader_refuses_malformed_files(tmp_path):
    """An unknown key, an unknown source or context, a repeated name, or a bad count raises."""
    good = {"name": "iwl_test", "source": "finetuned", "context": "uninformative", "label": "IWL test"}

    def write(name, conditions, **extra):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump({"name": name, "conditions": conditions, **extra}))

    write("fine", [good])
    assert load_preset("fine", root=tmp_path, contexts_per_pair=2)[0].contexts_per_pair == 2
    write("unknown_key", [{**good, "scoring_reference": "familiar_ordering"}])
    write("missing_key", [{key: value for key, value in good.items() if key != "label"}])
    write("bad_source", [{**good, "source": "novel"}])
    write("bad_context", [{**good, "context": "reverse"}])
    write("repeated", [good, {**good, "context": "congruent"}])
    write("empty", [])
    write("extra_top_level", [good], scoring_reference="shown_ordering")
    write("misnamed", [good])
    (tmp_path / "misnamed.yaml").write_text((tmp_path / "misnamed.yaml").read_text().replace("misnamed", "other"))
    for name in ("unknown_key", "missing_key", "bad_source", "bad_context", "repeated", "empty", "extra_top_level", "misnamed"):
        with pytest.raises(ValueError):
            load_preset(name, root=tmp_path, contexts_per_pair=2)
    with pytest.raises(ValueError):
        load_preset("fine", root=tmp_path, contexts_per_pair=0)


def _shuffled_trials(smoke_law, contexts_per_pair, seed=0):
    """The shuffled condition of the content preset at `contexts_per_pair` contexts per pair."""
    conditions = load_preset("content", contexts_per_pair=contexts_per_pair)
    shuffled = tuple(condition for condition in conditions if condition.context is ContextKind.SHUFFLED)
    assert [condition.name for condition in shuffled] == ["frequent_cue_shuffled"]
    return _by_condition(_build(smoke_law, shuffled, design=None, seed=seed))["frequent_cue_shuffled"]


@pytest.mark.parametrize("contexts_per_pair", [10, 2, 3, 1])
def test_shuffled_agreement_is_balanced_at_every_shown_distance(smoke_law, contexts_per_pair):
    """Half the trials agree at every shown distance; within every ordered pair when even."""
    its_trials = _shuffled_trials(smoke_law, contexts_per_pair)
    assert len(its_trials) == 42 * contexts_per_pair
    by_distance = defaultdict(list)
    by_pair = defaultdict(list)
    for trial in its_trials:
        p, q = trial.shown_positions
        assert {p, q} <= set(QUERYABLE_RANKS) and set(trial.query_ranks) <= set(QUERYABLE_RANKS)
        assert trial.d == abs(p - q)
        agree = trial.shown_sign == trial.familiar_sign
        assert trial.shown_sign == (1 if p < q else -1) and trial.reference_sign == trial.shown_sign
        by_distance[trial.d].append(agree)
        by_pair[(p, q)].append(agree)
    assert set(by_pair) == set(battery.ordered_rank_pairs(QUERYABLE_RANKS))
    for d, agreements in by_distance.items():
        assert len(agreements) == PAIRS_AT_DISTANCE[d] * contexts_per_pair
        assert 2 * sum(agreements) == len(agreements), d
    for pair, agreements in by_pair.items():
        expected = [battery.shuffled_query(pair, index).agree for index in range(contexts_per_pair)]
        assert agreements == expected, pair
        if contexts_per_pair % 2 == 0:
            assert 2 * sum(agreements) == contexts_per_pair, pair


def test_shuffled_query_alternates_from_opposite_sides_for_the_two_orders():
    """The first context of an upward pair conflicts and that of a downward pair agrees."""
    assert [battery.shuffled_query((2, 5), index).agree for index in range(4)] == [False, True, False, True]
    assert [battery.shuffled_query((5, 2), index).agree for index in range(4)] == [True, False, True, False]
    assert battery.shuffled_query((5, 2), 0).positions == (5, 2)


def test_shuffled_familiar_distance_spans_every_shown_distance(smoke_law):
    """Across seeds, every familiar distance occurs at every shown distance, near uniformly."""
    combinations = Counter()
    for seed in range(8):
        for trial in _shuffled_trials(smoke_law, CONTEXTS_PER_PAIR, seed=seed):
            combinations[(trial.d, abs(trial.query_ranks[0] - trial.query_ranks[1]))] += 1
    assert set(combinations) == {(d, k) for d in range(1, 7) for k in range(1, 7)}
    familiar = Counter()
    for (_, k), count in combinations.items():
        familiar[k] += count
    total = sum(familiar.values())
    assert total == 8 * 420
    assert all(abs(familiar[k] / total - 1 / 6) < 0.02 for k in range(1, 7))


# The digest of every condition but the shuffled one, per preset and seed, under the smoke law
# at ten contexts per pair; the shuffled construction changes none of them.
UNSHUFFLED_DIGESTS = {
    (0, "content", "frequent_cue_uninformative"): "f797698e0360530e3615b5daa9c64bb2c9afb7e14f70596cef218a924f4b4980",
    (0, "content", "frequent_cue_congruent"): "22f5be58484d88c2915d887efa3f30faf9fa76dae45110b622c64e036c4b7958",
    (0, "content", "frequent_cue_reversed"): "75cdf52012067b0ebf3453efd9d994f1a2cef4b4c0a15f797998db51d5720bb6",
    (0, "content", "icl_test"): "22a4de587b55e768815aba70ffbb3c2e0381468136d0b705bda262130b68f676",
    (0, "finetuned", "iwl_test"): "b913788e38b6d57ba86d0fc0b557d0e9792165bd0fcc1f7e788717e4804ac5fb",
    (0, "finetuned", "icl_test"): "167e4a2db76fe97ed4b0c1bca157d7003e7cbe4998749f915640094c479ff68e",
    (0, "finetuned", "icl_test_training_cue"): "2aa0cad3a109e39d42e4f66ae947cad220a063db03869d6d87e6135b7ba60bf7",
    (0, "finetuned", "baseline_test"): "063e9b54fc0b8b3f88db108e89ae886b48011dd9a2163524924c13078eb675c4",
    (0, "finetuned", "arbitrage_test"): "e971609277deeb992045eb556730bb3ccb1a2d365a633694c77ef3e54e9ec04f",
    (0, "pretrained", "iwl_test"): "da9779fc4b192fbb8e4b6605fa319c49cfa2becdce5b8f53d08d6dbe3ce988f2",
    (0, "pretrained", "icl_test"): "22a4de587b55e768815aba70ffbb3c2e0381468136d0b705bda262130b68f676",
    (0, "pretrained", "icl_test_training_cue"): "1350d3ee112eea2891051302f0e9b6b38a0405a792f968938cf1e3a64c09b86c",
    (3, "content", "frequent_cue_uninformative"): "28fce4de0da319e93ce9edaf45580bb62a1be845ed48dffbd531d2def4dd752d",
    (3, "content", "frequent_cue_congruent"): "d0a4d843037b09af8bb9e7ee9a9350b8a1e7c5cb6701350b348d3d71d23d7367",
    (3, "content", "frequent_cue_reversed"): "32f63d65cce93c711f22eb85d3dc93cb719ca05828ab0edbffd65e200867a6ae",
    (3, "content", "icl_test"): "ca95e7eadba94d85f6d2993554b0aed38bfbd54c6444b9e9237a2078853d8605",
    (3, "finetuned", "iwl_test"): "aaaf5aba43cf645d31fb60182f846e1c27f4de6f3bd4fa02c0d4fb657d4f79be",
    (3, "finetuned", "icl_test"): "22e3634dc3fd6eea5086dc43640c8a6b3710aaef3fe805d30d15960c79a79a2d",
    (3, "finetuned", "icl_test_training_cue"): "2c07e3a57b16bcc78b0e2e9124437ab2e7542eac06418e0de96709af6d942070",
    (3, "finetuned", "baseline_test"): "61e8b3cde6148fe53eb3636fd4a4583ddc5ebc329e80b5b4eda95e91f8541a3a",
    (3, "finetuned", "arbitrage_test"): "ff95fcd359b163d75f8a66fb3a6f361cc8e45f77621ec661c6358e48efa28887",
    (3, "pretrained", "iwl_test"): "ad4b34a146751e96727dfb1889964e1b99471dfe7ee4b744ded65072d025a374",
    (3, "pretrained", "icl_test"): "ca95e7eadba94d85f6d2993554b0aed38bfbd54c6444b9e9237a2078853d8605",
    (3, "pretrained", "icl_test_training_cue"): "d5cda2fa1a1ddce564751234054293eaad7ba9e18c2222b95f9222b0edc9aeb1",
}


def test_conditions_other_than_shuffled_keep_their_fixed_digests(smoke_law, smoke_design):
    """Every trial of every other condition of the three presets hashes to its fixed digest."""
    found = {}
    for seed in (0, 3):
        for preset, design in (("content", None), ("finetuned", smoke_design), ("pretrained", None)):
            running = {}
            for name, trial in _build(smoke_law, preset, design=design, seed=seed).trials:
                running.setdefault(name, hashlib.sha256()).update(battery._trial_line(name, trial).encode())
            found.update({(seed, preset, name): digest.hexdigest() for name, digest in running.items()})
    shuffled = {key for key in found if key[2] == "frequent_cue_shuffled"}
    assert len(shuffled) == 2
    assert {key: found[key] for key in found if key not in shuffled} == UNSHUFFLED_DIGESTS

