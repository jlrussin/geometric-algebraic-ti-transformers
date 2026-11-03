"""The training loop, checkpoints, determinism, and run directories."""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch.nn import functional as F

from geo_alg_ti import battery, config, training
from geo_alg_ti.chains import generate_chain
from geo_alg_ti.model import ModelOutput
from geo_alg_ti.orderings import OrderingLaw, cue_weights, uniform_cue_weights
from geo_alg_ti.sequences import Target, encode
from geo_alg_ti.streams import Batch, Stage, pretraining_stream, stack
from geo_alg_ti.trials import sample_pretraining_trial

# A configuration small enough to train and evaluate in seconds, and short enough to read a
# cadence off.
OVERRIDES = (
    "task.item_pool_size=48",
    "task.n_training_cues=6",
    "task.n_reserved_cues=2",
    "model.n_layers=2",
    "model.d_model=32",
    "model.d_ff=64",
    "pretraining.steps=4",
    "pretraining.batch_size=8",
    "pretraining.eval_every=2",
    "finetuning.steps=1",
    "finetuning.eval_every=1",
    "evaluation.max_reasoning_steps=0",
    "evaluation.battery=pretrained",
    "logging.log_every=1",
    "seeds=[0]",
)


def _config(*extra: str) -> config.Config:
    """The small configuration, with any further overrides applied after it."""
    return config.load_config(None, OVERRIDES + extra)


def _law(cfg: config.Config, seed: int = 0) -> OrderingLaw:
    """The ordering law a run of this configuration uses."""
    return training.build_law(cfg, seed)


def _steps(directory) -> set[int]:
    """The steps a directory of per-step tables holds."""
    return {int(path.stem.split("_")[1]) for path in directory.glob("step_*.parquet")}


def _weights(path) -> dict[str, torch.Tensor]:
    """The parameters a checkpoint holds."""
    return training.load_checkpoint(path).model_state


def _mixed_batch(cfg: config.Config, n_chains: int = 8) -> tuple[list, Batch]:
    """Chain targets, some reached through a chain, followed by one answer-only target."""
    vocab = training.build_vocabulary(cfg)
    rng = np.random.default_rng(0)
    examples = []
    weights = uniform_cue_weights(cfg.task.n_training_cues)
    for _ in range(n_chains):
        trial = sample_pretraining_trial(_law(cfg), rng, weights=weights)
        examples.append(encode(trial, generate_chain(trial), vocab, target=Target.CHAIN))
    examples.append(encode(trial, None, vocab, target=Target.ANSWER_ONLY))
    assert any(example.answer_index > example.prompt_len + 2 for example in examples[:-1])
    return examples, stack(examples, vocab).to_batch("cpu")


def _signs(batch: Batch, *, right_at: torch.Tensor) -> torch.Tensor:
    """Scalar predictions wrong at every supervised relation except the slots in `right_at`."""
    predicted = torch.zeros_like(batch.scalars)
    rows, slots = torch.nonzero(batch.scalar_loss, as_tuple=True)
    predicted[rows, slots - 1] = -batch.scalars[rows, slots]
    rows = torch.arange(len(right_at))
    predicted[rows, right_at - 1] = batch.scalars[rows, right_at]
    return predicted


def _logits(batch: Batch, vocab_size: int, *, right_where: torch.Tensor) -> torch.Tensor:
    """Logits predicting the next token wherever `right_where` holds and another token elsewhere."""
    target = batch.tokens[:, 1:]
    right = F.one_hot(target, vocab_size).float()
    wrong = F.one_hot((target + 1) % vocab_size, vocab_size).float()
    logits = torch.zeros((*batch.tokens.shape, vocab_size))
    logits[:, :-1] = torch.where(right_where[:, 1:, None], right, wrong)
    return logits


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    """One pretraining stage, trained once for the whole file."""
    out = tmp_path_factory.mktemp("pretraining")
    cfg = _config()
    summary = training.run_stage(cfg, Stage.PRETRAINING, seed=0, init_from=None, out_dir=out, device="cpu")
    return cfg, summary


def test_the_vocabulary_hides_the_cue_when_the_configuration_does():
    """`task.show_cue` reaches the one vocabulary every stream, battery, and probe encodes with."""
    assert training.build_vocabulary(_config()).show_cue
    assert not training.build_vocabulary(_config("task.show_cue=false")).show_cue


def test_the_network_is_built_from_the_spec_the_configuration_gives():
    """`build_model` builds the architecture `build_spec` describes, its heads included."""
    cfg = _config("model.n_layers=3")
    vocab = training.build_vocabulary(cfg)
    spec = training.build_spec(cfg, vocab)
    net = training.build_model(cfg, vocab)
    assert net.spec == spec and len(net.blocks) == spec.n_layers == 3
    assert all(block.attention.n_heads == spec.n_heads for block in net.blocks)


def test_run_stage_writes_config_fingerprint_and_checkpoint(tmp_path, pretrained):
    """The run directory holds config.yaml, fingerprint.json, and a checkpoint."""
    cfg, summary = pretrained
    out = summary.out_dir
    assert summary.stage is Stage.PRETRAINING and summary.steps == cfg.pretraining.steps
    assert summary.checkpoint.exists() and summary.checkpoint == out / "checkpoints" / "final.pt"
    written = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert written == config.to_dict(cfg)
    fingerprint = json.loads((out / "fingerprint.json").read_text(encoding="utf-8"))
    assert fingerprint["config_hash"] == config.config_hash(cfg)
    assert fingerprint["run_id"] == out.name
    assert fingerprint["package_version"] and fingerprint["torch_version"] and fingerprint["host"]
    assert float(fingerprint["wall_time_s"]) >= 0.0
    saved = training.load_checkpoint(summary.checkpoint)
    assert saved.step == cfg.pretraining.steps and saved.stage is Stage.PRETRAINING
    assert config.config_hash(saved.config) == config.config_hash(cfg)
    assert (out / "train_metrics.parquet").exists() and (out / "log.jsonl").exists()


def test_checkpoint_round_trip_restores_weights_and_optimizer(tmp_path):
    """Saving and loading reproduces the model and optimizer state."""
    cfg = _config()
    vocab = training.build_vocabulary(cfg)
    torch.manual_seed(0)
    model = training.build_model(cfg, vocab)
    optimizer = training.build_optimizer(model, cfg, Stage.PRETRAINING)
    batch = next(pretraining_stream(cfg.task, _law(cfg), vocab, seed=0, batch_size=4))
    training.train_step(model, optimizer, batch)
    path = tmp_path / "checkpoints" / "final.pt"
    training.save_checkpoint(path, model=model, optimizer=optimizer, cfg=cfg, step=7, stage=Stage.PRETRAINING)

    saved = training.load_checkpoint(path)
    assert saved.step == 7 and saved.stage is Stage.PRETRAINING
    torch.manual_seed(1)
    restored = training.build_model(cfg, vocab)
    assert not all(torch.equal(restored.state_dict()[name], value) for name, value in model.state_dict().items())
    restored.load_state_dict(saved.model_state)
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    fresh = training.build_optimizer(restored, cfg, Stage.PRETRAINING)
    fresh.load_state_dict(saved.optimizer_state)
    assert fresh.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    for key, state in optimizer.state_dict()["state"].items():
        assert torch.equal(fresh.state_dict()["state"][key]["exp_avg"], state["exp_avg"])
        assert torch.equal(fresh.state_dict()["state"][key]["exp_avg_sq"], state["exp_avg_sq"])


def test_same_seed_gives_identical_weights(tmp_path):
    """Two runs at the same seed end with identical parameters."""
    cfg = _config()
    runs = {}
    for name, seed in (("first", 0), ("second", 0), ("other", 1)):
        summary = training.run_stage(
            cfg, Stage.PRETRAINING, seed=seed, init_from=None, out_dir=tmp_path / name, device="cpu"
        )
        runs[name] = _weights(summary.checkpoint)
    for name, value in runs["first"].items():
        assert torch.equal(runs["second"][name], value)
    assert any(not torch.equal(runs["other"][name], value) for name, value in runs["first"].items())


def test_loss_decreases_on_a_small_overfit_batch():
    """Repeated steps on one batch lower the loss."""
    cfg = _config()
    vocab = training.build_vocabulary(cfg)
    torch.manual_seed(0)
    model = training.build_model(cfg, vocab)
    optimizer = training.build_optimizer(model, cfg, Stage.PRETRAINING)
    batch = next(pretraining_stream(cfg.task, _law(cfg), vocab, seed=0, batch_size=4))
    history = []
    for _ in range(60):
        out, terms = training.train_step(model, optimizer, batch)
        assert not any(t.requires_grad for t in (out.logits, out.scalar_pred, terms.total))
        history.append(training.train_metrics(out, terms, batch))
    assert history[-1]["loss"] < 0.5 * history[0]["loss"]
    assert history[-1]["answer_accuracy"] > history[0]["answer_accuracy"]
    assert set(history[0]) == set(training.TRAIN_METRICS) == {
        "loss",
        "token_cross_entropy",
        "scalar_squared_error",
        "answer_accuracy",
        "chain_token_accuracy",
    }
    assert all(isinstance(value, float) for value in history[0].values())


def test_answer_accuracy_scores_the_answer_relation_and_not_its_repeats():
    """The sign is read at the answer slot the encoder names; the padding repeats are not scored."""
    cfg = _config()
    vocab = training.build_vocabulary(cfg)
    examples, batch = _mixed_batch(cfg)
    assert batch.answer_index.tolist() == [example.answer_index for example in examples]
    assert any(index > example.prompt_len + 2 for index, example in zip(batch.answer_index, examples))
    empty = ModelOutput(logits=torch.zeros((*batch.tokens.shape, vocab.size)), scalar_pred=None)
    right_at_answer = replace(empty, scalar_pred=_signs(batch, right_at=batch.answer_index))
    assert training.answer_accuracy(right_at_answer, batch) == 1.0
    # Right at every relation but the answer: the repeats of the answer earn nothing.
    wrong_at_answer = replace(empty, scalar_pred=-_signs(batch, right_at=batch.answer_index))
    assert training.answer_accuracy(wrong_at_answer, batch) == 0.0
    # Half the rows right at the answer: the score is a mean over rows.
    half = _signs(batch, right_at=batch.answer_index)
    half[: len(examples) // 2] *= -1
    assert training.answer_accuracy(replace(empty, scalar_pred=half), batch) == pytest.approx(
        1 - (len(examples) // 2) / len(examples)
    )


def test_chain_token_accuracy_scores_the_chain_emissions_alone():
    """Token accuracy covers the emissions before the answer, not the answer or its repeats."""
    cfg = _config()
    vocab = training.build_vocabulary(cfg)
    examples, batch = _mixed_batch(cfg)
    positions = torch.arange(batch.tokens.shape[1])
    chain = torch.stack(
        [(positions >= example.prompt_len) & (positions < example.answer_index - 2) for example in examples]
    )
    assert bool(chain.any()) and not bool(chain[-1].any())
    scalar_pred = torch.zeros_like(batch.scalars)
    right_on_chain = ModelOutput(logits=_logits(batch, vocab.size, right_where=chain), scalar_pred=scalar_pred)
    assert training.chain_token_accuracy(right_on_chain, batch) == 1.0
    right_off_chain = ModelOutput(logits=_logits(batch, vocab.size, right_where=~chain), scalar_pred=scalar_pred)
    assert training.chain_token_accuracy(right_off_chain, batch) == 0.0
    # An answer-only batch carries no chain emission, so there is nothing to score.
    answer_only = stack(examples[-1:], vocab).to_batch("cpu")
    everywhere = ModelOutput(
        logits=_logits(answer_only, vocab.size, right_where=torch.ones_like(answer_only.token_loss)),
        scalar_pred=torch.zeros_like(answer_only.scalars),
    )
    assert torch.isnan(training.chain_token_accuracy(everywhere, answer_only))


def test_finetuning_starts_from_the_named_checkpoint(tmp_path, pretrained):
    """Finetuning loads the pretraining checkpoint it is pointed at."""
    _, summary = pretrained
    # At a negligible learning rate the stage leaves the parameters where it found them.
    cfg = _config("finetuning.learning_rate=1.0e-12")
    started = training.run_stage(
        cfg, Stage.FINETUNING, seed=0, init_from=summary.checkpoint, out_dir=tmp_path / "from", device="cpu"
    )
    scratch = training.run_stage(
        cfg, Stage.FINETUNING, seed=0, init_from=None, out_dir=tmp_path / "scratch", device="cpu"
    )
    pretrained_weights = _weights(summary.checkpoint)
    for name, value in _weights(started.checkpoint).items():
        assert torch.allclose(value, pretrained_weights[name], atol=1e-9, rtol=0.0)
    assert any(
        not torch.allclose(value, pretrained_weights[name], atol=1e-9, rtol=0.0)
        for name, value in _weights(scratch.checkpoint).items()
    )


def test_evaluation_fires_at_the_declared_cadence(tmp_path):
    """Evaluation runs before the first update, every `eval_every` steps, and at the final step."""
    cfg = _config("pretraining.steps=5", "pretraining.eval_every=2")
    summary = training.run_stage(
        cfg, Stage.PRETRAINING, seed=0, init_from=None, out_dir=tmp_path / "cadence", device="cpu"
    )
    out = summary.out_dir
    assert _steps(out / "eval") == {0, 2, 4, 5}
    # Per-trial records and the per-pair table built from them are written at the final step alone.
    assert _steps(out / "trials") == {5}
    assert _steps(out / "eval_pairs") == {5}
    log = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry["step"] for entry in log if entry["event"] == "evaluation"] == [0, 2, 4, 5]
    assert all(entry["rows"] > 0 for entry in log if entry["event"] == "evaluation")
    assert [entry["step"] for entry in log if entry["event"] == "train"] == [1, 2, 3, 4, 5]


def test_finetuning_scores_the_network_it_starts_from(tmp_path, pretrained):
    """Finetuning's step-0 evaluation scores the checkpoint it starts from."""
    _, summary = pretrained
    cfg = _config("finetuning.steps=2", "finetuning.eval_every=2")
    out = training.run_stage(
        cfg, Stage.FINETUNING, seed=0, init_from=summary.checkpoint, out_dir=tmp_path / "zero", device="cpu"
    ).out_dir
    assert _steps(out / "eval") == {0, 2}
    assert _steps(out / "trials") == {2}
    assert _steps(out / "eval_pairs") == {2}
    first = pd.read_parquet(out / "eval" / "step_00000000.parquet")
    assert (first["step"] == 0).all() and (first["tag"] == "finetuning").all() and len(first) > 0
    log = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [(entry["event"], entry["step"]) for entry in log if entry["event"] in {"evaluation", "train"}]
    assert events == [("evaluation", 0), ("train", 1), ("train", 2), ("evaluation", 2)]


def test_run_stage_scores_the_battery_the_configuration_declares(tmp_path):
    """The evaluated battery takes its cue weights and contexts per pair from the configuration."""
    cfg = _config(
        "task.cue_distribution.kind=zipf",
        "task.cue_distribution.skew_alpha=2.0",
        "evaluation.contexts_per_pair=3",
        "pretraining.steps=1",
        "pretraining.eval_every=1",
    )
    summary = training.run_stage(
        cfg, Stage.PRETRAINING, seed=0, init_from=None, out_dir=tmp_path / "skewed", device="cpu"
    )
    out = summary.out_dir
    log = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    recorded = next(entry for entry in log if entry["event"] == "battery")
    vocab = training.build_vocabulary(cfg)
    expected = battery.Battery.from_config(
        cfg, _law(cfg), vocab, seed=0, stage=Stage.PRETRAINING, design=None
    )
    assert recorded["digest"] == expected.digest()
    assert recorded["trials"] == len(expected.trials) == 3 * 42 * 3
    # The same preset under uniform cue weights, or at the preset's own count, is another battery.
    skewed = cue_weights("zipf", cfg.task.n_training_cues, skew_alpha=2.0)
    for weights, per_pair in ((uniform_cue_weights(cfg.task.n_training_cues), 3), (skewed, 10)):
        other = battery.Battery.build(
            battery.load_preset(cfg.pretraining.battery, contexts_per_pair=per_pair),
            _law(cfg),
            vocab,
            seed=0,
            design=None,
            budgets=cfg.evaluation.reasoning_steps(),
            cue_weights=weights,
        )
        assert other.digest() != recorded["digest"]
    # Every aggregate counts three contexts per ordered pair: 2 * (7 - d) pairs at distance d.
    table = pd.read_parquet(out / "eval" / "step_00000001.parquet")
    accuracy = table[table["metric"] == "accuracy"]
    for d, rows in accuracy.groupby("d"):
        assert set(rows["n"]) == {2 * (7 - int(d)) * 3}


def test_the_run_records_how_long_the_stage_and_its_evaluations_took(tmp_path):
    """Evaluations log their duration, train events the time so far, the fingerprint the total."""
    cfg = _config("pretraining.steps=5", "pretraining.eval_every=2")
    summary = training.run_stage(
        cfg, Stage.PRETRAINING, seed=0, init_from=None, out_dir=tmp_path / "timed", device="cpu"
    )
    out = summary.out_dir
    log = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    durations = [entry["duration_s"] for entry in log if entry["event"] == "evaluation"]
    assert len(durations) == 4 and all(duration >= 0.0 for duration in durations)
    elapsed = [entry["elapsed_s"] for entry in log if entry["event"] == "train"]
    assert elapsed == sorted(elapsed) and elapsed[0] >= 0.0
    fingerprint = json.loads((out / "fingerprint.json").read_text(encoding="utf-8"))
    # The fingerprint is rewritten at the end, so its wall time covers every evaluation and step.
    assert fingerprint["wall_time_s"] >= elapsed[-1]
    assert fingerprint["wall_time_s"] + 1e-2 >= sum(durations)
    assert summary.wall_time_s + 1e-2 >= fingerprint["wall_time_s"]


def test_a_run_that_stops_early_keeps_its_curve_and_a_checkpoint(tmp_path, monkeypatch):
    """A stage that raises mid-run keeps the metrics so far and the last updated parameters."""
    cfg = _config("pretraining.steps=6", "pretraining.eval_every=3")
    stream = training.pretraining_stream

    def cut_short(*args, **kwargs):
        for count, batch in enumerate(stream(*args, **kwargs)):
            if count == 2:
                raise RuntimeError("the stream stopped")
            yield batch

    monkeypatch.setattr(training, "pretraining_stream", cut_short)
    out = tmp_path / "cut"
    with pytest.raises(RuntimeError, match="stopped"):
        training.run_stage(cfg, Stage.PRETRAINING, seed=0, init_from=None, out_dir=out, device="cpu")
    assert pd.read_parquet(out / "train_metrics.parquet")["step"].tolist() == [1, 2]
    assert not (out / "checkpoints" / "final.pt").exists()
    saved = training.load_checkpoint(out / "checkpoints" / "step_00000002.pt")
    assert saved.step == 2 and saved.stage is Stage.PRETRAINING
    # The fingerprint is rewritten after the checkpoint on the way out, so it times the stopped run.
    checkpoint_written = (out / "checkpoints" / "step_00000002.pt").stat().st_mtime
    assert (out / "fingerprint.json").stat().st_mtime >= checkpoint_written
    # The parameters are those of a run declared to stop at that step.
    monkeypatch.setattr(training, "pretraining_stream", stream)
    short = training.run_stage(
        _config("pretraining.steps=2"), Stage.PRETRAINING, seed=0, init_from=None, out_dir=tmp_path / "two", device="cpu"
    )
    assert short.checkpoint.name == "final.pt"
    for name, value in _weights(short.checkpoint).items():
        assert torch.equal(saved.model_state[name], value)


def test_checkpoint_names_say_whether_a_stage_finished(repo_root):
    """A finished stage writes the final checkpoint; a stopped one names the step reached."""
    declared = sorted(path.parent.name for path in (repo_root / "experiments").glob("*/experiment.yaml"))
    continuing = [config.load_experiment(name) for name in declared]
    assert any(experiment.depends_on is not None for experiment in continuing)
    assert training.checkpoint_name(3000, 3000) == training.CHECKPOINT_NAME
    assert training.checkpoint_name(2999, 3000) == "step_00002999"


def test_existing_output_directory_is_not_overwritten_silently(tmp_path):
    """A directory that holds results raises unless overwrite is set."""
    (tmp_path / "config.yaml").write_text("x: 1\n")
    with pytest.raises(FileExistsError):
        training.prepare_out_dir(tmp_path, overwrite=False)


@pytest.mark.parametrize(("cores", "workers"), [(1, 0), (2, 1), (8, training.PREFETCH_WORKERS)])
def test_prefetch_workers_feed_a_gpu_only(monkeypatch, cores, workers):
    """CPU training builds batches in line; a GPU gets workers, leaving one core to drive it."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cores)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    assert training.prefetch_workers("cpu") == 0
    assert training.prefetch_workers("cuda") == workers


def test_a_stage_fed_by_a_worker_process_trains_the_same_network(tmp_path, monkeypatch):
    """Batches built ahead by a worker process leave the trained parameters unchanged."""
    cfg = _config("pretraining.steps=4", "pretraining.eval_every=4")
    in_line = training.run_stage(cfg, Stage.PRETRAINING, seed=0, init_from=None, out_dir=tmp_path / "line", device="cpu")
    monkeypatch.setattr(training, "prefetch_workers", lambda device: 1)
    ahead = training.run_stage(cfg, Stage.PRETRAINING, seed=0, init_from=None, out_dir=tmp_path / "ahead", device="cpu")
    for name, value in _weights(in_line.checkpoint).items():
        assert torch.equal(_weights(ahead.checkpoint)[name], value)
    assert pd.read_parquet(in_line.out_dir / "train_metrics.parquet").equals(
        pd.read_parquet(ahead.out_dir / "train_metrics.parquet")
    )
