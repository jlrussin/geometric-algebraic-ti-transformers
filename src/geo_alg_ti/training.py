"""One training loop for both stages, with evaluation hooks and checkpoints.

Realizes the extended methods' "Architecture and optimization" paragraph:
Adam with betas (0.9, 0.95), batches of 64, a constant learning rate, no
weight decay, gradient norm clipped at 1. Pretraining runs 100,000 updates from scratch; finetuning
continues from a named pretraining checkpoint for 1,600 updates at learning
rate 0.0001 with answer-only targets and replay. On a GPU, worker processes
build the batches ahead of the network. The test battery runs before the
first update, at a fixed cadence, and at the final step. Determinism is
enforced, and every run directory records the resolved configuration and a
fingerprint: `config.yaml`, `fingerprint.json`, `checkpoints/`,
`train_metrics.parquet`, `eval/step_*.parquet`, `eval_pairs/step_*.parquet`,
`trials/step_*.parquet`, `log.jsonl`. The log also records how long each
evaluation took and how long the stage had run at each logged step, and the
fingerprint's wall time covers the whole stage. A stage that stops before its
last step keeps the training curve so far and a checkpoint named by the step
it reached rather than `final`. A directory that already holds results is
never overwritten unless `--overwrite` is given.

Finetuning can let the network reason before it answers: a trial marked to
reason is rewritten, before the update, as its prompt, the reasoning steps the
network writes greedily itself, and the answer emission, so the answer is
scored after those steps exactly as the test battery scores it. Finetuning can
also ablate attention heads on every fixed-ordering trial, replayed pretraining
trials left intact: the heads that attend most to the context while the
network starting the stage reasons, or a random set, each replaced by its
mean output on that network. With ablated scoring, every evaluation also
scores the battery with those heads ablated (`ablated_eval/`,
`ablated_eval_pairs/`). A finetuning run with any of these settings also writes
the answer accuracy of its fixed-ordering trials by queried pair over each
logging window (`train_pairs.parquet`) and the per-pair table at every
evaluation; a run with none of them writes its per-pair table at its final
evaluation only.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from geo_alg_ti import battery, components, config, evaluate, tables, wandb_mirror
from geo_alg_ti.chains import PAD_TO
from geo_alg_ti.config import Config
from geo_alg_ti.model import LossTerms, ModelOutput, ModelSpec, RelationalTransformer, greedy_rollout, loss_terms
from geo_alg_ti.orderings import OrderingLaw
from geo_alg_ti.sequences import EMISSION_LEN, Target, Vocabulary, scored_answer_index, scoring_length, sequence_length
from geo_alg_ti.streams import (
    Batch,
    FinetuningDesign,
    Layout,
    RowKind,
    Stage,
    build_finetuning_design,
    design_rng,
    finetuning_stream,
    pretraining_stream,
    stage_section,
    training_cue_weights,
)
from geo_alg_ti.trials import ITEMS_PER_ORDERING

ADAM_BETAS: tuple[float, float] = (0.9, 0.95)
WEIGHT_DECAY: float = 0.0
GRAD_CLIP: float = 1.0
CHECKPOINT_NAME: str = "final"
# Processes that build batches ahead of a GPU; fewer when the allocation has fewer cores to spare.
PREFETCH_WORKERS: int = 2
# The columns of `train_metrics.parquet`, in the order `train_metrics` fills them.
TRAIN_METRICS: tuple[str, ...] = (
    "loss",
    "token_cross_entropy",
    "scalar_squared_error",
    "answer_accuracy",
    "chain_token_accuracy",
)


@dataclass
class RunSummary:
    """What a finished stage produced, and how many seconds it took."""

    out_dir: Path
    stage: Stage
    steps: int
    checkpoint: Path
    wall_time_s: float


@dataclass(frozen=True)
class HeadAblation:
    """The heads a finetuning run ablates, ranked and averaged on the network the stage starts from.

    With `during_training`, the heads are replaced on every fixed-ordering trial of the stage,
    through `rows`; with ablated scoring, every evaluation also scores the battery with them
    replaced.
    """

    heads: tuple[components.Head, ...]
    means: components.Means
    scores: components.HeadScores
    rows: components.RowAblation
    during_training: bool

    @property
    def label(self) -> str:
        """The heads' labels joined in order of attention rank, as the tables name them."""
        return "+".join(one.label for one in self.heads)


@dataclass
class Checkpoint:
    """A saved model with its optimizer state, resolved configuration, and step."""

    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    config: Config
    step: int
    stage: Stage


def set_determinism(seed: int) -> None:
    """Seed every generator and enable deterministic algorithms."""
    # A deterministic reduction workspace has to be chosen before the first matrix product.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_law(cfg: Config, seed: int) -> OrderingLaw:
    """The cued-ordering law one seed's training data and test battery are drawn from."""
    return OrderingLaw(
        seed=int(seed),
        pool_size=cfg.task.item_pool_size,
        n_training_cues=cfg.task.n_training_cues,
        n_reserved_cues=cfg.task.n_reserved_cues,
    )


def build_vocabulary(cfg: Config) -> Vocabulary:
    """Vocabulary for the configured pool and cue counts (training plus reserved cues)."""
    return Vocabulary(
        pool_size=cfg.task.item_pool_size,
        n_cues=cfg.task.n_training_cues + cfg.task.n_reserved_cues,
        show_cue=cfg.task.show_cue,
    )


def build_spec(cfg: Config, vocab: Vocabulary) -> ModelSpec:
    """The network's architecture from the model section of the configuration."""
    return ModelSpec(
        vocab_size=vocab.size,
        max_len=_max_len(cfg),
        n_layers=cfg.model.n_layers,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        feedforward=cfg.model.feedforward == "ffn",
    )


def build_model(cfg: Config, vocab: Vocabulary) -> RelationalTransformer:
    """Instantiate the network `build_spec` describes."""
    return RelationalTransformer(build_spec(cfg, vocab), vocab)


def build_optimizer(model: RelationalTransformer, cfg: Config, stage: Stage) -> torch.optim.Optimizer:
    """Adam with the stage's learning rate, betas (0.9, 0.95), and no weight decay."""
    learning_rate = stage_section(cfg, stage).learning_rate
    return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)


def train_step(
    model: RelationalTransformer,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    *,
    grad_clip: float = GRAD_CLIP,
    ablation: HeadAblation | None = None,
) -> tuple[ModelOutput, LossTerms]:
    """One update: forward, loss, clipped backward, step.

    Heads ablated during training are replaced on the batch's fixed-ordering rows.
    Returns the network's output and the detached loss terms without bringing either to the host,
    so a step never waits on the device; `train_metrics` reads them when a step is logged.
    """
    model.train()
    scope: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    if ablation is not None and ablation.during_training:
        if batch.row_kind is None or batch.reasoning is None:
            raise ValueError("a batch without row kinds cannot say which rows the ablation acts on")
        fixed = batch.row_kind != int(RowKind.PRETRAINING)
        scope = components.ablated_rows(
            model,
            ablation.rows,
            rows=fixed,
            query_at=batch.answer_index - 2,
            starts=batch.reasoning.query_starts,
        )
    with scope:
        out = model(batch.tokens, batch.scalars)
    terms = loss_terms(out, batch)
    optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    detached = LossTerms(
        token_cross_entropy=terms.token_cross_entropy.detach(),
        scalar_squared_error=terms.scalar_squared_error.detach(),
        total=terms.total.detach(),
    )
    return ModelOutput(logits=out.logits.detach(), scalar_pred=out.scalar_pred.detach()), detached


def train_metrics(out: ModelOutput, terms: LossTerms, batch: Batch) -> dict[str, float]:
    """The logged scalars of one update, brought to the host in a single transfer."""
    with torch.no_grad():
        values = torch.stack(
            [
                terms.total,
                terms.token_cross_entropy,
                terms.scalar_squared_error,
                answer_accuracy(out, batch),
                chain_token_accuracy(out, batch),
            ]
        )
    return dict(zip(TRAIN_METRICS, values.tolist(), strict=True))


def with_reasoning(model: RelationalTransformer, batch: Batch, *, ablation: HeadAblation | None = None) -> Batch:
    """The batch with every row marked to reason rewritten to answer after its own reasoning.

    Such a row becomes its prompt, the reasoning steps the network writes greedily from it, and
    the answer emission, with the loss on the answer's relation and separator alone, so the
    answer is scored at the position the test battery reads after that many steps. The steps
    are written without gradients, with the heads replaced when they are ablated during
    training; every other row is unchanged, and the batch is padded to its longest row. Which
    rows reason is read from the batch's plan, decided on the host, so nothing waits on the
    device.
    """
    plan = batch.reasoning
    if plan is None or plan.steps == 0 or plan.rows.numel() == 0:
        return batch
    rows, steps, prompt_len = plan.rows, plan.steps, plan.prompt_len
    prompt_tokens = batch.tokens[rows, :prompt_len]
    prompt_scalars = batch.scalars[rows, :prompt_len]
    answer_tokens = batch.tokens[rows, prompt_len : prompt_len + EMISSION_LEN]
    answer_scalars = batch.scalars[rows, prompt_len : prompt_len + EMISSION_LEN]
    scope: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    if ablation is not None and ablation.during_training:
        scope = components.ablated(model, ablation.heads, ablation.means)
    with scope:
        generated = greedy_rollout(model, prompt_tokens, prompt_scalars, answer_tokens[:, :2], max_emissions=steps)
    length = prompt_len + EMISSION_LEN * (steps + 1)
    width = max(int(batch.tokens.shape[1]), length)

    def widened(tensor: torch.Tensor, fill: float | bool) -> torch.Tensor:
        grown = tensor.new_full((tensor.shape[0], width), fill)
        grown[:, : tensor.shape[1]] = tensor
        return grown

    tokens = widened(batch.tokens, model.vocab.pad)
    scalars = widened(batch.scalars, 0.0)
    token_loss = widened(batch.token_loss, False)
    scalar_loss = widened(batch.scalar_loss, False)
    tokens[rows] = model.vocab.pad
    scalars[rows] = 0.0
    tokens[rows, :length] = torch.cat((prompt_tokens, generated.tokens, answer_tokens), dim=1)
    scalars[rows, :length] = torch.cat((prompt_scalars, generated.scalars.to(scalars.dtype), answer_scalars), dim=1)
    answer_at = scored_answer_index(prompt_len, steps)
    token_loss[rows] = False
    token_loss[rows, answer_at : answer_at + 2] = True
    scalar_loss[rows] = False
    scalar_loss[rows, answer_at] = True
    answer_index = batch.answer_index.clone()
    answer_index[rows] = answer_at
    return Batch(
        tokens=tokens,
        scalars=scalars,
        token_loss=token_loss,
        scalar_loss=scalar_loss,
        answer_index=answer_index,
        index=batch.index,
        row_kind=batch.row_kind,
        query_ranks=batch.query_ranks,
        reasoning_steps=batch.reasoning_steps,
        reasoning=plan,
    )


def build_head_ablation(
    cfg: Config, model: RelationalTransformer, law: OrderingLaw, vocab: Vocabulary, *, seed: int, budgets: tuple[int, ...]
) -> HeadAblation | None:
    """The heads a finetuning run ablates or scores ablated, from the network the stage starts at.

    The heads are ranked by their attention to the context on the chains the network writes on
    the in-context conditions of the pretraining stage's own battery preset, whatever preset the
    run is scored on; the means come from its rollouts of the pretraining distribution, to every
    reasoning-step value of `budgets`. A run that ablates `random` heads draws them from outside
    the top twice as many, keyed by the seed; any other run takes the heads ranked highest. None
    when the run neither ablates nor scores ablated. The configuration has already checked that
    the set fits the network.
    """
    section = cfg.finetuning
    if section.ablated_heads == "none" and not cfg.evaluation.ablated_scoring:
        return None
    # The pretraining preset is built directly: `Battery.from_config` would take
    # `evaluation.battery` first, and the heads are ranked on chains of the pretraining stage.
    preset = cfg.pretraining.battery
    conditions = battery.load_preset(preset, contexts_per_pair=cfg.evaluation.contexts_per_pair)
    weights = training_cue_weights(cfg.task)
    chains = battery.Battery.build(conditions, law, vocab, seed=int(seed), design=None, budgets=budgets, cue_weights=weights)
    scores, means = components.reading_heads(
        model, chains, vocab, law, conditions=components.chain_conditions(preset), cue_weights=weights, seed=int(seed)
    )
    size = int(section.n_ablated_heads)
    if section.ablated_heads == "random":
        chosen = components.random_set(scores.ranked, size, seed=int(seed), draw=0)
    else:
        chosen = tuple(scores.ranked[:size])
    return HeadAblation(
        heads=chosen,
        means=means,
        scores=scores,
        rows=components.RowAblation(model, chosen, means),
        during_training=section.ablated_heads != "none",
    )


def fixed_ordering_answers(out: ModelOutput, batch: Batch) -> torch.Tensor:
    """One row per row of the batch: its kind, reasoning steps, queried ranks, and whether the
    answer's sign was right, (rows, 5), kept on the device; `train_pairs_table` keeps the
    fixed-ordering rows."""
    if batch.row_kind is None or batch.query_ranks is None or batch.reasoning_steps is None:
        raise ValueError("a batch without row kinds holds no fixed-ordering trials to record")
    right = answer_right(out, batch)
    columns = (batch.row_kind, batch.reasoning_steps, batch.query_ranks[:, 0], batch.query_ranks[:, 1], right.long())
    return torch.stack(columns, dim=1)


def train_pairs_table(records: list[tuple[int, torch.Tensor]], *, seed: int, window: int, last: int) -> pd.DataFrame:
    """The per-pair answer accuracy of the fixed-ordering trials over each window of `window`
    updates, labelled by the window's last update (`last`, the last update the stage reached,
    closes the final one), brought to the host in one transfer."""
    if not records:
        return pd.DataFrame(columns=[name for name in tables.TRAIN_PAIRS_SCHEMA if name != "experiment"])
    values = torch.cat(
        [
            torch.cat((torch.full_like(answers[:, :1], min(-(-int(step) // window) * window, last)), answers), dim=1)
            for step, answers in records
        ]
    ).cpu().numpy()
    values = values[values[:, 1] != int(RowKind.PRETRAINING)]
    names = {int(kind): kind.name.lower() for kind in RowKind}
    answered = pd.DataFrame(
        {
            "step": values[:, 0],
            "context": [names[int(kind)] for kind in values[:, 1]],
            "reasoning_steps": values[:, 2],
            "rank_a": values[:, 3],
            "rank_b": values[:, 4],
            "right": values[:, 5],
        }
    )
    keys = ["step", "context", "reasoning_steps", "rank_a", "rank_b"]
    grouped = answered.groupby(keys, as_index=False)["right"].agg(["mean", "size"])
    grouped = grouped.rename(columns={"mean": "value", "size": "n"})
    grouped.insert(0, "seed", int(seed))
    return grouped


def run_stage(
    cfg: Config,
    stage: Stage,
    *,
    seed: int,
    init_from: Path | None,
    out_dir: Path,
    device: str = "auto",
    overwrite: bool = False,
    experiment: str = "",
) -> RunSummary:
    """Train one stage, evaluating at the configured cadence and writing the run directory.

    Each stage evaluates once before its first update, so its curve starts at the network the
    stage begins from. `experiment` names the run in the optional mirror; a run declared by no
    experiment is named by its run id alone.
    """
    stage = Stage(stage)
    out_dir = Path(out_dir)
    started = time.perf_counter()

    def write_fingerprint() -> None:
        elapsed = time.perf_counter() - started
        _write_json(out_dir / "fingerprint.json", config.fingerprint(cfg, run_id=out_dir.name, wall_time_s=elapsed))

    prepare_out_dir(out_dir, overwrite=overwrite)
    set_determinism(seed)
    where = resolve_device(device)

    vocab = build_vocabulary(cfg)
    model = build_model(cfg, vocab).to(where)
    if init_from is not None:
        model.load_state_dict(load_checkpoint(Path(init_from), map_location=where).model_state)
    # Each stage optimizes from scratch at its own learning rate.
    optimizer = build_optimizer(model, cfg, stage)

    law = build_law(cfg, seed)
    design = build_design(cfg, law, seed=seed) if stage is Stage.FINETUNING else None
    section = stage_section(cfg, stage)
    steps, eval_every = section.steps, section.eval_every
    workers = prefetch_workers(where)
    stream = (
        pretraining_stream(
            cfg.task, law, vocab, seed=int(seed), batch_size=cfg.pretraining.batch_size, device=where, workers=workers
        )
        if stage is Stage.PRETRAINING
        else finetuning_stream(
            cfg.task,
            law,
            vocab,
            design=design,
            seed=int(seed),
            batch_size=cfg.pretraining.batch_size,
            replay_proportion=cfg.finetuning.replay_prop,
            uninformative_proportion=cfg.finetuning.uninformative_prop,
            reasoning_proportion=cfg.finetuning.reasoning_prop,
            reasoning_steps=cfg.finetuning.reasoning_steps,
            device=where,
            workers=workers,
        )
    )
    tests = battery.Battery.from_config(cfg, law, vocab, seed=int(seed), stage=stage, design=design)
    finetuning = stage is Stage.FINETUNING
    ablation = build_head_ablation(cfg, model, law, vocab, seed=int(seed), budgets=tests.budgets) if finetuning else None
    # Reasoning or ablated heads while finetuning, or ablated scoring, also record the per-pair
    # table at every evaluation and the answers of the fixed-ordering trials by pair.
    by_pair = finetuning and (
        cfg.finetuning.reasoning_prop > 0 or cfg.finetuning.ablated_heads != "none" or cfg.evaluation.ablated_scoring
    )

    config.dump_config(cfg, out_dir / "config.yaml")
    write_fingerprint()
    mirror = wandb_mirror.start(cfg, out_dir=out_dir, name=f"{experiment}/{out_dir.name}" if experiment else out_dir.name)
    log = (out_dir / "log.jsonl").open("w", encoding="utf-8")
    # The digest names the battery this run scored, so a rebuilt battery can be checked against it.
    _write_line(log, {"event": "battery", "stage": str(stage), "digest": tests.digest(), "trials": len(tests.trials)})

    def evaluate_at(step: int) -> None:
        began = time.perf_counter()
        scored, aggregates = evaluate.evaluate_battery(
            model,
            tests,
            vocab,
            step=step,
            tag=str(stage),
            chain_algorithm=cfg.task.chain_algorithm,
            record_trials=step == steps or by_pair,
        )
        # The scores reach the host inside the call, so the clock covers the device work.
        duration = round(time.perf_counter() - began, 3)
        if ablation is not None and cfg.evaluation.ablated_scoring:
            began = time.perf_counter()
            with components.ablated(model, ablation.heads, ablation.means):
                ablated_trials, ablated_aggregates = evaluate.evaluate_battery(
                    model, tests, vocab, step=step, tag=str(stage), chain_algorithm=cfg.task.chain_algorithm
                )
            ablated_duration = round(time.perf_counter() - began, 3)
            labelled = ablated_aggregates.assign(heads=ablation.label)
            tables.write_run_table(out_dir, "ablated_eval", step, labelled)
            pairs = evaluate.aggregate_by_pair(ablated_trials).assign(heads=ablation.label)
            tables.write_run_table(out_dir, "ablated_eval_pairs", step, pairs)
            event = {"event": "ablated_evaluation", "stage": str(stage), "step": step, "heads": ablation.label}
            _write_line(log, {**event, "rows": len(labelled), "duration_s": ablated_duration})
            wandb_mirror.log_table(mirror, "ablated_eval", labelled, step=step)
        tables.write_run_table(out_dir, "eval", step, aggregates)
        if step == steps:
            tables.write_run_table(out_dir, tables.TRIALS_TABLE, step, scored)
        if step == steps or by_pair:
            # The per-pair table is aggregated from the recorded trials of the final evaluation,
            # or of every evaluation when the run records by pair.
            tables.write_run_table(out_dir, "eval_pairs", step, evaluate.aggregate_by_pair(scored))
        _write_line(
            log,
            {"event": "evaluation", "stage": str(stage), "step": step, "rows": len(aggregates), "duration_s": duration},
        )
        wandb_mirror.log_table(mirror, "eval", aggregates, step=step)

    rows: list[dict[str, Any]] = []
    answers: list[tuple[int, torch.Tensor]] = []
    reached = 0
    try:
        if ablation is not None:
            ranked = components.head_scores_table(ablation.scores, seed=int(seed), step=0)
            tables.write_run_table(out_dir, "head_scores", 0, ranked)
            _write_line(log, {"event": "ablated_heads", "heads": ablation.label, "during_training": ablation.during_training})
        # Each stage scores the network it starts from, before any update.
        evaluate_at(0)
        for step, batch in zip(range(1, steps + 1), stream, strict=False):
            if finetuning:
                batch = with_reasoning(model, batch, ablation=ablation)
            out, terms = train_step(model, optimizer, batch, ablation=ablation)
            if by_pair:
                answers.append((step, fixed_ordering_answers(out, batch)))
            reached = step
            if step % cfg.logging.log_every == 0 or step == steps:
                metrics = train_metrics(out, terms, batch)
                rows.append({"seed": int(seed), "stage": str(stage), "step": step, **metrics})
                elapsed = round(time.perf_counter() - started, 3)
                _write_line(log, {"event": "train", "stage": str(stage), "step": step, "elapsed_s": elapsed, **metrics})
                wandb_mirror.log_scalars(mirror, step, metrics)
            if step % eval_every == 0 or step == steps:
                evaluate_at(step)
    finally:
        stream.close()
        log.close()
        wandb_mirror.finish(mirror)
        # A run that stops early keeps the curve so far and the parameters at its last update.
        if rows:
            tables.write_train_table(out_dir, pd.DataFrame(rows))
        checkpoint = config.checkpoint_path(out_dir, checkpoint_name(reached, steps))
        save_checkpoint(checkpoint, model=model, optimizer=optimizer, cfg=cfg, step=reached, stage=stage)
        try:
            # The answers by pair come after the checkpoint, which a failure here cannot cost.
            if answers:
                pairs = train_pairs_table(answers, seed=int(seed), window=cfg.logging.log_every, last=reached)
                tables.write_train_table(out_dir, pairs, name="train_pairs")
        finally:
            # Written again so that the wall time covers the stage, whether it finished or stopped.
            write_fingerprint()
    return RunSummary(
        out_dir=out_dir, stage=stage, steps=steps, checkpoint=checkpoint, wall_time_s=time.perf_counter() - started
    )


def checkpoint_name(reached: int, steps: int) -> str:
    """`final` after the last step; a run that stopped short names its checkpoint by the step."""
    return CHECKPOINT_NAME if int(reached) == int(steps) else f"step_{int(reached):08d}"


def build_design(cfg: Config, law: OrderingLaw, *, seed: int) -> FinetuningDesign:
    """Draw the fixed ordering(s) a seed finetunes on, the same way in every process."""
    return build_finetuning_design(
        law,
        design_rng(seed),
        layout=Layout(cfg.finetuning.layout),
        n_orderings=cfg.finetuning.n_orderings,
    )


def resolve_device(device: str) -> str:
    """The device a run trains on; `auto` takes a GPU when one is present."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def prefetch_workers(device: str) -> int:
    """How many processes build batches ahead of the network on `device`.

    A GPU is fed by up to `PREFETCH_WORKERS` processes, leaving one core to drive it; on CPU
    the network needs the cores itself, so the batches are built in line.
    """
    if torch.device(device).type != "cuda":
        return 0
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(0, min(PREFETCH_WORKERS, cores - 1))


def save_checkpoint(path: Path, *, model: RelationalTransformer, optimizer: torch.optim.Optimizer, cfg: Config, step: int, stage: Stage) -> None:
    """Write a checkpoint embedding the resolved configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "step": int(step),
            "stage": str(Stage(stage)),
        },
        path,
    )


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> Checkpoint:
    """Read a checkpoint."""
    saved = torch.load(Path(path), map_location=map_location, weights_only=False)
    return Checkpoint(
        model_state=saved["model"],
        optimizer_state=saved["optimizer"],
        config=saved["config"],
        step=int(saved["step"]),
        stage=Stage(saved["stage"]),
    )


def prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    """Create the run directory; refuse one that already holds results unless `overwrite`."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} already holds results; run with overwrite to replace them")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _max_len(cfg: Config) -> int:
    """The longest sequence the network sees: a padded chain, or the longest scoring pass."""
    n_context = ITEMS_PER_ORDERING - 1
    return max(
        sequence_length(n_context, PAD_TO, Target.CHAIN),
        scoring_length(n_context, cfg.evaluation.max_reasoning_steps),
    )


def answer_right(out: ModelOutput, batch: Batch) -> torch.Tensor:
    """Per row, whether the predicted sign at the answer relation matches its target.

    A chain target repeats the answer emission up to the padded length; those repeats are copies
    of an emission the network was already given, and are not scored.
    """
    at = batch.answer_index[:, None]
    predicted = out.scalar_pred.gather(1, at - 1)
    target = batch.scalars.gather(1, at)
    return ((predicted > 0) == (target > 0)).flatten()


def answer_accuracy(out: ModelOutput, batch: Batch) -> torch.Tensor:
    """How often the predicted sign at the answer relation matches its target (`answer_right`)."""
    return answer_right(out, batch).float().mean()


def chain_token_accuracy(out: ModelOutput, batch: Batch) -> torch.Tensor:
    """How often the most likely token is the supervised one, over the chain emissions.

    The chain emissions are the supervised positions before the answer emission; an answer-only
    target has none, so a batch without a chain target scores nothing.
    """
    positions = torch.arange(batch.tokens.shape[1], device=batch.tokens.device)
    # The answer emission opens two positions before its relation slot.
    answer_opens = (batch.answer_index - 2)[:, None]
    mask = (batch.token_loss & (positions[None, :] < answer_opens))[:, 1:]
    predicted = out.logits[:, :-1].argmax(dim=-1)
    right = ((predicted == batch.tokens[:, 1:]) & mask).float().sum()
    return right / mask.float().sum()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    """Write one JSON document."""
    Path(path).write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")


def _write_line(handle: Any, record: dict[str, Any]) -> None:
    """Append one line to the run log."""
    handle.write(json.dumps(record, default=str) + "\n")
    handle.flush()

