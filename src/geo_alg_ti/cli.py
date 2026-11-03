"""One executable with one subcommand per step of the reproduction path.

`pretrain` and `finetune` run the two training stages, `evaluate` runs the
test battery over the reasoning-step sweep, `represent` fits rank probes and
applies rank-direction edits, `ablate` ranks the attention heads by how much
they attend to the context while the network reasons and scores the battery
with heads replaced by their mean outputs, `runs` turns an experiment
declaration into an array size, its stages, or one resolved configuration,
`export` writes the tidy tables, `figure` redraws a figure's panels, `numbers`
recomputes the results the paper prints about the transformers and writes them
to `results/numbers.csv`, and `smoke` runs the whole path at laptop scale. Every
subcommand but `runs` refuses to start on a cluster login node.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.util
import os
import socket
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from geo_alg_ti import battery, components, config, evaluate, orderings, represent, tables, training
from geo_alg_ti.model import ModelSpec, RelationalTransformer
from geo_alg_ti.sequences import Vocabulary
from geo_alg_ti.streams import Stage, training_cue_weights

LOGIN_NODE_BYPASS_ENV = "GEO_ALG_TI_ALLOW_LOGIN_NODE"

# An experiment declaring one of these stages trains its own network and is scored on it.
_TRAINING_STAGES: frozenset[str] = frozenset({"pretrain", "finetune"})

# The experiments the smoke path runs, and the directory under `--out` that holds its results.
SMOKE_PRETRAINING: str = "canonical_pretraining"
SMOKE_FINETUNING: str = "symbolic_distance"
SMOKE_DIRECTORY: str = "smoke"
SMOKE_FIGURE: str = "symbolic_distance"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to one subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in _GUARDED:
        assert_not_login_node()
    _, handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except (ValueError, IndexError, FileNotFoundError, FileExistsError) as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the executable and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="geo-alg-ti",
        description="Transformer experiments for the paper 'Geometric and algebraic inference in humans and neural networks'.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for verb, (summary, _) in _COMMANDS.items():
        subcommand = subcommands.add_parser(verb, help=summary, description=summary)
        if verb == "numbers":
            add_numbers_arguments(subcommand)
            continue
        add_config_arguments(subcommand)
        if verb == "runs":
            # One of these modes at a time; --index comes from the shared arguments.
            modes = subcommand.add_mutually_exclusive_group()
            modes.add_argument("--count", action="store_true", help="print how many runs the experiment declares")
            modes.add_argument("--stages", action="store_true", help="print the stages the experiment declares, in order")
            modes.add_argument(
                "--covered",
                type=Path,
                default=None,
                metavar="MANIFEST",
                help="check that an export's manifest covers exactly the declared runs, intact",
            )
            subcommand.add_argument(
                "--print",
                dest="print_config",
                action="store_true",
                help="print the resolved configuration of the run at --index",
            )
        if verb == "figure":
            subcommand.add_argument("--name", required=True, help="the figure topic: figures/fig_<name>.py")
            subcommand.set_defaults(out=None)
    return parser


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared arguments: `--config`, repeatable `--set section.field=value`,
    `--experiment`, `--index`, `--seed`, `--out`, `--device`, `--overwrite`.
    """
    parser.add_argument("--config", type=Path, default=None, help="configuration file applied as a delta on configs/base.yaml")
    parser.add_argument(
        "--set",
        action="append",
        metavar="section.field=value",
        help="override one setting; repeat to override several, applied left to right",
    )
    parser.add_argument("--experiment", default=None, help="experiment declared under experiments/")
    parser.add_argument("--index", type=int, default=None, help="array task index, resolved to one run of the experiment")
    parser.add_argument("--seed", type=int, default=None, help="run this seed alone")
    parser.add_argument("--out", type=Path, default=Path("results"), help="directory results are written under")
    parser.add_argument("--device", default="auto", help="torch device, or auto to choose one")
    parser.add_argument("--overwrite", action="store_true", help="replace results already written for this run")


def add_numbers_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the `numbers` arguments: `--results`, the directory holding the tables, and `--out`,
    the file the table is written to.
    """
    parser.add_argument("--results", type=Path, default=None, help="directory holding <experiment>/tables/ (default: results/)")
    parser.add_argument("--out", type=Path, default=None, help="file the table is written to (default: numbers.csv under --results)")


def assert_not_login_node() -> None:
    """Exit on a login node (host `login*`, no SLURM job) unless the bypass variable is set."""
    if os.environ.get(LOGIN_NODE_BYPASS_ENV):
        return
    if os.environ.get("SLURM_JOB_ID"):
        return
    if socket.gethostname().startswith("login"):
        raise SystemExit("this command belongs on a compute node; submit it through the scheduler")


def _resolve_config(args: argparse.Namespace) -> config.Config:
    """The configuration a subcommand runs: the base, a file delta, then the overrides in order."""
    overrides = list(args.set or ())
    if args.seed is not None:
        overrides.append(f"seeds=[{args.seed}]")
    return config.load_config(args.config, tuple(overrides))


def _cmd_train(args: argparse.Namespace, stage: Stage) -> int:
    """Run one training stage for one resolved run: pretraining, or finetuning from upstream."""
    _, run = _resolve_run(args)
    summary = _train_run(run, stage=stage, out=args.out, device=args.device, overwrite=args.overwrite)
    print(f"{summary.stage} finished at step {summary.steps} in {summary.out_dir} after {summary.wall_time_s:.0f} s")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the test battery on a checkpoint over the reasoning-step sweep."""
    declaration, run = _resolve_run(args)
    written = _evaluate_run(declaration, run, out=args.out, device=args.device)
    print(f"the test battery wrote {written}")
    return 0


def cmd_represent(args: argparse.Namespace) -> int:
    """Fit rank probes by layer and apply the erase and write edits with their controls."""
    declaration, run = _resolve_run(args)
    written = _represent_run(declaration, run, out=args.out, device=args.device)
    print(f"the rank probes and edits wrote {written}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """Rank the heads by their attention to the context; score the battery with heads ablated."""
    declaration, run = _resolve_run(args)
    written = _ablate_run(declaration, run, out=args.out, device=args.device)
    print(f"the ablations wrote {written}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """Print an experiment's run count, its stages, its run ids, or one resolved configuration,
    or check that an exported manifest covers exactly the runs the experiment declares.
    """
    if args.experiment is None:
        raise SystemExit("runs needs an experiment: geo-alg-ti runs --experiment <name>")
    base = _resolve_config(args)
    experiment = config.load_experiment(args.experiment)
    if args.stages:
        print(" ".join(experiment.stages))
        return 0
    if args.covered is not None:
        tables.check_export(args.covered, [run.run_id for run in config.enumerate_runs(experiment, base)])
        return 0
    if args.count:
        print(config.n_runs(experiment, base))
        return 0
    if args.index is not None:
        run = config.run_for_index(experiment, args.index, base)
        if args.print_config:
            print(yaml.safe_dump(config.to_dict(run.config), sort_keys=False), end="")
        else:
            print(run.run_id)
        return 0
    for run in config.enumerate_runs(experiment, base):
        print(run.run_id)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Turn an experiment's run directories into tidy tables and a manifest."""
    if args.experiment is None:
        raise SystemExit("export needs an experiment: geo-alg-ti export --experiment <name>")
    base = _resolve_config(args)
    for path in tables.export_experiment(args.experiment, results_root=args.out, base=base).values():
        print(path)
    return 0


def cmd_figure(args: argparse.Namespace) -> int:
    """Regenerate one figure's panels from the tidy tables.

    Without `--out`, the figure script `figures/fig_<name>.py` reads the committed tables and
    writes to `figures/out/`, as it does when run directly; with `--out`, it reads
    `<--out>/<experiment>/tables/` and writes beside the experiments, under `<--out>/figures/`.
    """
    for path in draw_figure(args.name, results_root=args.out):
        print(path)
    return 0


def cmd_numbers(args: argparse.Namespace) -> int:
    """Recompute the results the paper prints about the transformers and write the table.

    Exits non-zero, naming each one, when a recomputed number differs from the printed one.
    """
    from geo_alg_ti import numbers

    results = args.results or numbers.RESULTS_ROOT
    table = numbers.numbers(results)
    path = args.out or (results / "numbers.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    differ = table[~table["match"]]
    print(f"{len(table)} numbers, {len(table) - len(differ)} as printed; wrote {path}")
    for row in differ.itertuples():
        print(f"  {row.id} ({row.site}): printed {row.printed}, computed {row.computed}")
    return 1 if len(differ) else 0


def figure_directory(results_root: Path | None) -> Path:
    """`figures/out/` for the committed tables; otherwise a `figures/` directory under the root."""
    return _load_figure_module("_shared").OUT_DIR if results_root is None else Path(results_root) / "figures"


def draw_figure(name: str, *, results_root: Path | None) -> list[Path]:
    """Build one figure script's figure from the tables under `results_root` and save it."""
    shared = _load_figure_module("_shared")
    script = _load_figure_module(f"fig_{name}")
    tables_root = shared.RESULTS_ROOT if results_root is None else Path(results_root)
    out_dir = figure_directory(results_root)
    fig = script.build(argparse.Namespace(results=tables_root, out=out_dir))
    return shared.save(fig, name, out_dir=out_dir)


def _load_figure_module(stem: str):
    """Import `figures/<stem>.py` by path; the figure scripts import their helpers as `_shared`."""
    path = config.repo_root() / "figures" / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no figure script at {path}")
    if stem == "_shared" and stem in sys.modules:
        return sys.modules[stem]
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    if stem != "_shared" and "_shared" not in sys.modules:
        _load_figure_module("_shared")
    spec.loader.exec_module(module)
    if stem == "_shared":
        # Registered once it has loaded, so a failed load is not mistaken for the helpers.
        sys.modules[stem] = module
    return module


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run pretraining, finetuning, the test battery, the export, and one figure at laptop scale.

    The path runs the canonical pretraining experiment and the symbolic-distance experiment that
    continues it, under `<--out>/smoke/`, replacing whatever an earlier call left there, and
    ends by drawing the figure "Symbolic distance effects (SDE)" into `<--out>/smoke/figures/`.
    """
    base = _resolve_config(args)
    root = Path(args.out) / SMOKE_DIRECTORY
    _, pretraining = _declared_run(SMOKE_PRETRAINING, base)
    finetuning_experiment, finetuning = _declared_run(SMOKE_FINETUNING, base)
    _train_run(pretraining, stage=Stage.PRETRAINING, out=root, device=args.device, overwrite=True)
    _train_run(finetuning, stage=Stage.FINETUNING, out=root, device=args.device, overwrite=True)
    _evaluate_run(finetuning_experiment, finetuning, out=root, device=args.device)
    for name in (SMOKE_PRETRAINING, SMOKE_FINETUNING):
        for path in tables.export_experiment(name, results_root=root, base=base).values():
            print(path)
    for path in draw_figure(SMOKE_FIGURE, results_root=root):
        print(path)
    return 0


# Every subcommand in the order `--help` lists them: its one-line summary and its handler.
_COMMANDS: Mapping[str, tuple[str, Callable[[argparse.Namespace], int]]] = {
    "pretrain": (
        "train one network from scratch on the pretraining distribution",
        functools.partial(_cmd_train, stage=Stage.PRETRAINING),
    ),
    "finetune": (
        "train one pretrained network on a fixed ordering",
        functools.partial(_cmd_train, stage=Stage.FINETUNING),
    ),
    "evaluate": ("run the test battery over the reasoning-step sweep", cmd_evaluate),
    "represent": ("fit rank probes by layer and apply the rank-direction edits", cmd_represent),
    "ablate": ("rank attention heads by their attention to the context; score the battery with heads ablated", cmd_ablate),
    "runs": ("print an experiment's run count, stages, identifiers, or one configuration", cmd_runs),
    "export": ("write an experiment's tidy tables and manifest", cmd_export),
    "figure": ("redraw one figure's panels from the tidy tables", cmd_figure),
    "numbers": ("recompute the results the paper prints about the transformers", cmd_numbers),
    "smoke": ("run the whole path at laptop scale", cmd_smoke),
}
VERBS: tuple[str, ...] = tuple(_COMMANDS)
# Everything but `runs` trains, evaluates, draws, or recomputes, and belongs on a compute node.
_GUARDED: frozenset[str] = frozenset(VERBS) - {"runs"}


def _resolve_run(args: argparse.Namespace) -> tuple[config.Experiment | None, config.Run]:
    """The experiment and the one run a subcommand acts on, from `--experiment` and `--index`."""
    base = _resolve_config(args)
    if args.experiment is None:
        return None, _undeclared_run(base)
    return _declared_run(args.experiment, base, args.index)


def _declared_run(name: str, base: config.Config, index: int | None = None) -> tuple[config.Experiment, config.Run]:
    """One run of a declared experiment: the run at `--index`, or the only run it declares."""
    experiment = config.load_experiment(name)
    if index is not None:
        return experiment, config.run_for_index(experiment, index, base)
    declared = config.enumerate_runs(experiment, base)
    if len(declared) != 1:
        raise ValueError(f"experiment '{name}' declares {len(declared)} runs; choose one with --index")
    return experiment, declared[0]


def _undeclared_run(cfg: config.Config) -> config.Run:
    """The run a configuration file describes on its own, outside any experiment.

    A run acts on one seed; a declared run always resolves to one, so the check lives here.
    """
    if len(cfg.seeds) != 1:
        raise ValueError(f"a run acts on one seed, not {len(cfg.seeds)}; choose one with --seed")
    seed = int(cfg.seeds[0])
    return config.Run(
        experiment="",
        run_id=f"seed-{config.format_axis_value(seed)}",
        index=0,
        axis_values={"seed": seed},
        config=cfg,
        depends_on_experiment=None,
        depends_on_run_id=None,
    )


def _train_run(run: config.Run, *, stage: Stage, out: Path, device: str, overwrite: bool) -> training.RunSummary:
    """Train one stage of one run into its own directory; finetuning continues the upstream run."""
    stage = Stage(stage)
    init_from = _upstream_checkpoint(run, out) if stage is Stage.FINETUNING else None
    return training.run_stage(
        run.config,
        stage,
        seed=run.axis_values["seed"],
        init_from=init_from,
        out_dir=config.run_directory(out, run.experiment, run.run_id),
        device=device,
        overwrite=overwrite,
        experiment=run.experiment,
    )


@dataclasses.dataclass(frozen=True)
class _ScoredNetwork:
    """One run's checkpoint loaded for scoring, with the law and the test battery that score it."""

    cfg: config.Config
    seed: int
    out_dir: Path
    checkpoint: training.Checkpoint
    vocab: Vocabulary
    model: RelationalTransformer
    law: orderings.OrderingLaw
    tests: battery.Battery
    preset: str


def _scored_network(
    experiment: config.Experiment | None, run: config.Run, *, out: Path, device: str, spec: ModelSpec | None = None
) -> _ScoredNetwork:
    """Load the checkpoint `_checkpoint_to_score` names and build the battery it is scored on.

    The network is built from `spec`, or from the spec its configuration gives when None.
    """
    cfg = run.config
    seed = run.axis_values["seed"]
    out_dir = config.run_directory(out, run.experiment, run.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    training.set_determinism(seed)
    where = training.resolve_device(device)
    checkpoint = training.load_checkpoint(_checkpoint_to_score(experiment, run, out), map_location=where)
    vocab = training.build_vocabulary(cfg)
    model = RelationalTransformer(spec or training.build_spec(cfg, vocab), vocab).to(where)
    model.load_state_dict(checkpoint.model_state)
    law = training.build_law(cfg, seed)
    # The fixed ordering(s) exist for a network that was finetuned on them, and only then.
    design = training.build_design(cfg, law, seed=seed) if checkpoint.stage is Stage.FINETUNING else None
    tests = battery.Battery.from_config(cfg, law, vocab, seed=seed, stage=checkpoint.stage, design=design)
    preset = battery.preset_name(cfg, checkpoint.stage)
    return _ScoredNetwork(cfg, seed, out_dir, checkpoint, vocab, model, law, tests, preset)


def _evaluate_run(experiment: config.Experiment | None, run: config.Run, *, out: Path, device: str) -> Path:
    """Score one run's checkpoint on its test battery and write the run's evaluation tables.

    The declaration decides which checkpoint that is; see `_checkpoint_to_score`.
    """
    network = _scored_network(experiment, run, out=out, device=device)
    checkpoint, out_dir = network.checkpoint, network.out_dir
    trials, aggregates = evaluate.evaluate_battery(
        network.model,
        network.tests,
        network.vocab,
        step=checkpoint.step,
        tag=str(checkpoint.stage),
        chain_algorithm=network.cfg.task.chain_algorithm,
    )
    tables.write_run_table(out_dir, "eval", checkpoint.step, aggregates)
    tables.write_run_table(out_dir, "eval_pairs", checkpoint.step, evaluate.aggregate_by_pair(trials))
    tables.write_run_table(out_dir, tables.TRIALS_TABLE, checkpoint.step, trials)
    return out_dir / "eval"


def _represent_run(experiment: config.Experiment | None, run: config.Run, *, out: Path, device: str) -> Path:
    """Probe one run's checkpoint by site and score it under every rank-direction edit.

    The checkpoint is the one `_checkpoint_to_score` names. Ranks are the frequent cue's over
    the whole pool. The probe table holds one row per site and metric; the edit table holds
    the content battery's aggregates under every erase gain and write strength, along the rank
    direction fit at every edit site and along the random control, at every reasoning-step
    value.
    """
    network = _scored_network(experiment, run, out=out, device=device)
    cfg, seed, out_dir, checkpoint = network.cfg, network.seed, network.out_dir, network.checkpoint
    vocab, model, law, tests = network.vocab, network.model, network.law, network.tests
    cue = orderings.FREQUENT_CUE
    ranks = law.ranks(cue, range(law.pool_size))
    prompts = represent.fit_prompts(law, vocab, cue, n_prompts=represent.N_FIT_PROMPTS, seed=seed)
    probe = represent.probe_by_layer(model, prompts, vocab, ranks=ranks, layers=range(model.n_sites), seed=seed)
    tables.write_run_table(out_dir, "probe", checkpoint.step, represent.probe_table(probe, seed=seed, step=checkpoint.step))
    rank_directions, calibrations = represent.fit_edit_directions(model, prompts, vocab, ranks=ranks)
    specs = represent.edit_specs(rank_directions, calibrations, ranks, seed=seed)
    edited = represent.edit_table(
        model, tests, vocab, specs, step=checkpoint.step, chain_algorithm=cfg.task.chain_algorithm
    )
    tables.write_run_table(out_dir, "edit", checkpoint.step, edited)
    return out_dir / "edit"


def _ablate_run(experiment: config.Experiment | None, run: config.Run, *, out: Path, device: str) -> Path:
    """Rank one run's attention heads and score its checkpoint under every set of ablated heads.

    The heads are ranked by their attention on the chains the intact network writes on the test
    battery's conditions whose answer needs the context; the means come from rollouts of the
    pretraining distribution the network was trained on. Writes the head-score and ablate tables.
    A network with too few heads for every set of `components.ablation_sets`, counted from the
    configuration, is refused before anything is loaded, scored, or written.
    """
    spec = training.build_spec(run.config, training.build_vocabulary(run.config))
    components.check_set_sizes(spec.n_layers * spec.n_heads)
    network = _scored_network(experiment, run, out=out, device=device, spec=spec)
    cfg, checkpoint = network.cfg, network.checkpoint
    scores, means = components.reading_heads(
        network.model,
        network.tests,
        network.vocab,
        network.law,
        conditions=components.chain_conditions(network.preset),
        cue_weights=training_cue_weights(cfg.task),
        seed=network.seed,
    )
    head_scores = components.head_scores_table(scores, seed=network.seed, step=checkpoint.step)
    tables.write_run_table(network.out_dir, "head_scores", checkpoint.step, head_scores)
    ablate = components.ablate_table(
        network.model,
        network.tests,
        network.vocab,
        means,
        scores,
        step=checkpoint.step,
        chain_algorithm=cfg.task.chain_algorithm,
    )
    tables.write_run_table(network.out_dir, "ablate", checkpoint.step, ablate)
    return network.out_dir / "ablate"


def _checkpoint_to_score(experiment: config.Experiment | None, run: config.Run, out: Path) -> Path:
    """The checkpoint a run is scored on, decided from the declaration rather than the disk.

    A run that trains, under an experiment with a training stage or outside any experiment, is
    scored on its own final checkpoint; a run that stopped short of its last step is refused, so
    the network it started from is never scored in its place. An experiment with no training
    stage is scored on the final checkpoint of the run its dependency names.
    """
    if experiment is not None and not _TRAINING_STAGES.intersection(experiment.stages):
        return _upstream_checkpoint(run, out)
    path = config.checkpoint_path(
        config.run_directory(out, run.experiment, run.run_id), training.CHECKPOINT_NAME
    )
    if path.is_file():
        return path
    partial = sorted(path.parent.glob("step_*.pt"))
    if partial:
        reached = int(partial[-1].stem.removeprefix("step_"))
        raise FileNotFoundError(
            f"no checkpoint at {path}: the run stopped at step {reached} ({partial[-1].name}); "
            "train it to its last step before scoring it"
        )
    raise FileNotFoundError(f"no checkpoint at {path}: the run was never trained")


def _upstream_checkpoint(run: config.Run, out: Path) -> Path:
    """The final checkpoint of the upstream run the declared dependency names."""
    if run.depends_on_experiment is None or run.depends_on_run_id is None:
        raise ValueError("this run continues a trained network; name an experiment that declares which one")
    upstream = config.run_directory(out, run.depends_on_experiment, run.depends_on_run_id)
    path = config.checkpoint_path(upstream, training.CHECKPOINT_NAME)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}; run the '{run.depends_on_experiment}' experiment first")
    return path
