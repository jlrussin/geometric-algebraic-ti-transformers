"""Configuration schema, delta loading, sweep enumeration, and run fingerprints.

Realizes the experiment configuration described in the methods: `configs/base.yaml`
is the canonical pretraining and finetuning configuration with its two-factor
test battery, and each directory under `experiments/` declares one paper
experiment as a named delta on that configuration plus the axes it sweeps.
Nothing is materialized ahead of time, so a resolved configuration and a seed
fully determine a run. `run_for_index` is the entry point of a cluster array task;
`run_id` is a pure function of the axis values and never of the array index.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from itertools import product
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from geo_alg_ti import __version__
from geo_alg_ti.chains import PAD_TO
from geo_alg_ti.trials import check_pool

# Attention heads per layer; the architecture fixes it, so the configuration names no key for it.
N_HEADS: int = 8

CueDistributionKind = Literal["uniform", "zipf", "frequent_cue"]
ChainAlgorithmName = Literal["main", "inferred_relations", "descend_first", "backtrack"]
StageName = Literal["pretrain", "finetune", "represent", "ablate", "evaluate"]


@dataclass(frozen=True)
class CueDistributionConfig:
    """How often each cue, and so each ordering, appears during pretraining.

    A Zipf distribution at skew alpha 0 is the uniform distribution, so `kind` is normalized to
    `uniform` in that case on every construction path, and no configuration names the uniform
    distribution twice.
    """

    kind: CueDistributionKind
    skew_alpha: float
    frequent_cue_prop: float

    def __post_init__(self) -> None:
        if self.kind == "zipf" and self.skew_alpha == 0:
            object.__setattr__(self, "kind", "uniform")


@dataclass(frozen=True)
class TaskConfig:
    """The task: item pool, cues, cue distribution, chain proportion and algorithm, cue display."""

    item_pool_size: int
    n_training_cues: int
    n_reserved_cues: int
    cue_distribution: CueDistributionConfig
    p_cot: float
    chain_algorithm: ChainAlgorithmName
    show_cue: bool


@dataclass(frozen=True)
class ModelConfig:
    """The transformer's depth, width, and feedforward sublayer."""

    n_layers: int
    d_model: int
    d_ff: int
    feedforward: Literal["ffn", "none"]


@dataclass(frozen=True)
class PretrainingConfig:
    """The pretraining schedule and its in-training evaluation."""

    steps: int
    batch_size: int
    learning_rate: float
    eval_every: int
    battery: str


@dataclass(frozen=True)
class FinetuningConfig:
    """The fixed-ordering schedule; the batch size follows pretraining.

    `uninformative_prop` is the proportion of fixed-ordering trials whose context shows other
    items instead of the ordering. `reasoning_prop` is the proportion of the others, whose
    context is congruent, on which the network first writes `reasoning_steps` reasoning steps of
    its own before the answer that is scored; the rest answer at once. `ablated_heads` replaces
    two (or `n_ablated_heads`) heads' outputs with their means on every fixed-ordering trial of
    the stage: the heads ranked highest by their attention to the context while the network
    reasons (`context_reading`), or as many heads drawn from outside twice as many (`random`).
    """

    steps: int
    learning_rate: float
    replay_prop: float
    uninformative_prop: float
    n_orderings: int
    layout: Literal["conflicting", "disjoint"]
    eval_every: int
    battery: str
    reasoning_prop: float
    reasoning_steps: int
    ablated_heads: Literal["none", "context_reading", "random"]
    n_ablated_heads: int


@dataclass(frozen=True)
class EvaluationConfig:
    """How the test battery is run: the reasoning-step sweep, contexts per pair, and the preset.

    With `ablated_scoring`, a finetuning run also scores the battery at every evaluation with
    heads ablated: the heads it ablates during the stage, or else the context-reading heads.
    """

    max_reasoning_steps: int
    contexts_per_pair: int
    battery: str | None
    ablated_scoring: bool

    def reasoning_steps(self) -> tuple[int, ...]:
        """Every reasoning-step value in the sweep, zero to the maximum."""
        return tuple(range(self.max_reasoning_steps + 1))


@dataclass(frozen=True)
class WandbConfig:
    """Optional mirror of the training scalars, off by default."""

    enabled: bool
    project: str | None
    entity: str | None


@dataclass(frozen=True)
class LoggingConfig:
    """How often training scalars are recorded, and the optional mirror."""

    log_every: int
    wandb: WandbConfig


@dataclass(frozen=True)
class Config:
    """One fully resolved configuration: base plus deltas plus overrides plus the seed list."""

    task: TaskConfig
    model: ModelConfig
    pretraining: PretrainingConfig
    finetuning: FinetuningConfig
    evaluation: EvaluationConfig
    seeds: tuple[int, ...]
    logging: LoggingConfig


@dataclass(frozen=True)
class Axis:
    """One swept setting; `path` sets a single dotted path, `set` sets several paths per value."""

    name: str
    values: tuple[Any, ...]
    path: str | None = None
    set: dict[Any, dict[str, Any]] | None = None


@dataclass(frozen=True)
class Dependency:
    """The experiment whose final checkpoint a run continues, and the axes that must agree."""

    experiment: str
    match: tuple[str, ...]


@dataclass(frozen=True)
class Experiment:
    """One paper experiment declared as a delta on the canonical configuration plus its axes."""

    name: str
    question: str
    stages: tuple[StageName, ...]
    delta: dict[str, Any]
    axes: tuple[Axis, ...]
    exclude: tuple[dict[str, Any], ...]
    depends_on: Dependency | None

    @property
    def swept_columns(self) -> tuple[str, ...]:
        """The axis names in declaration order: the columns a tidy table adds to its schema."""
        return tuple(axis.name for axis in self.axes)


@dataclass(frozen=True)
class Run:
    """One point of an experiment's sweep: axis values, identifier, and resolved configuration."""

    experiment: str
    run_id: str
    index: int
    axis_values: dict[str, Any]
    config: Config
    depends_on_experiment: str | None
    depends_on_run_id: str | None


# When this process began, so that a fingerprint can record elapsed wall time.
_PROCESS_START = time.time()

_STAGE_NAMES: tuple[str, ...] = get_args(StageName)
_EXPERIMENT_KEYS = frozenset({"name", "question", "stages", "delta", "axes", "exclude", "depends_on"})
_AXIS_KEYS = frozenset({"name", "values", "path", "set"})
_DEPENDENCY_KEYS = frozenset({"experiment", "match"})
_SEED_AXIS = "seed"


@functools.lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    """Resolved type hints of one configuration dataclass."""
    return get_type_hints(cls)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse one YAML document into plain data."""
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} does not hold a mapping")
    return document


def _build(cls: type, data: Mapping[str, Any], prefix: str = "") -> Any:
    """Construct one configuration dataclass from plain data, coercing every field."""
    declared = {field.name for field in dataclasses.fields(cls)}
    unexpected = sorted(set(data) - declared)
    if unexpected:
        raise ValueError(f"unknown configuration key '{prefix}{unexpected[0]}'")
    hints = _hints(cls)
    values = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            raise ValueError(f"missing configuration key '{prefix}{field.name}'")
        values[field.name] = _coerce(hints[field.name], data[field.name], f"{prefix}{field.name}")
    return cls(**values)


def _coerce(hint: Any, value: Any, path: str) -> Any:
    """Turn one plain value into the type its field declares."""
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, Mapping):
            raise ValueError(f"configuration section '{path}' must be a mapping")
        return _build(hint, value, f"{path}.")
    origin = get_origin(hint)
    if origin is Literal:
        allowed = get_args(hint)
        if value not in allowed:
            raise ValueError(f"'{path}' must be one of {allowed}, not {value!r}")
        return value
    if origin in (Union, UnionType):
        if value is None:
            return None
        inner = [arg for arg in get_args(hint) if arg is not type(None)]
        return _coerce(inner[0], value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"'{path}' must be a list, not {value!r}")
        return tuple(_coerce(get_args(hint)[0], item, path) for item in value)
    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError(f"'{path}' must be true or false, not {value!r}")
    if hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        number = _number(path, value)
        if not number.is_integer():
            raise ValueError(f"'{path}' must be an integer, not {value!r}")
        return int(number)
    if hint is float:
        return _number(path, value)
    if hint is str:
        return str(value)
    raise TypeError(f"unsupported configuration type for '{path}'")


def _number(path: str, value: Any) -> float:
    """Read a numeric field's value; a boolean or a non-numeric string is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"'{path}' must be a number, not {value!r}")
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"'{path}' must be a number, not {value!r}") from None


def _as_data(obj: Any) -> Any:
    """Render one configuration dataclass as nested plain data."""
    if dataclasses.is_dataclass(obj):
        return {field.name: _as_data(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return [_as_data(item) for item in obj]
    return obj


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Turn a nested mapping into dotted paths, one entry per leaf."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set one dotted path in nested plain data; a path absent from the base raises."""
    parts = path.split(".")
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"unknown configuration path '{path}'")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ValueError(f"unknown configuration path '{path}'")
    node[parts[-1]] = value


def _get_path(data: Mapping[str, Any], path: str) -> Any:
    """Read one dotted path from nested plain data."""
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError(f"unknown configuration path '{path}'")
        node = node[part]
    return node


def _values_equal(left: Any, right: Any) -> bool:
    """Compare two axis or configuration values, allowing for floating-point representation."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


@functools.lru_cache(maxsize=None)
def _base_data() -> dict[str, Any]:
    """The canonical configuration as plain data, used to check that a dotted path exists."""
    return _read_yaml(repo_root() / "configs" / "base.yaml")


@functools.lru_cache(maxsize=None)
def _preset_names(root: Path) -> frozenset[str]:
    """The test battery presets declared under one directory."""
    return frozenset(path.stem for path in Path(root).glob("*.yaml"))


def _check_path(path: str) -> None:
    """Raise unless a dotted path names a setting of the canonical configuration."""
    _get_path(_base_data(), path)


def _in_unit_interval(name: str, value: float) -> None:
    """Raise unless a proportion lies between zero and one."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie between 0 and 1, not {value}")


def _positive(name: str, value: float) -> None:
    """Raise unless a count or a rate is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, not {value}")


def repo_root() -> Path:
    """Directory holding `configs/` and `experiments/`, found by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "experiments").is_dir():
            return parent
    raise FileNotFoundError("no directory above this file holds configs/ and experiments/")


def run_directory(results_root: Path, experiment: str, run_id: str) -> Path:
    """Where one run's artifacts go: `<root>/<experiment>/runs/<run_id>/`.

    A run outside any experiment, described by a configuration file alone, goes under
    `<root>/runs/<run_id>/`.
    """
    root = Path(results_root) / experiment if experiment else Path(results_root)
    return root / "runs" / run_id


def tables_directory(results_root: Path, experiment: str) -> Path:
    """Where an experiment's tidy tables and manifest go: `<root>/<experiment>/tables/`."""
    return Path(results_root) / experiment / "tables"


def checkpoint_path(run_dir: Path, name: str) -> Path:
    """Where a run keeps the checkpoint called `name`: `<run_dir>/checkpoints/<name>.pt`."""
    return Path(run_dir) / "checkpoints" / f"{name}.pt"


def load_base(path: Path | None = None) -> Config:
    """Read `configs/base.yaml` into a validated `Config`; the path defaults to `repo_root`."""
    path = Path(path) if path is not None else repo_root() / "configs" / "base.yaml"
    config = _build(Config, _read_yaml(path))
    validate(config)
    return config


def load_config(config_path: Path | None = None, overrides: tuple[str, ...] = ()) -> Config:
    """Read a configuration file and apply repeated `section.field=value` overrides in order.

    A configuration file is a delta: its values are merged onto `configs/base.yaml`, so a file
    names only the settings it changes. `configs/smoke.yaml` is such a delta.
    """
    delta: dict[str, Any] = {}
    if config_path is not None:
        delta.update(_flatten(_read_yaml(Path(config_path))))
    for text in overrides:
        path, value = parse_override(text)
        delta[path] = value
    return apply_delta(load_base(), delta)


def parse_override(text: str) -> tuple[str, Any]:
    """Split one `section.field=value` override; the value is parsed as YAML."""
    path, separator, raw = text.partition("=")
    if not separator:
        raise ValueError(f"an override reads section.field=value, not '{text}'")
    return path.strip(), yaml.safe_load(raw)


def apply_delta(config: Config, delta: dict[str, Any]) -> Config:
    """Return a copy with each dotted path set; a path absent from the base raises."""
    data = _as_data(config)
    for path, value in delta.items():
        _set_path(data, path, value)
    updated = _build(Config, data)
    validate(updated)
    return updated


def validate(config: Config) -> None:
    """Raise if a resolved configuration is internally inconsistent.

    Skew and a frequent cue are mutually exclusive, proportions lie in [0, 1], a skew or a
    frequent cue is declared under the kind that draws it, the item pool holds the finetuning
    design, and every named preset exists.
    """
    cue = config.task.cue_distribution
    if cue.skew_alpha < 0:
        raise ValueError(f"skew_alpha must not be negative, not {cue.skew_alpha}")
    _in_unit_interval("frequent_cue_prop", cue.frequent_cue_prop)
    if cue.skew_alpha > 0 and cue.frequent_cue_prop > 0:
        raise ValueError("skew and a frequent cue are mutually exclusive")
    if cue.kind == "uniform" and cue.skew_alpha > 0:
        raise ValueError("a uniform distribution draws no skew; declare the kind zipf")
    if cue.kind == "uniform" and cue.frequent_cue_prop > 0:
        raise ValueError("a uniform distribution draws no frequent cue; declare the kind frequent_cue")
    if cue.kind == "zipf" and cue.frequent_cue_prop > 0:
        raise ValueError("a Zipf distribution has no frequent cue")
    if cue.kind == "frequent_cue" and cue.skew_alpha > 0:
        raise ValueError("a frequent-cue distribution has no skew")
    _in_unit_interval("p_cot", config.task.p_cot)
    _in_unit_interval("replay_prop", config.finetuning.replay_prop)
    _in_unit_interval("finetuning.uninformative_prop", config.finetuning.uninformative_prop)
    _positive("item_pool_size", config.task.item_pool_size)
    _positive("n_training_cues", config.task.n_training_cues)
    _positive("n_reserved_cues", config.task.n_reserved_cues)
    _positive("n_layers", config.model.n_layers)
    _positive("d_model", config.model.d_model)
    _positive("d_ff", config.model.d_ff)
    _positive("pretraining.steps", config.pretraining.steps)
    _positive("pretraining.batch_size", config.pretraining.batch_size)
    _positive("pretraining.learning_rate", config.pretraining.learning_rate)
    _positive("pretraining.eval_every", config.pretraining.eval_every)
    _positive("finetuning.steps", config.finetuning.steps)
    _positive("finetuning.learning_rate", config.finetuning.learning_rate)
    _positive("finetuning.n_orderings", config.finetuning.n_orderings)
    _positive("finetuning.eval_every", config.finetuning.eval_every)
    _in_unit_interval("finetuning.reasoning_prop", config.finetuning.reasoning_prop)
    _positive("finetuning.n_ablated_heads", config.finetuning.n_ablated_heads)
    if config.finetuning.reasoning_steps < 0:
        raise ValueError("finetuning.reasoning_steps must not be negative")
    # The steps and the answer emission after them fit the longest sequence the network takes: a
    # padded chain of PAD_TO emissions, or the longest scoring pass, whose reads the means reach.
    steps, longest = config.finetuning.reasoning_steps, config.evaluation.max_reasoning_steps
    if config.finetuning.reasoning_prop > 0 and (steps > longest or steps + 1 > max(PAD_TO, longest)):
        raise ValueError(
            f"finetuning.reasoning_steps {steps} leaves no room for the answer: it can be at most "
            f"evaluation.max_reasoning_steps ({longest}) and at most {max(PAD_TO, longest) - 1}"
        )
    _check_ablated_heads(config)
    _positive("evaluation.contexts_per_pair", config.evaluation.contexts_per_pair)
    _positive("logging.log_every", config.logging.log_every)
    if config.evaluation.max_reasoning_steps < 0:
        raise ValueError("max_reasoning_steps must not be negative")
    check_pool(
        config.task.item_pool_size, layout=config.finetuning.layout, n_orderings=config.finetuning.n_orderings
    )
    if config.finetuning.layout == "conflicting" and config.finetuning.n_orderings > config.task.n_training_cues:
        raise ValueError(
            f"finetuning.n_orderings {config.finetuning.n_orderings} exceeds task.n_training_cues "
            f"{config.task.n_training_cues}: the conflicting layout gives every ordering its own cue"
        )
    if not config.seeds:
        raise ValueError("a configuration needs at least one seed")
    declared = _preset_names(repo_root() / "configs" / "battery")
    for name in (config.pretraining.battery, config.finetuning.battery, config.evaluation.battery):
        if name is not None and name not in declared:
            raise ValueError(f"no test battery preset is named '{name}'")


def _check_ablated_heads(config: Config) -> None:
    """Raise unless the heads a finetuning run ablates, or scores ablated, fit its network, and its
    answers by pair are recorded on one fixed ordering.

    A random set draws its heads from outside the top twice as many, so it needs three times its
    size; the heads ranked highest need their own number.
    """
    section = config.finetuning
    if section.ablated_heads == "none" and not config.evaluation.ablated_scoring:
        records_pairs = section.reasoning_prop > 0
    else:
        records_pairs = True
        n_heads = config.model.n_layers * N_HEADS
        needed = 3 * section.n_ablated_heads if section.ablated_heads == "random" else section.n_ablated_heads
        if needed > n_heads:
            raise ValueError(
                f"{section.ablated_heads} sets of {section.n_ablated_heads} heads need {needed} heads; "
                f"the network has {n_heads}"
            )
    if records_pairs and section.n_orderings > 1:
        raise ValueError(
            "reasoning, ablated heads, and ablated scoring record answers by pair on one fixed ordering; "
            f"finetuning.n_orderings is {section.n_orderings}"
        )


def to_dict(config: Config) -> dict[str, Any]:
    """Render a configuration as plain data in the key order `configs/base.yaml` uses.

    A run's `config.yaml` is this rendering, so `load_config` reads it back unchanged; a run
    records the one seed it ran as a one-element `seeds`.
    """
    return dict(_as_data(config))


def dump_config(config: Config, path: Path) -> Path:
    """Write the resolved configuration next to a run's results and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(config), handle, sort_keys=False)
    return path


def load_experiment(name: str, root: Path | None = None) -> Experiment:
    """Read `experiments/<name>/experiment.yaml` and validate its grammar.

    The default root comes from `repo_root`, so an experiment resolves from any directory.
    """
    root = Path(root) if root is not None else repo_root() / "experiments"
    document = _read_yaml(root / name / "experiment.yaml")
    unexpected = sorted(set(document) - _EXPERIMENT_KEYS)
    if unexpected:
        raise ValueError(f"unknown experiment key '{unexpected[0]}'")
    if document.get("name") != name:
        raise ValueError(f"the declaration in {name}/ names itself '{document.get('name')}'")
    question = document.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"experiment '{name}' states no question")
    stages = tuple(document.get("stages") or ())
    for stage in stages:
        if stage not in _STAGE_NAMES:
            raise ValueError(f"experiment '{name}' names an unknown stage '{stage}'")
    delta = dict(document.get("delta") or {})
    for path in delta:
        _check_path(path)
    axes = tuple(_build_axis(name, entry) for entry in document.get("axes") or ())
    axis_names = {axis.name for axis in axes}
    if len(axis_names) != len(axes):
        raise ValueError(f"experiment '{name}' repeats an axis name")
    if _SEED_AXIS in axis_names:
        raise ValueError(f"experiment '{name}' declares the implicit seed axis")
    exclude = tuple(dict(entry) for entry in document.get("exclude") or ())
    for entry in exclude:
        unknown = sorted(set(entry) - axis_names - {_SEED_AXIS})
        if unknown:
            raise ValueError(f"experiment '{name}' excludes an unknown axis '{unknown[0]}'")
    depends_on = None
    if document.get("depends_on"):
        depends_on = _build_dependency(name, document["depends_on"], axis_names)
    return Experiment(
        name=name,
        question=" ".join(question.split()),
        stages=stages,
        delta=delta,
        axes=axes,
        exclude=exclude,
        depends_on=depends_on,
    )


def _build_axis(experiment: str, entry: Mapping[str, Any]) -> Axis:
    """Read one axis declaration: its name, its values, and the paths each value sets."""
    unexpected = sorted(set(entry) - _AXIS_KEYS)
    if unexpected:
        raise ValueError(f"unknown axis key '{unexpected[0]}' in experiment '{experiment}'")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"an axis of experiment '{experiment}' has no name")
    values = tuple(entry.get("values") or ())
    if not values:
        raise ValueError(f"axis '{name}' of experiment '{experiment}' has no values")
    path = entry.get("path")
    assignments = entry.get("set")
    if (path is None) == (assignments is None):
        raise ValueError(f"axis '{name}' sets either one path or several paths per value")
    if path is not None:
        _check_path(path)
        return Axis(name=name, values=values, path=path, set=None)
    resolved: dict[Any, dict[str, Any]] = {}
    for value in values:
        if value not in assignments:
            raise ValueError(f"axis '{name}' sets no paths for the value {value!r}")
        for target in assignments[value]:
            _check_path(target)
        resolved[value] = dict(assignments[value])
    return Axis(name=name, values=values, path=None, set=resolved)


def _build_dependency(experiment: str, entry: Mapping[str, Any], axis_names: set[str]) -> Dependency:
    """Read the upstream experiment and the axes whose values must agree."""
    unexpected = sorted(set(entry) - _DEPENDENCY_KEYS)
    if unexpected:
        raise ValueError(f"unknown dependency key '{unexpected[0]}' in experiment '{experiment}'")
    upstream = entry.get("experiment")
    if not isinstance(upstream, str) or not upstream:
        raise ValueError(f"experiment '{experiment}' depends on no named experiment")
    match = tuple(entry.get("match") or ())
    unknown = sorted(set(match) - axis_names - {_SEED_AXIS})
    if unknown:
        raise ValueError(f"experiment '{experiment}' matches an unknown axis '{unknown[0]}'")
    return Dependency(experiment=upstream, match=match)


def _axis_assignments(axis: Axis, value: Any) -> dict[str, Any]:
    """The dotted paths one axis value sets."""
    if axis.path is not None:
        return {axis.path: value}
    assignments = axis.set or {}
    if value not in assignments:
        raise ValueError(f"axis '{axis.name}' sets no paths for the value {value!r}")
    return dict(assignments[value])


def _excluded(experiment: Experiment, axis_values: Mapping[str, Any]) -> bool:
    """Whether one combination of axis values is removed from the sweep."""
    for entry in experiment.exclude:
        matched = all(
            name in axis_values and _values_equal(axis_values[name], value)
            for name, value in entry.items()
        )
        if matched:
            return True
    return False


def enumerate_runs(experiment: Experiment, base: Config | None = None) -> list[Run]:
    """Expand the declared axes in declaration order, seeds last, minus excluded combinations.

    The seed is an implicit final axis taken from the base configuration's `seeds`, which no
    `experiment.yaml` declares. It varies fastest, appears in `Run.axis_values` under `seed`, and
    each run's configuration carries just that one seed.
    """
    base = base if base is not None else load_base()
    seeds = base.seeds
    upstream = None
    if experiment.depends_on is not None:
        upstream = load_experiment(experiment.depends_on.experiment)
    names = [axis.name for axis in experiment.axes]
    runs: list[Run] = []
    for combination in product(*(axis.values for axis in experiment.axes)):
        declared = dict(zip(names, combination, strict=True))
        for seed in seeds:
            axis_values = {**declared, _SEED_AXIS: int(seed)}
            if _excluded(experiment, axis_values):
                continue
            delta = dict(experiment.delta)
            for axis, value in zip(experiment.axes, combination, strict=True):
                delta.update(_axis_assignments(axis, value))
            delta["seeds"] = [int(seed)]
            run = Run(
                experiment=experiment.name,
                run_id=run_id(experiment, axis_values),
                index=len(runs),
                axis_values=axis_values,
                config=apply_delta(base, delta),
                depends_on_experiment=None,
                depends_on_run_id=None,
            )
            if upstream is not None:
                run = dataclasses.replace(
                    run,
                    depends_on_experiment=upstream.name,
                    depends_on_run_id=_upstream_run_id(experiment.depends_on, run, upstream),
                )
            runs.append(run)
    if upstream is not None:
        _check_inherited_sections(experiment, runs, enumerate_runs(upstream, base))
    return runs


def _check_inherited_sections(experiment: Experiment, runs: Sequence[Run], upstream_runs: Sequence[Run]) -> None:
    """Raise unless every run agrees with the upstream run it continues at each section it inherits.

    The network a run continues from was made under the upstream run's task, model, and
    pretraining settings, and under its finetuning settings too unless this run finetunes itself;
    a run that declared any of them differently would record a configuration its network never had.
    """
    by_id = {upstream_run.run_id: upstream_run for upstream_run in upstream_runs}
    inherited = ["task", "model", "pretraining"] + ([] if "finetune" in experiment.stages else ["finetuning"])
    for run in runs:
        upstream_run = by_id.get(run.depends_on_run_id)
        if upstream_run is None:
            raise ValueError(
                f"run '{run.run_id}' of '{experiment.name}' continues from '{run.depends_on_run_id}', "
                f"which '{run.depends_on_experiment}' does not declare"
            )
        mine, theirs = _as_data(run.config), _as_data(upstream_run.config)
        for section in inherited:
            if mine[section] != theirs[section]:
                key = next(name for name in mine[section] if mine[section][name] != theirs[section][name])
                raise ValueError(
                    f"run '{run.run_id}' of '{experiment.name}' sets {section}.{key} differently from "
                    f"'{upstream_run.run_id}' of '{run.depends_on_experiment}', whose network it continues"
                )


def n_runs(experiment: Experiment, base: Config | None = None) -> int:
    """Count the runs an experiment declares; this feeds the array size."""
    return len(enumerate_runs(experiment, base))


def run_for_index(experiment: Experiment, index: int, base: Config | None = None) -> Run:
    """Resolve one array task index to one run; the entry point of a cluster task."""
    runs = enumerate_runs(experiment, base)
    if not 0 <= index < len(runs):
        raise IndexError(f"experiment '{experiment.name}' declares {len(runs)} runs, so {index} is out of range")
    return runs[index]


def run_id(experiment: Experiment, axis_values: dict[str, Any]) -> str:
    """Build the identifier from axis values alone; it never depends on the index."""
    fields = []
    for axis in experiment.axes:
        if axis.name not in axis_values:
            raise ValueError(f"axis '{axis.name}' has no value to build an identifier from")
        fields.append(f"{axis.name}-{format_axis_value(axis_values[axis.name])}")
    if _SEED_AXIS in axis_values:
        fields.append(f"{_SEED_AXIS}-{format_axis_value(axis_values[_SEED_AXIS])}")
    return "_".join(fields)


def format_axis_value(value: Any) -> str:
    """Render one axis value for an identifier: fixed point for floats, `m` for a minus sign."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        digits = f"{abs(value):.12f}".rstrip("0").rstrip(".") or "0"
        return f"m{digits}" if value < 0 else digits
    if isinstance(value, int):
        return f"m{abs(value)}" if value < 0 else str(value)
    return str(value)


def _upstream_run_id(dependency: Dependency, run: Run, upstream: Experiment) -> str:
    """The identifier of the upstream run whose checkpoint one run continues from.

    The upstream run agrees with this one at every axis named in `depends_on.match`. An upstream
    axis outside `match` is resolved by reading this run's configuration at that axis's `path`,
    so an axis the downstream experiment does not sweep still selects the right upstream run.
    """
    values: dict[str, Any] = {}
    for axis in upstream.axes:
        if axis.name in dependency.match:
            if axis.name not in run.axis_values:
                raise ValueError(f"experiment '{run.experiment}' has no axis '{axis.name}' to match")
            values[axis.name] = run.axis_values[axis.name]
        else:
            values[axis.name] = _axis_value_of(axis, run.config)
    values[_SEED_AXIS] = run.axis_values.get(_SEED_AXIS, run.config.seeds[0])
    return run_id(upstream, values)


def _axis_value_of(axis: Axis, config: Config) -> Any:
    """The value of one axis that a resolved configuration already holds."""
    data = _as_data(config)
    for value in axis.values:
        assignments = _axis_assignments(axis, value)
        if all(_values_equal(_get_path(data, path), setting) for path, setting in assignments.items()):
            return value
    raise ValueError(f"no value of axis '{axis.name}' matches the resolved configuration")


def config_hash(config: Config) -> str:
    """Hash the resolved configuration, stable across processes and key order."""
    rendered = json.dumps(to_dict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _git_sha() -> str | None:
    """The commit this checkout sits on, when it is a checkout at all."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _torch_versions() -> tuple[str | None, str | None]:
    """The installed torch version, and its CUDA build once torch has been imported."""
    torch = sys.modules.get("torch")
    if torch is not None:
        return str(torch.__version__), getattr(torch.version, "cuda", None)
    try:
        return metadata.version("torch"), None
    except metadata.PackageNotFoundError:
        return None, None


def fingerprint(
    config: Config, *, run_id: str | None = None, wall_time_s: float | None = None
) -> dict[str, Any]:
    """What produced a result: config hash, commit, package and torch versions, host, wall time.

    `wall_time_s` is the seconds given, or else the seconds since this process started.
    """
    torch_version, cuda_version = _torch_versions()
    elapsed = time.time() - _PROCESS_START if wall_time_s is None else float(wall_time_s)
    return {
        "run_id": run_id,
        "config_hash": config_hash(config),
        "git_sha": _git_sha(),
        "package_version": __version__,
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "host": socket.gethostname(),
        "wall_time_s": round(elapsed, 3),
    }
