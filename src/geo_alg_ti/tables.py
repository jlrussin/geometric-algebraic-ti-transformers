"""Tidy table schemas, validation, export, and the reasoning-step estimands.

One long-form row per (experiment, condition, metric, symbolic distance,
reasoning-step value, seed, step). `value` on an accuracy row is agreement
with the condition's scoring reference; `context_consistent_frac` and
`prior_consistent_frac` are the companion columns. The symbolic distance of a
shuffled row is the one in the ordering shown; `split_shuffled` divides a
shuffled row into the accuracy of its agreeing and of its conflicting trials.
Holds the five reasoning-step estimands the paper uses -- a pinned budget, the
mean over one to five steps, the best over the sweep, the first budget reaching
ninety percent of that seed's maximum within a distance, reported as a step
count in its own column and, on request, left missing where the best accuracy
is not above chance, and the running best up to each budget, which keeps the
reasoning-step axis -- and the trial-weighted pool over distances two and
above. `pair_type_by_seed` gives one pair type's accuracy (the ICL test, or the IWL test's
trained, end-item or interior pairs) of a reasoning-during-finetuning run scored as it was
finetuned (`finetuned_pair_rows`). Every estimand is taken within a seed and raises without
that axis; seeds are the only averaged axis, so the seed mean raises when seeds are grouped or when
the symbolic distances were never pooled. `export_experiment` turns an
experiment's run directories into the tables under `results/<experiment>/tables/`
plus a manifest of row counts and hashes.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Collection, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geo_alg_ti import CHANCE_ACCURACY, __version__, config, trials

EVAL_SCHEMA: Mapping[str, str] = {
    "experiment": "string",
    "tag": "string",
    "condition": "string",
    "source": "string",
    "context": "string",
    "metric": "string",
    "d": "int64",
    "K": "int64",
    "seed": "int64",
    "step": "int64",
    "value": "float64",
    "n": "int64",
    "context_consistent_frac": "float64",
    "prior_consistent_frac": "float64",
}
PAIR_SCHEMA: Mapping[str, str] = {**EVAL_SCHEMA, "rank_a": "int64", "rank_b": "int64"}
TRAIN_SCHEMA: Mapping[str, str] = {
    "experiment": "string",
    "seed": "int64",
    "stage": "string",
    "step": "int64",
    "loss": "float64",
    "token_cross_entropy": "float64",
    "scalar_squared_error": "float64",
    "answer_accuracy": "float64",
    "chain_token_accuracy": "float64",
}
PROBE_SCHEMA: Mapping[str, str] = {
    "experiment": "string",
    "kind": "string",
    "metric": "string",
    "layer": "int64",
    "seed": "int64",
    "step": "int64",
    "value": "float64",
    "n": "int64",
}
EDIT_SCHEMA: Mapping[str, str] = {
    **EVAL_SCHEMA,
    "edit_kind": "string",
    "direction": "string",
    "target": "string",
    "gain": "float64",
    "strength": "float64",
    "written_consistent_frac": "float64",
    "spearman_vs_familiar": "float64",
}
# The evaluation columns under one set of ablated attention heads. `set_kind` is intact, head
# (one head alone), top (the heads ranked highest by their attention to the context while the
# network reasons), or random (as many heads drawn from outside the top twice as many);
# `n_heads` counts the heads (zero when intact) and `heads` joins their labels, numbered from
# one, in order of attention rank. A single head's rows also carry its label, layer, and
# attention rank (empty, missing, and -1 on every other row); a random set's rows carry its
# draw (-1 elsewhere).
ABLATE_SCHEMA: Mapping[str, str] = {
    **EVAL_SCHEMA,
    "set_kind": "string",
    "head": "string",
    "layer": "float64",
    "n_heads": "int64",
    "heads": "string",
    "draw": "int64",
    "attention_rank": "int64",
}
# Each attention head's mean attention mass from the reasoning steps its network writes onto
# the context relations that mention the steps' items, pooled over the scored steps of every
# trial, and its rank within the network (one for the highest). `n_steps` counts the scored
# steps; `n_trials` counts every trial of the conditions scored, including trials whose chain
# has no step to score.
HEAD_SCORES_SCHEMA: Mapping[str, str] = {
    "experiment": "string",
    "head": "string",
    "layer": "int64",
    "seed": "int64",
    "step": "int64",
    "score": "float64",
    "attention_rank": "int64",
    "n_steps": "int64",
    "n_trials": "int64",
}
# The test battery scored with a run's set of heads ablated, the heads named by their labels.
ABLATED_EVAL_SCHEMA: Mapping[str, str] = {**EVAL_SCHEMA, "heads": "string"}
ABLATED_PAIR_SCHEMA: Mapping[str, str] = {**PAIR_SCHEMA, "heads": "string"}
# The answer accuracy on the fixed-ordering trials of each logging window of finetuning updates
# (`step` is the window's last update), by their context (congruent: it shows the ordering;
# uninformative: other items), the reasoning steps the network wrote before answering, and the
# queried ranks as the per-pair table numbers them, zero-based in the nine-item ordering the test
# battery queries (the main items at one to seven); `n` counts the trials.
TRAIN_PAIRS_SCHEMA: Mapping[str, str] = {
    "experiment": "string",
    "seed": "int64",
    "step": "int64",
    "context": "string",
    "reasoning_steps": "int64",
    "rank_a": "int64",
    "rank_b": "int64",
    "value": "float64",
    "n": "int64",
}
# The column the criterion estimand writes: a count of reasoning steps, not an accuracy.
K_TO_CRITERION: str = "k_to_criterion"
COLUMN_LABELS: Mapping[str, str] = {
    "d": "Symbolic distance",
    "K": "Reasoning steps",
    "value": "Accuracy",
    K_TO_CRITERION: "Reasoning steps",
    "skew_alpha": "Skew (α)",
    "p_cot": "p(CoT)",
    "frequent_cue_prop": "Frequent-cue proportion",
    "n_training_cues": "Number of orderings",
    "step": "Step",
}
# Each exported table's schema and where its rows sit inside one run directory. A source ending
# in `.parquet` is the one file the training loop writes; any other source is a directory named
# after the table holding one `step_NNNNNNNN.parquet` file per evaluation step.
_TABLES: Mapping[str, tuple[Mapping[str, str], str]] = {
    "eval": (EVAL_SCHEMA, "eval"),
    "eval_pairs": (PAIR_SCHEMA, "eval_pairs"),
    "train": (TRAIN_SCHEMA, "train_metrics.parquet"),
    "probe": (PROBE_SCHEMA, "probe"),
    "edit": (EDIT_SCHEMA, "edit"),
    "ablate": (ABLATE_SCHEMA, "ablate"),
    "head_scores": (HEAD_SCORES_SCHEMA, "head_scores"),
    "ablated_eval": (ABLATED_EVAL_SCHEMA, "ablated_eval"),
    "ablated_eval_pairs": (ABLATED_PAIR_SCHEMA, "ablated_eval_pairs"),
    "train_pairs": (TRAIN_PAIRS_SCHEMA, "train_pairs.parquet"),
}
TABLE_NAMES: tuple[str, ...] = tuple(_TABLES)
# The scored trials a run keeps beside its per-step tables; they are not exported.
TRIALS_TABLE: str = "trials"
# The shuffled context's condition and the two it splits into: trials whose shown order agrees
# with the frequent cue's ordering, and trials whose shown order conflicts with it.
SHUFFLED_CONDITION: str = "frequent_cue_shuffled"
SHUFFLED_SPLIT: Mapping[str, tuple[str, str]] = {
    "agree": ("frequent_cue_shuffled_agree", "shuffled_agree"),
    "conflict": ("frequent_cue_shuffled_conflict", "shuffled_conflict"),
}
_STEP_FILE = "step_{step:08d}.parquet"
_STEP_COLUMN = "K"
_DISTANCE_COLUMN = "d"
# The reasoning-step estimands: one pinned budget, the mean over budgets, the best budget.
STEP_STATISTICS: tuple[str, ...] = ("pin", "mean", "best")
# The budgets the mean over reasoning steps covers unless a caller names others.
DEFAULT_MEAN_BUDGETS: tuple[int, ...] = (1, 2, 3, 4, 5)
# The columns that name a reasoning-during-finetuning run's condition: the heads it ablates while
# it finetunes, and the proportion of congruent-context trials on which it reasons first.
HEADS_COLUMN: str = "ablated_heads"
REASONING_COLUMN: str = "reasoning_prop"
# The finetuning conditions, as (heads ablated while finetuning, proportion of congruent-context
# trials that reason first): no reasoning, reasoning, and reasoning with the context-reading heads
# or as many random heads ablated.
FINETUNING_CONDITIONS: tuple[tuple[str, float], ...] = (("none", 0.0), ("none", 1.0), ("context_reading", 1.0), ("random", 1.0))
# The ICL test, then the IWL test's trained pairs, its non-adjacent pairs that hold the first or
# last main item, and its other non-adjacent pairs.
PAIR_TYPES: tuple[str, ...] = ("icl_test", "trained", "end_item", "interior")
_PAIR_TYPE_NAMES: Mapping[str, str] = {
    "icl_test": "ICL test",
    "trained": "IWL test trained",
    "end_item": "IWL test end-item",
    "interior": "IWL test interior",
}
# The column `finetuned_pair_rows` adds: each row's one of `PAIR_TYPES`, empty on every other
# condition.
PAIR_TYPE_COLUMN: str = "pair_type"


def _expected_columns(schema: Mapping[str, str], swept_columns: Sequence[str]) -> list[str]:
    """The columns a table holds: its schema followed by one column per swept axis."""
    columns = list(schema)
    columns.extend(name for name in swept_columns if name not in schema)
    return columns


def _dtype_holds(series: pd.Series, declared: str) -> bool:
    """Whether one column's type is the declared one."""
    if declared == "string":
        return pd.api.types.is_string_dtype(series)
    if declared == "int64":
        return pd.api.types.is_integer_dtype(series)
    if declared == "float64":
        return pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series)
    return str(series.dtype) == declared


def validate_table(df: pd.DataFrame, *, schema: Mapping[str, str], swept_columns: Sequence[str] = ()) -> None:
    """Raise if columns are missing, unexpected, or of the wrong type.

    A table is checked when it is written and when it is loaded. The estimand helpers below take
    whatever columns they need and never validate, so they also serve partial frames.
    """
    expected = _expected_columns(schema, swept_columns)
    missing = [name for name in expected if name not in df.columns]
    if missing:
        raise ValueError(f"the table is missing the columns {missing}")
    unexpected = [name for name in df.columns if name not in expected]
    if unexpected:
        raise ValueError(f"the table carries the undeclared columns {unexpected}")
    for name, declared in schema.items():
        if not _dtype_holds(df[name], declared):
            raise ValueError(f"column '{name}' holds {df[name].dtype}, not {declared}")


def _conform(df: pd.DataFrame, schema: Mapping[str, str], swept_columns: Sequence[str]) -> pd.DataFrame:
    """Order a table's columns and give each the type its schema declares."""
    expected = _expected_columns(schema, swept_columns)
    ordered = df.loc[:, expected].copy()
    return ordered.astype({name: declared for name, declared in schema.items()})


def write_table(df: pd.DataFrame, path: Path, *, schema: Mapping[str, str], swept_columns: Sequence[str] = ()) -> Path:
    """Validate, conform to the declared types, and write a parquet table."""
    validate_table(df, schema=schema, swept_columns=swept_columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _conform(df, schema, swept_columns).to_parquet(path, index=False)
    return path


def load_table(path: Path, *, schema: Mapping[str, str] | None = None, swept_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a parquet table, validating against `schema` when given.

    Naming `swept_columns` allows exactly those axes beyond the schema, so a column belonging to
    neither is rejected. Left out, any column beyond the schema is taken to be a swept axis.
    """
    df = pd.read_parquet(Path(path))
    if schema is not None:
        inferred = [name for name in df.columns if name not in schema]
        allowed = inferred if swept_columns is None else list(swept_columns)
        validate_table(df, schema=schema, swept_columns=allowed)
    return df


def _run_source(run_dir: Path, name: str) -> Path:
    """Where one table's rows sit inside one run directory."""
    if name == TRIALS_TABLE:
        return Path(run_dir) / name
    if name not in _TABLES:
        raise ValueError(f"no run table is named '{name}'")
    return Path(run_dir) / _TABLES[name][1]


def write_run_table(run_dir: Path, name: str, step: int, df: pd.DataFrame) -> None:
    """Write one evaluation step's rows of a per-step table under `<run_dir>/<name>/`.

    `name` is one of the per-step tables or `trials`. The file is named by the step so that the
    steps read back in order. An empty frame writes nothing.
    """
    source = _run_source(run_dir, name)
    if source.suffix == ".parquet":
        raise ValueError(f"the '{name}' table is written once by the training loop, not per step")
    if df is None or df.empty:
        return
    source.mkdir(parents=True, exist_ok=True)
    df.to_parquet(source / _STEP_FILE.format(step=int(step)), index=False)


def write_train_table(run_dir: Path, df: pd.DataFrame, *, name: str = "train") -> Path:
    """Write a table the training loop writes once at the end of a stage: the training curve,
    or the per-pair answer accuracy of the finetuning trials."""
    source = _run_source(run_dir, name)
    if source.suffix != ".parquet":
        raise ValueError(f"the '{name}' table is written per evaluation step, not once")
    source.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(source, index=False)
    return source


def read_run_table(run_dir: Path, name: str) -> pd.DataFrame | None:
    """One run's rows of one table, or None when the run wrote none.

    A per-step table is its step files concatenated in step order; the training curve is its
    one file.
    """
    source = _run_source(run_dir, name)
    if source.suffix == ".parquet":
        return pd.read_parquet(source) if source.is_file() else None
    if not source.is_dir():
        return None
    parts = [pd.read_parquet(part) for part in sorted(source.glob("step_*.parquet"))]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _complete(df: pd.DataFrame, *, experiment: str, run: config.Run, schema: Mapping[str, str], swept_columns: Sequence[str]) -> pd.DataFrame:
    """Add the experiment name, the swept axis values, and the seed a run's own rows omit."""
    completed = df.copy()
    completed["experiment"] = experiment
    for name in swept_columns:
        completed[name] = run.axis_values[name]
    if "seed" in schema and "seed" not in df.columns:
        completed["seed"] = run.axis_values["seed"]
    return completed


def _sha256(path: Path) -> str:
    """The digest of one written table."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_experiment(
    experiment: str, *, results_root: Path, base: config.Config | None = None
) -> dict[str, Path]:
    """Collect every run directory of an experiment into the tidy tables and write the manifest.

    The runs are those the experiment declares on `base`, the configuration the runs were made
    with (the canonical configuration when None), so the seeds a delta or override chose are found.
    Each run's aggregates gain the `experiment` name and one column per swept axis, taken from
    the run's axis values, which completes the schema each table is then validated against.
    """
    declaration = config.load_experiment(experiment)
    swept = declaration.swept_columns
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLE_NAMES}
    sources: dict[str, list[str]] = {name: [] for name in TABLE_NAMES}
    missing: list[str] = []
    for run in config.enumerate_runs(declaration, base):
        run_dir = config.run_directory(results_root, experiment, run.run_id)
        if not run_dir.is_dir():
            missing.append(run.run_id)
            continue
        for name, (schema, _) in _TABLES.items():
            rows = read_run_table(run_dir, name)
            if rows is None or rows.empty:
                continue
            parts[name].append(_complete(rows, experiment=experiment, run=run, schema=schema, swept_columns=swept))
            sources[name].append(run.run_id)
    if not any(parts.values()):
        raise FileNotFoundError(f"no run of '{experiment}' under {results_root} holds results to export")
    if missing:
        print(f"{len(missing)} of the declared runs are not written yet: {' '.join(missing)}", file=sys.stderr)
    out_dir = config.tables_directory(results_root, experiment)
    written: dict[str, Path] = {}
    entries: list[Mapping[str, object]] = []
    for name, (schema, _) in _TABLES.items():
        if not parts[name]:
            continue
        table = pd.concat(parts[name], ignore_index=True)
        path = write_table(table, out_dir / f"{name}.parquet", schema=schema, swept_columns=swept)
        written[name] = path
        entries.append(
            {
                "table": name,
                "path": path.name,
                "rows": int(len(table)),
                "sha256": _sha256(path),
                "axes": list(swept),
                "runs": sources[name],
            }
        )
    write_manifest(out_dir / "MANIFEST.yaml", entries)
    return written


def check_export(manifest: Path, declared: Collection[str]) -> None:
    """Raise unless every table of an export covers exactly the `declared` run ids and is intact.

    Each table's file must sit beside the manifest with the digest the manifest records, so a
    table copied while it was being rewritten is refused with the manifest that names it.
    """
    manifest = Path(manifest)
    try:
        with manifest.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"the manifest {manifest} is not readable: {error}") from None
    entries = document.get("tables") if isinstance(document, Mapping) else None
    wanted = set(declared)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"the export covers none of the {len(wanted)} declared runs")
    for entry in entries:
        entry = entry if isinstance(entry, Mapping) else {}
        runs = set(entry.get("runs") or ())
        name = entry.get("table", "unnamed")
        if wanted - runs:
            raise ValueError(f"the {name} table covers {len(wanted & runs)} of the {len(wanted)} declared runs")
        if runs - wanted:
            raise ValueError(f"the {name} table holds {len(runs - wanted)} runs the experiment does not declare")
        path = manifest.parent / str(entry.get("path", ""))
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise ValueError(f"the {name} table is missing or differs from the digest its manifest records")


def write_manifest(path: Path, entries: Sequence[Mapping[str, object]]) -> Path:
    """Write `MANIFEST.yaml`: row counts, hashes, axes, source run ids, package version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"package_version": __version__, "tables": [dict(entry) for entry in entries]}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)
    return path


def pick(df: pd.DataFrame, **filters: object) -> pd.DataFrame:
    """Column-equality filter; a collection of values selects membership."""
    selected = df
    for name, wanted in filters.items():
        if name not in selected.columns:
            raise ValueError(f"the table has no column '{name}' to filter on")
        if isinstance(wanted, (list, tuple, set, frozenset)):
            selected = selected[selected[name].isin(list(wanted))]
        else:
            selected = selected[selected[name] == wanted]
    return selected.copy()


def select(df: pd.DataFrame, *, fname: str = "panel", **filters: object) -> pd.DataFrame:
    """Rows matching every filter; raises naming the filter that left nothing."""
    rows = pick(df, **filters)
    if rows.empty:
        wanted = ", ".join(f"{name}={value!r}" for name, value in filters.items())
        raise ValueError(f"{fname}: no rows match {wanted}")
    return rows


def apply_swept_pins(df: pd.DataFrame, *, fixed: Mapping[str, object], swept_columns: Sequence[str], fname: str = "panel") -> pd.DataFrame:
    """Apply pins, then raise if any swept axis still has more than one value."""
    selected = pick(df, **dict(fixed))
    for name in swept_columns:
        if name not in selected.columns:
            continue
        remaining = selected[name].dropna().unique()
        if len(remaining) > 1:
            raise ValueError(f"{fname} draws one setting at a time; pin '{name}', which still holds {sorted(remaining)}")
    return selected


def _forbid(group_cols: Sequence[str], column: str, over: str) -> None:
    """Raise when an axis a helper reduces over is also a grouping column."""
    if column in group_cols:
        raise ValueError(f"'{column}' is reduced over {over} and cannot also group the rows")


def _require_seed(group_cols: Sequence[str], estimand: str, seed_col: str = "seed") -> None:
    """Raise when an estimand taken within a seed is asked to reduce across seeds instead."""
    if seed_col not in group_cols:
        raise ValueError(f"{estimand} is taken within a seed; keep '{seed_col}' among the grouping columns")


def _carry_counts(frame: pd.DataFrame, group: list[str], n_col: str = "n") -> pd.DataFrame:
    """The trial count of each group, unchanged by a reduction over reasoning-step values."""
    return frame.groupby(group, dropna=False, sort=False)[n_col].max().reset_index()


def seed_mean_se(df: pd.DataFrame, *, group_cols: Sequence[str], value_col: str = "value", seed_col: str = "seed") -> pd.DataFrame:
    """Mean and standard error over seeds; raises on a grouped seed or an unpooled distance."""
    group = list(group_cols)
    _forbid(group, seed_col, "seeds")
    if _DISTANCE_COLUMN in df.columns and _DISTANCE_COLUMN not in group:
        raise ValueError(
            f"'{_DISTANCE_COLUMN}' is neither grouped nor pooled; pool it with pool_over_distance, "
            "whose mean is trial-weighted, before averaging over seeds"
        )
    per_seed = df.groupby([*group, seed_col], dropna=False, sort=False)[value_col].mean().reset_index()
    grouped = per_seed.groupby(group, dropna=False, sort=False)[value_col]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out[f"{value_col}_mean"] = out["mean"]
    out[f"{value_col}_se"] = out["std"] / np.sqrt(out["count"])
    out["n_seeds"] = out["count"].astype("int64")
    return out.drop(columns=["mean", "std", "count"])


def pool_over_distance(df: pd.DataFrame, *, group_cols: Sequence[str], value_col: str = "value", n_col: str = "n", d_min: int = 2) -> pd.DataFrame:
    """Trial-weighted mean over distances at least `d_min`; raises if `d` is grouped.

    A row with a missing value carries no weight, so a group's mean covers only the rows that
    hold a value, and a group with none has a missing mean; the trial count still counts the
    trials of every pooled row.
    """
    group = list(group_cols)
    _forbid(group, _DISTANCE_COLUMN, "symbolic distances")
    pooled = df[df[_DISTANCE_COLUMN] >= d_min]
    if pooled.empty:
        raise ValueError(f"no rows lie at symbolic distance {d_min} or above")
    weight = pooled[n_col].where(pooled[value_col].notna(), 0)
    pooled = pooled.assign(_weighted=pooled[value_col] * pooled[n_col], _weight=weight)
    grouped = pooled.groupby(group, dropna=False, sort=False)
    out = grouped.agg(_weighted=("_weighted", "sum"), _weight=("_weight", "sum"), _trials=(n_col, "sum")).reset_index()
    out[value_col] = out["_weighted"] / out["_weight"]
    out[n_col] = out["_trials"]
    return out.drop(columns=["_weighted", "_weight", "_trials"])


def pin_k(df: pd.DataFrame, k: int, *, seed_col: str = "seed") -> pd.DataFrame:
    """Rows at one reasoning-step value; raises if the value or the seed axis is absent."""
    if seed_col not in df.columns:
        raise ValueError(f"a pinned budget is taken within a seed; the rows carry no '{seed_col}' column")
    present = set(df[_STEP_COLUMN].unique())
    if k not in present:
        raise ValueError(f"the table holds reasoning-step values {sorted(present)}, not {k}")
    return df[df[_STEP_COLUMN] == k].copy()


def accuracy_at_budget(df: pd.DataFrame, *, k: int, d_min: int = 2, group_cols: Sequence[str] = ("condition",)) -> pd.DataFrame:
    """Accuracy at budget `k`, pooled over distances at least `d_min` by trials within each seed.

    One row per group of `group_cols` and seed, which `seed_mean_se` over `group_cols` averages;
    the estimand of the accuracy bars, of the numbers written beside a reasoning-step panel's
    lines, and of the accuracies at the largest budget that `geo_alg_ti.numbers` recomputes.
    """
    return pool_over_distance(pin_k(df, int(k)), group_cols=[*group_cols, "seed"], d_min=int(d_min))


def mean_over_k(df: pd.DataFrame, *, group_cols: Sequence[str], k_values: Sequence[int] = DEFAULT_MEAN_BUDGETS, value_col: str = "value") -> pd.DataFrame:
    """Mean over the given reasoning-step values within each group, seed and distance first."""
    group = list(group_cols)
    _forbid(group, _STEP_COLUMN, "reasoning-step values")
    _require_seed(group, "the mean over reasoning steps")
    swept = df[df[_STEP_COLUMN].isin(list(k_values))]
    if swept.empty:
        raise ValueError(f"the table holds none of the reasoning-step values {list(k_values)}")
    out = swept.groupby(group, dropna=False, sort=False)[value_col].mean().reset_index()
    if "n" in swept.columns:
        out = out.merge(_carry_counts(swept, group), on=group, how="left")
    return out


def best_over_k(df: pd.DataFrame, *, group_cols: Sequence[str], value_col: str = "value") -> pd.DataFrame:
    """Maximum over reasoning-step values per group, before any pooling or seed averaging."""
    group = list(group_cols)
    _forbid(group, _STEP_COLUMN, "reasoning-step values")
    _require_seed(group, "the best over reasoning steps")
    out = df.groupby(group, dropna=False, sort=False)[value_col].max().reset_index()
    if "n" in df.columns:
        out = out.merge(_carry_counts(df, group), on=group, how="left")
    return out


def running_best_over_k(df: pd.DataFrame, *, group_cols: Sequence[str], value_col: str = "value") -> pd.DataFrame:
    """Best over the budgets up to each reasoning-step value within a group; the budget stays.

    Each row's value becomes the maximum over its group's rows at reasoning-step values no
    larger than its own (a cumulative maximum in budget order within each seed), so a curve
    over reasoning steps never falls. Unlike the other estimands the reasoning-step column is
    kept: the result is drawn against it.
    """
    group = list(group_cols)
    _forbid(group, _STEP_COLUMN, "reasoning-step values")
    _require_seed(group, "the running best over reasoning steps")
    out = df.sort_values(_STEP_COLUMN, kind="stable").copy()
    out[value_col] = out.groupby(group, dropna=False, sort=False)[value_col].cummax()
    return out


def reduce_over_k(
    df: pd.DataFrame, *, k_stat: str, k_pin: int | None, k_values: Sequence[int], group_cols: Sequence[str], fname: str = "panel"
) -> pd.DataFrame:
    """One reasoning-step estimand within each group, which always includes the seed: the budget
    `k_pin` (zero when None), the mean over `k_values`, or the best budget."""
    if k_stat not in STEP_STATISTICS:
        raise ValueError(f"{fname}: k_stat is one of {STEP_STATISTICS}, not {k_stat!r}")
    if k_stat == "pin":
        return pin_k(df, 0 if k_pin is None else int(k_pin))
    if k_stat == "mean":
        return mean_over_k(df, group_cols=list(group_cols), k_values=list(k_values))
    return best_over_k(df, group_cols=list(group_cols))


def finetuned_pair_rows(pairs: pd.DataFrame, ablated_pairs: pd.DataFrame, *, fname: str = "panel") -> pd.DataFrame:
    """Every run's per-pair accuracy rows as it was finetuned: a run that ablated heads while it
    finetuned is read from `ablated_pairs`, scored with the same heads still ablated, and every
    other run from `pairs`, scored intact. Each row carries its pair type in `PAIR_TYPE_COLUMN`,
    so the rows are split into pair types once; the column comes only from here, so rows that
    carry it are rows scored as each run was finetuned."""
    for frame, name in ((pairs, "pairs"), (ablated_pairs, "ablated_pairs")):
        if HEADS_COLUMN not in frame.columns or REASONING_COLUMN not in frame.columns:
            raise ValueError(f"{fname}: the {name} rows carry no '{HEADS_COLUMN}' and '{REASONING_COLUMN}' columns")
    if "heads" not in ablated_pairs.columns:
        raise ValueError(f"{fname}: the ablated_pairs rows name no ablated heads; pass the table scored with heads ablated")
    intact = pairs[pairs[HEADS_COLUMN] == "none"]
    ablated = ablated_pairs[ablated_pairs[HEADS_COLUMN] != "none"]
    run = ["seed", *(name for name in ablated.columns if name not in ABLATED_PAIR_SCHEMA)]
    sets = ablated.groupby(run, dropna=False, observed=True)["heads"].nunique()
    if (sets > 1).any():
        raise ValueError(f"{fname}: a run's ablated rows name more than one head set: {sets[sets > 1].index.tolist()}")
    rows = select(pd.concat([intact, ablated.drop(columns=["heads"])], ignore_index=True), fname=fname, metric="accuracy")
    return rows.assign(**{PAIR_TYPE_COLUMN: _label_pair_types(rows, fname=fname)})


def _label_pair_types(rows: pd.DataFrame, *, fname: str = "panel") -> pd.Series:
    """Each per-pair row's one of `PAIR_TYPES`, empty on every other condition. The end items are
    the first and last main items the battery queries (`trials.QUERYABLE_RANKS`), whichever pairs
    the rows hold; an IWL test pair outside the main items raises."""
    first, last = trials.QUERYABLE_RANKS[0], trials.QUERYABLE_RANKS[-1]
    low = np.minimum(rows["rank_a"], rows["rank_b"]).to_numpy()
    high = np.maximum(rows["rank_a"], rows["rank_b"]).to_numpy()
    iwl = (rows["condition"] == "iwl_test").to_numpy()
    if (iwl & ((low < first) | (high > last))).any():
        raise ValueError(f"{fname}: an IWL test pair lies outside the main items, ranks {first} to {last}")
    gap = high - low
    end = (low == first) | (high == last)
    labels = np.select(
        [(rows["condition"] == "icl_test").to_numpy(), iwl & (gap == 1), iwl & (gap >= 2) & end, iwl & (gap >= 2)],
        list(PAIR_TYPES),
        default="",
    )
    return pd.Series(labels, index=rows.index, name=PAIR_TYPE_COLUMN)


def pair_type_rows(rows: pd.DataFrame, measure: str, *, fname: str = "panel") -> pd.DataFrame:
    """The rows of one of `PAIR_TYPES`: the ICL test's pairs, or the IWL test's trained pairs, its
    non-adjacent pairs that hold the first or last main item, or its other non-adjacent pairs, as
    the pair-type column of `finetuned_pair_rows` names them."""
    if measure not in PAIR_TYPES:
        raise ValueError(f"{fname}: a measure is one of {PAIR_TYPES}, not {measure!r}")
    if PAIR_TYPE_COLUMN not in rows.columns:
        raise ValueError(f"{fname}: the rows carry no '{PAIR_TYPE_COLUMN}' column; pass the rows of finetuned_pair_rows")
    sub = rows[rows[PAIR_TYPE_COLUMN] == measure]
    if sub.empty:
        raise ValueError(f"{fname}: the rows hold no {_PAIR_TYPE_NAMES[measure]} pairs")
    return sub


def pair_type_by_seed(
    rows: pd.DataFrame,
    measure: str,
    *,
    group_cols: Sequence[str] = (),
    k_stat: str = "best",
    k_pin: int | None = None,
    k_values: Sequence[int] = DEFAULT_MEAN_BUDGETS,
    d_min: int = 2,
    fname: str = "panel",
) -> pd.DataFrame:
    """One pair type's accuracy within each group and seed, from rows of one finetuning condition:
    the pairs pooled by trials at each distance and budget, `reduce_over_k` within each
    distance, then the trial-weighted pool over distances at least `d_min` (trained pairs are all
    adjacent, so their pool keeps distance one). A pair with a missing value carries no weight
    and its trials are not counted, so the trials behind a distance can differ from budget to
    budget; under the best budget, the distance carries the trials of its best budget's value
    (the smallest of tied budgets), and under the mean, the most trials of any budget."""
    sub = pair_type_rows(rows, measure, fname=fname)
    within = [*group_cols, "seed", _DISTANCE_COLUMN]
    weight = sub["n"].where(sub["value"].notna(), 0)
    weighted = sub.assign(_weighted=(sub["value"] * sub["n"]).fillna(0.0), _weight=weight).groupby([*within, _STEP_COLUMN], dropna=False, sort=False)
    pooled = weighted.agg(_weighted=("_weighted", "sum"), n=("_weight", "sum")).reset_index()
    pooled["value"] = pooled["_weighted"] / pooled["n"].where(pooled["n"] > 0)
    if k_stat == "best":
        by_budget = pooled.sort_values(_STEP_COLUMN, kind="stable")
        best = by_budget.groupby(within, dropna=False, sort=False)["value"].transform("max")
        own = by_budget.assign(_own=by_budget["n"].where(by_budget["value"] == best)).groupby(within, dropna=False, sort=False)["_own"]
        pooled["n"] = own.transform("first").fillna(0).astype("int64")
    reduced = reduce_over_k(pooled.drop(columns=["_weighted"]), k_stat=k_stat, k_pin=k_pin, k_values=k_values, group_cols=within, fname=fname)
    return pool_over_distance(reduced, group_cols=[*group_cols, "seed"], d_min=1 if measure == "trained" else int(d_min))


def round_half_up(value: float, decimals: int) -> Decimal:
    """`value` at `decimals` places, a half rounded away from zero, as the paper writes numbers.

    The value is read at twelve significant digits first, so a binary value just below a
    decimal half rounds as the half does. The figures write their numbers with it and
    `geo_alg_ti.numbers` compares the printed ones with it.
    """
    exact = Decimal(f"{float(value):.12g}")
    return exact.quantize(Decimal(1).scaleb(-int(decimals)), rounding=ROUND_HALF_UP)


def wilson_lower_bound(accuracy: float, n: float, *, z: float = 1.96) -> float:
    """The lower end of the Wilson score interval of an accuracy over `n` trials."""
    p, n, z = float(accuracy), float(n), float(z)
    if n <= 0:
        raise ValueError(f"a Wilson interval needs at least one trial, not {n:g}")
    center = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((center - spread) / (1 + z * z / n))


def first_k_to_criterion(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    fraction: float = 0.9,
    value_col: str = "value",
    n_col: str = "n",
    unreached_below_chance: bool = False,
    unreached_z: float = 1.96,
) -> pd.DataFrame:
    """Smallest budget reaching `fraction` of the (seed, distance) maximum.

    A group whose accuracy is undefined at every budget has no maximum and is dropped.

    The result is a count of reasoning steps, so it lands in `K_TO_CRITERION` rather than in the
    accuracy column it is computed from, and the trial count comes along, which lets it feed
    `pool_over_distance` and `seed_mean_se` with `value_col=K_TO_CRITERION`.

    With `unreached_below_chance`, a group whose best accuracy over the sweep is not above
    chance has no criterion value: when the lower end of the Wilson score interval
    (`wilson_lower_bound` at z = `unreached_z`) of that best accuracy over the group's trial
    count does not exceed one half, the group is kept with its count and a missing value, which
    `pool_over_distance` leaves out of its mean.
    """
    group = list(group_cols)
    _forbid(group, _STEP_COLUMN, "reasoning-step values")
    _require_seed(group, "the first budget to criterion")
    carry_counts = n_col in df.columns
    if unreached_below_chance and not carry_counts:
        raise ValueError(f"the chance floor needs each accuracy's trial count; the rows carry no '{n_col}'")
    reached = []
    for keys, rows in df.groupby(group, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows = rows.sort_values(_STEP_COLUMN)
        best = rows[value_col].max()
        above = rows[rows[value_col] >= fraction * best]
        if above.empty:
            continue
        record = dict(zip(group, keys, strict=True))
        record[K_TO_CRITERION] = int(above[_STEP_COLUMN].iloc[0])
        if unreached_below_chance:
            floor = wilson_lower_bound(best, rows[n_col].max(), z=unreached_z)
            if not floor > CHANCE_ACCURACY:
                record[K_TO_CRITERION] = np.nan
        if carry_counts:
            record[n_col] = int(rows[n_col].max())
        reached.append(record)
    columns = [*group, K_TO_CRITERION, *([n_col] if carry_counts else [])]
    return pd.DataFrame(reached, columns=columns)


def split_shuffled(df: pd.DataFrame, *, tolerance: float = 1e-12) -> pd.DataFrame:
    """The shuffled context's accuracy on its agreeing and on its conflicting trials, row by row.

    Every shuffled row holds as many trials whose shown order agrees with the frequent cue's
    ordering as trials whose shown order conflicts with it, and every prediction is one sign or
    the other. A correct answer follows both orderings on an agreeing trial and only the shown
    one on a conflicting trial, so with `value` the accuracy and `prior_consistent_frac` the
    fraction of answers following the frequent cue's ordering, the agreeing trials are
    answered correctly at `value + prior_consistent_frac - 1/2` and the conflicting trials at
    `value - prior_consistent_frac + 1/2`.

    Takes only accuracy rows of the `frequent_cue_shuffled` condition, from the evaluation or
    the per-pair table; refuses any other row, an odd trial count, and a result outside [0, 1]
    beyond `tolerance`. Returns two rows per row, conditions `frequent_cue_shuffled_agree`
    and `frequent_cue_shuffled_conflict` with contexts `shuffled_agree` and
    `shuffled_conflict`, every other column kept except these: `value` is the accuracy of
    that half, `n` its trial count (half the row's), `context_consistent_frac` the new value,
    and `prior_consistent_frac` the new value on the agreeing row and one minus it on the
    conflicting row. It is applied only where a caller asks for it, and before any estimand
    over reasoning steps: a running best keeps the consistency columns of its own budget while
    taking the best value, so a split after it would mix budgets.
    """
    fname = "split_shuffled"
    for name in ("condition", "metric", "value", "n", "prior_consistent_frac"):
        if name not in df.columns:
            raise ValueError(f"{fname}: the rows carry no column '{name}'")
    if df.empty:
        raise ValueError(f"{fname}: there are no rows to split")
    others = df[(df["condition"] != SHUFFLED_CONDITION) | (df["metric"] != "accuracy")]
    if not others.empty:
        found = sorted({f"{c}/{m}" for c, m in zip(others["condition"], others["metric"], strict=True)})
        raise ValueError(f"{fname} splits only accuracy rows of '{SHUFFLED_CONDITION}', not {found}")
    counts = df["n"].to_numpy()
    if (counts % 2 != 0).any():
        raise ValueError(f"{fname}: a row of an odd trial count cannot hold agree and conflict in equal numbers")
    value = df["value"].to_numpy(dtype=float)
    prior = df["prior_consistent_frac"].to_numpy(dtype=float)
    halves = {"agree": value + prior - CHANCE_ACCURACY, "conflict": value - prior + CHANCE_ACCURACY}
    parts = []
    for half, accuracy in halves.items():
        inside = (accuracy >= -tolerance) & (accuracy <= 1 + tolerance)
        if not inside.all():
            raise ValueError(f"{fname}: the {half} accuracy leaves [0, 1]; the rows are not balanced shuffled rows")
        accuracy = np.clip(accuracy, 0.0, 1.0)
        condition, context = SHUFFLED_SPLIT[half]
        part = df.copy()
        part["condition"] = pd.Series(condition, index=part.index, dtype=df["condition"].dtype)
        if "context" in part.columns:
            part["context"] = pd.Series(context, index=part.index, dtype=df["context"].dtype)
        part["value"] = accuracy
        part["n"] = (counts // 2).astype(df["n"].dtype)
        if "context_consistent_frac" in part.columns:
            part["context_consistent_frac"] = accuracy
        part["prior_consistent_frac"] = accuracy if half == "agree" else 1.0 - accuracy
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
