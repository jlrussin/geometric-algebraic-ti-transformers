"""The committed tidy tables: each manifest matches its tables, and every figure script builds
from them (the predictions need none)."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pytest
import yaml

from geo_alg_ti import __version__, cli, config, tables

RESULTS = config.repo_root() / "results"
MANIFESTS = sorted(RESULTS.glob("*/tables/MANIFEST.yaml"))
FIGURES = sorted(path.stem.removeprefix("fig_") for path in (config.repo_root() / "figures").glob("fig_*.py"))


def test_every_table_directory_has_a_manifest():
    """Each experiment's tables under `results/` come with their manifest."""
    directories = sorted(RESULTS.glob("*/tables"))
    assert directories and directories == [manifest.parent for manifest in MANIFESTS]


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda path: path.parent.parent.name)
def test_each_manifest_matches_its_tables(manifest):
    """Every table the manifest lists sits beside it with the digest and row count it records,
    covers exactly the runs the experiment declares, and carries the experiment's swept axes;
    no table sits there unlisted."""
    experiment = config.load_experiment(manifest.parent.parent.name)
    tables.check_export(manifest, [run.run_id for run in config.enumerate_runs(experiment)])
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert document["package_version"] == __version__
    entries = document["tables"]
    assert sorted(entry["path"] for entry in entries) == sorted(path.name for path in manifest.parent.glob("*.parquet"))
    for entry in entries:
        assert entry["path"] == f"{entry['table']}.parquet" and entry["table"] in tables.TABLE_NAMES
        assert pq.ParquetFile(manifest.parent / entry["path"]).metadata.num_rows == entry["rows"]
        assert entry["axes"] == list(experiment.swept_columns)


@pytest.mark.parametrize("name", FIGURES)
def test_each_figure_draws_from_the_committed_tables(name, tmp_path):
    """Each figure script builds its figure, from the committed tables where it reads any, and
    saves it."""
    shared = cli._load_figure_module("_shared")
    script = cli._load_figure_module(f"fig_{name}")
    fig = script.build(argparse.Namespace(results=RESULTS, out=tmp_path))
    try:
        paths = shared.save(fig, name, out_dir=tmp_path)
    finally:
        plt.close(fig)
    assert [path.suffix for path in paths] == [".pdf", ".png"]
    assert all(path.stat().st_size > 0 for path in paths)
