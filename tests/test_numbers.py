"""The paper's numbers: the committed `results/numbers.csv` is what the committed tidy tables
give, every number is as printed, and each value is written at the printed precision."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from geo_alg_ti import cli, numbers

COLUMNS = ["id", "site", "printed", "computed", "source", "match"]


def _committed(repo_root) -> pd.DataFrame:
    return pd.read_csv(repo_root / "results" / "numbers.csv", dtype=str, keep_default_na=False)


def test_every_committed_number_is_as_printed(repo_root):
    """The committed table lists every number once, and each matches its printed value."""
    table = _committed(repo_root)
    assert list(table.columns) == COLUMNS
    assert len(table) == len(numbers.PAPER_NUMBERS)
    assert (table["match"] == "True").all() and (table["printed"] == table["computed"]).all()


def test_the_numbers_command_reproduces_the_committed_table(repo_root, tmp_path):
    """`numbers` rewrites the committed table byte for byte."""
    out = tmp_path / "numbers.csv"
    assert cli.main(["numbers", "--out", str(out)]) == 0
    assert out.read_text() == (repo_root / "results" / "numbers.csv").read_text()


def test_the_numbers_command_refuses_a_login_node(tmp_path, monkeypatch):
    """Like every verb but `runs`, `numbers` exits on a login node before writing anything."""
    monkeypatch.setattr("socket.gethostname", lambda: "login9")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv(cli.LOGIN_NODE_BYPASS_ENV, raising=False)
    with pytest.raises(SystemExit):
        cli.main(["numbers", "--out", str(tmp_path / "numbers.csv")])
    assert not (tmp_path / "numbers.csv").exists()


def test_ids_are_numbered_in_order_and_sites_are_known():
    """Ids run tf-1, tf-2, ... in table order; every site names a place the paper prints."""
    ids = [row[0] for row in numbers.PAPER_NUMBERS]
    assert ids == [f"tf-{n}" for n in range(1, len(ids) + 1)]
    assert all(row[1] in numbers.SITES for row in numbers.PAPER_NUMBERS)
    assert all(row[2] for row in numbers.PAPER_NUMBERS)


@pytest.mark.parametrize("value, printed, expected", [
    (0.8777, "0.88", "0.88"), (0.125, "0.13", "0.13"), (0.000436, "0.00044", "0.00044"),
    (3, "Three", "Three"), (4, "four", "four"), (2.5, "four", "2.5"), (1022, "1,022", "1,022"),
    (0.93, ">= 0.90", ">= 0.90"), (0.85, ">= 0.90", "0.85"), (0.84, "< 0.90", "< 0.90"),
    ("two-layer", "two-layer", "two-layer"), (float("nan"), "3", "nan"),
])
def test_values_are_written_at_the_printed_precision(value, printed, expected):
    assert numbers.format_like(value, printed) == expected


def test_the_numbers_module_loads_without_torch_or_matplotlib():
    """Recomputing the numbers needs only the tables: the module imports neither library."""
    code = "import sys, geo_alg_ti.numbers; print(sorted({'torch', 'matplotlib'} & set(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
