"""The cluster launcher and its job scripts: settings, resources, syntax, and permissions."""

from __future__ import annotations

import os
import re
import subprocess

import pytest

SCRIPTS = ("submit.sh", "array.sbatch", "export.sbatch", "fetch_tables.sh", "job_env.sh")


@pytest.fixture
def slurm_dir(repo_root):
    """The directory holding the launcher, the job scripts, and the settings template."""
    return repo_root / "scripts" / "slurm"


def _text(path) -> str:
    return path.read_text(encoding="utf-8")


def _sbatch_options(path) -> set[str]:
    """The long option names on the `#SBATCH` lines of a job script."""
    return set(re.findall(r"^#SBATCH\s+(--[a-z-]+)", _text(path), flags=re.MULTILINE))


def test_the_template_names_every_setting_the_scripts_read(slurm_dir):
    """Every CLUSTER_*, RESULTS_ROOT, or LOG_ROOT key a script reads is exported by the template."""
    exported = set(re.findall(r"^export ([A-Z_]+)=", _text(slurm_dir / "env.example.sh"), flags=re.MULTILINE))
    read: set[str] = set()
    for name in SCRIPTS:
        read |= set(re.findall(r"\b(CLUSTER_[A-Z_]+|RESULTS_ROOT|LOG_ROOT)\b", _text(slurm_dir / name)))
    assert read and read <= exported


def test_the_job_scripts_leave_the_partition_and_the_gpu_to_the_launcher(slurm_dir):
    """The job scripts fix only their shape; the launcher chooses partition, account, and GPU."""
    for name in ("array.sbatch", "export.sbatch"):
        options = _sbatch_options(slurm_dir / name)
        assert {"--time", "--mem", "--cpus-per-task"} <= options
        assert not options & {"--partition", "--account", "--gres", "--constraint", "--array"}


def test_the_launcher_gives_every_array_a_gpu_and_the_export_the_cpu_partition(slurm_dir):
    """An experiment array asks for one GPU on the GPU partition; a table export uses the CPU."""
    launcher = _text(slurm_dir / "submit.sh")
    assert "--gres=gpu:1" in launcher and "CLUSTER_GPU_CONSTRAINT" in launcher
    assert "scripts/slurm/array.sbatch" in launcher and "scripts/slurm/export.sbatch" in launcher
    assert "--device cuda" in _text(slurm_dir / "array.sbatch")
    tables_branch = launcher[launcher.index('if [ "$TABLES" = 1 ]; then\n  require') :]
    assert tables_branch.index("CLUSTER_CPU_PARTITION") < tables_branch.index("CLUSTER_GPU_PARTITION")


def test_every_task_checks_the_launched_commit_and_is_not_requeued(slurm_dir):
    """A job runs only the commit the launcher validated, and a failed task stays failed."""
    shared = _text(slurm_dir / "job_env.sh")
    assert "LAUNCH_COMMIT" in shared and "git rev-parse HEAD" in shared
    # The scheduler runs a copy of a job script, so the settings are found through the submit dir.
    for name in ("array.sbatch", "export.sbatch"):
        text = _text(slurm_dir / name)
        (source_line,) = re.findall(r"^\. .*job_env\.sh.*$", text, flags=re.MULTILINE)
        assert "SLURM_SUBMIT_DIR" in source_line and "BASH_SOURCE" not in text and "$0" not in text
    assert "--no-requeue" in _sbatch_options(slurm_dir / "array.sbatch")
    launcher = _text(slurm_dir / "submit.sh")
    assert 'COMMIT=$(git rev-parse HEAD)' in launcher and "CLUSTER_EXCLUDE_NODES" in launcher


def test_the_cluster_scripts_parse_as_bash(slurm_dir):
    """Every script and the template pass `bash -n`."""
    for name in SCRIPTS + ("env.example.sh",):
        subprocess.run(["bash", "-n", str(slurm_dir / name)], check=True)


def test_the_cluster_scripts_are_executable(slurm_dir):
    """The two scripts run by hand carry the executable bit."""
    for name in ("submit.sh", "fetch_tables.sh"):
        assert os.access(slurm_dir / name, os.X_OK)
