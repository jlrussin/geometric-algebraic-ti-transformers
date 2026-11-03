#!/bin/bash
# One task exports one experiment's tidy tables and manifest.
#
# Every run directory the experiment declares is collected from the results
# root named in .env; a run that has not been written yet is skipped and named
# in the log, so a sweep still in progress exports what exists. The partition
# and the log paths come from scripts/slurm/submit.sh.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

EXPERIMENT="${1:?a table export needs the experiment to collect}"
LAUNCH_COMMIT="${2:?a table export needs the commit the launcher validated}"

# The scheduler runs a copy of this script; the settings are read from the submitting checkout.
# shellcheck disable=SC1091
. "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/job_env.sh"

echo "host $(hostname)"
uv run --no-sync geo-alg-ti export --experiment "$EXPERIMENT" --out "$RESULTS_ROOT"
