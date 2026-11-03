#!/bin/bash
# One array task runs one run of one experiment.
#
# The task index resolves to a run through `geo-alg-ti runs --index`, which is
# a pure function of the experiment's declared axes, so the same index always
# names the same run. The stages the launcher resolved run in order on that
# index, writing under the results root named in .env. The account, the
# partition, the GPU request, and the log paths come from scripts/slurm/submit.sh;
# a longer time limit than the one below goes on the sbatch command line. A task
# that fails is not requeued: its run is missing until it is launched again.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail

EXPERIMENT="${1:?an array task needs the experiment to run}"
LAUNCH_COMMIT="${2:?an array task needs the commit the launcher validated}"
STAGES="${3:?an array task needs the stages to run}"
OVERWRITE="${4:-}"
INDEX="${SLURM_ARRAY_TASK_ID:?an array task needs SLURM_ARRAY_TASK_ID}"

# The scheduler runs a copy of this script; the settings are read from the submitting checkout.
# shellcheck disable=SC1091
. "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/job_env.sh"

echo "host $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

echo "$EXPERIMENT index $INDEX; stages: $STAGES"
for STAGE in $STAGES; do
  echo "--- $STAGE"
  # The task runs on the GPU it was given and fails rather than fall back to the CPU.
  COMMAND=(uv run --no-sync geo-alg-ti "$STAGE" --experiment "$EXPERIMENT" --index "$INDEX" --out "$RESULTS_ROOT")
  COMMAND+=(--device cuda)
  if [ "$OVERWRITE" = "--overwrite" ]; then
    COMMAND+=(--overwrite)
  fi
  "${COMMAND[@]}"
done
