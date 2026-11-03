#!/bin/bash
# Launch one experiment as a SLURM array, or export its tables, from a clean checkout.
#
#   scripts/slurm/submit.sh [--dry-run] [--overwrite] [--tables] <experiment> [sbatch options...]
#
# The array holds one task per run the experiment declares, and each task runs
# that run's stages in order on one GPU of the GPU partition, under the node
# constraint when .env names one. `--tables` submits one task on the CPU
# partition that exports the experiment's tidy tables instead. `--overwrite`
# lets the tasks replace results already written. `--dry-run` prints the
# sbatch command line instead of running it. Options after the experiment
# name go to sbatch as they are; an `--array` given there replaces the one
# computed from the run count, so one run (`--array=0`) or a throttled array
# (`--array=0-99%20`) launches the same way, and `--time` raises the limit.
#
# The account, the partitions, the nodes to avoid, and the results root come
# from .env, which is not under version control; copy scripts/slurm/env.example.sh
# to .env at the top of the checkout and fill it in. Every result on cluster
# disk traces to a commit on origin/main, so a checkout that is on another
# branch, carries uncommitted changes, or sits at another commit is refused,
# and every task checks that it still runs the commit that was launched.

set -euo pipefail

usage() {
  echo "usage: scripts/slurm/submit.sh [--dry-run] [--overwrite] [--tables] <experiment> [sbatch options...]" >&2
}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

DRY_RUN=0
OVERWRITE=0
TABLES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --tables) TABLES=1; shift ;;
    -*) echo "unknown option '$1'" >&2; usage; exit 2 ;;
    *) break ;;
  esac
done
EXPERIMENT="${1:-}"
if [ -z "$EXPERIMENT" ]; then
  usage
  exit 2
fi
shift
if [ ! -f "experiments/$EXPERIMENT/experiment.yaml" ]; then
  echo "no experiment is declared at experiments/$EXPERIMENT/" >&2
  exit 2
fi
if [ "$TABLES" = 1 ] && [ "$OVERWRITE" = 1 ]; then
  echo "--overwrite has no effect on a table export" >&2
  exit 2
fi
# An `--array` after the experiment name replaces the computed one; its indices must exist.
ARRAY_SPEC=""
PREVIOUS=""
for OPTION in "$@"; do
  case "$PREVIOUS" in
    --array | -a) ARRAY_SPEC="$OPTION" ;;
  esac
  case "$OPTION" in
    --array=*) ARRAY_SPEC="${OPTION#--array=}" ;;
    -a?*) ARRAY_SPEC="${OPTION#-a}" ;;
  esac
  PREVIOUS="$OPTION"
done
if [ "$TABLES" = 1 ] && [ -n "$ARRAY_SPEC" ]; then
  echo "a table export is one task; --array does not apply" >&2
  exit 2
fi

if [ ! -f .env ]; then
  echo "no .env at $REPO_ROOT; copy scripts/slurm/env.example.sh to .env and fill it in" >&2
  exit 2
fi
# Exporting while sourcing puts every key of .env into the environment the job inherits.
set -a
# shellcheck disable=SC1091
. ./.env
set +a

# A launch is refused unless the checkout is main, clean, and at origin/main;
# a dry run reports what a launch would refuse and goes on.
refuse() {
  if [ "$DRY_RUN" = 1 ]; then
    echo "launch guard: would refuse, $1" >&2
  else
    echo "$1" >&2
    exit 3
  fi
}
if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
  refuse "this checkout is not on main"
fi
if [ -n "$(git status --porcelain)" ]; then
  refuse "this checkout has uncommitted changes"
fi
if ! git fetch --quiet origin main; then
  refuse "origin is unreachable, so the checkout cannot be compared with origin/main"
elif [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  refuse "this checkout differs from origin/main"
fi

require() {
  if [ -z "${!1:-}" ]; then
    echo "$1 is empty in .env" >&2
    exit 2
  fi
}
require CLUSTER_ACCOUNT
require RESULTS_ROOT
if [ "$TABLES" = 1 ]; then
  require CLUSTER_CPU_PARTITION
  PARTITION="$CLUSTER_CPU_PARTITION"
else
  require CLUSTER_GPU_PARTITION
  PARTITION="$CLUSTER_GPU_PARTITION"
fi
# A relative root would write into the checkout, which the guard above then refuses forever.
outside_checkout() { # name, value
  case "$2" in
    /*) ;;
    *) echo "$1 must be an absolute path, not '$2'" >&2; exit 2 ;;
  esac
  case "$2" in
    "$REPO_ROOT" | "$REPO_ROOT"/*)
      echo "$1 lies inside the checkout; $3 belong on cluster storage" >&2
      exit 2 ;;
  esac
}
outside_checkout RESULTS_ROOT "$RESULTS_ROOT" results
# Logs sit beside the checkout unless .env names a place, so a job leaves the tree untouched.
if [ -n "${LOG_ROOT:-}" ]; then
  outside_checkout LOG_ROOT "$LOG_ROOT" logs
fi
LOG_DIR="${LOG_ROOT:-${REPO_ROOT}-logs}/$EXPERIMENT"

COUNT=$(uv run --no-sync geo-alg-ti runs --experiment "$EXPERIMENT" --count)
if [ "$COUNT" -lt 1 ]; then
  echo "experiment $EXPERIMENT declares no run to launch" >&2
  exit 2
fi
if [ -n "$ARRAY_SPEC" ]; then
  # The largest index in `a-b:s,c,...`, ignoring a `%n` throttle, must name a declared run.
  LARGEST=0
  INDICES="${ARRAY_SPEC%%%*}"
  for RANGE in ${INDICES//,/ }; do
    RANGE="${RANGE%%:*}"
    LAST="${RANGE##*-}"
    if ! [ "$LAST" -ge 0 ] 2>/dev/null; then
      echo "cannot read the array indices '$ARRAY_SPEC'" >&2
      exit 2
    fi
    if [ "$LAST" -gt "$LARGEST" ]; then LARGEST="$LAST"; fi
  done
  if [ "$LARGEST" -ge "$COUNT" ]; then
    echo "index $LARGEST is out of range: $EXPERIMENT declares $COUNT runs, indices 0-$((COUNT - 1))" >&2
    exit 2
  fi
fi
COMMIT=$(git rev-parse HEAD)
if [ "$DRY_RUN" = 0 ]; then
  mkdir -p "$RESULTS_ROOT" "$LOG_DIR"
fi

CMD=(sbatch --account="$CLUSTER_ACCOUNT" --partition="$PARTITION")
if [ -n "${CLUSTER_EXCLUDE_NODES:-}" ]; then
  CMD+=(--exclude="$CLUSTER_EXCLUDE_NODES")
fi
if [ "$TABLES" = 1 ]; then
  CMD+=(--job-name="$EXPERIMENT-tables" --output="$LOG_DIR/tables-%j.out" --error="$LOG_DIR/tables-%j.err")
  CMD+=("$@" scripts/slurm/export.sbatch "$EXPERIMENT" "$COMMIT")
  WHAT="tables"
else
  CMD+=(--gres=gpu:1)
  if [ -n "${CLUSTER_GPU_CONSTRAINT:-}" ]; then
    CMD+=(--constraint="$CLUSTER_GPU_CONSTRAINT")
  fi
  CMD+=(--job-name="$EXPERIMENT" --output="$LOG_DIR/%A_%a.out" --error="$LOG_DIR/%A_%a.err")
  STAGES=$(uv run --no-sync geo-alg-ti runs --experiment "$EXPERIMENT" --stages)
  if [ -z "$ARRAY_SPEC" ]; then
    CMD+=(--array="0-$((COUNT - 1))")
  fi
  CMD+=("$@" scripts/slurm/array.sbatch "$EXPERIMENT" "$COMMIT" "$STAGES")
  if [ "$OVERWRITE" = 1 ]; then
    CMD+=(--overwrite)
  fi
  WHAT="$COUNT runs"
fi

echo "$EXPERIMENT: $WHAT at ${COMMIT:0:7}, partition $PARTITION, logs under $LOG_DIR"
if [ "$DRY_RUN" = 1 ]; then
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi
"${CMD[@]}"
