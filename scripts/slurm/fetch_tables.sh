#!/bin/bash
# Copy one experiment's exported tables from the cluster into this checkout.
#
#   scripts/slurm/fetch_tables.sh <experiment>
#
# The tables land under results/<experiment>/tables/, where the figure scripts
# read them, as an exact copy of the exported set. An export is refused when it
# has no manifest or when any of its tables does not cover exactly the runs
# this checkout declares, so a sweep still in progress, or one made from
# another declaration, never replaces a complete set. The host, the user, and
# the results root come from .env.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

EXPERIMENT="${1:-}"
if [ -z "$EXPERIMENT" ]; then
  echo "usage: scripts/slurm/fetch_tables.sh <experiment>" >&2
  exit 2
fi
if [ ! -f "experiments/$EXPERIMENT/experiment.yaml" ]; then
  echo "no experiment is declared at experiments/$EXPERIMENT/" >&2
  exit 2
fi
if [ ! -f .env ]; then
  echo "no .env at $REPO_ROOT; copy scripts/slurm/env.example.sh to .env and fill it in" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a
for KEY in CLUSTER_HOST RESULTS_ROOT; do
  if [ -z "${!KEY:-}" ]; then
    echo "$KEY is empty in .env" >&2
    exit 2
  fi
done

REMOTE="${CLUSTER_USER:+$CLUSTER_USER@}$CLUSTER_HOST"
DESTINATION="results/$EXPERIMENT/tables"
STAGING="$DESTINATION.fetch"
PREVIOUS="$DESTINATION.previous"
# The staging copy never outlives the script, and tables set aside by an interrupted swap go
# back in place: at the start, and again if this run stops before its own swap completes.
cleanup() {
  rm -rf "$STAGING"
  if [ ! -d "$DESTINATION" ] && [ -d "$PREVIOUS" ]; then
    mv "$PREVIOUS" "$DESTINATION"
  fi
}
cleanup
rm -rf "$PREVIOUS"
trap cleanup EXIT
mkdir -p "$STAGING"
rsync -a --delete "$REMOTE:$RESULTS_ROOT/$EXPERIMENT/tables/" "$STAGING/"
if [ ! -f "$STAGING/MANIFEST.yaml" ]; then
  echo "the export of $EXPERIMENT has no manifest; nothing was replaced" >&2
  exit 3
fi
# Every table must hold exactly the declared runs and match the digest its manifest records.
if ! uv run --no-sync geo-alg-ti runs --experiment "$EXPERIMENT" --covered "$STAGING/MANIFEST.yaml"; then
  echo "the export of $EXPERIMENT was not fetched; nothing was replaced" >&2
  exit 3
fi
if [ -d "$DESTINATION" ]; then
  mv "$DESTINATION" "$PREVIOUS"
fi
mv "$STAGING" "$DESTINATION"
rm -rf "$PREVIOUS"
trap - EXIT
echo "$EXPERIMENT: tables in $DESTINATION/"
