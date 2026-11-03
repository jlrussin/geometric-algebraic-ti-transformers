#!/bin/bash
# Settings every job script reads before its work: the checkout this file lives in, its .env,
# and the commit the launcher validated. Sourced by the job scripts, not run; the caller sets
# LAUNCH_COMMIT from its arguments first.

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [ ! -f .env ]; then
  echo "no .env in $PWD; a job reads its settings from the checkout that holds its scripts" >&2
  exit 2
fi
# The job reads the same settings the launcher did, whatever the scheduler exports.
set -a
# shellcheck disable=SC1091
. ./.env
set +a
RESULTS_ROOT="${RESULTS_ROOT:?a job reads and writes under the results root named in .env}"
# A task that starts after the checkout moved on would run code the launcher never validated.
LAUNCH_COMMIT="${LAUNCH_COMMIT:?a job needs the commit the launcher validated}"
if [ "$(git rev-parse HEAD)" != "$LAUNCH_COMMIT" ]; then
  echo "the checkout is at $(git rev-parse --short HEAD), not at the launched commit ${LAUNCH_COMMIT:0:7}" >&2
  exit 3
fi
# A batch job may start with a minimal PATH, so uv is named where it installs itself.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "no uv on PATH in this job" >&2; exit 127; }
