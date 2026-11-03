#!/bin/bash
# Template for the local .env. Copy it to the top of the checkout, fill it in,
# and keep it out of version control. The launcher sources .env with `set -a`,
# and each job sources it again, so every key below reaches the job.

export CLUSTER_ACCOUNT=""         # SLURM account passed as --account
export CLUSTER_GPU_PARTITION=""   # partition for the experiment arrays (every stage runs on a GPU)
export CLUSTER_CPU_PARTITION=""   # partition for the table export
export CLUSTER_GPU_CONSTRAINT=""  # node feature the CUDA build needs; empty to omit
export CLUSTER_EXCLUDE_NODES=""   # nodes never to schedule on, comma-separated; empty to omit
export CLUSTER_HOST=""            # login host, or an ssh alias, for rsync
export CLUSTER_USER=""            # user name on the cluster; empty uses the alias
export RESULTS_ROOT=""            # absolute results root on cluster storage
export LOG_ROOT=""                # absolute directory for job logs; empty puts them beside the checkout

# Settings without an empty form stay commented out rather than exported empty.
# export UV_CACHE_DIR=""          # uv cache on a filesystem with room for it
# export WANDB_API_KEY=""         # only when logging.wandb.enabled is true
