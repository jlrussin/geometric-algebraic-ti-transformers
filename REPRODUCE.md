# Reproducing the transformer results

## Environment

`uv sync` installs the pinned environment from `uv.lock` (PyTorch 2.5.1): a CPU
wheel on Apple-silicon macOS, a CUDA 12.1 wheel on x86-64 Linux, the two
platforms the lock file covers. Logging to Weights & Biases needs
`uv sync --extra wandb`. `uv run pytest` runs the test suite (about 70 seconds
on a laptop CPU).

## Smoke test

`uv run geo-alg-ti smoke --config configs/smoke.yaml` exercises pretraining,
finetuning, the test battery, export, and one figure at laptop scale.

## Running one configuration locally

`uv run geo-alg-ti pretrain --set task.p_cot=0.5 --seed 0 --out results/scratch`
runs the canonical configuration with one setting overridden; `--config` names
a file of such overrides applied on top of `configs/base.yaml`, as
`configs/smoke.yaml` is. Each stage of a declared run is one subcommand:
`uv run geo-alg-ti pretrain --experiment canonical_pretraining --index 0`, then
`uv run geo-alg-ti finetune --experiment symbolic_distance --index 0`
(`geo-alg-ti runs --experiment <name>` lists the runs in index order, and
`--stages` the stages).

## Running an experiment on a cluster

The launcher targets SLURM. Copy `scripts/slurm/env.example.sh` to `.env` at
the top of the checkout and fill in the account, a GPU partition, a CPU
partition, the node feature the CUDA build needs (if any), the nodes to avoid
(if any), the login host, and an absolute results root on cluster storage. Run
`uv sync` once on a login host (`uv sync --extra wandb` when logging to Weights
& Biases); the jobs use that environment without syncing again.

`scripts/slurm/submit.sh <experiment>` launches one array whose size is
`geo-alg-ti runs --experiment <experiment> --count`; each task resolves its
index to one run and runs the declared stages in order on one GPU of the GPU
partition, writing under the results root; a task whose GPU cannot be used
fails rather than continuing on the CPU, and a failed task is not requeued.
The launcher refuses a checkout that is not on `main`, has uncommitted
changes, or differs from `origin/main`, so every result traces to a commit,
and every task checks when it starts that the checkout still holds the
launched commit. Options after the experiment name go to sbatch as they are:

- `--array=0` launches the first run alone (every index must name a
  declared run);
- `--dependency=afterok:<job id> --kill-on-invalid-dep=yes` waits for the
  experiment whose checkpoints this one continues from;
- `--array=0-99%20` throttles a large array;
- `--time=12:00:00` raises the eight-hour limit of the job script.

The launcher's own flags go before the experiment name: `--dry-run` prints
the sbatch command line without submitting, and `--overwrite` lets the tasks
replace results already written. `symbolic_distance`,
`finetuning_hyperparameters`, `conflicting_vs_disjoint`,
`finetuning_reasoning`, and `finetuning_reasoning_hyperparameters` continue
from `canonical_pretraining`, and `representation` and `components` score
`content_effects`, so those launch after the experiment they depend on has
finished.

`scripts/slurm/submit.sh --tables <experiment>` exports the tidy tables as one
task on the CPU partition, naming any declared run it could not find, and
`scripts/slurm/fetch_tables.sh <experiment>` copies them into
`results/<experiment>/tables/` on a laptop (whose own `.env` names the login
host and the results root), where the figure scripts read them; an export
whose tables do not cover exactly the runs the checkout declares, or that has
no manifest, is refused.

Results are bitwise reproducible on the same GPU model with the same installed
environment. Across GPU models, the RTX 3090 and the RTX A5500 match bitwise,
while the RTX A5000 differs in the last bits of a floating-point reduction, as
other GPU models may.

## Exporting the tidy tables

`uv run geo-alg-ti export --experiment <experiment>` writes the tables and the
manifest under `results/<experiment>/tables/`. It reads the run directories
under `--out` (default `results/`) and writes there, so with the default a local
export replaces the committed tables.

## Building the figures

From the top of the checkout, `uv run python figures/fig_<topic>.py` (or
`uv run geo-alg-ti figure --name <topic>`) reads only the tidy tables under
`results/` and writes a PDF at print size plus a PNG preview to
`figures/out/`. The predictions in the figures "Symbolic distance effects
(SDE)", "Content effects in transitive inference", and "Data-distribution
effects" (`fig_formalization.py`) need no tables. Text is set in Helvetica or
Arial when one is installed, otherwise in DejaVu Sans; the predictions use the
bundled Source Sans 3.

## Checking the printed results

`uv run geo-alg-ti numbers` recomputes the results the paper prints about the
transformers from the tidy tables under `results/` and writes
`results/numbers.csv`: one row per number, with where the paper prints it, the
printed and the recomputed value, its source, and whether the two agree at the
printed precision. It exits non-zero, naming each number that differs.

## Which experiment produces which figure

Each script reads the tables of the experiments listed; an experiment that
continues another's checkpoints (`symbolic_distance`,
`finetuning_hyperparameters`, `conflicting_vs_disjoint`,
`finetuning_reasoning`, and `finetuning_reasoning_hyperparameters` from
`canonical_pretraining`; `representation` and `components` from
`content_effects`) runs after the one it depends on, as the cluster section
says.

| Topic (`fig_<topic>.py`) | Figure | Experiments |
|---|---|---|
| `symbolic_distance` | Symbolic distance effects (SDE) | `symbolic_distance` |
| `skew` | Data-distribution effects | `skew` |
| `content_effects` | Content effects in transitive inference | `content_effects` |
| `mechanisms` | Mechanisms underlying the geometric and algebraic strategies | `representation`, `components`, `finetuning_reasoning` |
| `basic_tests` | Additional symbolic-distance results (appendix) | `symbolic_distance` |
| `cot_algorithms` | Alternative chain-of-thought algorithms (appendix) | `chain_algorithms` |
| `architectures` | Architecture and learning-rate sweep (appendix) | `architectures` |
| `finetuning_hyperparameters` | Replay during finetuning (appendix) | `finetuning_hyperparameters` |
| `skew_learning` | Learning over pretraining by skew (appendix) | `skew` |
| `number_of_orderings_and_conflict` | Additional data-distribution manipulations (appendix) | `number_of_orderings`, `conflicting_vs_disjoint` |
| `cot_proportion` | Chain-of-thought supervision improves sample efficiency (appendix) | `cot_proportion` |
| `content_grid` | Content effects across pretraining distributions (appendix) | `content_effects` |
| `no_cue` | Experiments without an explicit cue (appendix) | `skew`, `skew_no_cue`, `content_effects_no_cue` |
| `content_grid_no_cue` | content grid without the cue (not in the paper) | `content_effects_no_cue` |
| `representation` | Additional mechanism results (appendix) | `representation`, `components` |
| `finetuning_reasoning` | Ablating context-reading heads during finetuning (appendix) | `finetuning_reasoning` |
| `finetuning_reasoning_hyperparameters` | Ablation during finetuning across learning rates and proportions of uninformative context trials (appendix) | `finetuning_reasoning_hyperparameters` |
| `formalization` | predictions in Symbolic distance effects (SDE), Content effects in transitive inference, and Data-distribution effects | none (the `formalization` module) |

## Expected runtime and storage

Measured on NVIDIA RTX 3090 nodes with four CPU cores per task, with the
canonical configuration, from the runs' own logs; the evaluations are already
counted in the run times, and a battery's time grows with the number of test
conditions it declares:

| Item | Time |
|---|---|
| Pretraining run (100,000 steps, 41 evaluations) | 40 minutes; 23 ms per step |
| Finetuning run (1,600 steps, 9 evaluations) | about a minute |
| Finetuning run of `finetuning_reasoning_hyperparameters`, each evaluation on 10 contexts per ordered pair scored intact and with heads ablated (1,600 steps, 5 evaluations) | about 7 minutes with eight reasoning steps on every fixed-ordering trial with a congruent context; about a minute without |
| Finetuning run of `finetuning_reasoning`, each evaluation on 50 contexts per ordered pair scored intact and with heads ablated (1,600 steps, 17 evaluations) | about 10 minutes with eight reasoning steps on every fixed-ordering trial with a congruent context; about 5 minutes without |
| Test battery of three test conditions (1,260 trials, reasoning steps 0 to 8) | 1.0 s on the GPU |
| Test battery of five test conditions (2,100 trials, reasoning steps 0 to 8) | 1.6 s on the GPU |

On a laptop CPU (Apple M5 Max, four threads) the two batteries take 2.4 s
and 3.6 s. A run directory takes about 40 MB, almost all of it the checkpoint
with its optimizer state (about 12 MB for the smallest architectures); the
tidy tables take about 50 KB per run (about 200 KB for `finetuning_reasoning`,
whose per-pair tables keep every evaluation). The cluster experiments come to
1,160 pretraining runs, 504 finetuning runs, and 40 representation runs: about
890 GPU hours as the scheduler accounts them, and about 62 GB of run
directories.
The 10 runs of `components` were scored on the same laptop CPU, three networks
at a time with three threads each: 11 to 15 minutes per network (about 6
minutes for one network alone). The committed probe and edit tables of
`representation` were scored on that laptop CPU too. To repeat this, copy the
`content_effects` checkpoints into
`results/content_effects/runs/<run>/checkpoints/`, score each declared run
with `--device cpu` (`represent` and `evaluate` for `representation`, `ablate`
for `components`), and export each experiment.
