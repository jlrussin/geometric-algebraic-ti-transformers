# geometric-algebraic-ti-transformers

Transformer experiments for the paper *Geometric and algebraic inference in
humans and neural networks*.

## What this is

One pretraining configuration, one finetuning configuration, one two-factor
test battery, and one named delta per paper experiment; everything the paper
reports about transformers trained from scratch is reproducible from this
repository.

## Install

`uv sync` with Python 3.12 on Apple-silicon macOS (a CPU wheel) or x86-64
Linux (a CUDA 12.1 wheel), the two platforms the lock file covers.

## Quickstart

`uv run geo-alg-ti smoke --config configs/smoke.yaml` runs the whole path in
seconds on CPU.

## Repository layout

`src/geo_alg_ti/` the package; `configs/` the canonical configuration, the
smoke delta, and the battery presets; `experiments/` one declaration per paper
experiment; `figures/` one script per figure; `scripts/slurm/` the cluster
launcher; `results/` committed tidy tables and `numbers.csv`; `tests/` the
test suite.

## Experiments

Each `experiments/<name>/` holds a declaration and a short README:
`canonical_pretraining`, `symbolic_distance`, `finetuning_hyperparameters`,
`skew`, `number_of_orderings`, `conflicting_vs_disjoint`, `content_effects`,
`representation`, `chain_algorithms`, `architectures`, `cot_proportion`,
`skew_no_cue`, `content_effects_no_cue`, `components`, `finetuning_reasoning`,
`finetuning_reasoning_hyperparameters`.

## Results and tables

Running an experiment saves each run's checkpoint and result tables under
`results/<experiment>/runs/`; `export` collects them into the tidy tables under
`results/<experiment>/tables/`, which are committed with a manifest, so every
figure and printed number can be redrawn without rerunning. `numbers`
recomputes from these tables the results the paper prints about the
transformers and writes them to `results/numbers.csv`.

## Figures

Panel functions live in `geo_alg_ti.plots`; `figures/fig_<topic>.py` redraws
each paper figure's transformer panels from the tidy tables.

## Reproducing the paper

See `REPRODUCE.md`.

## Related repositories

Code for the paper's other experiments:

- Humans: https://github.com/jlrussin/geometric-algebraic-ti-humans
- Large language models: https://github.com/joonhwakim/geometric-algebraic-ti-llms

## Citation

Russin, J., Kim, J., Pavlick, E., & Frank, M. J. (2026). Geometric and algebraic
inference in humans and neural networks. arXiv:2611.00387.
https://arxiv.org/abs/2611.00387

See `CITATION.cff`.

## License

MIT. The Source Sans 3 fonts in `src/geo_alg_ti/formalization/fonts/` are under
the SIL Open Font License 1.1 (`LICENSE.md` there).

**Development notes.** This code was developed with substantial assistance
from AI coding tools (Claude Code), used for drafting code, tests,
documentation, and analysis scripts. The human authors designed the
experiments, specified and reviewed all code, and are responsible for its
correctness.
