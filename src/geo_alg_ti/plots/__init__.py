"""Panel functions for the paper's figures.

Every function is `make_<name>(ax, df, **settings) -> dict` (the summary of the
rank-direction edits and the head ablations takes both tables): it draws onto the
given axes only, never creates a figure or saves a file, takes every
aesthetic and pin as a keyword with the current behavior as its default, and
returns a dictionary of the statistics it plotted, including the seed count.
Seeds are the only axis a panel averages over across networks (random heads
ablated are the mean over a network's random draws within the seed); distances
are shown individually or pooled over two and above; reasoning steps are pinned,
averaged over one to five, maximized, summarized by the first budget to
criterion, or replaced by the running best up to each budget, the five
estimands of `geo_alg_ti.tables`. A swept axis of the experiment that a
panel neither draws nor loops over must be pinned first; the panel raises
naming it otherwise.
Depends only on numpy, pandas, matplotlib, and `geo_alg_ti.tables`.
"""

from geo_alg_ti.plots import content, distance, distribution, learning, placement, representation, style
from geo_alg_ti.plots.content import (
    make_context_accuracy_by_distance,
    make_context_accuracy_by_reasoning_steps,
)
from geo_alg_ti.plots.distance import (
    make_accuracy_bars,
    make_accuracy_by_distance,
    make_accuracy_numbers,
    make_accuracy_by_reasoning_steps,
    make_pair_accuracy,
    make_response_classes_by_reasoning_steps,
    make_steps_to_criterion_by_distance,
)
from geo_alg_ti.plots.distribution import (
    make_accuracy_vs_skew,
    make_emergence_by_skew,
    make_icl_iwl_dissociation,
    make_ordering_frequency_schematic,
)
from geo_alg_ti.plots.learning import (
    make_accuracy_by_step,
    make_pair_type_bars,
    make_pair_type_by_step,
    make_pooled_accuracy_vs_sweep,
    make_reasoning_gain_by_step,
)
from geo_alg_ti.plots.representation import (
    make_edit_accuracy_by_distance,
    make_head_ablation_scatter,
    make_intervention_bars,
    make_probe_accuracy_by_layer,
    make_write_by_strength,
)

__all__ = [
    "content",
    "distance",
    "distribution",
    "learning",
    "placement",
    "representation",
    "style",
    "make_accuracy_bars",
    "make_accuracy_by_distance",
    "make_accuracy_numbers",
    "make_accuracy_by_reasoning_steps",
    "make_accuracy_by_step",
    "make_accuracy_vs_skew",
    "make_context_accuracy_by_distance",
    "make_context_accuracy_by_reasoning_steps",
    "make_edit_accuracy_by_distance",
    "make_emergence_by_skew",
    "make_head_ablation_scatter",
    "make_icl_iwl_dissociation",
    "make_intervention_bars",
    "make_ordering_frequency_schematic",
    "make_pair_accuracy",
    "make_pair_type_bars",
    "make_pair_type_by_step",
    "make_pooled_accuracy_vs_sweep",
    "make_probe_accuracy_by_layer",
    "make_reasoning_gain_by_step",
    "make_response_classes_by_reasoning_steps",
    "make_steps_to_criterion_by_distance",
    "make_write_by_strength",
]
