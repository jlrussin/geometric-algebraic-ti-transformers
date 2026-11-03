# conflicting_vs_disjoint

**Question.** Does finetuning on several orderings of the same items under
different cues degrade readout differently from finetuning on item-disjoint
orderings under one cue?

**Delta from base.** Evaluation at zero reasoning steps only.

**Axes.** layout {conflicting, disjoint} x n_orderings {1..10}, minus the
duplicate single-ordering design, x 10 seeds (19 configurations).

**Feeds.** The appendix figure "Additional data-distribution manipulations".
