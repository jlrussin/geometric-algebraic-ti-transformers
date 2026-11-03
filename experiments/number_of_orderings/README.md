# number_of_orderings

**Question.** How many distinct cued orderings must pretraining contain before
the network orders novel items from context?

**Delta from base.** None beyond the swept axis; the 16 reserved cues and the
pool of 300 items are unchanged.

**Axes.** n_training_cues {1, 2, 5, 10, 100, 1000} x 10 seeds.

**Feeds.** The appendix figure "Additional data-distribution manipulations".
