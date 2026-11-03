# skew

**Question.** How does skew in the cue-frequency distribution during pretraining
trade off in-context ordering against reading out the orderings the network has
learned?

**Delta from base.** Cue distribution Zipf over the 100 training cues; p(CoT)
0.9, so that the orderings the network has learned can be read out at 0
reasoning steps.

**Axes.** skew_alpha {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10} x 10
seeds.

**Feeds.** The figure "Data-distribution effects"; the appendix figures
"Learning over pretraining by skew" and "Experiments without an explicit cue".
