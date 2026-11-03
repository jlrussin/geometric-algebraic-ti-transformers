# skew_no_cue

**Question.** When every sequence shows the same cue token, so that a trial does
not say which ordering its items follow, how does skew in the cue-frequency
distribution trade off in-context ordering against reading out the orderings the
network has learned?

**Delta from base.** As `skew` (cue distribution Zipf over the 100 training cues,
p(CoT) 0.9), with the cue hidden: the cue position holds the token of cue 0, the
most frequent cue under Zipf skew, in every sequence. The orderings and their
frequencies are those of `skew`.

**Axes.** skew_alpha {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10} x 10
seeds.

**Feeds.** The appendix figure "Experiments without an explicit cue".
