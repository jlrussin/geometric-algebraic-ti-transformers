# content_effects_no_cue

**Question.** When every sequence shows the same cue token, so that a trial does
not say whether its items follow the frequent cue's ordering, when does the
network follow the ordering shown in context and when does it follow the ordering
it has learned?

**Delta from base.** As `content_effects` (one frequent cue at the swept
proportion; the content test battery), with the cue hidden: the cue position holds
the frequent cue's token in every sequence. At proportion 1.0 every pretraining
trial is drawn under the frequent cue, so its pretraining sequences are those of
`content_effects`; the ICL test still differs, showing the frequent cue's token
for a novel cue.

**Axes.** frequent_cue_prop {0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0} x p_cot {0.5,
0.9, 0.99} x 10 seeds.

**Feeds.** The appendix figure "Experiments without an explicit cue" (proportion
0.2, p(CoT) 0.9).
