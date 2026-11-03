# content_effects

**Question.** As one cue becomes frequent during pretraining, when does the
network follow the ordering shown in context and when does it follow the
ordering it has learned?

**Delta from base.** One frequent cue at the swept proportion; the content test
battery (uninformative, congruent, reversed, shuffled contexts on the frequent
cue's ordering, plus the ICL test). The shuffled context shows the ordering's
items in a new order with both queried items interior to it; its symbolic
distance is the one shown, the distance in the frequent cue's ordering is drawn
independently of it, and the shown order of the queried pair agrees with the
frequent cue's ordering on exactly half the queries at every shown distance.

**Axes.** frequent_cue_prop {0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0} x p_cot {0.5,
0.9, 0.99} x 10 seeds.

**Feeds.** The figure "Content effects in transitive inference" (proportion 0.2,
p(CoT) 0.9); the appendix figure "Content effects across pretraining
distributions".
