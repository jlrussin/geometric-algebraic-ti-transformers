# finetuning_reasoning

**Question.** When the network may reason over the context before answering
while it finetunes on one fixed ordering, does it learn that ordering less in
its weights, and does ablating the heads that read the context while it
reasons restore that learning?

**Delta from base.** Finetuning learning rate 0.001 and uninformative_prop
0.01 (the proportion of fixed-ordering trials with an uninformative context);
the battery scored every 100 steps on 50 contexts per ordered pair, also with
heads ablated: the heads the run ablates while it finetunes, or else the two
context-reading heads of the network it starts from.

**Axes.** ablated_heads {none, context_reading, random} (none, the two
context-reading heads, or two random heads outside the network's four
highest-ranked context-reading heads, ablated while finetuning) x
reasoning_prop {0, 1} (the proportion of fixed-ordering trials with a
congruent context on which the network writes eight reasoning steps of its
own before its scored answer) x 10 seeds; heads are ablated at proportion 1
only (4 configurations).

**Feeds.** The figure "Mechanisms underlying the geometric and algebraic
strategies"; the appendix figure "Ablating context-reading heads during
finetuning".
