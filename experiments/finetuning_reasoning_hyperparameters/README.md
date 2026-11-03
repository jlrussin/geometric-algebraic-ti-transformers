# finetuning_reasoning_hyperparameters

**Question.** When the network may reason over the context before answering
while it finetunes on one fixed ordering, at which finetuning learning rates
and proportions of fixed-ordering trials with an uninformative context does it
learn that ordering less in its weights, and does ablating the heads that read
the context while it reasons restore that learning?

**Delta from base.** The battery scored every 400 finetuning steps, also with
heads ablated: the heads the run ablates while it finetunes, or else the two
context-reading heads of the network it starts from.

**Axes.** learning_rate {0.0001, 0.0003, 0.001} x uninformative_prop {0, 0.01,
0.03, 0.1} (the proportion of fixed-ordering trials with an uninformative
context) x ablated_heads {none, context_reading, random} (none, the two
context-reading heads, or two random heads outside the network's four
highest-ranked context-reading heads, ablated while finetuning) x
reasoning_prop {0, 1} (the proportion of fixed-ordering trials with a
congruent context on which the network writes eight reasoning steps of its
own before its scored answer) x seeds 0-2; heads are ablated at proportion 1
only (48 configurations).

**Feeds.** The appendix figure "Ablation during finetuning across learning rates
and proportions of uninformative context trials".
