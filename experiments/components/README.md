# components

**Question.** Which attention heads read the context while the network writes
its reasoning steps, and what do direct and reasoned answers lose without them?

**Delta from base.** The content-effects networks with one frequent cue at
proportion 0.2 and p(CoT) 0.9, scored on the content test battery at every
reasoning-step value 0..8.

**Axes.** 10 seeds.

**Context-reading heads.** On every trial of the ICL test and of the reversed
context, the intact network writes its chain to eight reasoning steps, and one
forward pass over the prompt and that chain gives each head's attention
probabilities. Every step written before the network first states the queried
relation is read from its four positions onto the context relations that
mention either of its items, four positions each; a malformed step, or one
whose items no context relation mentions, is left out. A head's score is its
attention mass onto those relations, averaged over the step's positions and
then over every step of every trial. The heads are ranked within each network
from the highest score; no ablation enters their order. Each head's score and
rank are in the `head_scores` table.

**Ablations.** A head's output is replaced by its mean at the same position,
over greedy rollouts of 1,024 pretraining-distribution prompts; the means of
the sequence and of the query items appended at each reasoning-step value are
kept apart. The battery is scored with the network intact, with each of its 32
heads ablated alone, with its k highest-ranked heads ablated together for k in
{1, 2, 4}, and, as their control, with k heads drawn at random from the heads
outside its top 2k, 20 draws for each k; the `ablate` table holds the battery
under every set.

**Feeds.** The figure "Mechanisms underlying the geometric and algebraic
strategies"; the appendix figure "Additional mechanism results".
