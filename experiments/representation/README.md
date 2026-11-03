# representation

**Question.** Is rank linearly decodable by layer, and does erasing, inverting,
or overwriting the rank direction change which ordering the network follows?

**Delta from base.** Probes and edits on the content-effects networks (one
frequent cue, p(CoT) 0.9); the content test battery at every reasoning-step
value 0..8.

**Axes.** frequent_cue_prop {0.01, 0.05, 0.1, 0.2} x 10 seeds (edits shown at 0.2).

**Probes.** One ridge regression of rank per site (the item embedding table,
each block's output, the final normalized state), fit on every item-token
occurrence of 500 prompts under the frequent cue whose query pairs cover every
queryable pool item (one more prompt for each item the 500 leave unqueried) and
whose context is uninformative (other items in the order the same cue assigns
them, which says nothing about the queried pair), and scored on held-out items:
r2 and Spearman (`rank_regression`) and pairwise ordering accuracy
(`rank_order`).

**Edits.** The edits act on the item tokens' residual state after the
embedding and after every block, in every forward pass of the rollout, with a
rank direction fit at each of those sites on the item-token occurrences of the
probe prompts; each site is edited along its rank direction or along a matched
random direction of its own. Erase gains {3, 2.5, 2, 1.5, 1, 0.75, 0.5, 0.25, 0,
-0.5, -1}: 1 leaves the rank coordinate, 0 erases it, -1 inverts it. Write
strengths {0, 0.25, 0.5, 0.75, 1} toward the familiar ordering, its reverse,
and an unrelated ordering (rank correlation with the familiar one at most 0.2
in absolute value) along the rank direction, and toward the unrelated ordering
along the random direction. Every edited battery runs at every reasoning-step
value 0..8; the `evaluate` stage writes the unedited baseline.

**Feeds.** The figure "Mechanisms underlying the geometric and algebraic
strategies"; the appendix figure "Additional mechanism results".
