"""The content grid without the cue, not in the paper: accuracy by distance under the four
contexts (the shuffled context's distance taken in the ordering shown), reversed-context
accuracy by reasoning steps, and the reasoning steps the congruent, reversed, and shuffled
contexts need to reach ninety percent of their best accuracy by distance, each on a grid of
p(CoT) by frequent-cue proportion, for networks pretrained with the same cue token in every
sequence.
Tables: content_effects_no_cue/eval.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import content_grid, main

EXPERIMENT = "content_effects_no_cue"


def build(args: argparse.Namespace) -> Figure:
    """Assemble the three blocks from the tidy tables."""
    return content_grid(EXPERIMENT, results_root=args.results)


if __name__ == "__main__":
    main(build, name="content_grid_no_cue")
