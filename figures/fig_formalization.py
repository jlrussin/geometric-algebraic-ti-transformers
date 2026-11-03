"""The prediction panels of the figures "Symbolic distance effects (SDE)", "Content effects in
transitive inference", and "Data-distribution effects": accuracy and reasoning steps by
distance for each strategy, accuracy under congruent, reversed, shuffled and uninformative
contexts and reasoning steps under the first three, and accuracy across skew alpha. No
tables; exact computation.

The strip is saved as `formalization` and each panel alone as `formalization_<panel>`
next to it, with the two panels of accuracy by reasoning-step budget per distance, which
are not in the strip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.figure import Figure

from _shared import OUT_DIR, main
from geo_alg_ti.formalization import panels, strategies


def build(args: argparse.Namespace) -> Figure:
    """Check the closed forms, draw the strip, and save the single panels beside it."""
    strategies.check_identities()
    fig, axes = panels.strip_figure()
    panels.build_strip(axes)
    out_dir = getattr(args, "out", None)
    panels.build_singles(Path(out_dir) if out_dir else OUT_DIR)
    return fig


if __name__ == "__main__":
    main(build, name="formalization")
