"""Reference values for a planted rank direction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geo_alg_ti import represent

REFERENCE = Path(__file__).with_suffix(".npz")
SEED = 0
N_ITEMS = 20
PER_ITEM = 30
WIDTH = 32
NOISE = 0.05
RIDGE_ALPHA = 1e-3


def planted_scenario(seed: int = SEED, *, n_items: int = N_ITEMS, per_item: int = PER_ITEM, width: int = WIDTH, noise: float = NOISE) -> dict[str, np.ndarray]:
    """Vectors carrying z-scored ranks along one unit direction plus noise, with their ranks."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(width)
    direction /= np.linalg.norm(direction)
    item_ranks = rng.choice(n_items, size=n_items, replace=False)
    items = np.repeat(np.arange(n_items), per_item)
    ranks = item_ranks[items]
    z = (ranks - ranks.mean()) / ranks.std()
    vectors = z[:, None] * direction[None, :] + noise * rng.standard_normal((items.size, width))
    item_rows = np.stack([vectors[items == item].mean(axis=0) for item in range(n_items)])
    return {
        "direction": direction,
        "items": items,
        "ranks": ranks,
        "item_ranks": item_ranks,
        "vectors": vectors,
        "item_rows": item_rows,
    }


def reference_arrays(scenario: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The rank-direction fit, its calibration, the ridge fit, and the matched random direction."""
    rank_direction = represent.fit_rank_direction(scenario["vectors"], scenario["ranks"], site=represent.EMBEDDING_SITE)
    z_ranks = (scenario["ranks"] - scenario["ranks"].mean()) / scenario["ranks"].std()
    coefficients, intercept = represent.ridge_fit(scenario["vectors"], z_ranks, RIDGE_ALPHA)
    return {
        "mean": rank_direction.mean,
        "covariance": represent.shrunk_covariance(scenario["vectors"]),
        "whitener": rank_direction.whitener,
        "basis_vector": rank_direction.direction,
        "calibration": represent.write_calibration(rank_direction, scenario["item_rows"], scenario["item_ranks"]),
        "ridge_coefficients": coefficients,
        "ridge_intercept": np.array(intercept),
        "ridge_alpha": np.array(RIDGE_ALPHA),
        "random_direction": represent.edit_direction(rank_direction, represent.Direction.RANDOM, represent.RANDOM_DIRECTION_SEED),
    }


def main() -> None:
    """Write the scenario and its reference arrays beside this file."""
    scenario = planted_scenario()
    np.savez(REFERENCE, **scenario, **reference_arrays(scenario))


if __name__ == "__main__":
    main()
