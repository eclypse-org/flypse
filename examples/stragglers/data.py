"""Synthetic regression data for the stragglers example.

The source is sorted by the input feature, so a uniform (contiguous) split
gives every entity a different region of the feature space: the federation is
statistically heterogeneous (non-IID) by construction.
"""

from __future__ import annotations

import random

Item = tuple[list[float], float]
"""An item of the data source: its features and its target."""

TRUE_WEIGHTS = [2.0]
"""The weights of the ground-truth linear model."""

TRUE_BIAS = 1.0
"""The bias of the ground-truth linear model."""


def make_source(n_samples: int, *, noise: float, seed: int) -> list[Item]:
    """Build a synthetic linear-regression data source.

    Args:
        n_samples (int): The number of samples.
        noise (float): The amplitude of the uniform noise added to the targets.
        seed (int): The seed of the noise.

    Returns:
        list[Item]: The data source, sorted by the input feature.
    """
    rng = random.Random(seed)
    source: list[Item] = []
    for i in range(n_samples):
        x = -1.0 + 2.0 * i / max(n_samples - 1, 1)
        y = TRUE_WEIGHTS[0] * x + TRUE_BIAS + rng.uniform(-noise, noise)
        source.append(([x], y))
    return source
