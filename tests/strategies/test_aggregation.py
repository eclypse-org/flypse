"""Tests for the aggregation strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypse.strategies.aggregation import (
    mean,
    median,
    trimmed_mean,
)


def _vectors(*rows) -> list[np.ndarray]:
    """Return one float vector per row."""
    return [np.asarray(row, dtype=float) for row in rows]


def test_mean_is_weighted_when_weights_are_given():
    """Without weights the vectors count the same."""
    vectors = _vectors([0.0, 2.0], [2.0, 4.0])

    assert mean(vectors).tolist() == [1.0, 3.0]
    assert mean(vectors, np.array([0.75, 0.25])).tolist() == [0.5, 2.5]


def test_median_and_trimmed_mean_work_by_coordinate():
    """An outlier in each coordinate, from different vectors, is left out."""
    vectors = _vectors([100.0, 1.0], [1.0, 2.0], [2.0, -100.0], [3.0, 3.0], [2.0, 2.0])

    assert median(vectors).tolist() == [2.0, 2.0]
    assert trimmed_mean(vectors, 0.2).tolist() == [7 / 3, 5 / 3]


def test_trimmed_mean_with_nothing_to_drop_is_the_mean():
    """⌊βn⌋ = 0 drops nothing."""
    vectors = _vectors([0.0], [10.0], [2.0])

    assert trimmed_mean(vectors, 0.2).tolist() == mean(vectors).tolist()


@pytest.mark.parametrize("beta", [-0.1, 0.5])
def test_trimmed_mean_refuses_beta_out_of_range(beta):
    """Dropping half of the values at each end would leave nothing."""
    with pytest.raises(ValueError, match="beta"):
        trimmed_mean(_vectors([0.0], [1.0]), beta)
