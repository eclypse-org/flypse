"""Tests for the aggregation strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypse.strategies.aggregation import (
    centered_clipping,
    geometric_median,
    masked_mean,
    meamed,
    mean,
    median,
    sign_majority,
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


def test_meamed_averages_the_values_closest_to_the_median():
    """For each coordinate, the n - f values nearest the median."""
    vectors = _vectors([0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [100.0, -50.0])

    assert meamed(vectors, 1).tolist() == [1.0, 11.0]
    with pytest.raises(ValueError, match="f must be"):
        meamed(vectors, 4)


def test_geometric_median_resists_an_outlier():
    """It lies between the two middle values, not near the mean."""
    vectors = _vectors([0.0], [1.0], [2.0], [100.0])

    assert 1.0 <= geometric_median(vectors)[0] <= 2.0
    assert geometric_median(vectors, iterations=1)[0] > 2.0


def test_geometric_median_honours_the_weights():
    """A heavy vector pulls the median onto itself."""
    vectors = _vectors([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])

    assert geometric_median(vectors, np.array([10.0, 1.0, 1.0])) == pytest.approx(
        [0.0, 0.0], abs=1e-5
    )
    with pytest.raises(ValueError, match="nu"):
        geometric_median(vectors, nu=0.0)


def test_centered_clipping_bounds_the_pull_of_each_contribution():
    """An outlier moves the result by at most τ / n."""
    vectors = _vectors([0.0], [1.0], [2.0], [100.0])

    once = centered_clipping(vectors, np.array([1.0]), tau=1.0)

    assert once.tolist() == [1.25]
    assert centered_clipping(vectors, np.array([1.0]), tau=0.0).tolist() == [1.0]
    with pytest.raises(ValueError, match="tau"):
        centered_clipping(vectors, np.array([1.0]), tau=-1.0)


def test_sign_majority_votes_by_coordinate():
    """The sign of the sum of the signs, 0 on a tie."""
    vectors = _vectors(
        [1.0, -2.0, 0.5, 1.0], [-1.0, -1.0, 0.2, -1.0], [3.0, 4.0, -1.0, 0.0]
    )

    assert sign_majority(vectors).tolist() == [1.0, -1.0, 1.0, 0.0]


def test_masked_mean_keeps_the_uncovered_coordinates():
    """A coordinate nobody covers keeps its previous value, never 0."""
    vectors = _vectors([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
    masks = [np.array([True, True, False]), np.array([False, True, False])]

    assert masked_mean(vectors, masks, np.full(3, 9.0)).tolist() == [1.0, 2.0, 9.0]
    weighted = masked_mean(vectors, masks, np.zeros(3), np.array([1.0, 3.0]))
    assert weighted.tolist() == [1.0, 2.5, 0.0]
