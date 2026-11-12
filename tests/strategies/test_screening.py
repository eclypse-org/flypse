"""Tests for the screening strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypse.strategies.screening import krum


def _vectors(*rows) -> list[np.ndarray]:
    """Return one float vector per row."""
    return [np.asarray(row, dtype=float) for row in rows]


def test_krum_scores_by_squared_distances():
    """With plain distances 8 and 3 would tie; squared distances favour 3."""
    vectors = _vectors([8.0], [9.0], [3.0], [1.0], [5.0])

    assert krum(vectors, f=1) == [2]


def test_multi_krum_leaves_the_byzantine_contributions_out():
    """The m vectors kept exclude a far-away one."""
    rng = np.random.default_rng(0)
    honest = [rng.normal(0.0, 0.1, 10) for _ in range(6)]

    kept = krum([*honest, np.full(10, 50.0)], f=1, m=5)

    assert len(kept) == 5
    assert 6 not in kept
    assert kept == sorted(kept)


@pytest.mark.parametrize(("n", "f", "m"), [(4, 1, 1), (5, -1, 1), (5, 1, 0), (5, 1, 6)])
def test_krum_checks_its_preconditions(n, f, m):
    """n > 2f + 2, f >= 0 and 1 <= m <= n."""
    with pytest.raises(ValueError):
        krum(_vectors(*([float(i)] for i in range(n))), f=f, m=m)
