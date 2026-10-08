"""Shared fixtures for the FLYPSE test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def source() -> list[int]:
    """Return a small integer data source."""
    return list(range(10))
