"""Shared fixtures for the FLYPSE test suite."""

from __future__ import annotations

import pytest
from eclypse.utils.constants import RND_SEED


@pytest.fixture
def source() -> list[int]:
    """Return a small integer data source."""
    return list(range(10))


@pytest.fixture(autouse=True)
def no_process_seed(monkeypatch):
    """Run every test without a process seed, restoring it afterwards.

    ECLYPSE exposes the seed of a process through an environment variable:
    setting then deleting it makes the teardown remove any seed a test sets.
    """
    monkeypatch.setenv(RND_SEED, "0")
    monkeypatch.delenv(RND_SEED)
