"""Tests for the seed of the federated process."""

from __future__ import annotations

from eclypse.utils.constants import RND_SEED

from flypse.seed import get_seed


def test_explicit_seed_wins(monkeypatch):
    """An explicit seed is returned as is."""
    monkeypatch.setenv(RND_SEED, "7")

    assert get_seed(3) == 3


def test_falls_back_to_the_process_seed(monkeypatch):
    """Without an explicit seed, the seed of the process is used."""
    monkeypatch.setenv(RND_SEED, "7")

    assert get_seed() == 7


def test_no_seed_outside_of_a_process():
    """Outside of a process there is no seed."""
    assert get_seed() is None
