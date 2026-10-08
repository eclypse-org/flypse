"""Tests for the top-level flypse package."""

from __future__ import annotations

import flypse


def test_version_is_exposed():
    """The package exposes its version."""
    assert flypse.__version__ == "0.1.0"
