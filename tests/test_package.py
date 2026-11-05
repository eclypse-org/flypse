"""Tests for the top-level flypse package."""

from __future__ import annotations

from importlib.metadata import version

import flypse


def test_version_is_exposed():
    """The package exposes its installed version."""
    assert flypse.__version__ == version("flypse")
