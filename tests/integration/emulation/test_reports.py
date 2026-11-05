"""Tests for the reports of the entities, through Ray."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("_reporting.py")
"""The script emulating the federation."""


@pytest.mark.integration
@pytest.mark.emulation
@pytest.mark.parametrize("crashing", [None, "a"])
def test_every_report_reaches_the_process(tmp_path, crashing):
    """All an entity sent is read back, even if another entity crashes."""
    subprocess.run(
        [sys.executable, SCRIPT, tmp_path, *([crashing] if crashing else [])],
        check=True,
        env={**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"},
        timeout=120,
    )

    reported = json.loads((tmp_path / "reported.json").read_text())
    for entity_id in ("hub", "a", "b"):
        sent = int((tmp_path / entity_id).read_text())
        mine = [r for r in reported if r["entity_id"] == entity_id]
        assert sent > 0
        assert sorted(r["n_report"] for r in mine) == list(range(1, sent + 1))
        assert {r["node_id"] for r in mine} == {entity_id}
    assert {r["loss"] for r in reported} == {0.5}
