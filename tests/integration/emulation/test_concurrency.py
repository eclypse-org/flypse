"""Tests for the slots of a device shared by several entities, through Ray."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("_concurrency.py")
"""The script emulating the federation."""


@pytest.mark.integration
@pytest.mark.emulation
def test_a_device_admits_its_limit_without_deadlock(tmp_path):
    """Two entities at most work at once, and every round involves every client."""
    subprocess.run(
        [sys.executable, SCRIPT, tmp_path],
        check=True,
        env={**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"},
        timeout=120,
    )

    reported = json.loads((tmp_path / "reported.json").read_text())
    assert {r["holders"] for r in reported} == {1, 2}
    rounds = Counter(r["entity_id"] for r in reported)
    assert rounds["server"] >= 3
    clients = [rounds[client_id] for client_id in ("a", "b", "c")]
    assert max(clients) - min(clients) <= 1
    assert min(clients) >= rounds["server"] - 1
