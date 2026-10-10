"""Runtime compatibility patch for the federated averaging example.

Ray 2.59 detects that the driver was started through ``uv run`` and starts
its workers with ``uv run`` as well, resolving the *packaged* copy of the
project inside a temporary working directory. The workers must not sync
that copy: it would rewrite the environment of the driver process from a
bare checkout of the project (replacing, e.g., its editable installs).
They must instead reuse the driver environment as-is: ``UV_PROJECT_ENVIRONMENT``
points ``uv run`` at it, and ``UV_NO_SYNC`` forbids any re-sync. Both
variables are inert unless a ``uv run`` resolves an environment.
"""

from __future__ import annotations

import os
import sys


def patch_ray_uv_workers() -> None:
    """Make Ray workers started through ``uv run`` reuse the driver environment."""
    os.environ["UV_PROJECT_ENVIRONMENT"] = sys.prefix
    os.environ["UV_NO_SYNC"] = "1"
