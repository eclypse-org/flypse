"""Module for the seed of the federated process.

A federated process has a single seed, which reaches your script and every device
through the ``ECLYPSE_RND_SEED`` environment variable. The FLYPSE components
given no seed fall back to it, so that the seed of the process reproduces all of
them.
"""

from __future__ import annotations

import os

from eclypse.utils.constants import RND_SEED


def get_seed(seed: int | None = None) -> int | None:
    """Return the given seed, or the seed of the running process.

    Args:
        seed (int | None): An explicit seed, which always wins. Defaults to
            None.

    Returns:
        int | None: The explicit seed if given, otherwise the seed of the
        process, or None outside of a process.
    """
    if seed is not None:
        return seed
    value = os.getenv(RND_SEED)
    return None if value is None else int(value)
