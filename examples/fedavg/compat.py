"""Runtime compatibility patches for the federated averaging example.

Two environment gaps keep the example from running today; both are patched
here so the documented command works, and both become no-ops (or harmless)
as soon as the underlying issue is fixed.

1. Ray 2.59 detects that the driver was started through ``uv run`` and
   starts its workers with ``uv run`` as well, syncing the *packaged* copy
   of the project inside a temporary working directory. That environment
   only contains the base dependencies -- ``ray`` is a test-group
   dependency of this repository -- so every worker dies with
   ``ModuleNotFoundError: No module named 'ray'`` and the simulation hangs
   on its first remote call. Pointing ``UV_PROJECT_ENVIRONMENT`` at the
   environment of the driver process makes the workers reuse it. The
   variable is inert unless a ``uv run`` resolves an environment.

2. PyPI ``eclypse`` 0.9.0 does not forward ``RemoteBootstrap.node_args``
   to the remote node actors, so ``Device.build`` never receives the
   federation dataset: the learners see no data, never send their models,
   and the aggregator blocks forever on ``recv``. The forwarding is fixed
   upstream but unreleased; the fixed implementation is applied here until
   an ``eclypse`` release includes it.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import (
    TYPE_CHECKING,
    Any,
)

from eclypse.remote import ray_backend
from eclypse.remote.bootstrap import RemoteBootstrap
from eclypse.remote.bootstrap.bootstrap import _create_remote

if TYPE_CHECKING:
    from eclypse.simulation import SimulationConfig

    from eclypse.graph import Infrastructure


def patch_ray_uv_workers() -> None:
    """Make Ray workers started through ``uv run`` reuse the driver environment."""
    os.environ["UV_PROJECT_ENVIRONMENT"] = sys.prefix


def _build_with_node_args(
    self: RemoteBootstrap,
    infrastructure: Infrastructure,
    simulation_config: SimulationConfig | None = None,
) -> Any:
    """Build the remote simulation, forwarding the node args to the nodes.

    This is the upstream (fixed) implementation of
    ``RemoteBootstrap.build``.

    Args:
        self (RemoteBootstrap): The bootstrap.
        infrastructure (Infrastructure): The infrastructure to emulate.
        simulation_config (SimulationConfig | None): The configuration of
            the simulation. Defaults to None.

    Returns:
        Any: The handle of the remote simulator.
    """
    ray_backend.init(runtime_env={"env_vars": self.env_vars})

    remote_nodes = [
        _create_remote(
            f"{infrastructure.id}/{node}",
            self._node_class,
            self.ray_options_factory,
            node,
            infrastructure.id,
            **self.node_args,
        )
        for node in infrastructure.nodes
    ]

    return _create_remote(
        f"{infrastructure.id}/manager",
        self._sim_class,
        self.ray_options_factory,
        infrastructure,
        simulation_config,
        remotes=remote_nodes,
    )


def patch_node_args_forwarding() -> None:
    """Forward ``RemoteBootstrap.node_args`` when the installed eclypse drops them."""
    if "self.node_args" not in inspect.getsource(RemoteBootstrap.build):
        RemoteBootstrap.build = _build_with_node_args
