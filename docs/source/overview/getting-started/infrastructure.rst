==================================
Choose where the federation runs
==================================

The third argument of :class:`~flypse.process.FederatedProcess`, ``system``,
is a :class:`~flypse.system.config.SystemConfig`: the infrastructure hosting the
federation, and the placement strategy mapping the entities onto it. Each node
of the infrastructure becomes a :class:`~flypse.system.device.Device` holding
the federation data, from which the entities deployed on it bind their own
portion.

The ``"singleton"`` and ``"mirror"`` infrastructures come with their own
placement strategy, inferred when none is given; any other infrastructure
requires an explicit one.

Singleton
---------

With ``"singleton"``, the infrastructure has a single node, with maximal
resources, and every entity is placed on it:

.. code-block:: python

   from flypse.system import SystemConfig

   process = FederatedProcess(federation, data, SystemConfig("singleton"))

Entities on the same device exchange messages without any modelled network
cost. It is the quickest way to check the logic of the entities.

Mirror
------

With ``"mirror"``, the default, the infrastructure copies the topology of the
federation: one device per entity and one link per connection, with maximal
resources. Each entity is placed on its own device:

.. code-block:: python

   process = FederatedProcess(federation, data)  # or SystemConfig("mirror")

The communication cost of a message then depends on the latency and the
bandwidth of the links it traverses, and on the size of the message.

Any ECLYPSE infrastructure
--------------------------

Any :class:`~eclypse.graph.infrastructure.Infrastructure` can host the
federation, for instance one built with the ECLYPSE builders, together with
the placement strategy deciding where each entity runs.

The entities and their connections can only be placed where their
requirements are satisfied. By default, a federation includes the ECLYPSE
default assets at their minimum, so it fits any infrastructure, while the
communication costs follow the links of the infrastructure:

.. code-block:: python

   from eclypse.builders.infrastructure import get_star

   from flypse.system import ExclusiveStrategy

   infrastructure = get_star(
       n_clients=5, symmetric=True, include_default_assets=True, seed=0
   )

   process = FederatedProcess(
       federation,
       data,
       SystemConfig(infrastructure, ExclusiveStrategy()),
   )

:class:`~flypse.system.placement.ExclusiveStrategy` places each entity on a
device of its own, among those satisfying its requirements, and places
nothing when that is not possible. Any ECLYPSE placement strategy can be used
instead.

The update policies of the infrastructure (e.g., failures, degradation,
topology changes) are applied during the emulation, so that the system
heterogeneity of the continuum affects the federation.

The builders of the singleton and mirror infrastructures, and their placement
strategies, are also available in :mod:`flypse.system`, to be customised.

.. seealso::

   - `ECLYPSE infrastructure builders <https://eclypse.readthedocs.io/en/latest/source/overview/concepts/builders.html>`_
   - `ECLYPSE placement strategies <https://eclypse.readthedocs.io/en/latest/source/overview/concepts/placement-strategy.html>`_
