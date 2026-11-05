==================================
Choose where the federation runs
==================================

The third argument of :class:`~flypse.process.FederatedProcess`, ``system``, is a
:class:`~flypse.system.config.SystemConfig`: the infrastructure hosting the
federation, and the placement strategy mapping the entities onto it. Each node of
the infrastructure becomes a :class:`~flypse.system.device.Device` holding the
federation data, from which the entities deployed on it bind their own portion.

There are three ways to choose the system:

- ``"singleton"``, a single device hosting every entity,
- ``"mirror"``, the default, a copy of the topology of the federation,
- any ECLYPSE infrastructure, with its placement strategy.

Check the logic on one device
-----------------------------

With ``"singleton"``, the infrastructure has a single node, with maximal
resources, and every entity is placed on it:

.. code-block:: python

   from flypse.system import SystemConfig

   process = FederatedProcess(
       federation,
       data,
       SystemConfig("singleton"),
       step_every_ms=100.0,
   )

It is the quickest way to check the logic of the entities.

Give each entity a device
-------------------------

With ``"mirror"``, the infrastructure copies the topology of the federation: one
device per entity and one link per connection, with maximal resources. Each
entity is placed on its own device:

.. code-block:: python

   process = FederatedProcess(federation, data, step_every_ms=100.0)

The communication cost of a message depends on the latency and the bandwidth of
the links it traverses, and on the size of the message.

Use any ECLYPSE infrastructure
------------------------------

Any :class:`~eclypse.graph.infrastructure.Infrastructure` can host the federation,
for instance one built with the ECLYPSE builders, together with the placement
strategy deciding where each entity runs:

.. code-block:: python

   from eclypse.builders.infrastructure import get_star

   from flypse.system import ExclusiveStrategy

   infrastructure = get_star(
       n_clients=5,
       symmetric=True,
       include_default_assets=True,
       resource_init="max",
       seed=0,
   )

   process = FederatedProcess(
       federation,
       data,
       SystemConfig(infrastructure, ExclusiveStrategy()),
       step_every_ms=100.0,
   )

:class:`~flypse.system.placement.ExclusiveStrategy` places each entity on a device
of its own, among those satisfying its requirements.

.. important::

   The ECLYPSE builders default to ``resource_init="min"``, which leaves no
   resources for the paths between the devices: the entities are placed, the
   placement is reset at every step, and nothing runs. Pass ``"max"`` unless you
   set the resources yourself.

.. seealso::

   - `ECLYPSE infrastructure builders <https://eclypse.readthedocs.io/en/latest/source/overview/concepts/builders.html>`_
   - `ECLYPSE placement strategies <https://eclypse.readthedocs.io/en/latest/source/overview/concepts/placement-strategy.html>`_
