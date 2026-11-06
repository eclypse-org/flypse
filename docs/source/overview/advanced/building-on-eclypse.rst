Building on ECLYPSE
===================

FLYPSE runs on `ECLYPSE <https://eclypse.readthedocs.io/en/latest/>`_, a framework for
emulating Cloud–Edge systems. You do not need to know it to use FLYPSE, and no other
page of this documentation relies on it. This page is for when the built-in systems
are not enough: everything on it uses ECLYPSE directly.

How the two relate
------------------

The classes of FLYPSE extend those of ECLYPSE, so what ECLYPSE offers works on them:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - FLYPSE
     - ECLYPSE
   * - :class:`~flypse.federation.federation.Federation`
     - :class:`~eclypse.graph.application.Application`
   * - :class:`~flypse.federation.entity.Entity`
     - :class:`~eclypse.remote.service.service.Service`
   * - :class:`~flypse.system.device.Device`
     - ``RemoteNode``
   * - :class:`~flypse.process.FederatedProcess`
     - :class:`~eclypse.simulation.simulation.Simulation`, always in remote mode
   * - Requirements and resources
     - Assets

The keyword arguments of a process are those of
:class:`~eclypse.simulation.config.SimulationConfig`. There are two exceptions:
``remote`` is set by the process, so passing it raises ``TypeError``, and
``default_strategy`` has no effect, as the strategy comes from the system.

Build an infrastructure
-----------------------

Any :class:`~eclypse.graph.infrastructure.Infrastructure` can host a federation,
together with a placement strategy:

.. tab-set::

   .. tab-item:: From a builder
      :sync: builder

      An ECLYPSE builder creates a whole topology with one call:

      .. code-block:: python

         from eclypse.builders.infrastructure import get_star as get_star_infrastructure

         from flypse.system import ExclusiveStrategy, SystemConfig

         infrastructure = get_star_infrastructure(
             n_clients=5,
             symmetric=True,
             include_default_assets=True,
             resource_init="max",
             seed=0,
         )
         system = SystemConfig(infrastructure, ExclusiveStrategy())

   .. tab-item:: By hand
      :sync: hand

      Build the graph yourself, and map the entities onto its nodes with a static
      strategy:

      .. code-block:: python

         from eclypse.graph import Infrastructure
         from eclypse.placement.strategies import StaticStrategy

         from flypse.system import SystemConfig

         infrastructure = Infrastructure(
             "edge-cloud",
             include_default_assets=True,
             resource_init="max",
             seed=0,
         )
         infrastructure.add_node("cloud")
         for i in range(3):
             infrastructure.add_node(f"edge_{i}")
             infrastructure.add_edge(
                 f"edge_{i}",
                 "cloud",
                 symmetric=True,
                 latency=20.0,
                 bandwidth=100.0,
             )

         strategy = StaticStrategy(
             {
                 "server": "cloud",
                 "client_0": "edge_0",
                 "client_1": "edge_1",
                 "client_2": "edge_2",
             }
         )
         system = SystemConfig(infrastructure, strategy)

.. important::

   The ECLYPSE builders default to ``resource_init="min"``, which leaves no
   resources for the paths between the devices: the entities are placed, the
   placement is reset at every process step, and nothing runs. Pass ``"max"`` unless you
   set the resources yourself.

.. _ready-made-policies:

Ready-made update policies
--------------------------

An :doc:`update policy <heterogeneity>` is a plain function, and ECLYPSE ships a
catalogue of them, for failures, noise, gradual degradation and schedules, in
``eclypse.policies``:

.. code-block:: python

   from eclypse import policies

   from flypse.system import get_mirror

   slow_down = policies.degrade.set_value(
       1000.0,
       edge_assets="latency",
       edge_ids=[("client_0", "server")],
   )
   infrastructure = get_mirror(
       federation,
       update_policies=[policies.at(30, slow_down)],
       seed=0,
   )

.. _custom-metrics:

Custom metrics
--------------

Any ECLYPSE metric works on a process, passed through ``events``. A service metric
with ``remote=True`` runs on the device, and receives the live entity:

.. code-block:: python

   from eclypse.report.metrics import metric

   from flypse.process import FederatedProcess


   @metric.service(remote=True)
   def n_versions(service):
       return len(service.model.versions)


   process = FederatedProcess(federation, data, step_every_ms=100.0, events=[n_versions])
   process.run(seconds=3)

   frame = process.report.service(service_ids="server")
   results = frame[frame["callback_id"] == "n_versions"]

.. warning::

   A metric must return a single scalar. A metric reading an entity also runs on
   another thread than its ``step``, so it should not change its state.

.. seealso::

   - :doc:`ECLYPSE infrastructure builders <eclypse:source/overview/concepts/builders>`
   - :doc:`ECLYPSE placement strategies <eclypse:source/overview/concepts/placement-strategy>`
   - :doc:`ECLYPSE assets <eclypse:source/overview/concepts/assets>`
   - :doc:`ECLYPSE update policies <eclypse:source/overview/concepts/update-policy>`
   - :doc:`ECLYPSE reporting <eclypse:source/overview/advanced/reporting>`
   - :doc:`ECLYPSE simulation configuration <eclypse:source/overview/concepts/simulation-configuration>`
