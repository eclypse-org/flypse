Heterogeneity
=============

A federation on the Cloud-Edge continuum is heterogeneous in two independent ways:

- *statistical* heterogeneity: the entities hold different data,
- *system* heterogeneity: the entities run on different devices, over different
  links, which change over time.

FLYPSE takes the first from the :doc:`federated datasets <../concepts/data>`, and
the second from ECLYPSE: the assets of an infrastructure and its update policies.

Statistical heterogeneity
-------------------------

The kind of split sets how the entities differ, and ``alpha`` how much:

.. code-block:: python

   data = DirichletDataset(source, class_fn=label, alpha=0.1, seed=0)

See :doc:`../concepts/data` for how to choose one.

What the system can do
----------------------

Each asset of an infrastructure acts on an emulation in its own way:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Assets
     - Effect on the emulation
   * - ``latency``, ``bandwidth``, ``processing_time``
     - Delay messages, which really take that long to arrive.
   * - ``cpu``, ``ram``, ``storage``, ``gpu``
     - Decide where entities can be placed. They do not slow a ``step`` down.
   * - ``availability``
     - A device at zero drops out of the paths, which
       :ref:`resets the placement <placement-reset>`.

A message costs its modelled time, computed from the route it follows:

.. code-block:: text

   cost = processing_time / 1000 + sum over the links of
          (bytes * 8e-6 / bandwidth + latency / 1000)

with ``processing_time`` and ``latency`` in milliseconds, and ``bandwidth`` in
megabits per second, for a cost in seconds. Nothing throttles the computation of an
entity: emulated devices run as fast as your machine.

Slow down a link
----------------

An update policy changes the infrastructure during a run. ECLYPSE policies target
assets by name, and a mirror link has one edge per direction, so one direction can
slow down alone. This policy sets the latency of the uplink of a client:

.. code-block:: python

   from eclypse import policies

   slow_down = policies.degrade.set_value(
       1000.0,
       edge_assets="latency",
       edge_ids=[("client_0", "server")],
   )

.. warning::

   The links of a mirror start with zero latency, so a policy multiplying it has no
   effect. Use ``set_value``.

Schedule changes
----------------

``policies.at`` runs a policy at given simulation steps, and ``policies.between``
within a range of them. Steps follow ``step_every_ms``, so at 100 ms per step, step
30 comes about three seconds into the run:

.. code-block:: python

   from flypse.system import SystemConfig, get_mirror, get_mirror_strategy

   recover = policies.degrade.set_value(
       0.0,
       edge_assets="latency",
       edge_ids=[("client_0", "server")],
   )
   infrastructure = get_mirror(
       federation,
       update_policies=[policies.at(30, slow_down), policies.at(60, recover)],
       seed=0,
   )

   process = FederatedProcess(
       federation,
       data,
       SystemConfig(infrastructure, get_mirror_strategy(federation)),
       step_every_ms=100.0,
   )

What failures do
----------------

.. warning::

   A failed device does not slow one client down: it resets the whole federation.
   When a device hosting an entity becomes unavailable, every entity is undeployed
   until a path is back, and messages in flight are lost. See
   :ref:`placement-reset`.

To model entities that fall behind or drop out of a round while the federation
carries on, degrade their links instead, and give the aggregator a
:ref:`timeout <messaging-timeouts>`.

.. seealso::

   - :doc:`ECLYPSE update policies <eclypse:source/overview/concepts/update-policy>`
