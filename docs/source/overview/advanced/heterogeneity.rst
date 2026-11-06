Heterogeneity
=============

A federation on the Cloud–Edge continuum is heterogeneous in two independent ways:

- *statistical* heterogeneity: the entities hold different data,
- *system* heterogeneity: the entities run on different devices, over different
  links, which change over time.

FLYPSE takes the first from the :doc:`federated datasets <../concepts/data>`, and
the second from the resources of the system and the update policies changing them.

Statistical heterogeneity
-------------------------

The kind of split sets how the entities differ, and ``alpha`` how much:

.. code-block:: python

   from flypse.data import DirichletDataset

   data = DirichletDataset(source, class_fn=label, alpha=0.1, seed=0)

See :doc:`../concepts/data` for how to choose one.

System heterogeneity
--------------------

Each device and link can have its own :ref:`resources <system-resources>`, set before
the run. Latency, bandwidth and processing time delay the messages, which really take
:ref:`that long <message-cost>` to arrive:

.. code-block:: python

   from flypse.system import get_mirror

   infrastructure = get_mirror(federation, seed=0)
   infrastructure.edges["client_0", "server"]["latency"] = 100.0

Slow down a link
----------------

An update policy changes the infrastructure during a run: it is a function, called at
every process step with the infrastructure. A link has one edge per direction, so one
direction can slow down alone. This policy sets the latency of the uplink of a client:

.. code-block:: python

   def slow_down(infrastructure):
       infrastructure.edges["client_0", "server"]["latency"] = 1000.0


   infrastructure = get_mirror(federation, update_policies=slow_down, seed=0)

.. warning::

   The links of a mirror start with zero latency, so a policy multiplying it has no
   effect. Set a value.

Schedule changes
----------------

To act at a given moment, a policy counts the process steps. They follow
``step_every_ms``, so at 100 ms per step, step 30 comes about three seconds into the
run:

.. code-block:: python

   from flypse.process import FederatedProcess
   from flypse.system import SystemConfig, get_mirror, get_mirror_strategy


   def slow_uplink(client_id, start, end):
       step = 0

       def update(infrastructure):
           nonlocal step
           if step == start:
               infrastructure.edges[client_id, "server"]["latency"] = 1000.0
           elif step == end:
               infrastructure.edges[client_id, "server"]["latency"] = 0.0
           step += 1

       return update


   infrastructure = get_mirror(
       federation,
       update_policies=slow_uplink("client_0", 30, 60),
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

A policy can also take a device down, by setting its availability to zero:

.. code-block:: python

   infrastructure.nodes["client_0"]["availability"] = 0.0

.. warning::

   A failed device does not slow one client down: it resets the whole federation.
   When a device hosting an entity becomes unavailable, every entity is undeployed
   until a path is back, and messages in flight are lost. See
   :ref:`placement-reset`.

To model entities that fall behind or drop out of a round while the federation
carries on, degrade their links instead, and give the server a
:ref:`timeout <messaging-timeouts>`. Leave those links without
:doc:`requirements <../concepts/federation>`: one that stops meeting a requirement
resets the placement too.

.. seealso::

   - :doc:`../examples/stragglers`
   - :ref:`ready-made-policies`
