================================
Choose where the federation runs
================================

The third argument of :class:`~flypse.process.FederatedProcess`, ``system``, is a
:class:`~flypse.system.config.SystemConfig`: the infrastructure hosting the
federation, and the placement strategy mapping the entities onto it. An infrastructure
is a graph of devices and the links between them. Each device holds the federation
data, from which the entities deployed on it bind their own portion.

There are two built-in systems, and a mirror you can adjust:

- ``"singleton"``, a single device hosting every entity,
- ``"mirror"``, the default, a copy of the topology of the federation,
- a mirror you adjust yourself, with the conditions you want to study.

Check the logic on one device
-----------------------------

With ``"singleton"``, the infrastructure has a single device, with maximal
resources, and every entity is placed on it:

.. code-block:: python

   from flypse.process import FederatedProcess
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

The time a message takes depends on the latency and the bandwidth of the links it
crosses, and on its size. The links of a mirror start with zero latency and maximal
bandwidth, so messages cost next to nothing.

Adjust the system
-----------------

:func:`~flypse.system.infrastructure.get_mirror` builds the same infrastructure as
``"mirror"``, for you to change before the run. A link has one edge per direction,
so this delays the messages from ``client_0`` to the server by 100 ms:

.. code-block:: python

   from flypse.system import get_mirror, get_mirror_strategy

   infrastructure = get_mirror(federation, seed=0)
   infrastructure.edges["client_0", "server"]["latency"] = 100.0

   process = FederatedProcess(
       federation,
       data,
       SystemConfig(infrastructure, get_mirror_strategy(federation)),
       step_every_ms=100.0,
   )

An infrastructure you build comes with its placement strategy:
:func:`~flypse.system.placement.get_mirror_strategy` places each entity on the device
named after it.

The server waits for every client, so one slow link holds the whole federation back:
a round now takes at least 100 ms, whatever the speed of your machine.

.. seealso::

   - :doc:`../concepts/system`
   - :doc:`../advanced/heterogeneity`
