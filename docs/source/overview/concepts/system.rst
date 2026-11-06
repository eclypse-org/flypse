System and placement
====================

The system is where a federation runs: an *infrastructure*, a graph of devices and
the links between them, and a *placement strategy*, mapping the entities onto the
devices. A :class:`~flypse.system.config.SystemConfig` keeps the two consistent.

There are two ways to choose the system:

.. grid:: 2

   .. grid-item::

      .. card:: Built-in systems
         :link: built-in-systems
         :link-type: ref

         Derive the infrastructure and the placement from the federation.

   .. grid-item::

      .. card:: Adjust a built-in system
         :link: adjust-a-built-in-system
         :link-type: ref

         Set the resources of its devices and links, and change its topology.

.. _built-in-systems:

Built-in systems
----------------

.. tab-set::

   .. tab-item:: Singleton
      :sync: singleton

      A single device, with maximal resources, hosts every entity. It is the
      quickest way to check the logic of the entities:

      .. code-block:: python

         from flypse.system import SystemConfig

         system = SystemConfig("singleton")

   .. tab-item:: Mirror
      :sync: mirror

      The default. The infrastructure copies the topology of the federation: one
      device per entity and one link per connection, with maximal resources:

      .. code-block:: python

         system = SystemConfig("mirror")

.. note::

   The links of a mirror start with zero latency and maximal bandwidth, so messages
   cost next to nothing until you adjust them.

.. _adjust-a-built-in-system:

Adjust a built-in system
------------------------

``SystemConfig("mirror")`` takes no options. To adjust a built-in system, build it
with :func:`~flypse.system.infrastructure.get_mirror` or
:func:`~flypse.system.infrastructure.get_singleton`, and pass it with its strategy:

.. code-block:: python

   from flypse.system import SystemConfig, get_mirror, get_mirror_strategy

   infrastructure = get_mirror(federation, seed=0)
   system = SystemConfig(infrastructure, get_mirror_strategy(federation))

For a singleton, the strategy is ``get_singleton_strategy(federation)``. An
infrastructure given without a strategy raises ``ValueError``, as there is no way to
tell where the entities go.

Set the resources
~~~~~~~~~~~~~~~~~

An infrastructure is a `networkx <https://networkx.org/>`_ graph, so its devices and
links can be read and changed like those of any graph. A link has one edge per
direction, and each direction has its own values:

.. code-block:: python

   infrastructure.edges["client_0", "server"]["latency"] = 100.0
   infrastructure.edges["client_0", "server"]["bandwidth"] = 10.0
   infrastructure.nodes["client_0"]["processing_time"] = 5.0

To change them during a run, see :doc:`../advanced/heterogeneity`.

Change the topology
~~~~~~~~~~~~~~~~~~~

Devices and links can be added and removed too, as long as connected entities can
still reach each other. This infrastructure routes every client through a gateway,
so that their messages cross two links:

.. code-block:: python

   infrastructure = get_mirror(federation, seed=0)

   infrastructure.add_node("gateway")
   infrastructure.add_edge("gateway", "server", symmetric=True, latency=50.0)
   for client_id in ("client_0", "client_1", "client_2"):
       infrastructure.remove_edge(client_id, "server")
       infrastructure.remove_edge("server", client_id)
       infrastructure.add_edge(client_id, "gateway", symmetric=True, latency=5.0)

   system = SystemConfig(infrastructure, get_mirror_strategy(federation))

For an infrastructure built from scratch, see :doc:`../advanced/building-on-eclypse`.

.. _system-resources:

Devices and links
-----------------

A :class:`~flypse.system.device.Device` is an emulated machine, running in a process
of its own and holding the dataset of the federation. Every node of the
infrastructure becomes a device, and each entity deployed on it binds its own
portion of the data. You never create one yourself.

Devices and links offer resources, which entities and connections can
:doc:`require <federation>`. Each resource acts on a run in its own way:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Resource
     - Of a
     - Effect on the run
   * - ``cpu``, ``ram``, ``storage``, ``gpu``
     - device
     - Decide where entities can be placed. They do not slow a ``step`` down.
   * - ``availability``
     - device
     - At zero, the device drops out of the paths, which
       :ref:`resets the placement <placement-reset>`.
   * - ``processing_time``
     - device
     - Delays the messages crossing the device, in milliseconds.
   * - ``latency``
     - link
     - Delays the messages crossing the link, in milliseconds.
   * - ``bandwidth``
     - link
     - Sets how long a message takes to cross the link, given its size, in
       megabits per second.

These delays make up the :ref:`cost of a message <message-cost>`. Nothing throttles
the computation of an entity: emulated devices run as fast as your machine.

Exclusive placement
-------------------

The built-in systems place the entities by themselves: on the only device of a
singleton, or on the device named after them in a mirror.

On an infrastructure with more devices than entities, where any device will do,
:class:`~flypse.system.placement.ExclusiveStrategy` places each entity on a distinct
device satisfying its requirements, as a maximum matching between entities and
devices, so an assignment is found whenever one exists:

.. code-block:: python

   from flypse.system import ExclusiveStrategy

   system = SystemConfig(infrastructure, ExclusiveStrategy())

.. important::

   The placement is all or nothing: if the entities cannot all get a device of
   their own, none of them is placed, and the process runs without entities.

.. _placement-reset:

When the placement is reset
---------------------------

At every process step, the process checks that the entities can still reach each
other and that their requirements still hold. If a path between two connected
entities disappears, for instance because a device fails, or if a resource falls short
of a requirement, the placement of the whole federation is reset: **every** entity is
undeployed, and all of them are deployed again once the problem is gone. Messages in
flight are lost, and each entity runs ``on_deploy`` again.

It is the same for one failed device or many: in a run that took one device out of
four down for three seconds, all four entities were deployed twice, and none of them
ran in between.

.. seealso::

   - :doc:`../advanced/heterogeneity`
   - :doc:`../advanced/building-on-eclypse`
