System and placement
====================

The system is where a federation runs. In FLYPSE, it is an ECLYPSE
:class:`~eclypse.graph.infrastructure.Infrastructure` whose nodes are
:class:`~flypse.system.device.Device` objects, plus the placement strategy mapping
the entities onto them. A :class:`~flypse.system.config.SystemConfig` keeps the two
consistent.

There are two ways to choose the system:

.. grid:: 2

   .. grid-item::

      .. card:: Built-in systems
         :link: built-in-systems
         :link-type: ref

         Derive the infrastructure and the placement from the federation.

   .. grid-item::

      .. card:: Any ECLYPSE infrastructure
         :link: any-eclypse-infrastructure
         :link-type: ref

         Bring your own infrastructure and placement strategy.

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

   The links of a mirror start with the best values of their assets, so messages
   cost next to nothing until you degrade a link with an update policy.

Change the built-in systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SystemConfig("mirror")`` takes no options. To customise a built-in system, for
instance to give it update policies, build it with
:func:`~flypse.system.infrastructure.get_mirror` or
:func:`~flypse.system.infrastructure.get_singleton`, and pass it with its strategy:

.. code-block:: python

   from flypse.system import SystemConfig, get_mirror, get_mirror_strategy

   infrastructure = get_mirror(federation, update_policies=policies, seed=0)
   system = SystemConfig(infrastructure, get_mirror_strategy(federation))

.. _any-eclypse-infrastructure:

Any ECLYPSE infrastructure
--------------------------

Any :class:`~eclypse.graph.infrastructure.Infrastructure` can host a federation,
together with an ECLYPSE placement strategy. Without one, ``SystemConfig`` raises
``ValueError``, as there is no way to tell where the entities go:

.. code-block:: python

   from eclypse.builders.infrastructure import get_star

   from flypse.system import ExclusiveStrategy, SystemConfig

   infrastructure = get_star(
       n_clients=5,
       symmetric=True,
       include_default_assets=True,
       resource_init="max",
       seed=0,
   )
   system = SystemConfig(infrastructure, ExclusiveStrategy())

.. important::

   The ECLYPSE builders default to ``resource_init="min"``, which leaves no
   resources for the paths between the devices: the entities are placed, the
   placement is reset at every step, and nothing runs. Pass ``"max"`` unless you
   set the resources yourself.

Exclusive placement
~~~~~~~~~~~~~~~~~~~

In a federation, a device usually hosts one participant.
:class:`~flypse.system.placement.ExclusiveStrategy` places each entity on a distinct
device satisfying its requirements, as a maximum matching between entities and
devices, so an assignment is found whenever one exists.

.. important::

   The placement is all or nothing: if the entities cannot all get a device of
   their own, none of them is placed, and the process runs without entities.

Devices
-------

A :class:`~flypse.system.device.Device` is an ECLYPSE remote node holding the
dataset of the federation. Every node of the infrastructure becomes a device, and
each entity deployed on it binds its own portion of the data. You never create one
yourself.

When the placement is reset
---------------------------

ECLYPSE checks at every step that the entities can still reach each other. If a
path between two connected entities disappears, for instance because a device
fails, the placement of the whole federation is reset: **every** entity is
undeployed, and all of them are deployed again once a path is back. Messages in
flight are lost, and each entity runs ``on_deploy`` again.

This is an ECLYPSE placement semantic, and it is the same for one failed device or
many: in a run that killed one device out of four for three seconds, all four
entities were deployed twice, and none of them ran in between.

.. seealso::

   - :doc:`ECLYPSE builders <eclypse:source/overview/concepts/builders>`
   - :doc:`ECLYPSE placement strategies <eclypse:source/overview/concepts/placement-strategy>`
