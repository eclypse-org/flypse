Federation
==========

In FLYPSE, a :class:`~flypse.federation.federation.Federation` is a graph whose
nodes are :doc:`entities <entities>` and whose edges are the connections between
them. It describes who takes part in the learning process and who can talk to whom,
not where they run nor the data they hold.

On this page, ``Server``, ``Client``, ``Aggregator`` and ``Peer`` stand for your own
subclasses of :class:`~flypse.federation.entity.Entity`.

There are two ways to build a federation:

.. grid:: 2

   .. grid-item::

      .. card:: Use a topology builder
         :link: use-a-topology-builder
         :link-type: ref

         Connect your entities into a common topology with one call.

   .. grid-item::

      .. card:: Build it by hand
         :link: build-it-by-hand
         :link-type: ref

         Add the entities and their connections one by one.

.. _use-a-topology-builder:

Use a topology builder
----------------------

Each builder takes the entities, already created, and connects them. Any other
argument goes to the :class:`~flypse.federation.federation.Federation`, such as
``federation_id`` or ``seed``.

.. tab-set::

   .. tab-item:: Star
      :sync: star

      :func:`~flypse.federation.topology.get_star` connects a centre to each of
      its leaves, as in client-server federated learning:

      .. code-block:: python

         from flypse.federation import get_star

         federation = get_star(
             Server("server"),
             [Client(f"client_{i}", "server") for i in range(5)],
             federation_id="ClientServer",
         )

   .. tab-item:: Tree
      :sync: tree

      :func:`~flypse.federation.topology.get_tree` connects each entity to its
      children, as in hierarchical federated learning:

      .. code-block:: python

         from flypse.federation import get_tree

         federation = get_tree(
             Server("server"),
             {
                 "server": [Aggregator("edge_0"), Aggregator("edge_1")],
                 "edge_0": [Client("client_0", "edge_0"), Client("client_1", "edge_0")],
                 "edge_1": [Client("client_2", "edge_1"), Client("client_3", "edge_1")],
             },
         )

      The keys are the *identifiers* of the parents, the values are the entities.
      A parent not reachable from the root raises ``ValueError``.

   .. tab-item:: Ring
      :sync: ring

      :func:`~flypse.federation.topology.get_ring` connects each entity to the
      next, as in decentralised federated learning. It requires at least three
      entities:

      .. code-block:: python

         from flypse.federation import get_ring

         federation = get_ring([Peer(f"peer_{i}") for i in range(4)])

.. _build-it-by-hand:

Build it by hand
----------------

Any other topology is a matter of adding the entities with
:meth:`~flypse.federation.federation.Federation.add_entity` and connecting them
with :meth:`~flypse.federation.federation.Federation.add_connection`, which takes
their identifiers or the entities themselves:

.. code-block:: python

   from flypse.federation import Federation

   federation = Federation(federation_id="Custom")

   federation.add_entity(Server("server"))
   federation.add_entity(Client("client_0", "server"))
   federation.add_entity(Client("client_1", "server"))

   federation.add_connection("server", "client_0")
   federation.add_connection("server", "client_1")

``entities`` holds the entities of the federation, keyed by identifier. Adding two
entities with the same identifier raises ``ValueError``.

Neighbours and direction
------------------------

The neighbours of an entity, returned by
:meth:`~flypse.federation.entity.Entity.get_neighbors`, are the entities it can
send messages to. A connection works in both directions by default.

.. important::

   With ``add_connection(source, target, symmetric=False)``, ``source`` can send
   to ``target`` but not the other way round: ``target`` does not have ``source``
   among its neighbours, and cannot reply.

Requirements
------------

Entities and connections require nothing by default, so a federation fits any
infrastructure. Keyword arguments of ``add_entity`` and ``add_connection`` set
what an entity or a connection requires, out of the
:ref:`resources of the system <system-resources>`:

.. code-block:: python

   federation.add_entity(Client("client_2", "server"), cpu=2, ram=4.0)
   federation.add_connection("server", "client_2", latency=50.0, bandwidth=10.0)

The :doc:`placement strategy <system>` then only uses devices and paths that
satisfy them. A federation also accepts ``update_policies``, functions changing its
requirements during the run, as an infrastructure does with its
:doc:`resources <../advanced/heterogeneity>`.

.. seealso::

   - :doc:`entities`
   - :doc:`system`
