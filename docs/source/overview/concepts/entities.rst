Entities
========

An :class:`~flypse.federation.entity.Entity` is the basic unit of a federation: an
ECLYPSE :class:`~eclypse.remote.service.service.Service` deployed onto a device,
bound to its portion of the data.

FLYPSE has no client or server class. What an entity is follows from what its
``step`` does, and the protocol of a federation emerges from the steps of its
entities: there are no rounds or barriers in the library.

Implement ``step``
------------------

``step`` runs in a loop for as long as the entity is deployed. This entity trains
on its data, sends the result to a server and waits for the next global model:

.. code-block:: python

   from flypse.federation import Entity


   class Client(Entity):
       def __init__(self, entity_id, server_id):
           super().__init__(entity_id)
           self.server_id = server_id
           self.model = LinearModel()

       async def step(self):
           train(self.model, self.data)
           record = self.model.update_version(self.model.get_parameters(), "local")
           await self.send(self.server_id, record, n_samples=len(self.data))
           message = await self.receive()
           self.model.load_version(message.record)

With ``store_step=True``, the value returned by ``step`` is available to the
:doc:`step_result metric <../getting-started/inspect-results>`.

.. important::

   An entity runs on an asyncio loop, and messages reach it on that loop: while
   its ``step`` runs without awaiting, nothing is delivered to it.

Lifecycle
---------

An entity lives in two places, and each method runs in only one of them:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Method
     - Runs on
     - What happens
   * - ``__init__``
     - your script
     - The entity is created. ``data`` is None and ``rng`` is not seeded yet.
   * - ``on_deploy``
     - the device
     - ``rng`` is seeded and ``data`` is bound.
   * - ``step``
     - the device
     - Runs in a loop until the entity is stopped.
   * - ``on_undeploy``
     - the device
     - ``data`` and ``splits`` are released.

Anything that depends on the data or on the seed belongs in ``on_deploy``, after
calling the method of the parent class:

.. code-block:: python

   class Client(Entity):
       def on_deploy(self):
           super().on_deploy()
           self.model.model = random_model(self.rng)
           self.splits = FractionSplitter({"train": 0.8, "test": 0.2}).apply(self.data)
           self.data = self.splits["train"]

An entity can be deployed more than once: when the placement of a federation is
reset, its entities are undeployed and deployed again, and they run ``on_deploy``
again. See :doc:`system` for when that happens.

Local data
----------

``self.data`` is the :class:`~flypse.data.view.DataView` of the entity: it can be
indexed, iterated and sliced, so ``self.data[:32]`` is a view of the first 32
items, with nothing copied. An entity may reassign it, for instance to one of its
parts.

An entity holding no data, such as an aggregator, is created with
``with_data=False``: it is left out of the :doc:`split <data>`, and its ``data``
stays None.

Randomness
----------

``self.rng`` is a ``random.Random`` seeded from the seed of the process and the
identifier of the entity, so that entities draw different but reproducible
sequences:

.. code-block:: python

   local_batch = self.rng.sample(range(len(self.data)), k=32)

Roles are behaviour
-------------------

Learner, aggregator, or both: the same class covers all of them. An aggregator is an entity with ``with_data=False``
that combines what it receives. A group aggregator receives from its children and
sends to its parent. A peer trains and aggregates in the same ``step``.

When a step fails
-----------------

.. warning::

   Only a ``RouteNotFoundError`` is survivable: the entity logs a warning and
   starts its next step. Any other exception ends the loop of the entity for good,
   and the entities waiting for its messages block forever, unless they use a
   :ref:`timeout <messaging-timeouts>`.

.. seealso::

   - :doc:`messaging`
   - :doc:`models`
