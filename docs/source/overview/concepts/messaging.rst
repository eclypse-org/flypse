Messaging
=========

Entities exchange versions of their models, and any data attached to them, with
three methods of :class:`~flypse.federation.entity.Entity`.

Send and receive
----------------

:meth:`~flypse.federation.entity.Entity.send` delivers a message to one or more
entities, and :meth:`~flypse.federation.entity.Entity.bcast` to all the neighbours.
Keyword arguments are attached to the message:

.. code-block:: python

   await self.send("server", record, n_samples=len(self.data), loss=loss)
   await self.send(["client_0", "client_1"], record)
   await self.bcast(record)

:meth:`~flypse.federation.entity.Entity.receive` waits for the next message, a
:class:`~flypse.federation.message.Message`:

.. code-block:: python

   message = await self.receive()
   message.sender  # the identifier of the sender
   message.record  # the ModelVersion sent
   message.data  # {"n_samples": 240, "loss": 0.04}

A record holds its parameters as one flat, read-only vector, so it is sized correctly
on the wire and cannot be altered by the entities sharing it.

Attach what the sender knows, such as how much data it trained on. How much a
collaborator weighs is decided by whoever aggregates, when it aggregates.

Control messages
----------------

A message without a model carries only its data, and its ``record`` is None:

.. code-block:: python

   await self.bcast(stop=True)

.. _messaging-timeouts:

Timeouts
--------

``receive`` waits for as long as it takes unless given a ``timeout``, in seconds.
It then raises ``TimeoutError``, and a message arriving later stays in the queue
for the next call:

.. code-block:: python

   try:
       message = await self.receive(timeout=2.0)
   except TimeoutError:
       ...

.. warning::

   ``receive(timeout=0)`` raises ``TimeoutError`` even with a message waiting. To
   poll the queue, use a small positive timeout.

.. _message-cost:

What a message costs
--------------------

Awaiting ``send`` or ``bcast`` takes the modelled time of the message, computed
from the route it follows through the infrastructure:

.. code-block:: text

   cost = sum over the devices of processing_time / 1000 +
          sum over the links of (bytes * 8e-6 / bandwidth + latency / 1000)

with ``processing_time`` and ``latency`` in milliseconds, and ``bandwidth`` in
megabits per second, for a cost in seconds. The devices and the links are those along
the route, both ends included. Entities on the same device exchange messages at no
cost.

The size of the message comes from the size of the record, so a bigger model
really does take longer to send. To several recipients, the wait lasts as long as
the slowest of them, so one slow link stalls the sender.

Ordering
--------

``receive`` returns messages as they arrive, whoever sent them: there is no way
to wait for a given sender. When the order matters, keep what arrives early aside on
the entity, one queue per sender:

.. code-block:: python

   import collections


   class Peer(Entity):
       def __init__(self, entity_id):
           super().__init__(entity_id)
           self.inbox = collections.defaultdict(collections.deque)

       async def receive_from(self, sender):
           while not self.inbox[sender]:
               message = await self.receive()
               self.inbox[message.sender].append(message)
           return self.inbox[sender].popleft()

Messages from the same sender keep their order, as long as the sender awaits each
``send``.

For the same reason, the result of an aggregation should not depend on who arrived
first. Floating-point rounding makes a sum follow the order of its terms, so sort
the messages before combining them:

.. code-block:: python

   messages.sort(key=lambda message: message.sender)

When a message cannot be delivered
----------------------------------

An entity can only reach its neighbours: sending to any other entity raises
``RouteNotFoundError``, as does sending when the infrastructure has no path to the
recipient. The entity logs a warning and starts
its next ``step``, so the rest of the step is skipped. A ``bcast`` can reach some
neighbours before it fails.

.. seealso::

   - :doc:`../examples/decentralized`
   - :doc:`../advanced/heterogeneity`
