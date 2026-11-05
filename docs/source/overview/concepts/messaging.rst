Messaging
=========

Entities exchange versions of their models, and any data attached to them, with
three methods of :class:`~flypse.federation.entity.Entity`. They wrap the ECLYPSE
MPI interface, so you never call it directly.

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

A record is one flat, read-only vector, so it is sized correctly on the wire and
cannot be altered by the entities sharing it.

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

What a message costs
--------------------

Awaiting ``send`` or ``bcast`` takes the modelled time of the message, which
ECLYPSE computes from the route it follows through the infrastructure:

.. code-block:: text

   cost = processing_time + sum over the links of (size / bandwidth + latency)

The size of the message comes from the size of the record, so a bigger model
really does take longer to send. To several recipients, the wait lasts as long as
the slowest of them, so one slow link stalls the sender.

Ordering
--------

``receive`` returns messages as they arrive, whoever sent them: there is no way
to wait for a given sender. When the order matters, keep what arrives early aside,
one queue per sender:

.. code-block:: python

   inbox = collections.defaultdict(collections.deque)

   while not inbox[neighbour]:
       message = await self.receive()
       inbox[message.sender].append(message)
   message = inbox[neighbour].popleft()

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
``eclypse.remote.communication.RouteNotFoundError``, as does sending when the
infrastructure has no path to the recipient. The entity logs a warning and starts
its next ``step``, so the rest of the step is skipped. A ``bcast`` can reach some
neighbours before it fails.

.. seealso::

   - :doc:`../examples/decentralized`
   - :doc:`ECLYPSE messaging <eclypse:source/overview/advanced/emulation/messaging>`
