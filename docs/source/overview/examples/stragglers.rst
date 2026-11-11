Stragglers
==========

The stragglers example slows down the uplink of one client for a few seconds, while a
server with a deadline carries on without it.

Use it when you want to understand:

- how to change an infrastructure during a run with update policies,
- how a slow link really delays the messages that cross it,
- how a server can stop waiting with ``receive(timeout=...)``.

The full code lives in the
`examples/stragglers <https://github.com/eclypse-org/flypse/tree/main/examples/stragglers>`_
directory.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.stragglers.main

Model and data
--------------

The model and the data are the ones of the :doc:`federated averaging example
<fedavg>`: a linear model trained with stochastic gradient descent, and a synthetic
regression source sorted by its input feature. The example keeps its own copy of
both, so that it reads on its own.

.. dropdown:: Model code

    .. literalinclude:: ../../../../examples/stragglers/model.py
        :language: python

.. dropdown:: Data code

    .. literalinclude:: ../../../../examples/stragglers/data.py
        :language: python

Entities
--------

``Client`` trains, sends its model to the server and waits for the next global one,
as in the federated averaging example. ``DeadlineServer`` closes a round when every
client has answered, or when the deadline expires.

A model generated for a previous round is late, so it does not take part in the
average. Everyone who wrote gets the latest model back, on time or not: every client
always has one request pending, so the clients never need to know about the deadline.

.. dropdown:: Entities code

    .. literalinclude:: ../../../../examples/stragglers/entities.py
        :language: python

Infrastructure
--------------

``slow_uplink`` creates an update policy, a plain function the process calls with the
infrastructure at every process step. It counts them, raises the latency of the link
from a client to the server at the first given step, and resets it at the second. At
100 ms per process step, the link slows down three seconds in, and recovers three
seconds later.

Only the direction towards the server slows down, as a mirror link has one edge per
direction. If the replies were slow too, the server would stall on its own ``send``,
and the deadline would not protect it. For the same reason, the example degrades a
link and does not fail a device, which would
:ref:`reset the whole federation <placement-reset>`.

The infrastructure is built with :func:`~flypse.system.infrastructure.get_mirror`,
as ``SystemConfig("mirror")`` takes no policies.

.. dropdown:: Infrastructure code

    .. literalinclude:: ../../../../examples/stragglers/main.py
        :language: python

What to inspect
---------------

The server reports each round, and the example counts them second by second.
The numbers change a little at every run:

.. code-block:: text

   second  rounds  mean participants  late models
   0       79      5.0                0
   1       82      5.0                0
   2       83      5.0                0
   3       5       4.2                0
   4       5       4.0                1
   5       5       4.0                1
   6       66      5.0                1
   7       82      5.0                0
   8       82      5.0                0
   9       78      5.0                0

   mean test loss at round 560: 0.003160

While the uplink is slow, rounds drop from about 80 per second to 5: the server waits
for the deadline of 0.2 seconds at every round, then closes it with the other four
clients. The straggler's model takes a whole second to arrive, one round per second
reports it as late, and the client gets the latest model back and tries again. When
the link recovers, the client rejoins and the federation goes back to full speed.

Without a deadline, the server would wait for the straggler at every round, and the
federation would run at about one round per second.

Unlike the other examples, this one depends on the wall clock, so the seed does not
reproduce it round by round.

This example is a good template when you want to study how a protocol reacts to the
conditions of the system.

Related concepts:

- :doc:`../advanced/heterogeneity`
- :doc:`../concepts/messaging`
- :doc:`../concepts/system`
