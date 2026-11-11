Decentralised federated averaging
=================================

The decentralised example has no server: five peers in a ring each train on their
own data and average their model with those of their two neighbours.

Use it when you want to understand:

- how a federation works with no central entity,
- how to keep what arrives early aside, one queue per sender,
- how a consensus forms through repeated local averaging.

The full code lives in the
`examples/decentralized <https://github.com/eclypse-org/flypse/tree/main/examples/decentralized>`_
directory.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.decentralized.main

Model and data
--------------

The model and the data are the ones of the :doc:`federated averaging example
<fedavg>`: a linear model trained with stochastic gradient descent, and a synthetic
regression source sorted by its input feature. The example keeps its own copy of
both, so that it reads on its own.

.. dropdown:: Model code

    .. literalinclude:: ../../../../examples/decentralized/model.py
        :language: python

.. dropdown:: Data code

    .. literalinclude:: ../../../../examples/decentralized/data.py
        :language: python

Entities
--------

``Peer`` is both a learner and an aggregator. At every step it trains, sends its model
to its neighbours, collects theirs and averages everything, its own model included,
weighted by the number of samples. Peers draw their initial model from their ``rng``
in ``on_deploy``, and keep 20% of their data aside as a test part.

Two details make the protocol work:

- One neighbour can be a round ahead of the other, so its next message may arrive
  first. ``receive`` returns messages as they come, so a peer keeps them in a queue
  per sender, and takes one message per neighbour at every round.
- A peer reads ``message.record`` and never loads it: ``load_version`` would adopt
  the counters of a neighbour, while the ``"global"`` version of a peer counts its
  own rounds.

.. dropdown:: Entities code

    .. literalinclude:: ../../../../examples/decentralized/entities.py
        :language: python

Process
-------

``main`` builds the ring with :func:`~flypse.federation.topology.get_ring` and runs
it on the default mirror infrastructure. Every ten rounds, peers evaluate their
model on their test part and report its parameters, from which the example measures
how far apart the models are.

.. dropdown:: Process code

    .. literalinclude:: ../../../../examples/decentralized/main.py
        :language: python

What to inspect
---------------

.. code-block:: text

   round   mean test loss   largest distance between peers
   10      0.137220         0.351514
   20      0.029629         0.149306
   30      0.009107         0.063738
   40      0.004956         0.027375
   50      0.004020         0.014334
   60      0.003772         0.008949

There is no global model here: every peer has its own, and the last column is the
distance between the two peers furthest apart. It shrinks as models cross the ring
one hop at a time, and the loss follows. A model has to go through several peers to
reach the other side, so a sparse graph like a ring converges more slowly than a star.

The process is seeded, and each round is made of the same messages in the same
order, so the table is the same on every run.

This example is a good template when you want to try another topology, or another
way of combining the models of the neighbours.

Related concepts:

- :doc:`../concepts/federation`
- :doc:`../concepts/messaging`
