Federated averaging
===================

The federated averaging example trains a linear model across five clients and a
server, the basic client-server pattern of federated learning.

Use it when you want to understand:

- how a synchronous protocol emerges from the ``step`` of each entity,
- how to wrap a model of your own into a federated one,
- how to hold out a test part on each client and evaluate the global model.

The full code lives in the
`examples/fedavg <https://github.com/eclypse-org/flypse/tree/main/examples/fedavg>`_
directory.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.fedavg.main

Model
-----

The model is plain Python, with no framework, as FLYPSE ships no model of its own.
``local_train`` runs stochastic gradient descent on a copy of the model, and
``FedAvgModel`` exposes its weights and its bias as numpy arrays.

.. dropdown:: Model code

    .. literalinclude:: ../../../../examples/fedavg/model.py
        :language: python

Data
----

The source is a synthetic regression task, sorted by its input feature. A
:class:`~flypse.data.uniform.UniformDataset` assigns contiguous blocks, so every
client sees a different region of the feature space and the federation is non-IID
by construction.

.. dropdown:: Data code

    .. literalinclude:: ../../../../examples/fedavg/data.py
        :language: python

Entities
--------

The protocol lives entirely in the entities. ``FedAvgClient`` trains, sends its
model to the server, and waits for the next global model. ``FedAvgServer`` waits for
a model from each of its neighbours, averages them weighted by their number of
samples, and broadcasts the result:

.. code-block:: text

   Θ = Σ_c (n_c / n) Θ_c

Clients draw their initial model from their ``rng`` in ``on_deploy``, and keep 20%
of their data aside as a test part. The server sorts the messages before averaging,
so that the result does not depend on the order they arrived in.

.. dropdown:: Entities code

    .. literalinclude:: ../../../../examples/fedavg/entities.py
        :language: python

Process
-------

``main`` builds the federation and its data, and runs a process on the default
mirror infrastructure, with one device per entity, for ten seconds.

.. dropdown:: Process code

    .. literalinclude:: ../../../../examples/fedavg/main.py
        :language: python

What to inspect
---------------

Every ten rounds, each client evaluates the global model on its test part. The
example prints the mean loss of the first evaluated rounds, and of the last one:

.. code-block:: text

   round   evaluations  mean test loss
   10      5            0.006386
   20      5            0.003138
   30      5            0.003159
   40      5            0.003160
   50      5            0.003160
   760     5            0.003160

The loss settles within a few dozen rounds. The process is seeded, so the values of
a round are the same on every run, while the last round depends on the speed of
your machine.

This example is a good template when you want to try a client-server protocol of
your own.

Related concepts:

- :doc:`../concepts/entities`
- :doc:`../concepts/models`
- :doc:`../concepts/data`
- :doc:`../advanced/evaluation`
