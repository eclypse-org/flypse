Hierarchical federated averaging
================================

The hierarchical example adds a tier between the clients and the server, the root of
the tree: group aggregators average their clients at every round, and the root
averages the groups every few rounds.

Use it when you want to understand:

- how to build a tree with :func:`~flypse.federation.topology.get_tree`,
- how three version levels record where a model comes from,
- why a version alone does not identify a model.

The full code lives in the
`examples/hierarchical <https://github.com/eclypse-org/flypse/tree/main/examples/hierarchical>`_
directory.

Run it from the repository root with:

.. code-block:: bash

   uv run python -m examples.hierarchical.main

Entities
--------

The clients are ``FedAvgClient`` and the root is ``FedAvgServer``, both from the
:doc:`federated averaging example <fedavg>`, with one more version level:
``("global", "group", "local")``. Only ``GroupAggregator`` is new.

At every round, a group aggregator averages the models of its clients with
``update_version(average, "group")``. Every ``GLOBAL_EVERY`` rounds, it also sends
the result to the root and loads the global model it gets back. Then it sends the
latest model to its clients.

It sends to its clients explicitly rather than with ``bcast``, as its neighbours
include the root. A plain ``receive`` is enough, with no queue per sender, because
the parties strictly alternate: while a group waits for the root, its clients are
waiting for the group.

.. dropdown:: Entities code

    .. literalinclude:: ../../../../examples/hierarchical/entities.py
        :language: python

Process
-------

``main`` runs four clients in two groups on the default mirror infrastructure. The
data comes from the same sorted source as the federated averaging example, so each
group holds one half of the feature range.

Clients evaluate every model they receive, and the example groups the records by
version.

.. dropdown:: Process code

    .. literalinclude:: ../../../../examples/hierarchical/main.py
        :language: python

What to inspect
---------------

.. code-block:: text

   version       evaluations  models  mean test loss
   (0, 1, 0)     4            2       0.189433
   (0, 2, 0)     4            2       0.144373
   (1, 0, 0)     4            1       0.446442
   (1, 1, 0)     4            2       0.066209
   (1, 2, 0)     4            2       0.051899
   (2, 0, 0)     4            1       0.140989
   (2, 1, 0)     4            2       0.024880

The ``models`` column counts the distinct ``model`` hashes behind a version. Group
versions show two: both groups generate ``(1, 1, 0)``, and they hold different
models, which is why a version alone is not an identity. Global versions show one,
as every client received the same model.

The loss goes up at each global version, because the data is non-IID: a group model
fits the half of the range its clients hold, while the global model has to cover
both. The jump shrinks round after round as the models converge.

The process is seeded, and the aggregators sort their messages, so the table is the
same on every run.

This example is a good template when you want to compare how often to aggregate at
each tier.

Related concepts:

- :doc:`../concepts/federation`
- :doc:`../concepts/models`
- :doc:`../advanced/evaluation`
