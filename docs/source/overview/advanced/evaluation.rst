Evaluation
==========

An entity decides what to evaluate, and when, in its own ``step``. There is no
global test set: every evaluation runs on the data of an entity, with a version of
a model it holds, and its results go straight to the
:doc:`reports <reporting>`.

Implement ``evaluate``
----------------------

:meth:`~flypse.federation.entity.Entity.evaluate` receives a version of a model and
some data, and returns the value of each metric by name. The version need not be
the loaded one, so read its parameters from the record:

.. code-block:: python

   class Client(Entity):
       def evaluate(self, record, data):
           parameters = self.model.get_parameters(flattened=False, version=record)
           return {"loss": test_loss(parameters, data)}

Returning None, the default, means that the entity does not evaluate.

Run an evaluation
-----------------

:meth:`~flypse.federation.entity.Entity.run_evaluation` calls ``evaluate`` on a
version and reports the results. Name one of the ``splits`` of the entity to
evaluate on it, or nothing to use its ``data``:

.. code-block:: python

   class Client(Entity):
       def on_deploy(self):
           super().on_deploy()
           splitter = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng)
           self.splits = splitter.apply(self.data)
           self.data = self.splits["train"]

       async def step(self):
           ...
           message = await self.receive()
           self.run_evaluation(message.record, "test")
           self.model.load_version(message.record)

The version can be any the entity holds: one it generated, or one it received from
a peer or a server, even without loading it.

.. note::

   ``evaluate`` runs inside ``step``, so it takes time from the round of the entity.
   Evaluate every few rounds when it is expensive. A failing evaluation is logged
   and reports nothing, and the entity carries on.

Read a record
-------------

The results are reported at once, as a single record:

.. code-block:: python

   process.reported[0]

.. code-block:: text

   {'round': 10, 'version': [10, 0], 'model': '0394059a834f6bb2', 'split': 'test', 'metrics': {'loss': 0.0057279877326380955}, 'node_id': 'client_2', 'entity_id': 'client_2', 'timestamp': '2026-11-05T23:35:35.910228', 'n_report': 1}

Compare across entities
-----------------------

Each record stands on its own, so a round can be compared across the entities by
grouping on ``round`` and ``split``:

- ``round`` is the coarsest counter of the version, unless ``run_evaluation`` is
  given another one, for a protocol that does not bump that level,
- ``version`` is the lineage of the model, but different models can share it, such
  as two clients generating ``(3, 1)``,
- ``model`` is a hash of the parameters, the same for all the entities evaluating
  the same model.

.. important::

   A model that stops changing keeps its hash, so tell evaluations apart by
   ``round`` too. The hash follows the exact bits of the parameters: it is only
   comparable across runs when they are computed the same way, see
   :doc:`reproducibility`.

Evaluation entities
-------------------

Holding data out across the federation is a matter of behaviour: an entity that
only evaluates is an entity whose ``step`` receives models and calls
``run_evaluation``, without training or sending anything.

.. seealso::

   - :doc:`reporting`
   - :doc:`../concepts/splitters`
