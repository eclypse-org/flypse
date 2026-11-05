===================
Inspect the results
===================

A :class:`~flypse.process.FederatedProcess` is an ECLYPSE simulation, so its
outputs are stored on disk. Its ``report`` exposes the ECLYPSE metrics, which
must be requested as there are none by default, and its ``reported`` what the
entities reported themselves.

Step results
------------

When an entity is created with ``store_step=True``, the value returned by its
``step`` method is reported, as the ``step_result`` metric of the entity. The
metric requires the default ECLYPSE metrics:

.. code-block:: python

   process = FederatedProcess(federation, data, include_default_metrics=True)
   process.run(seconds=5)

   frame = process.report.service(service_ids="server")
   results = frame[frame["callback_id"] == "step_result"]

Each row of ``results`` is a value returned by the server, in the ``value``
column, with the timestamp at which it was reported.

Entity reports
--------------

An entity can report anything at any moment of its ``step`` with ``report``. Its
values are pushed straight to the simulator, without waiting for it, and with no
metric reading them:

.. code-block:: python

   class Server(Entity):
       async def step(self):
           ...
           self.report(round=version[0], participants=len(messages), loss=loss)

Once the process has finished, ``reported`` holds one record per report, oldest
first. The values of a report stay together, with who reported them, from where
and when:

.. code-block:: python

   process = FederatedProcess(federation, data)
   process.run(seconds=5)

   process.reported[0]
   # {"round": 1, "participants": 5, "loss": 0.04, "node_id": "server",
   #  "entity_id": "server", "timestamp": "2026-11-05T16:48:30.160476",
   #  "n_report": 1}

``entity_id`` and ``n_report``, the progressive number of the report for that
entity, identify a record. These two names, ``node_id`` and ``timestamp`` are
reserved. A report is complete as soon as it is made: nothing an entity reported
is lost when the process stops, or when the entity is undeployed.

Values can be anything JSON can represent, and scalars or arrays of the numerical
libraries, which become plain numbers and lists. Anything else raises a
``TypeError`` from ``report``.

Every report is a call to the simulator, which sustains a few thousand of them
per second: an entity reporting faster than that loses nothing, but the process
takes longer to stop, and stops later, until the simulator catches up.

The records are stored in the JSON report (``json/service.jsonl`` in the output
directory of the process) whatever the ``report_format``, as they do not fit the
columns of the other formats. ``process.report`` shows them only with
``report_format="json"``, as the ``entity_report`` callback, and only with the
``pandas`` backend: the polars ones cannot read a report mixing them with the
numbers of the metrics. ``reported`` works in all cases.

Evaluation
----------

An entity decides what to evaluate, and when, in its own ``step``. It calls
``run_evaluation`` with a version of a model and, optionally, the name of one of
its ``splits``. The version can be any the entity holds: one it generated, or
one it received from a peer or a server, even without loading it. The evaluation
itself is the ``evaluate`` method of the entity, which returns the value of each
metric by name (``None``, the default, means it does not evaluate):

.. code-block:: python

   class Learner(Entity):
       def on_deploy(self):
           super().on_deploy()
           splitter = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng)
           self.splits = splitter.apply(self.data)
           self.data = self.splits["train"]

       def evaluate(self, record, data):
           params = self.model.get_parameters(flattened=False, version=record)
           return {"loss": test_loss(params, data)}

       async def step(self):
           message = await self.receive()
           self.run_evaluation(message.record, "test")
           self.model.load_version(message.record)
           ...

``evaluate`` runs inside ``step``, so it takes time from the round of the entity.
A failing evaluation is logged and reports nothing, and the entity carries on.

The results are reported at once, as a single record:

.. code-block:: python

   process.reported[0]
   # {"round": 10, "version": [10, 0], "model": "fdaeb36482741f8b",
   #  "split": "test", "metrics": {"loss": 0.012}, "node_id": "client_0",
   #  "entity_id": "client_0", "timestamp": "2026-11-05T16:49:55.431911",
   #  "n_report": 1}

Each record stands on its own, so a round can be compared across the entities by
grouping on ``round`` and ``split``:

- ``round`` is the coarsest counter of the version, unless ``run_evaluation`` is
  given another one, e.g. for a protocol that does not bump that level;
- ``version`` is the lineage of the model, but different models can share it,
  such as two clients generating ``(3, 1)``;
- ``model`` is a hash of the parameters, so it is the same for all the entities
  evaluating the same model, and differs for different ones. A model that no
  longer changes between rounds keeps its hash, so tell evaluations apart by
  ``round`` too. The hash follows the exact bits of the parameters: it is not
  comparable across runs unless the protocol computes them identically, which an
  aggregation following the arrival order of the messages does not.

Report format and location
--------------------------

The report format, its backend, and its location on disk are options of the
ECLYPSE simulation configuration, passed to the process as keyword arguments
(e.g., ``report_format="parquet"``, ``path="runs/mean-estimation"``).

.. seealso::

   - `ECLYPSE: inspect the results <https://eclypse.readthedocs.io/en/latest/source/overview/getting-started/inspect-results.html>`_
