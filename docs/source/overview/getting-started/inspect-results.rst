===================
Inspect the results
===================

A :class:`~flypse.process.FederatedProcess` is an ECLYPSE simulation, so its
outputs are stored on disk and exposed through its ``report``.

Step results
------------

When an entity is created with ``store_step=True``, the value returned by its
``step`` method is reported, as the ``step_result`` metric of the entity. The
metric requires the default ECLYPSE metrics:

.. code-block:: python

   process = FederatedProcess(
       federation,
       data,
       SystemConfig(),
       include_default_metrics=True,
   )
   process.run(seconds=5)

   frame = process.report.service(service_ids="server")
   results = frame[frame["callback_id"] == "step_result"]

Each row of ``results`` is a value returned by the server, in the ``value``
column, with the timestamp at which it was reported.

Federation metrics
------------------

By default, a process also reports the federation metrics of every entity,
read from the live entity after each step of the process:

- ``steps``: the number of steps the entity completed;
- ``messages_sent`` and ``messages_received``: the messages exchanged, those
  sent counted once per recipient;
- ``bytes_sent`` and ``bytes_received``: the bytes of model they carried;
- ``send_time`` and ``receive_time``: the seconds spent waiting for messages
  to be delivered, and for messages to arrive.

The values only grow, so the difference between two reports gives the
activity in between. The same counters are available to the entity itself as
its ``traffic``. They are left out with ``include_federation_metrics=False``:

.. code-block:: python

   frame = process.report.service(service_ids="server")
   received = frame[frame["callback_id"] == "bytes_received"]

Evaluation
----------

An entity evaluates itself by overriding ``evaluate``, returning the value of
each evaluation metric by name (``None``, the default, means it does not
evaluate). For example, a learner can keep a test part of its own data and
evaluate on it the latest version of its model:

.. code-block:: python

   class Learner(Entity):
       def on_deploy(self):
           super().on_deploy()
           parts = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng)
           split = parts.apply(self.data)
           self.train, self.test = split["train"], split["test"]

       def evaluate(self):
           record = self.model.versions[-1]
           return {"loss": test_loss(record.parameters, self.test)}

``evaluate`` runs on the device while the entity keeps stepping, so it should
evaluate a snapshot of the model, such as a recorded version, rather than the
live model.

Every process has an ``evaluate`` event, which evaluates the entities when
triggered and keeps the results in their ``eval_metrics``. The names given as
``eval_metrics`` to the process are reported right after each evaluation:

.. code-block:: python

   process = FederatedProcess(federation, data, SystemConfig(), eval_metrics=["loss"])
   process.start()
   ...
   process.trigger("evaluate")
   ...
   process.stop()

   frame = process.report.service()
   losses = frame[frame["callback_id"] == "loss"]

Entities that do not evaluate, or whose evaluation lacks a name, report no
row for it.

Report format and location
--------------------------

The report format, its backend, and its location on disk are options of the
ECLYPSE simulation configuration, passed to the process as keyword arguments
(e.g., ``report_format="parquet"``, ``path="runs/mean-estimation"``).

.. seealso::

   - `ECLYPSE: inspect the results <https://eclypse.readthedocs.io/en/latest/source/overview/getting-started/inspect-results.html>`_
