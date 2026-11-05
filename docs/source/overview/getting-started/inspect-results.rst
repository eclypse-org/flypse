===================
Inspect the results
===================

A :class:`~flypse.process.FederatedProcess` gives you two views of a finished run:

- ``reported``, the records the entities pushed during the run,
- ``report``, the metrics collected by ECLYPSE, which must be requested.

Read the records of the entities
--------------------------------

An entity can report anything at any moment of its ``step`` with
:meth:`~flypse.federation.entity.Entity.report`. Its values are pushed straight
to the simulator, without waiting for it, and with no metric reading them:

.. code-block:: python

   class Server(Entity):
       async def step(self):
           ...
           self.report(round=record.version[0], estimate=average[0])

Once the process has finished, ``reported`` holds one record per report, oldest
first. The values of a report stay together, with who reported them, from where
and when:

.. code-block:: python

   process = FederatedProcess(federation, data, step_every_ms=100.0, seed=0)
   process.run(seconds=5)

   print(process.reported[0])

.. code-block:: text

   {'round': 1, 'estimate': 24.75, 'node_id': 'server', 'entity_id': 'server', 'timestamp': '2026-11-05T23:27:10.992829', 'n_report': 1}

``entity_id`` and ``n_report``, the progressive number of the report for that
entity, identify a record.

Read the ECLYPSE metrics
------------------------

``report`` exposes the ECLYPSE metrics as frames. To collect the value returned
by the ``step`` of an entity, create the entity with ``store_step=True``, return
the value, and request the default metrics:

.. code-block:: python

   class Server(Entity):
       def __init__(self, entity_id):
           super().__init__(entity_id, store_step=True, with_data=False)
           self.model = MeanModel({"mean": np.zeros(1)})

       async def step(self):
           ...
           return float(average[0])


   process = FederatedProcess(
       federation,
       data,
       step_every_ms=100.0,
       seed=0,
       include_default_metrics=True,
   )
   process.run(seconds=5)

   frame = process.report.service(service_ids="server")
   results = frame[frame["callback_id"] == "step_result"]
   print(results[["n_event", "timestamp", "value"]].tail(2))

.. code-block:: text

         n_event                   timestamp      value
   1027       49  2026-11-05T23:27:37.599781  49.499989
   1048       50  2026-11-05T23:27:37.698982  49.499989

Each row of ``results`` is a value returned by the server, in the ``value``
column, with the timestamp at which it was collected.

.. note::

   With no metric requested, there is no tabular report and ``process.report``
   raises ``FileNotFoundError``, while ``reported`` still works.

.. seealso::

   - :doc:`infrastructure`
   - :doc:`../advanced/reporting`
   - :doc:`../advanced/evaluation`
   - `ECLYPSE: inspect the results <https://eclypse.readthedocs.io/en/latest/source/overview/getting-started/inspect-results.html>`_
