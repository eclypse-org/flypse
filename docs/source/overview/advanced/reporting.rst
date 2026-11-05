Reporting
=========

A run of a federation produces two kinds of results, stored apart:

- the values pushed by the entities, available from ``process.reported``,
- the metrics collected by ECLYPSE, available from ``process.report``.

Report from an entity
---------------------

:meth:`~flypse.federation.entity.Entity.report` pushes values straight to the
simulator, from anywhere in a ``step``, without waiting for it and with no metric
reading them. The values of a call stay together:

.. code-block:: python

   class Server(Entity):
       async def step(self):
           ...
           self.report(round=round_, participants=len(messages), loss=loss)

Values can be anything JSON can represent, and scalars or arrays of the numerical
libraries, which become plain numbers and lists. Anything else raises a
``TypeError`` from ``report``, in the ``step`` of the entity.

Read the records
----------------

Once the process has finished,
:attr:`~flypse.process.FederatedProcess.reported` holds one record per report,
oldest first. A record has the values of the report, and who reported them, from
where and when:

.. code-block:: python

   for record in process.reported:
       print(record["entity_id"], record["round"], record["loss"])

.. important::

   ``entity_id``, ``node_id``, ``timestamp`` and ``n_report`` are reserved: a value
   with one of these names is overwritten silently.

``entity_id`` and ``n_report``, the progressive number of the report for that
entity, identify a record. A report is complete as soon as it is made: nothing an
entity reported is lost when the process stops, or when the entity is undeployed.

Step results
------------

An entity created with ``store_step=True`` keeps the value returned by each
``step``, and the default ECLYPSE metrics collect them as the ``step_result`` of the
entity:

.. code-block:: python

   process = FederatedProcess(
       federation,
       data,
       step_every_ms=100.0,
       include_default_metrics=True,
   )
   process.run(seconds=5)

   frame = process.report.service(service_ids="server")
   results = frame[frame["callback_id"] == "step_result"]

.. warning::

   ``step_result`` takes one value per simulation step, the oldest one. An entity
   faster than the simulation steps falls behind: the frame lags, and the latest
   values are never collected. For a value per round, use ``report``.

Custom metrics
--------------

Any ECLYPSE metric works on a process, passed through ``events``. A service metric
with ``remote=True`` runs on the device, and receives the live entity:

.. code-block:: python

   from eclypse.report.metrics import metric


   @metric.service(remote=True)
   def n_versions(service):
       return len(service.model.versions)


   process = FederatedProcess(federation, data, step_every_ms=100.0, events=[n_versions])
   process.run(seconds=3)

   frame = process.report.service(service_ids="server")
   results = frame[frame["callback_id"] == "n_versions"]

.. warning::

   A metric must return a single scalar. A metric reading an entity also runs on
   another thread than its ``step``, so it should not change its state.

Storage and formats
-------------------

The records of the entities are stored in the JSON report, ``json/service.jsonl``
in the output directory of the process, whatever the ``report_format``, as they do
not fit the columns of the other formats. ``reported`` works in all cases.

``process.report`` shows them too, as the ``entity_report`` callback, only with
``report_format="json"`` and the ``pandas`` backend: the polars backends cannot read
a report mixing them with the numbers of the metrics.

With no metric requested, there is no tabular report, so ``process.report`` raises
``FileNotFoundError``.

Limits
------

Every report is a call to the simulator, which sustains a few thousand of them per
second: an entity reporting faster than that loses nothing, but the process takes
longer to stop, until the simulator catches up.

.. seealso::

   - :doc:`evaluation`
   - :doc:`ECLYPSE reporting <eclypse:source/overview/advanced/reporting>`
