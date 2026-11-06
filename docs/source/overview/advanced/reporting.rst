Reporting
=========

A run of a federation produces two kinds of results, stored apart:

- the values pushed by the entities, available from ``process.reported``,
- the metrics collected at every process step, available from ``process.report``.

Report from an entity
---------------------

:meth:`~flypse.federation.entity.Entity.report` pushes values straight to the
process, from anywhere in a ``step``, without waiting for it. The values of a call
stay together:

.. code-block:: python

   from flypse.federation import Entity


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

Metrics
-------

``report`` covers what an entity knows. A process can also collect metrics at every
process step, and ``include_default_metrics=True`` asks for the default ones.
``process.report`` exposes them as frames, in which an entity is called a *service*.

Among them, ``step_result`` holds the values returned by the ``step`` of the entities
created with ``store_step=True``:

.. code-block:: python

   from flypse.process import FederatedProcess

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

   ``step_result`` takes one value per process step, the oldest one. An entity
   faster than the process falls behind: the frame lags, and the latest values are
   never collected. For a value per round, use ``report``.

Storage and formats
-------------------

The records of the entities are stored in the JSON report, ``json/service.jsonl``
in the output directory of the process, whatever the ``report_format``, as they do
not fit the columns of the other formats. ``reported`` works in all cases.

``process.report`` shows them too, with ``callback_id`` equal to ``entity_report``, only with
``report_format="json"`` and the ``pandas`` backend: the ``polars`` backends cannot read
a report mixing them with the numbers of the metrics.

With no metric requested, there is no tabular report, so ``process.report`` raises
``FileNotFoundError``.

Limits
------

Every report is a call to the process, which sustains a few thousand of them per
second: an entity reporting faster than that loses nothing, but the process takes
longer to stop, until it catches up.

.. seealso::

   - :doc:`evaluation`
   - :ref:`custom-metrics`
