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

Report format and location
--------------------------

The report format, its backend, and its location on disk are options of the
ECLYPSE simulation configuration, passed to the process as keyword arguments
(e.g., ``report_format="parquet"``, ``path="runs/mean-estimation"``).

.. seealso::

   - `ECLYPSE: inspect the results <https://eclypse.readthedocs.io/en/latest/source/overview/getting-started/inspect-results.html>`_
