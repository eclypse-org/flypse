Federated process
=================

A :class:`~flypse.process.FederatedProcess` is the emulation of a federation: it
binds together the federation, its data and the system it runs on, so you only call
``run``.

Run a process
-------------

The first three arguments are the :doc:`federation <federation>`, its
:doc:`data <data>` and the :doc:`system <system>`, the mirror infrastructure when
omitted. The keyword arguments configure the run:

.. code-block:: python

   from flypse.process import FederatedProcess

   process = FederatedProcess(federation, data, step_every_ms=100.0, seed=0)
   process.run(seconds=10)

.. important::

   ``step_every_ms`` is required to run for a given time. A process takes no steps on
   its own, and entities are deployed by the first step: without it, ``run`` waits,
   nothing is deployed, and no results come out.

A process emulates a single federation, and registers it by itself. Calling
``register`` with any other federation raises ``ValueError``.

What happens at runtime
-----------------------

#. Creating the process splits the data across the entities holding it, builds the
   system, and starts one Ray actor per device, each with a copy of the dataset.
#. ``run`` fires the first step, which places the entities and deploys them. Each
   entity seeds its ``rng``, binds its data, and starts looping on its ``step``.
#. Every ``step_every_ms``, the process takes a step: it applies the update policies,
   checks the placement, and collects the metrics. Entities keep running in between,
   at their own pace.
#. When the process stops, the entities are stopped and undeployed, and the reports
   are written to disk.

.. note::

   A process step is neither the ``step`` of an entity nor a round of learning.
   ``step_every_ms`` paces the process, not the entities: a federation can go through
   many rounds between two process steps, or none at all.

Configuration
-------------

These are the options that matter most:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Meaning
   * - ``step_every_ms``
     - How often the process takes a step, in milliseconds. Required, unless you
       use ``run(steps=N)``.
   * - ``timeout``
     - Stops the process after this many seconds, counted from its creation.
   * - ``max_steps``
     - Stops the process after this many steps. Whichever of ``timeout`` and
       ``max_steps`` comes first wins.
   * - ``seed``
     - The seed of the whole process, random if omitted. See
       :doc:`../advanced/reproducibility`.
   * - ``events``
     - Metrics of your own. See :ref:`custom metrics <custom-metrics>`.
   * - ``include_default_metrics``
     - Whether to collect the default metrics, such as ``step_result``. See
       :doc:`../advanced/reporting`.
   * - ``path``
     - Where the outputs are written.
   * - ``report_format``, ``report_backend``
     - The format and the backend of the tabular report. Parquet and the ``polars``
       backends need ``polars``, which is not installed with FLYPSE.
   * - ``reporters``
     - Additional ECLYPSE reporters, see :doc:`../advanced/building-on-eclypse`. The
       JSON reporter is always included.
   * - ``log_level``, ``log_to_file``
     - The verbosity of the logs, and whether to keep them in a file.

Stopping
--------

There are three ways to stop a process:

- ``run(seconds=10)`` stops it 10 seconds after the call,
- ``timeout`` and ``max_steps``, with a plain ``run()``,
- ``run(steps=N)`` takes N process steps and stops straight away.

.. warning::

   ``run(steps=N)`` does not wait for the entities: they are deployed by the first
   process step and stopped right after the last one, so they get almost no time to
   run. Use ``seconds`` to let a federation learn.

Stopping does not wait for a round to end: entities are stopped wherever they are.
Those waiting for a message are cancelled, so ``Task was destroyed but it is
pending!`` lines at the end of the log are expected.

Where outputs go
----------------

Outputs are written under ``path``, ``~/eclypse-sim`` by default. If that directory
already exists, a timestamp is appended to the name, so read the actual location
from ``process.path``. The report format, its backend and its location can be set
with their options, for instance ``report_format="json"`` or
``path="runs/mean-estimation"``.

.. seealso::

   - :doc:`../getting-started/inspect-results`
   - :doc:`../advanced/reporting`
