===================
Inspect the results
===================

This page shows how to read what the entities reported during a run.

We will:

#. report values from an entity,
#. read them as records once the process has finished,
#. turn the records into a table.

The following example uses:

- :meth:`~flypse.federation.entity.Entity.report`, to push values from an entity,
- :attr:`~flypse.process.FederatedProcess.reported`, to read them back.

Report from an entity
---------------------

An entity can report any value JSON can represent, at any moment of its ``step``, with
:meth:`~flypse.federation.entity.Entity.report`. Its values go straight to the
process, without waiting for it:

.. code-block:: python

   class Server(Entity):
       async def step(self):
           ...
           self.report(round=record.version[0], estimate=average[0])

Read the records
----------------

Once the process has finished, ``reported`` holds one record per report, oldest
first. The values of a report stay together, with who reported them, from where
and when:

.. code-block:: python

   from flypse.process import FederatedProcess

   process = FederatedProcess(federation, data, step_every_ms=100.0, seed=0)
   process.run(seconds=5)

   print(process.reported[0])

.. code-block:: text

   {'round': 1, 'estimate': 24.75, 'node_id': 'server', 'entity_id': 'server', 'timestamp': '2026-11-05T23:27:10.992829', 'n_report': 1}

``entity_id`` and ``n_report``, the progressive number of the report for that
entity, identify a record.

Turn them into a table
----------------------

A record is a plain dictionary, so ``reported`` feeds any tool you like. pandas,
which is installed with FLYPSE, turns it into a table in one call:

.. code-block:: python

   import pandas as pd

   frame = pd.DataFrame(process.reported)
   print(frame[["round", "estimate"]].tail(2))

.. code-block:: text

        round   estimate
   568    569  49.499989
   569    570  49.499989

.. seealso::

   - :doc:`infrastructure`
   - :doc:`../advanced/reporting`
   - :doc:`../advanced/evaluation`
