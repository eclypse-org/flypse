===============
Getting started
===============

.. toctree::
   :maxdepth: 2
   :hidden:

   first-federation
   inspect-results
   infrastructure

Let's get started with FLYPSE.

FLYPSE runs every federation as an ECLYPSE remote emulation: the entities are
remote services, deployed onto the devices of an infrastructure and
communicating through ECLYPSE's MPI interface. The guided paths below take
you from a first run to the results it produces, and to the system it runs on.

Happy paths
-----------

.. grid:: 1

   .. grid-item::

      .. card:: 1. Run your first federation
         :link: first-federation
         :link-type: doc

         Implement the entities of a small client-server federation, split
         the data across them, and emulate the process.

   .. grid-item::

      .. card:: 2. Inspect the results
         :link: inspect-results
         :link-type: doc

         Read the records the entities reported during the run, and the
         metrics collected by ECLYPSE.

   .. grid-item::

      .. card:: 3. Choose where the federation runs
         :link: infrastructure
         :link-type: doc

         Place the federation on a single device, on a copy of its own
         topology, or on any ECLYPSE infrastructure.

.. tip::

   If you have not yet installed FLYPSE, refer to the :doc:`Installation <../install>` page.
