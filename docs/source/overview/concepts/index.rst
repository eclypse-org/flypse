========
Concepts
========

.. toctree::
   :maxdepth: 2
   :hidden:

   federation
   entities
   messaging
   models
   data
   splitters
   system
   process

This section collects the building blocks of FLYPSE.

Use these pages once you already know the basic workflow and want to customise
who takes part in a federation, what they exchange, the data they hold, and the
system they run on.

.. grid:: 2

   .. grid-item::

      .. card:: Federation
         :link: federation
         :link-type: doc

         Connect entities into a star, a tree, a ring, or any graph of your own.

   .. grid-item::

      .. card:: Entities
         :link: entities
         :link-type: doc

         Implement what each participant does, and follow its lifecycle on a
         device.

   .. grid-item::

      .. card:: Messaging
         :link: messaging
         :link-type: doc

         Exchange models and control messages, and understand what a message
         costs.

   .. grid-item::

      .. card:: Models and versions
         :link: models
         :link-type: doc

         Expose the parameters of any model as numpy arrays, and track their
         lineage.

   .. grid-item::

      .. card:: Federated data
         :link: data
         :link-type: doc

         Split a data source across the entities, uniformly or with label or
         quantity skew.

   .. grid-item::

      .. card:: View splitters
         :link: splitters
         :link-type: doc

         Split the data of an entity into parts, such as train and test.

   .. grid-item::

      .. card:: System and placement
         :link: system
         :link-type: doc

         Choose the devices that host the federation, and how the entities are
         placed on them.

   .. grid-item::

      .. card:: Federated process
         :link: process
         :link-type: doc

         Configure, run and stop the emulation of a federation.
