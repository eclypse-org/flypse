View splitters
==============

A :class:`~flypse.data.dataset.FederatedDataset` splits the data *across* the
entities. A :class:`~flypse.data.splitter.ViewSplitter` splits the data *of* an
entity into named parts, such as train, evaluation and test. Each part is a
:class:`~flypse.data.view.DataView` of the view it was split from.

Together, they give two ways of holding data out:

- across entities, by dedicating some entities to evaluation: nothing more than a
  different behaviour in their ``step``,
- within an entity, by splitting its own view, as this page shows.

There is no global test set: every evaluation happens on the data of an entity.

There are two ways to split a view:

.. grid:: 2

   .. grid-item::

      .. card:: Split by fractions
         :link: split-by-fractions
         :link-type: ref

         Use :class:`~flypse.data.splitter.FractionSplitter`, the default.

   .. grid-item::

      .. card:: Write your own
         :link: write-your-own-splitter
         :link-type: ref

         Subclass :class:`~flypse.data.splitter.ViewSplitter`.

.. _split-by-fractions:

Split by fractions
------------------

:class:`~flypse.data.splitter.FractionSplitter` sizes the parts by fractions, which
must sum to 1. An entity calls it once its data is bound, in ``on_deploy``:

.. code-block:: python

   from flypse.data import FractionSplitter


   class Client(Entity):
       def on_deploy(self):
           super().on_deploy()
           splitter = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng)
           self.splits = splitter.apply(self.data)
           self.data = self.splits["train"]

Passing ``self.rng`` as the seed makes each entity split differently, but
reproducibly. Storing the parts in ``self.splits`` makes them available to the
:doc:`evaluation <../advanced/evaluation>` of the entity
by name.

.. note::

   Parts are sized with the largest remainder method, so a part can be empty when
   the view is too small.

Stratify
~~~~~~~~

With ``stratify``, a function returning the class of an item, the items are split
class by class, so each part keeps the class proportions of the view:

.. code-block:: python

   splitter = FractionSplitter({"train": 0.8, "test": 0.2}, stratify=label, seed=self.rng)

Keep the order
~~~~~~~~~~~~~~

With ``order``, a function returning the sorting key of an item, the items are not
shuffled: the first parts get the earliest items. For time series, no part then
precedes the earlier ones:

.. code-block:: python

   def timestamp(item):
       return item["time"]


   splitter = FractionSplitter({"train": 0.7, "test": 0.3}, order=timestamp)

``stratify`` and ``order`` can be combined.

Split again
~~~~~~~~~~~

A part is a view, so it can be split in turn:

.. code-block:: python

   parts = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng).apply(self.data)
   parts |= FractionSplitter({"train": 0.75, "eval": 0.25}, seed=self.rng).apply(parts["train"])

Out of 100 items, this gives 60 for training, 20 for evaluation and 20 for test.

.. _write-your-own-splitter:

Write your own
--------------

Implement :meth:`~flypse.data.splitter.ViewSplitter.split`, returning the positions
within the view assigned to each part.
:meth:`~flypse.data.splitter.ViewSplitter.apply` then builds the views:

.. code-block:: python

   from flypse.data import ViewSplitter


   class LastItem(ViewSplitter):
       """Keeps the last item of the view for test."""

       def split(self, view):
           n = len(view)
           return {"train": range(n - 1), "test": [n - 1]}

.. seealso::

   - :doc:`data`
