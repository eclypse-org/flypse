Federated data
==============

In FLYPSE, a :class:`~flypse.data.dataset.FederatedDataset` embodies the data of
the whole federation together with the way it is partitioned across the entities.
It wraps a data source, anything with ``__len__`` and ``__getitem__``: a list, an
array, a dataframe, a torch dataset. The items it yields are opaque to FLYPSE.

The process splits the dataset across the entities holding data, and each entity
binds its own portion when it is deployed.

There are two ways to choose how the data is split:

.. grid:: 2

   .. grid-item::

      .. card:: Choose a split
         :link: choose-a-split
         :link-type: ref

         Use a ready-made dataset for the common kinds of heterogeneity.

   .. grid-item::

      .. card:: Write your own split
         :link: write-your-own-split
         :link-type: ref

         Subclass ``FederatedDataset`` and assign the indices yourself.

.. _choose-a-split:

Choose a split
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Dataset
     - The entities differ in
   * - :class:`~flypse.data.uniform.UniformDataset`
     - Nothing but what the order of the source gives them
   * - :class:`~flypse.data.dirichlet.DirichletDataset`
     - The classes they hold
   * - :class:`~flypse.data.quantity.QuantitySkewDataset`
     - How much data they hold, not its distribution

Uniform
~~~~~~~

:class:`~flypse.data.uniform.UniformDataset` assigns contiguous blocks of equal
size, up to one item: 10 items across 3 entities become blocks of 4, 3 and 3.

.. code-block:: python

   from flypse.data import UniformDataset

   data = UniformDataset(source)

.. note::

   The blocks are contiguous and not shuffled. On a source sorted by class or by
   feature, this split is non-IID. Shuffle the source first if that is not what you
   want.

Label skew
~~~~~~~~~~

:class:`~flypse.data.dirichlet.DirichletDataset` draws, for each class, the
proportions of its items assigned to the entities from a Dirichlet distribution of
concentration ``alpha``. It needs a function returning the class of an item:

.. code-block:: python

   from flypse.data import DirichletDataset


   def label(item):
       return item[1]


   data = DirichletDataset(source, class_fn=label, alpha=0.5, seed=0)

``alpha`` is the knob of the heterogeneity. On 1000 items of 10 balanced classes
across 5 entities, the largest class of an entity holds, on average:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - ``alpha``
     - Share of the largest class
     - Smallest entity
   * - 0.1
     - 47%
     - 65 items
   * - 1
     - 22%
     - 157 items
   * - 100
     - 11%
     - 190 items

.. warning::

   With a very low ``alpha``, an entity can end up with a single class, or with no
   data at all. Guard the code that assumes otherwise, such as an average over
   ``self.data``.

Quantity skew
~~~~~~~~~~~~~

:class:`~flypse.data.quantity.QuantitySkewDataset` draws the size of each entity
from a Dirichlet distribution, and assigns the items at random. Every entity gets
at least ``min_size`` items:

.. code-block:: python

   from flypse.data import QuantitySkewDataset

   data = QuantitySkewDataset(source, alpha=0.5, min_size=10, seed=0)

With these values, 1000 items are split into blocks of 146, 154, 11, 639 and 50.

.. _write-your-own-split:

Write your own split
--------------------

Splitting behaviour lives in the subclasses of
:class:`~flypse.data.dataset.FederatedDataset`: implement
:meth:`~flypse.data.dataset.FederatedDataset.split`, returning the indices of the
source assigned to each entity:

.. code-block:: python

   from flypse.data import FederatedDataset


   class ByOwner(FederatedDataset):
       """Assigns each item to the entity named in the item."""

       def split(self, entity_ids):
           splits = {entity_id: [] for entity_id in entity_ids}
           for index in range(len(self.source)):
               splits[self.source[index]["owner"]].append(index)
           return splits

The entities are given at split time, not at construction, so the same dataset
works for any federation.

Views
-----

A :class:`~flypse.data.view.DataView` is what an entity gets in ``self.data``: it
delegates to the source, so the ``i``-th item of the view is the source item at the
``i``-th index of the entity. It can be indexed, iterated and sliced, and a slice is
another view, with nothing copied.

.. important::

   Every device receives a copy of the whole dataset, source included, and each
   entity then keeps only its own view. Keep this in mind with large in-memory
   sources: a source that loads its items lazily avoids the cost.

.. seealso::

   - :doc:`splitters`
