========
Examples
========

.. toctree::
   :maxdepth: 2
   :hidden:

   fedavg
   hierarchical

The examples section complements the getting-started guides with runnable
federations you can inspect and adapt.

The code for the documented examples is available in the
:icon:`fa-brands fa-github`
`examples <https://github.com/eclypse-org/flypse/tree/main/examples/>`_
subdirectory of the repository. Run them from the repository root, after
``make setup``.

.. grid:: 1

   .. grid-item::

      .. card:: :octicon:`broadcast;1em;info` **Federated averaging**
         :link-type: doc
         :link: fedavg

         A server and five clients learn a linear model from non-IID data, the
         starting point of the other examples.

   .. grid-item::

      .. card:: :octicon:`git-merge;1em;info` **Hierarchical federated averaging**
         :link-type: doc
         :link: hierarchical

         Group aggregators between the clients and the root, with three version
         levels recording where a model comes from.
