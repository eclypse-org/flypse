Models and versions
===================

FLYPSE is model-agnostic: how a model computes and trains is left to its
framework. A :class:`~flypse.federation.model.FederatedModel` exposes the
parameters of any model as numpy arrays, so combining them is plain numpy, and
records every version it goes through.

Wrap a model
------------

Subclasses implement two hooks: ``_read_parameters`` yields the parameters to
exchange, and ``_write_parameter`` writes one of them back into the wrapped model,
which stays available as ``model`` for local training and inference.

.. tab-set::

   .. tab-item:: Plain Python
      :sync: plain

      .. code-block:: python

         import numpy as np

         from flypse.federation import FederatedModel


         class MeanModel(FederatedModel):
             def _read_parameters(self):
                 yield "mean", self.model["mean"]

             def _write_parameter(self, name, value):
                 self.model[name] = value


         model = MeanModel({"mean": np.zeros(1)})

   .. tab-item:: PyTorch
      :sync: torch

      .. code-block:: python

         import torch

         from flypse.federation import FederatedModel


         class TorchModel(FederatedModel):
             def _read_parameters(self):
                 for name, value in self.model.state_dict().items():
                     if value.is_floating_point():
                         yield name, value.detach().cpu().numpy()

             def _write_parameter(self, name, value):
                 self.model.state_dict()[name].copy_(torch.from_numpy(value))


         model = TorchModel(torch.nn.Linear(4, 2))

The entries yielded by the hook are read once and fixed: their names, order and
shapes must stay the same. Only floating-point entries are accepted, so integer
counters must be skipped, and a model with no parameters to expose fails at
initialization. Choosing what to yield is a decision of the algorithm: yielding
BatchNorm statistics or not is the only difference between FedAvg and FedBN.

.. important::

   The hook is first called by ``__init__``, so any attribute it uses must be set
   before calling ``super().__init__``.

Read and write parameters
-------------------------

:meth:`~flypse.federation.model.FederatedModel.get_parameters` reads a fresh copy
of the parameters, as one flat vector or as arrays keyed by name. Local training
is always visible, so a pseudogradient is a subtraction:

.. code-block:: python

   before = model.get_parameters()
   train(model.model, data)
   delta = model.get_parameters() - before

   model.get_parameters(flattened=False)  # {"mean": array([...], dtype=float32)}

There are two ways to write parameters, depending on who generated them:

- :meth:`~flypse.federation.model.FederatedModel.update_version` is for the entity
  generating a new model, at a given level,
- :meth:`~flypse.federation.model.FederatedModel.load_version` is for the entity
  receiving one, which adopts its version and timestamp as they are.

Both return the recorded :class:`~flypse.federation.model.ModelVersion`, ready to
be sent.

Version levels and lineage
--------------------------

A version has one counter per level, from the coarsest to the finest,
``("global", "local")`` by default. ``update_version`` increments the counter of
its level and resets the finer ones, so a version tells where a model comes from.
A three-tier federation only needs one more level:

.. code-block:: python

   model = MeanModel({"mean": np.zeros(1)}, version_levels=("global", "group", "local"))

.. list-table::
   :widths: 50 20
   :header-rows: 1

   * - Event
     - Version
   * - A client loads the model of its group
     - ``(1, 3, 0)``
   * - It trains locally, twice: ``update_version(params, "local")``
     - ``(1, 3, 2)``
   * - Its group aggregates: ``update_version(params, "group")``
     - ``(1, 4, 0)``
   * - The server aggregates: ``update_version(params, "global")``
     - ``(2, 0, 0)``

The history
-----------

``versions`` holds the records of a model, oldest first, and ``version`` the
current one. Each record is a read-only copy, so neither local training nor the
caller can alter it.

.. warning::

   The history is unbounded by default, and every record holds all the parameters.
   For long runs or large models, pass ``max_versions`` to drop the oldest records:
   ``MeanModel(model, max_versions=10)``.

Reading any version
-------------------

``get_parameters`` also reads a past version, by its counters or by its record:

.. code-block:: python

   model.get_parameters(version=(1, 3, 0))  # from the history of this model
   model.get_parameters(version=message.record)  # from the record, as it is

.. important::

   A version is not unique across entities: two clients can both generate
   ``(3, 1)``. Counters look up the history of the model reading them, so to read
   the record of a peer, pass the record.

.. seealso::

   - :doc:`messaging`
