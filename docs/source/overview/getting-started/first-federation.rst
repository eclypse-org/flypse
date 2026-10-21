==========================
Run your first federation
==========================

This page builds a small client-server federation that estimates the mean of
a dataset split across three clients. Each client moves the global estimate
towards the mean of its own data, and the server averages the estimates of
the clients, weighted by the size of their data.

The model
---------

A :class:`~flypse.federation.model.FederatedModel` wraps a model of any
framework and exposes its parameters as numpy arrays. Subclasses implement
two hooks: ``_read_parameters`` yields the parameters to exchange, and
``_write_parameter`` writes one of them back into the wrapped model.

Here, the wrapped model is a dictionary holding a single estimate:

.. code-block:: python

   import numpy as np

   from flypse.federation import FederatedModel


   class MeanModel(FederatedModel):
       """A model whose only parameter is an estimate of the mean."""

       def _read_parameters(self):
           yield "mean", self.model["mean"]

       def _write_parameter(self, name, value):
           self.model[name] = value

Every write is recorded as a new version, a
:class:`~flypse.federation.model.ModelVersion` holding the version counters,
a timestamp, and the parameters as a flat vector. ``update_version`` is used
by the entity generating a new model, at a given level (``"local"`` or
``"global"`` by default), while ``load_version`` is used by the entity
receiving it.

The entities
------------

An :class:`~flypse.federation.entity.Entity` is an ECLYPSE service: its
behaviour is the ``step`` method, executed repeatedly once the entity is
deployed. When deployed, the entity binds its portion of the federation data,
available as ``self.data``. Entities exchange versions of their models with
``send``, ``bcast`` and ``receive``, attaching any other data as keyword
arguments, and find their neighbours with ``get_neighbors``. A message sent
without a model is a control message, carrying only its data (e.g.,
``await self.bcast(stop=True)``), whose ``record`` is None. ``receive``
waits for the next message, for at most ``timeout`` seconds when given. The
value returned by ``step`` is stored in the report when ``store_step`` is
True.

.. code-block:: python

   from flypse.federation import Entity


   class Client(Entity):
       """Moves the global estimate towards the mean of its own data."""

       def __init__(self, entity_id, server_id):
           super().__init__(entity_id, store_step=True)
           self.server_id = server_id
           self.model = MeanModel({"mean": np.zeros(1)})

       async def step(self):
           local_mean = np.mean([self.data[i] for i in range(len(self.data))])
           estimate = self.model.get_parameters()
           self.model.update_version(estimate + 0.5 * (local_mean - estimate), "local")
           await self.send(
               self.server_id, self.model.versions[-1], n_samples=len(self.data)
           )
           message = await self.receive()
           self.model.load_version(message.record)


   class Server(Entity):
       """Averages the estimates of its clients, weighted by their data size."""

       def __init__(self, entity_id):
           super().__init__(entity_id, store_step=True, with_data=False)
           self.model = MeanModel({"mean": np.zeros(1)})

       async def step(self):
           messages = [await self.receive() for _ in await self.get_neighbors()]
           n_total = sum(m.data["n_samples"] for m in messages)
           average = (
               sum(m.record.parameters * m.data["n_samples"] for m in messages)
               / n_total
           )
           self.model.update_version(average, "global")
           await self.bcast(self.model.versions[-1])
           return float(average[0])

The synchronization of the federation emerges from the entities themselves:
each client waits for the global estimate after sending its own, and the
server waits for an estimate from each of its neighbours.

The federation and its data
---------------------------

A :class:`~flypse.federation.federation.Federation` is an ECLYPSE application
whose nodes are the entities and whose edges are their connections. It can be
built entity by entity with ``add_entity`` and ``add_connection``, or with a
topology builder: :func:`~flypse.federation.topology.get_star`,
:func:`~flypse.federation.topology.get_tree` (e.g., for hierarchical
aggregation) or :func:`~flypse.federation.topology.get_ring` (e.g., for
decentralized peers). A :class:`~flypse.data.uniform.UniformDataset` splits a data source (anything
with ``__len__`` and ``__getitem__``) into contiguous blocks, one per entity:

.. code-block:: python

   from flypse.data import UniformDataset
   from flypse.federation import get_star

   federation = get_star(
       Server("server"),
       [Client(f"client_{i}", "server") for i in range(3)],
       federation_id="MeanEstimation",
   )

   data = UniformDataset(list(range(100)))

The data is split across the entities holding data: the server is created with
``with_data=False``, so the 100 items are split among the three clients.

The data of an entity can be split further into named parts with a
:class:`~flypse.data.splitter.ViewSplitter`. The default
:class:`~flypse.data.splitter.FractionSplitter` sizes the parts by fractions,
optionally stratifying by class or following the order of the items (e.g., by
time, so that no part precedes the earlier ones). Each part is a view of the
entity's data, and can be split again:

.. code-block:: python

   from flypse.data import FractionSplitter

   splitter = FractionSplitter({"train": 0.8, "test": 0.2}, seed=0)
   parts = splitter.apply(self.data)  # e.g., in the on_deploy of an entity
   parts["train"][0], len(parts["test"])

Subclasses of :class:`~flypse.data.splitter.ViewSplitter` define their own
splits by implementing ``split``.

Reproducibility
---------------

A process has a single seed, the ``seed`` argument of the simulation
configuration. Every component given no seed of its own falls back to it: the
split of the data, the ``"singleton"`` and ``"mirror"`` infrastructures, the
federation, and the splitters. For its own randomness, an entity draws from
its ``rng``, seeded from the seed of the process and the identifier of the
entity, so that entities draw different but reproducible sequences; it can
also drive a splitter, e.g., ``FractionSplitter(parts, seed=self.rng)``.

Run the process
---------------

A :class:`~flypse.process.FederatedProcess` emulates the federation: it splits
the data, gives it to the devices of the infrastructure described by a
:class:`~flypse.system.config.SystemConfig`, and registers the federation. Any
other argument is passed to the ECLYPSE simulation configuration.

.. code-block:: python

   from flypse.process import FederatedProcess
   from flypse.system import SystemConfig

   process = FederatedProcess(
       federation,
       data,
       SystemConfig(),
       step_every_ms=100.0,
       include_default_metrics=True,
   )
   process.run(seconds=5)

With the default ``"mirror"`` infrastructure, each entity runs on its own
device. The clients hold the items from 0 to 99, so the estimate of the server
converges to their mean, 49.5.

.. note::

   Each device runs in its own Ray worker. When the script is launched with
   ``uv run``, Ray starts its workers through ``uv`` as well, setting up a new
   environment for each of them. Setting the ``RAY_ENABLE_UV_RUN_RUNTIME_ENV``
   environment variable to ``0`` before Ray is imported makes the workers
   reuse the current environment instead.

.. seealso::

   - :doc:`infrastructure`
   - :doc:`inspect-results`
