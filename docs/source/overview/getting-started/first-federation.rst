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
whose nodes are the entities and whose edges are their connections. A
:class:`~flypse.data.uniform.UniformDataset` splits a data source (anything
with ``__len__`` and ``__getitem__``) into contiguous blocks, one per entity:

.. code-block:: python

   from flypse.data import UniformDataset
   from flypse.federation import Federation

   federation = Federation("MeanEstimation")
   federation.add_entity(Server("server"))
   for i in range(3):
       federation.add_entity(Client(f"client_{i}", "server"))
       federation.add_connection("server", f"client_{i}")

   data = UniformDataset(list(range(100)))

The data is split across the entities holding data: the server is created with
``with_data=False``, so the 100 items are split among the three clients.

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
