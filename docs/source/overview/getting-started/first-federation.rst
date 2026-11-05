==========================
Run your first federation
==========================

This page shows the shortest path to a first federated run.

We will:

#. wrap a model so that its parameters can be exchanged,
#. implement a client and a server,
#. build the federation and split the data across it,
#. run the process and read what the server reported.

The federation estimates the mean of a dataset split across three clients. Each
client moves the global estimate towards the mean of its own data, and the
server averages the estimates of the clients, weighted by the size of their data.

The following example uses:

- :class:`~flypse.federation.model.FederatedModel`, to exchange the parameters
  of a model,
- :class:`~flypse.federation.entity.Entity`, to implement the behaviour of the
  client and the server,
- :func:`~flypse.federation.topology.get_star`, to connect them,
- :class:`~flypse.data.uniform.UniformDataset`, to split the data,
- :class:`~flypse.process.FederatedProcess`, to emulate the federation.

Define the model
----------------

A :class:`~flypse.federation.model.FederatedModel` wraps a model of any
framework and exposes its parameters as numpy arrays. Subclasses implement two
hooks: ``_read_parameters`` yields the parameters to exchange, and
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

Implement the entities
----------------------

An :class:`~flypse.federation.entity.Entity` is an ECLYPSE service: its behaviour
is the ``step`` method, which runs repeatedly once the entity is deployed.
``self.data`` holds the portion of the data assigned to the entity.

.. code-block:: python

   from flypse.federation import Entity


   class Client(Entity):
       """Moves the global estimate towards the mean of its own data."""

       def __init__(self, entity_id, server_id):
           super().__init__(entity_id)
           self.server_id = server_id
           self.model = MeanModel({"mean": np.zeros(1)})

       async def step(self):
           local_mean = np.mean(list(self.data))
           estimate = self.model.get_parameters()
           record = self.model.update_version(
               estimate + 0.5 * (local_mean - estimate), "local"
           )
           await self.send(self.server_id, record, n_samples=len(self.data))
           message = await self.receive()
           self.model.load_version(message.record)


   class Server(Entity):
       """Averages the estimates of its clients, weighted by their data size."""

       def __init__(self, entity_id):
           super().__init__(entity_id, with_data=False)
           self.model = MeanModel({"mean": np.zeros(1)})

       async def step(self):
           messages = [await self.receive() for _ in await self.get_neighbors()]
           n_total = sum(m.data["n_samples"] for m in messages)
           average = (
               sum(m.record.parameters * m.data["n_samples"] for m in messages)
               / n_total
           )
           record = self.model.update_version(average, "global")
           self.report(round=record.version[0], estimate=average[0])
           await self.bcast(record)

``update_version`` records a new version of a model, which ``send`` and ``bcast``
deliver to other entities, and ``load_version`` adopts a received one.

The synchronisation of the federation emerges from the entities themselves: each
client waits for the global estimate after sending its own, and the server waits
for an estimate from each of its neighbours.

Build the federation and its data
---------------------------------

A :class:`~flypse.federation.federation.Federation` is an ECLYPSE application
whose nodes are the entities and whose edges are their connections.
:func:`~flypse.federation.topology.get_star` connects a server to its clients,
and a :class:`~flypse.data.uniform.UniformDataset` splits a data source (anything
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

The server is created with ``with_data=False``, so the 100 items are split among
the three clients.

Run the process
---------------

A :class:`~flypse.process.FederatedProcess` emulates the federation: it splits
the data, gives it to the devices of an infrastructure, and registers the
federation. Any other argument goes to the ECLYPSE simulation configuration:

.. code-block:: python

   from flypse.process import FederatedProcess

   process = FederatedProcess(federation, data, step_every_ms=100.0, seed=0)
   process.run(seconds=5)

   print(process.reported[-1])

.. code-block:: text

   {'round': 511, 'estimate': 49.4999885559082, 'node_id': 'server', 'entity_id': 'server', 'timestamp': '2026-11-05T23:27:15.781024', 'n_report': 511}

The clients hold the items from 0 to 99, so the estimate of the server converges
to their mean, 49.5. The number of rounds depends on the speed of your machine.

.. important::

   ``step_every_ms`` is required: it is how often ECLYPSE enacts the placement,
   and without it the entities are never deployed.

.. seealso::

   - :doc:`inspect-results`
   - :doc:`infrastructure`
