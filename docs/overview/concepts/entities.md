# Entities

An [Entity][flypse.federation.entity.Entity] is the basic unit of a federation: a
participant deployed on a device, bound to its portion of the data.

FLYPSE has no client or server class. What an entity is follows from what its
`step` does, and the protocol of a federation emerges from the steps of its
entities: there are no rounds or barriers in the library.

## Implement `step`

`step` runs in a loop for as long as the entity is deployed. This entity trains
on its data, sends the result to a server and waits for the next global model:

```python
from flypse.federation import Entity


class Client(Entity):
    def __init__(self, entity_id, server_id):
        super().__init__(entity_id)
        self.server_id = server_id
        self.model = MeanModel({"mean": np.zeros(1)})

    async def step(self):
        train(self.model.model, self.data)
        record = self.model.update_version(self.model.get_parameters(), "local")
        await self.send(self.server_id, record, n_samples=len(self.data))
        message = await self.receive()
        self.model.load_version(message.record)
```

!!! info "Important"

    An entity runs on an asyncio loop, and messages reach it on that loop: while
    its `step` runs without awaiting, nothing is delivered to it.

## Lifecycle

An entity lives in two places, and each method runs in only one of them:

| Method | Runs on | What happens |
| --- | --- | --- |
| `__init__` | your script | The entity is created. `data` is None, and the generators are seeded once the process is created. |
| `on_deploy` | the device | `data` is bound. |
| `step` | the device | Runs in a loop until the entity is stopped. |
| `on_undeploy` | the device | `data` and `splits` are released. |

Anything that depends on the data or on the seed belongs in `on_deploy`, after
calling the method of the parent class:

```python
from flypse.data import FractionSplitter


class Client(Entity):
    def on_deploy(self):
        super().on_deploy()
        splitter = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng)
        self.splits = splitter.apply(self.data)
        self.data = self.splits["train"]
```

An entity can be deployed more than once: when the placement of a federation is reset, its
entities are undeployed and deployed again, and they run `on_deploy` again. The attributes
of the entity, such as its model, are kept, so `on_deploy` should not reset what a new
deployment must not lose. See [When the placement is reset](system.md#placement-reset) for
when that happens.

## Local data

`self.data` is the [DataView][flypse.data.view.DataView] of the entity: it can be
indexed, iterated and sliced, so `self.data[:32]` is a view of the first 32
items, with nothing copied. An entity may reassign it, for instance to one of its
parts.

An entity holding no data, such as an aggregator, is created with
`with_data=False`: it is left out of the [split](data.md), and its `data`
stays None.

## Randomness

`self.rng` is a `random.Random`, and `self.np_rng` a numpy generator. Both are seeded from
the seed of the process and the identifier of the entity, so that entities draw different
but reproducible sequences. A redeployed entity goes on with its sequences, so a reset of
the federation does not replay its draws:

```python
positions = self.rng.sample(range(len(self.data)), k=32)
batch = [self.data[i] for i in positions]

noise = self.np_rng.normal(size=delta.shape)
```

## Ask about the federation

An entity keeps no copy of the federation. It asks the process, which may change it during
the run:

| Method | Returns |
| --- | --- |
| [get_neighbors][flypse.federation.entity.Entity.get_neighbors] | The entities it sends to, or, with `"in"`, those sending to it |
| [get_connections][flypse.federation.entity.Entity.get_connections] | The same, with the attributes of each connection and the degree of the neighbour |
| [get_device][flypse.federation.entity.Entity.get_device] | The resources of [its device](devices.md) |

The answers are those of the moment of the call, so call again rather than keep them.

## Roles are behaviour

Training, aggregating, or both: the same class covers all of them. An aggregator is
an entity with `with_data=False` that combines what it receives, usually with the
[strategies](../strategies/index.md). A group aggregator receives from its
[children](federation.md#parents-and-children) and sends to its parent. A peer trains and
aggregates in the same `step`.

## When a step fails

!!! warning

    An exception raised by `step` ends the loop of the entity for good. It is logged, the
    other entities carry on, and those waiting for its messages wait forever, unless they
    use a [timeout](messaging.md#timeouts). This includes a `TimeoutError` raised by
    `receive`: catch it where you expect it.

!!! note "See also"

    - [Messaging](messaging.md)
    - [Models](models.md)
    - [Devices](devices.md)
    - [Strategies](../strategies/index.md)
    - [Reporting](../advanced/reporting.md)
