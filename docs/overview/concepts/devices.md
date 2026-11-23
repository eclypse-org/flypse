# Devices

A [Device][flypse.system.device.Device] is an emulated machine, running in a process of its
own and holding the dataset of the federation. Every node of the
[infrastructure](system.md) becomes a device, and each entity deployed on it binds its own
portion of the data. You never create one yourself.

An entity does not know its device beforehand, as the placement decides it. It asks for what
it needs while it runs.

## Read the resources

[get_device][flypse.federation.entity.Entity.get_device] returns the
[resources](system.md#system-resources) of the device of an entity, by name:

```python
device = await self.get_device()
device["cpu"], device["ram"], device["availability"]
```

They are asked of the process at every call, so they follow its
[update policies](../advanced/heterogeneity.md): nothing is stored on the entity, and
nothing gets out of date.

## Emulate the speed of a device

Every device runs on your machine, so a `step` takes the same time on a slow device as on a
fast one. [compute][flypse.federation.entity.Entity.compute] adds the time the device would
take: `work / cpu` seconds, for a work given in seconds of a single core.

```python
train(self.model.model, self.data)
await self.compute(len(self.data) * 1e-3)
```

The same work takes four times as long on 2 cores as on 8. Set the cores of a device on the
infrastructure:

```python
from flypse.system import SystemConfig, get_mirror, get_mirror_strategy

infrastructure = get_mirror(federation, seed=0)
infrastructure.nodes["client_0"]["cpu"] = 2
infrastructure.nodes["client_1"]["cpu"] = 8

system = SystemConfig(infrastructure, get_mirror_strategy(federation))
```

!!! info "Important"

    The devices of the built-in systems start with maximal resources, so `compute` returns
    almost at once until you set their `cpu`.

## Share a device

Several entities can run on one device, as on a singleton. They all train at once, unless
the device admits only a few of them at a time. `max_concurrency` is the number of its
slots:

```python
system = SystemConfig("singleton", max_concurrency=2)
```

An entity holds a slot within a block on its
[concurrency][flypse.federation.entity.Entity.concurrency]. The others wait for theirs, in
the order they asked:

```python
async def step(self):
    message = await self.receive()
    self.model.load_version(message.record)
    async with self.concurrency:
        train(self.model.model, self.data)
        await self.compute(len(self.data) * 1e-3)
    record = self.model.update_version(self.model.get_parameters(), "local")
    await self.send(message.sender, record, n_samples=len(self.data))
```

Only the block is limited: code outside it is not, and a federation with
no such block ignores the limit.

!!! warning

    An entity cannot `receive` within the block, which raises `RuntimeError`: waiting for
    another entity while keeping a slot could wait forever. Asking for a second slot raises
    `RuntimeError` too. Sending is allowed.

An entity that is undeployed gives its slot back, even from the middle of a block.

!!! note "See also"

    - [System and placement](system.md)
    - [Heterogeneity](../advanced/heterogeneity.md)
