# Assignment

Assignment decides **what each selected entity is asked to do**: the instructions sent
along with the model. It is apart from [selection](selection.md) because the two vary
independently: how long a client trains has nothing to do with how it was chosen.

The module [flypse.strategies.assignment][] has no member yet. An instruction is data
attached to a message, and the patterns below need nothing more.

## A local budget

Keyword arguments of `send` reach the client in `message.data`, so a fixed budget is one
argument:

```python
await self.send(selected, record, epochs=5)
```

```python
async def step(self):
    message = await self.receive()
    self.model.load_version(message.record)
    for _ in range(message.data["epochs"]):
        train(self.model.model, self.data)
    ...
```

To give each client a budget of its own, send one message per client, all at once:

```python
import asyncio

await asyncio.gather(
    *(self.send(client, record, epochs=epochs[client]) for client in selected)
)
```

## A deadline

A deadline is a budget in time. The client can check that its device is fast enough before
it starts, and answer with no model otherwise:

```python
async def step(self):
    message = await self.receive()
    work = len(self.data) * 1e-3
    device = await self.get_device()
    if work / device["cpu"] > message.data["deadline"]:
        await self.send(message.sender, declined=True)
        return
    ...
```

The server then leaves out the answers with no model:

```python
messages = [m for m in messages if m.record is not None]
```

The server should still keep its own clock, with the `timeout` of `receive`: see the
[stragglers example](../examples/stragglers.md).

## A part of the model

A weak device can train some entries only. The entries carry
[labels](../concepts/models.md#labels-and-masks), so a message names them rather than
carrying a mask:

```python
await self.send("weak_client", record, parts=["body"])
```

```python
async def step(self):
    message = await self.receive()
    parts = message.data["parts"]
    self.model.load_version(message.record)
    train(self.model.model, self.data, parts)
    record = self.model.update_version(self.model.get_parameters(), "local")
    await self.send(message.sender, record, parts=parts)
```

The whole model still travels and is loaded, as the entries left alone must follow the
server too, and your training has to leave them alone. The server averages each coordinate over the clients that
trained it, with [masked_mean][flypse.strategies.aggregation.masked_mean]:

```python
masks = [self.model.mask(*m.data["parts"]) for m in messages]
new = aggregation.masked_mean(vectors, masks, previous=self.model.get_parameters())
```

!!! note

    [HeteroFL](https://arxiv.org/abs/2010.01264) slices the channels *inside* each entry,
    which labels cannot name. Compute those masks yourself: `masked_mean` takes any boolean
    vectors.

!!! note "See also"

    - [Selection](selection.md)
    - [Aggregation](aggregation.md)
    - [Devices](../concepts/devices.md)
