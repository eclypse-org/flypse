# Optimisation

Optimisation decides **how the result changes the model of the entity**: adopt it, move
part of the way towards it, or take an optimiser step along it.

There are three ways to use a result:

<div class="grid cards" markdown>

-   **[Adopt it](#adopt-the-result)**

    ---

    The result becomes the new model, as in federated averaging.

-   **[Move towards it](#move-towards-the-result)**

    ---

    The new model is a mix of the two, as in asynchronous federated learning.

-   **[Step along it](#step-along-the-update)**

    ---

    The difference from the model is an update, as in adaptive federated optimisation.

</div>

## Adopt the result

Adopting a result is
[update_version][flypse.federation.model.FederatedModel.update_version] itself, with no
strategy:

```python
await self.bcast(self.model.update_version(average, "global"))
```

Adopting only a part of a received model, as
[FedPer](https://arxiv.org/abs/1912.00818) and [FedBN](https://arxiv.org/abs/2102.07623)
do, is [load_version][flypse.federation.model.FederatedModel.load_version] with a
[mask](../concepts/models.md#labels-and-masks).

## Move towards the result

[interpolate][flypse.strategies.optimization.interpolate] returns
`(1 - alpha) own + alpha other`. In [FedAsync](https://arxiv.org/abs/1903.03934), the model
of a client is mixed in as soon as it arrives, with an `alpha` that falls with its
staleness:

```python
from flypse.strategies import optimization, weighting

message = await self.receive()
lag = self.model.lag(message.record, "global")
alpha = 0.6 * weighting.polynomial_staleness([lag])[0]
new = optimization.interpolate(
    self.model.get_parameters(), message.record.parameters, alpha
)
await self.send(message.sender, self.model.update_version(new, "global"))
```

## Step along the update

The update is the result minus the model of the entity, such as the mean of the
differences between the models of the clients and the one they started from. Every
optimiser steps *along* it.

| Optimiser | Step | As in |
| --- | --- | --- |
| [SGD][flypse.strategies.optimization.SGD] | `v ← β v + Δ`, `θ ← θ + η v` | [FedAvgM](https://arxiv.org/abs/1909.06335), [FedBuff](https://arxiv.org/abs/2106.06639); [DiLoCo](https://arxiv.org/abs/2311.08105) with `nesterov=True` |
| [Adam][flypse.strategies.optimization.Adam] | Scaled by the moments of the update | [FedAdam](https://arxiv.org/abs/2003.00295) |
| [Yogi][flypse.strategies.optimization.Yogi] | The same, with a second moment that shrinks slowly, so the steps do not suddenly grow | [FedYogi](https://arxiv.org/abs/2003.00295) |
| [Adagrad][flypse.strategies.optimization.Adagrad] | The same, with a second moment that only grows | [FedAdagrad](https://arxiv.org/abs/2003.00295) |
| [fedexp][flypse.strategies.optimization.fedexp] | Along the mean update, longer when the updates disagree | [FedExP](https://arxiv.org/abs/2301.09604) |

An optimiser keeps its state between steps, so it lives on the entity:

```python
class Server(Entity):
    def __init__(self, entity_id):
        super().__init__(entity_id, with_data=False)
        self.optimizer = optimization.Adam(lr=0.1)
        ...

    async def step(self):
        ...
        own = self.model.get_parameters()
        deltas = [m.record.parameters - own for m in messages]
        delta = aggregation.mean(deltas, weights)
        self.model.update_version(self.optimizer.step(own, delta), "global")
```

[fedexp][flypse.strategies.optimization.fedexp] has no state, and takes the updates rather
than their mean, as their spread sets the length of the step:

```python
self.model.update_version(optimization.fedexp(own, deltas), "global")
```

!!! info "Important"

    The optimisers have the form of the paper that introduced them for federated learning:
    the second moment starts from `τ²`, and there is no bias correction. Their step differs
    from that of the Adam of a deep learning framework, especially in the first rounds.

### Leave coordinates out

Some entries are not meant to be optimised, such as the statistics of batch normalisation:
average them, and step on the others. Use the same mask at every step, as the state of the
optimiser has one value per coordinate it sees:

```python
keep = ~self.model.mask("statistics")
new = own + delta
new[keep] = self.optimizer.step(own[keep], delta[keep])
self.model.update_version(new, "global")
```

!!! note "See also"

    - [Aggregation](aggregation.md)
    - [Models](../concepts/models.md)
    - [Compose an algorithm](algorithms.md)
