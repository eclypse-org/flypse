# Aggregation

Aggregation decides **how the contributions combine into one vector**, with a rule such as
the weighted mean or the median. It is apart from [weighting](weighting.md) because the
same weights can feed different rules.

| Rule | Result | Weights |
| --- | --- | --- |
| [mean][flypse.strategies.aggregation.mean] | The weighted mean | Yes |
| [geometric_median][flypse.strategies.aggregation.geometric_median] | The vector minimising the sum of distances ([Pillutla et al., 2022](https://arxiv.org/abs/1912.13445)) | Yes |
| [masked_mean][flypse.strategies.aggregation.masked_mean] | Each coordinate, over the contributions covering it ([Diao et al., 2021](https://arxiv.org/abs/2010.01264)) | Yes |
| [median][flypse.strategies.aggregation.median] | The median of each coordinate ([Yin et al., 2018](https://arxiv.org/abs/1803.01498)) | No |
| [trimmed_mean][flypse.strategies.aggregation.trimmed_mean] | The mean of each coordinate, without its extremes ([Yin et al., 2018](https://arxiv.org/abs/1803.01498)) | No |
| [meamed][flypse.strategies.aggregation.meamed] | The mean of each coordinate, around its median ([Xie et al., 2018](https://arxiv.org/abs/1802.10116)) | No |
| [centered_clipping][flypse.strategies.aggregation.centered_clipping] | A step from the previous aggregate, clipped ([Karimireddy et al., 2021](https://arxiv.org/abs/2012.10333)) | No |
| [sign_majority][flypse.strategies.aggregation.sign_majority] | The majority sign of each coordinate ([Bernstein et al., 2019](https://arxiv.org/abs/1810.05291)) | No |

## Average

```python
from flypse.strategies import aggregation, weighting

weights = weighting.normalise(weighting.by_samples(messages))
average = aggregation.mean([m.record.parameters for m in messages], weights)
```

Any vector can be averaged: the models themselves, or their updates, for an
[optimiser](optimization.md) to step along.

## Resist bad contributions

The robust rules bound how far a minority of contributions can move the result. Most of
them ignore weights on purpose: a weight would let one contribution outweigh the others,
which is what they guard against.

```python
delta = aggregation.median(deltas)
delta = aggregation.trimmed_mean(deltas, beta=0.2)
delta = aggregation.geometric_median(deltas, weights)
```

!!! warning

    [trimmed_mean][flypse.strategies.aggregation.trimmed_mean] drops `⌊βn⌋` values at each end
    of every coordinate. With fewer than `1 / β` contributions that is zero, and the rule is
    the plain mean.

[centered_clipping][flypse.strategies.aggregation.centered_clipping] starts from the
previous aggregate, so the entity keeps it. No contribution moves the result by more than
`τ / n` at each step:

```python
class Server(Entity):
    def __init__(self, entity_id):
        super().__init__(entity_id, with_data=False)
        self.reference = 0.0
        ...

    async def step(self):
        ...
        self.reference = aggregation.centered_clipping(deltas, self.reference, tau=10.0)
```

To drop contributions before they are combined, see [Screening](screening.md).

## Combine partial models

When contributions cover different parts of the model, each coordinate is averaged over
those that cover it. One that nobody covers keeps its previous value:

```python
masks = [self.model.mask(*m.data["parts"]) for m in messages]
new = aggregation.masked_mean(vectors, masks, previous=self.model.get_parameters())
```

See [Assignment](assignment.md#a-part-of-the-model) for where the parts come from.

## Vote on signs

[sign_majority][flypse.strategies.aggregation.sign_majority] returns a vector in
`{-1, 0, 1}`, and the entity steps along it with a learning rate of its own:

```python
from flypse.strategies import optimization

optimizer = optimization.SGD(lr=1e-3)
new = optimizer.step(own, aggregation.sign_majority(deltas))
```

!!! note "See also"

    - [Screening](screening.md)
    - [Optimisation](optimization.md)
