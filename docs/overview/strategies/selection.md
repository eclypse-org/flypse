# Selection

Selection decides **whom to engage**: which of the candidates are contacted at all. An
entity that is not selected receives no message, keeps no state about it and waits in its
`receive`, so there is nothing to activate or to switch off.

The members of [flypse.strategies.selection][] return two things: the selected entities,
and the probability of *every* candidate to be selected.

| Strategy | Selects | Probability of a candidate |
| --- | --- | --- |
| [uniform][flypse.strategies.selection.uniform] | `k` candidates, without replacement | `k / n` |
| [poisson][flypse.strategies.selection.poisson] | Each candidate independently, with probability `q` | `q` |
| [proportional][flypse.strategies.selection.proportional] | `k` draws with replacement, each in proportion to a size ([Li et al., 2020](https://arxiv.org/abs/1907.02189)) | `1 - (1 - n_i / n)^k` |
| [clustered][flypse.strategies.selection.clustered] | One candidate from each of `m` clusters of equal size ([Fraboni et al., 2021](https://arxiv.org/abs/2105.05883)) | Depends on its clusters |
| [round_robin][flypse.strategies.selection.round_robin] | `k` candidates at a time, in turn | 1 or 0 |

## Select some neighbours

The candidates usually are the neighbours of the entity, and the randomness its own:

```python
from flypse.strategies import selection

async def step(self):
    selected, _ = selection.uniform(await self.get_neighbors(), 10, self.rng)
    await self.send(selected, self.model.versions[-1])
    messages = [await self.receive(source=client) for client in selected]
    ...
```

The candidates are sorted first, so the selection depends on the seed and not on the order
they are given in.

With a selection, the server opens the round: a client waits for a model, trains it, and
answers whoever sent it.

```python
async def step(self):
    message = await self.receive()
    self.model.load_version(message.record)
    train(self.model.model, self.data)
    record = self.model.update_version(self.model.get_parameters(), "local")
    await self.send(message.sender, record, n_samples=len(self.data))
```

!!! note

    [poisson][flypse.strategies.selection.poisson] can select nobody. Guard the code that
    averages what it receives.

## Select in proportion to the data

[proportional][flypse.strategies.selection.proportional] and
[clustered][flypse.strategies.selection.clustered] take the size of each candidate, such
as the number of samples it last reported. They draw *with replacement*: a candidate can
appear twice, and the contributions are averaged with equal weights, one per draw.

```python
from collections import Counter

import numpy as np

drawn, _ = selection.clustered(self.sizes, 10, self.rng)
draws = Counter(drawn)

await self.send(sorted(draws), self.model.versions[-1])
messages = [await self.receive(source=client) for client in sorted(draws)]

vectors = [m.record.parameters for m in messages]
weights = np.array([draws[m.sender] for m in messages])
average = aggregation.mean(vectors, weights)
```

Both give every candidate the same expected number of draws. Those of
[clustered][flypse.strategies.selection.clustered] vary less, as each cluster is drawn from
once.

!!! warning

    With one weight per draw, do not weigh by the number of samples too: the sizes already
    drive the draws, and would count twice.

## Correct for the selection

A candidate selected more often than its relevance deserves biases the aggregate. The
probabilities undo it, through
[inverse_probability][flypse.strategies.weighting.inverse_probability]. Here every distinct
candidate counts once, whatever its number of draws, so its samples weigh it too:

```python
drawn, pi = selection.proportional(self.sizes, 10, self.rng)
selected = sorted(set(drawn))
...
weights = weighting.normalise(
    weighting.by_samples(messages),
    weighting.inverse_probability([pi[m.sender] for m in messages]),
)
```

With [uniform][flypse.strategies.selection.uniform] and
[poisson][flypse.strategies.selection.poisson] every probability is the same, and a normalised
average does not change. A [round_robin][flypse.strategies.selection.round_robin] cannot be
corrected within a turn, as the candidates it skips have probability 0. See
[Weighting](weighting.md#by-the-probability-to-be-selected) for the estimators this gives.

!!! note "See also"

    - [Assignment](assignment.md)
    - [Weighting](weighting.md)
    - [Messaging](../concepts/messaging.md)
