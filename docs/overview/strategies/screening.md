# Screening

Screening decides **which of the received contributions to use**, and in which groups. A
filter keeps some and drops the others, to guard against contributions that are broken or
malicious. A grouping splits them, so that each group is combined on its own.

There are two kinds of members in [flypse.strategies.screening][]:

<div class="grid cards" markdown>

-   **[Filters](#filters)**

    ---

    Return the positions of the contributions they keep.

-   **[Groupings](#groupings)**

    ---

    Return groups of positions, each combined on its own.

</div>

!!! tip

    Screen *updates*, `message.record.parameters - base.parameters`, when the contributions
    started from different models, or when a rule looks at norms or angles: the models
    themselves all have nearly the same. Distances between models that share their start are
    those between their updates.

## Filters

`f` is the number of contributions that may be adversarial, and `n` the number of vectors. The
guarantees of a filter hold only when there are enough honest vectors, and only
[krum][flypse.strategies.screening.krum] and [bulyan][flypse.strategies.screening.bulyan]
raise `ValueError` when there are too few. The others check their arguments and no more:
have `n > 2f` for [mda][flypse.strategies.screening.mda], and `n >= 11f` for
[monna][flypse.strategies.screening.monna].

| Strategy | Keeps | Raises unless |
| --- | --- | --- |
| [krum][flypse.strategies.screening.krum] | The `m` vectors closest to their neighbours ([Blanchard et al., 2017](https://arxiv.org/abs/1703.02757)) | `n > 2f + 2` |
| [bulyan][flypse.strategies.screening.bulyan] | `n - 2f` vectors, chosen one at a time by Krum ([El Mhamdi et al., 2018](https://arxiv.org/abs/1802.07927)) | `n >= 4f + 3` |
| [mda][flypse.strategies.screening.mda] | The `n - f` vectors of smallest diameter | `f < n` |
| [monna][flypse.strategies.screening.monna] | The `n - f` vectors nearest to the entity's own ([Farhadkhani et al., 2023](https://arxiv.org/abs/2209.10931)) | `f < n` |
| [dnc][flypse.strategies.screening.dnc] | The vectors that deviate the least along the main direction of spread ([Shejwalkar and Houmansadr, 2021](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/)) | `c f < n` |
| [norm_filter][flypse.strategies.screening.norm_filter] | The vectors whose norm is at most a bound | |

The positions keep vectors, senders and weights aligned, so a filter sits in front of any
aggregation:

```python
from flypse.strategies import aggregation, screening

own = self.model.get_parameters()
deltas = [m.record.parameters - own for m in messages]

kept = screening.krum(deltas, f=1, m=3)
delta = aggregation.mean([deltas[i] for i in kept])
```

The contributions that are left count the same: a number of samples is what a client says of
itself, and an adversarial one can say anything.

With `m=1`, [krum][flypse.strategies.screening.krum] keeps a single vector; with more, it
is Multi-Krum. [bulyan][flypse.strategies.screening.bulyan] comes with its own
aggregation:

```python
kept = screening.bulyan(deltas, f=1)
delta = aggregation.meamed([deltas[i] for i in kept], 2)  # 2f
```

A bound on the norm is often a multiple of the median one:

```python
import numpy as np

norms = [np.linalg.norm(delta) for delta in deltas]
kept = screening.norm_filter(deltas, bound=2 * np.median(norms))
```

A peer screens what it received together with its own model, at position 0 here:

```python
vectors = [record.parameters, *(m.record.parameters for m in messages)]
kept = screening.monna(vectors, own=0, f=1)
self.model.update_version(aggregation.mean([vectors[i] for i in kept]), "global")
```

!!! warning

    [mda][flypse.strategies.screening.mda] tries every subset of `n - f` vectors: it suits a
    few tens of contributions and a small `f`.

## Groupings

| Strategy | Groups |
| --- | --- |
| [cosine_bipartition][flypse.strategies.screening.cosine_bipartition] | The two groups least similar to each other ([Sattler et al., 2020](https://arxiv.org/abs/1910.01991)) |
| [bucketing][flypse.strategies.screening.bucketing] | Random buckets of a given size ([Karimireddy et al., 2022](https://arxiv.org/abs/2006.09365)) |
| [nnm][flypse.strategies.screening.nnm] | Each vector with its nearest, as overlapping groups ([Allouah et al., 2023](https://arxiv.org/abs/2302.01772)) |

### Clusters of clients

In clustered federated learning, clients that pull in different directions get different
models. [cosine_bipartition][flypse.strategies.screening.cosine_bipartition] splits their
updates in two:

```python
for group in screening.cosine_bipartition(deltas):
    members = [messages[i].sender for i in group]
    average = aggregation.mean([messages[i].record.parameters for i in group])
    ...
```

Clustered federated learning splits a cluster when the mean of its updates is small while
some of them are large: its members have nothing left to agree on, but still want to move.

### Pre-aggregation

On non-IID data, honest contributions differ from each other too, which robust rules take
for attacks. Averaging small groups first makes the contributions more alike:

=== "Bucketing"

    ```python
    buckets = screening.bucketing(len(deltas), size=2, rng=self.np_rng)
    means = [aggregation.mean([deltas[i] for i in bucket]) for bucket in buckets]
    delta = aggregation.median(means)
    ```

=== "Nearest-neighbour mixing"

    ```python
    groups = screening.nnm(deltas, f=1)
    mixed = [aggregation.mean([deltas[i] for i in group]) for group in groups]
    delta = aggregation.median(mixed)
    ```

!!! note

    A bucket with one adversarial member is adversarial as a whole: the rule that follows
    has fewer vectors, and as many bad ones.

!!! note "See also"

    - [Aggregation](aggregation.md), for the rules that are robust by themselves
    - [Weighting](weighting.md), for the weights that lower a contribution instead of
      dropping it
