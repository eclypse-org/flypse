# Decentralised federated averaging

The decentralised example has no server: five peers in a ring each train on their
own data and average their model with those of their two neighbours.

Use it when you want to understand:

- how a federation works with no central entity,
- how to receive from a given neighbour, leaving the others queued,
- how a consensus forms through repeated local averaging.

The full code lives in the
[examples/decentralized](https://github.com/eclypse-org/flypse/tree/main/examples/decentralized)
directory.

Run it from the repository root with:

```bash
uv run python -m examples.decentralized.main
```

## Model and data

The model and the data are the ones of the [federated averaging example](fedavg.md): a
linear model trained with stochastic gradient descent, and a synthetic regression source
sorted by its input feature. The example keeps its own copy of both, so that it reads on
its own.

??? example "Model code"

    ```python
    --8<-- "examples/decentralized/model.py"
    ```

??? example "Data code"

    ```python
    --8<-- "examples/decentralized/data.py"
    ```

## Entities

`Peer` both trains and aggregates. At every step it trains, sends its model
to its neighbours, collects theirs and averages everything, its own model included,
weighted by the number of samples. In `on_deploy`, peers draw their initial model from their
`rng`, only the first time, and keep 20% of their data aside as a test part, the same at
every deployment.

Two details make the protocol work:

- One neighbour can be a round ahead of the other, so its next message may arrive first. A
  peer receives from one neighbour at a time, with `receive(source=...)`: the messages of
  the other stay queued until their turn, and every round takes one message per neighbour.
- A peer reads `message.record` and never loads it: `load_version` would adopt
  the counters of a neighbour, while the `"global"` version of a peer counts its
  own rounds.

??? example "Entities code"

    ```python
    --8<-- "examples/decentralized/entities.py"
    ```

## Process

`main` builds the ring with [get_ring][flypse.federation.topology.get_ring] and runs
it on the default mirror infrastructure. Every ten rounds, peers evaluate their
model on their test part and report its parameters, from which the example measures
how far apart the models are.

??? example "Process code"

    ```python
    --8<-- "examples/decentralized/main.py"
    ```

## What to inspect

```text
round   mean test loss   largest distance between peers
10      0.114046         0.347883
20      0.022369         0.145257
30      0.006378         0.060791
40      0.003578         0.025407
50      0.003091         0.012001
60      0.003008         0.006894
```

There is no global model here: every peer has its own, and the last column is the
distance between the two peers furthest apart. It shrinks as models cross the ring
one hop at a time, and the loss follows. A model has to go through several peers to
reach the other side, so a sparse graph like a ring converges more slowly than a star.

The process is seeded, and each round is made of the same messages in the same
order, so the table is the same on every run.

This example is a good template when you want to try another topology, or another
way of combining the models of the neighbours.

Related concepts:

- [Federation](../concepts/federation.md)
- [Messaging](../concepts/messaging.md)
