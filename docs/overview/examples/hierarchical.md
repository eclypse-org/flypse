# Hierarchical federated averaging

The hierarchical example adds a tier between the clients and the server, the root of
the tree: group aggregators average their clients at every round, and the root
averages the groups every few rounds.

Use it when you want to understand:

- how to build a tree with [get_tree][flypse.federation.topology.get_tree],
- how an entity finds its parent and its children,
- how three version levels record where a model comes from,
- why a version alone does not identify a model.

The full code lives in the
[examples/hierarchical](https://github.com/eclypse-org/flypse/tree/main/examples/hierarchical)
directory.

Run it from the repository root with:

```bash
uv run python -m examples.hierarchical.main
```

## Model and data

The model and the data are the ones of the [federated averaging example](fedavg.md): a
linear model trained with stochastic gradient descent, and a synthetic regression source
sorted by its input feature. The example keeps its own copy of both, so that it reads on
its own.

??? example "Model code"

    ```python
    --8<-- "examples/hierarchical/model.py"
    ```

??? example "Data code"

    ```python
    --8<-- "examples/hierarchical/data.py"
    ```

## Entities

`Client` trains, sends its model to its group aggregator and waits for the next
one. `Root` averages the models of the groups and broadcasts the global one. They
work as the client and the server of the federated averaging example, with one more
version level: `("global", "group", "local")`. `GroupAggregator` sits between
them.

At every round, a group aggregator averages the models of its clients with
`update_version(average, "group")`. Every `GLOBAL_EVERY` rounds, it also sends
the result to the root and loads the global model it gets back. Then it sends the
latest model to its clients.

It sends to its clients explicitly rather than with `bcast`, as its neighbours
include the root. No entity is given the identifiers of the others: `related` asks
the process for the connections of an entity with
[get_connections][flypse.federation.entity.Entity.get_connections], and keeps those with a
given relation, `"parent"` or `"child"`, as `get_tree` labels them (see
[Parents and children](../concepts/federation.md#parents-and-children)).

A plain `receive` is enough, with no `source`, because the entities strictly alternate:
while a group waits for the root, its clients are waiting for the group.

??? example "Entities code"

    ```python
    --8<-- "examples/hierarchical/entities.py"
    ```

## Process

`main` runs four clients in two groups on the default mirror infrastructure. The
source is sorted by its input feature, so each group holds one half of the feature
range.

Clients evaluate every model they receive, and the example groups the records by
version.

??? example "Process code"

    ```python
    --8<-- "examples/hierarchical/main.py"
    ```

## What to inspect

```text
version       evaluations  models  mean test loss
(0, 1, 0)     4            2       0.188751
(0, 2, 0)     4            2       0.141911
(1, 0, 0)     4            1       0.416864
(1, 1, 0)     4            2       0.058675
(1, 2, 0)     4            2       0.045519
(2, 0, 0)     4            1       0.122991
(2, 1, 0)     4            2       0.021488
```

The `models` column counts the distinct `model` hashes behind a version. Group
versions show two: both groups generate `(1, 1, 0)`, and they hold different
models, which is why a version alone is not an identity. Global versions show one,
as every client received the same model.

The loss goes up at each global version, because the data is non-IID: a group model
fits the half of the range its clients hold, while the global model has to cover
both. The jump shrinks round after round as the models converge.

The process is seeded, and the aggregators sort their messages, so the table is the
same on every run.

This example is a good template when you want to compare how often to aggregate at
each tier.

Related concepts:

- [Federation](../concepts/federation.md)
- [Versions](../concepts/versions.md)
- [Evaluation](../advanced/evaluation.md)
