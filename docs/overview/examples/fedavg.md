# Federated averaging

The federated averaging example trains a linear model across five clients and a
server, the basic client-server pattern of federated learning.

Use it when you want to understand:

- how a synchronous protocol emerges from the `step` of each entity,
- how to wrap a model of your own into a federated one,
- how to hold out a test part on each client and evaluate the global model.

The full code lives in the
[examples/fedavg](https://github.com/eclypse-org/flypse/tree/main/examples/fedavg)
directory.

Run it from the repository root with:

```bash
uv run python -m examples.fedavg.main
```

## Model

The model is plain Python, with no framework, as FLYPSE ships no model of its own.
`local_train` runs stochastic gradient descent on a copy of the model, and
`FedAvgModel` exposes its weights and its bias as numpy arrays.

??? example "Model code"

    ```python
    --8<-- "examples/fedavg/model.py"
    ```

## Data

The source is a synthetic regression task, sorted by its input feature. A
[UniformDataset][flypse.data.uniform.UniformDataset] assigns contiguous blocks, so every
client sees a different region of the feature space and the federation is non-IID
by construction.

??? example "Data code"

    ```python
    --8<-- "examples/fedavg/data.py"
    ```

## Entities

The protocol lives entirely in the entities. `FedAvgClient` trains, sends its
model to the server, and waits for the next global model. `FedAvgServer` waits for
a model from each of its neighbours, averages them weighted by their number of
samples, and broadcasts the result:

```text
Θ = Σ_c (n_c / n) Θ_c
```

In `on_deploy`, clients draw their initial model from their `rng`, only the first time, and
keep 20% of their data aside as a test part, the same at every deployment: a client that is
[deployed again](../concepts/system.md#placement-reset) goes on with what it has learnt. The
server sorts the messages before averaging, so that the result does not depend on the order
they arrived in.

??? example "Entities code"

    ```python
    --8<-- "examples/fedavg/entities.py"
    ```

## Process

`main` builds the federation and its data, and runs a process on the default
mirror infrastructure, with one device per entity, for ten seconds.

??? example "Process code"

    ```python
    --8<-- "examples/fedavg/main.py"
    ```

## What to inspect

Every ten rounds, each client evaluates the global model on its test part. The
example prints the mean loss of the first evaluated rounds, and of the last one:

```text
round   evaluations  mean test loss
10      5            0.006079
20      5            0.002994
30      5            0.002992
40      5            0.002992
50      5            0.002992
510     5            0.002992
```

The loss settles within a few dozen rounds. The process is seeded, so the values of
a round are the same on every run, while the last round depends on the speed of
your machine.

This example is a good template when you want to try a client-server protocol of
your own.

Related concepts:

- [Entities](../concepts/entities.md)
- [Models](../concepts/models.md)
- [Federated data](../concepts/data.md)
- [Evaluation](../advanced/evaluation.md)
