# Models

FLYPSE is model-agnostic: how a model computes and trains is left to its
framework. A [FederatedModel][flypse.federation.model.FederatedModel] exposes the
parameters of any model as numpy arrays, so combining them is plain numpy. It also records
every [version](versions.md) the model goes through.

## Wrap a model

Subclasses implement two hooks: `_read_parameters` yields the parameters to
exchange, and `_write_parameter` writes one of them back into the wrapped model,
which stays available as `model` for local training and inference.

=== "Plain Python"

    ```python
    import numpy as np

    from flypse.federation import FederatedModel


    class MeanModel(FederatedModel):
        def _read_parameters(self):
            yield "mean", self.model["mean"]

        def _write_parameter(self, name, value):
            self.model[name] = value


    model = MeanModel({"mean": np.zeros(1)})
    ```

=== "PyTorch"

    ```python
    import torch

    from flypse.federation import FederatedModel


    class TorchModel(FederatedModel):
        def _read_parameters(self):
            for name, value in self.model.state_dict().items():
                if value.is_floating_point():
                    yield name, value.detach().cpu().numpy()

        def _write_parameter(self, name, value):
            self.model.state_dict()[name].copy_(torch.from_numpy(value))


    model = TorchModel(torch.nn.Linear(4, 2))
    ```

The entries yielded by the hook are read once and fixed: their names, order and
shapes must stay the same. Only floating-point entries are accepted, so integer
counters must be skipped, and a model with no parameters to expose fails at
initialisation. Choosing what to yield is a decision of the algorithm: FedBN is FedAvg
that does not yield the entries of the batch normalisation layers, neither their learnt
scale and shift nor their statistics.

!!! info "Important"

    The hook is first called by `__init__`, so any attribute it uses must be set
    before calling `super().__init__`.

## Read and write parameters

[get_parameters][flypse.federation.model.FederatedModel.get_parameters] reads a fresh copy
of the parameters, as one flat vector or as arrays keyed by name. Local training
is always visible, so the update of a local training is a subtraction:

```python
before = model.get_parameters()
train(model.model, data)
delta = model.get_parameters() - before

model.get_parameters(flattened=False)  # {"mean": array([...], dtype=float32)}
```

There are two ways to write parameters, depending on who generated them:

- [update_version][flypse.federation.model.FederatedModel.update_version] is for the entity
  generating a new model, at a given level,
- [load_version][flypse.federation.model.FederatedModel.load_version] is for the entity
  receiving one, which adopts its version and timestamp as they are.

Both return the recorded [ModelVersion][flypse.federation.model.ModelVersion], ready to
be sent.

## One vector, many shapes

A flat vector has one value per coordinate of the parameters, in the order the hook yields
them. [unflatten][flypse.federation.model.FederatedModel.unflatten] gives any such vector,
not only parameters, the shapes of the entries:

```python
correction = model.unflatten(global_control - local_control)
correction["mean"]  # a view of the vector, shaped as the entry
```

It is how a term computed on flat vectors, such as a control variate, reaches a training
loop working on the layers of a framework.

## Labels and masks

The read hook can label an entry with a third element, a name or several. The labels say
what an entry is for:

```python
class TwoPartModel(FederatedModel):
    def _read_parameters(self):
        yield "body", self.model["body"], "body"
        yield "head", self.model["head"], "personal"
        yield "running_mean", self.model["running_mean"], "statistics"

    def _write_parameter(self, name, value):
        self.model[name] = value
```

[mask][flypse.federation.model.FederatedModel.mask] turns labels into a boolean vector, True
on the coordinates carrying any of them:

```python
model.mask("statistics")
model.mask("personal", "statistics")
~model.mask("personal")  # everything else
```

A mask selects the coordinates a strategy works on, such as the ones a server optimiser
[steps on](../strategies/optimization.md#leave-coordinates-out), or the ones a weak device
[trains](../strategies/assignment.md#a-part-of-the-model). A label that no entry carries
raises `ValueError`.

## Load a part of a model

With a mask, `load_version` writes the selected coordinates only, and the others keep
their value. A client with a personal head keeps it through every round:

```python
shared = ~self.model.mask("personal")
self.model.load_version(message.record, mask=shared)
```

The version is adopted all the same: the model it names is the shared part.

!!! note "See also"

    - [Versions](versions.md)
    - [Messaging](messaging.md)
    - [Strategies](../strategies/index.md)
