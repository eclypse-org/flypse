# Versions

A [FederatedModel][flypse.federation.model.FederatedModel] records every version it goes
through. A version tells where a model comes from and how stale it is, so an
aggregator can find the model a contribution was trained from.

## Levels and lineage

A version has one counter per level, from the coarsest to the finest,
`("global", "local")` by default.
[update_version][flypse.federation.model.FederatedModel.update_version] increments the
counter of its level and resets the finer ones. A three-tier federation only needs one more
level:

```python
model = MeanModel({"mean": np.zeros(1)}, version_levels=("global", "group", "local"))
```

| Event | Version |
| --- | --- |
| A client loads the model of its group | `(1, 3, 0)` |
| It trains locally, twice: `update_version(params, "local")` | `(1, 3, 2)` |
| Its group aggregates: `update_version(params, "group")` | `(1, 4, 0)` |
| The server aggregates: `update_version(params, "global")` | `(2, 0, 0)` |

## The history

`versions` holds the records of a model, oldest first, and `version` the counters of the
current one. A record is a [ModelVersion][flypse.federation.model.ModelVersion]: its
counters, when it was generated, and its parameters as one flat, read-only vector. Neither
local training nor the caller can alter it.

!!! warning

    The history is unbounded by default, and every record holds all the parameters.
    For long runs or large models, limit it with `max_versions`.

`max_versions` limits the records in all, or level by level. A level left out keeps every
record, and the level of a record is the finest one with a non-zero counter:

```python
MeanModel({"mean": np.zeros(1)}, max_versions=10)
MeanModel({"mean": np.zeros(1)}, max_versions={"local": 1})
```

The second form suits a client, whose local versions are of no use once sent, while the
global ones may still be evaluated.

## Read any version

[get_parameters][flypse.federation.model.FederatedModel.get_parameters] also reads a past
version, by its counters or by its record:

```python
model.get_parameters(version=(1, 3, 0))  # from the history of this model
model.get_parameters(version=message.record)  # from the record, as it is
```

!!! info "Important"

    A version is not unique across entities: two clients can both generate
    `(3, 1)`. Counters look up the history of the model reading them, so to read
    the record of a peer, pass the record.

## Base and lag

Two questions about a received version have their answer in its counters.

[base][flypse.federation.model.FederatedModel.base] finds the version it was generated from,
at a given level, in the history of the model. The update of a client is relative to it,
even if the model of the server has moved on since:

```python
base = self.model.base(message.record, "global")
delta = message.record.parameters - base.parameters
```

[lag][flypse.federation.model.FederatedModel.lag] counts how many versions it is behind, at a
given level. At the coarsest level, it is the staleness of a contribution:

```python
lag = self.model.lag(message.record, "global")
```

With `self.model.version == (7, 0)`, a client answering with `(5, 1)` has
`base(...)` at `(5, 0)` and a `lag` of 2.

!!! note

    `base` looks the version up, so it must still be in the history: with a limit on it,
    keep at least one more version than the largest staleness you accept.

## Updates that are not models

A compressed or encoded update is not a version of a model, so it travels as data, with the
version it was computed at. `base` and `lag` take bare counters too:

```python
await self.send(message.sender, update=encoded, version=record.version)
```

```python
version = message.data["version"]
base = self.model.base(version, "global")
lag = self.model.lag(version, "global")
```

!!! note "See also"

    - [Models](models.md)
    - [Compose an algorithm](../strategies/algorithms.md)
    - [Hierarchical federated averaging](../examples/hierarchical.md)
