# Inspect the results

This page shows how to read what the entities reported during a run.

We will:

1. report values from an entity,
1. read them as records once the process has finished,
1. turn the records into a table.

The following example uses:

- [report][flypse.federation.entity.Entity.report], to push values from an entity,
- [reported][flypse.process.FederatedProcess.reported], to read them back.

## Report from an entity

An entity can report any value JSON can represent, at any moment of its `step`, with
[report][flypse.federation.entity.Entity.report]. Its values go straight to the
process, without waiting for it:

```python
class Server(Entity):
    async def step(self):
        ...
        self.report(round=record.version[0], estimate=average[0])
```

## Read the records

Once the process has finished, `reported` holds one record per report, oldest
first. The values of a report stay together, with who reported them, from where
and when:

```python
from flypse.process import FederatedProcess

process = FederatedProcess(federation, data, step_every_ms=100.0, seed=0)
process.run(seconds=5)

print(process.reported[0])
```

```text
{'round': 1, 'estimate': 24.75, 'node_id': 'server', 'entity_id': 'server', 'timestamp': '2026-11-05T23:27:10.992829', 'n_report': 1}
```

`entity_id` and `n_report`, the progressive number of the report for that
entity, identify a record.

## Turn them into a table

A record is a plain dictionary, so `reported` feeds any tool you like. pandas,
which is installed with FLYPSE, turns it into a table in one call:

```python
import pandas as pd

frame = pd.DataFrame(process.reported)
print(frame[["round", "estimate"]].tail(2))
```

```text
     round   estimate
568    569  49.499989
569    570  49.499989
```

!!! note "See also"

    - [Choose where the federation runs](infrastructure.md)
    - [Reporting](../advanced/reporting.md)
    - [Evaluation](../advanced/evaluation.md)
