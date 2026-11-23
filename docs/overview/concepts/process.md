# Federated process

A [FederatedProcess][flypse.process.FederatedProcess] is the emulation of a federation: it
binds together the federation, its data and the system it runs on, so you only call
`run`.

## Run a process

The first three arguments are the [federation](federation.md), its
[data](data.md) and the [system](system.md), the mirror infrastructure when
omitted. The keyword arguments configure the run:

```python
from flypse.process import FederatedProcess

process = FederatedProcess(federation, data, step_every_ms=100.0, seed=0)
process.run(seconds=10)
```

!!! info "Important"

    `step_every_ms` is required to run for a given time. A process takes no steps on
    its own, and entities are deployed by the first step: without it, `run` waits,
    nothing is deployed, and no results come out.

A process emulates a single federation, and registers it by itself. Calling
`register` with any other federation raises `ValueError`.

!!! note

    A script runs a single process: creating a second one after the first has run raises
    `RuntimeError`. To compare settings, run the script once for each of them.

## What happens at runtime

1. Creating the process splits the data across the entities holding it, seeds the
   generators of the entities, builds the system, and starts one Ray actor per device, each
   with a copy of the dataset. 2. `run` fires the first step, which places the entities and
   deploys them. Each entity binds its data and starts looping on its `step`. 3. Every
   `step_every_ms`, the process takes a step: it applies the update policies, checks the
   placement, and collects the metrics. Entities keep running in between, at their own pace.
   4. When the process stops, the entities are stopped and undeployed, and the reports are
   written to disk.

!!! note

    A process step is neither the `step` of an entity nor a round of learning.
    `step_every_ms` paces the process, not the entities: a federation can go through
    many rounds between two process steps, or none at all.

## Configuration

These are the options that matter most:

| Parameter | Meaning |
| --- | --- |
| `step_every_ms` | How often the process takes a step, in milliseconds. Required, unless you use `run(steps=N)`. |
| `timeout` | Stops the process after this many seconds, counted from its creation. |
| `max_steps` | Stops the process after this many steps. Whichever of `timeout` and `max_steps` comes first wins. |
| `seed` | The seed of the whole process, random if omitted. See [Reproducibility](../advanced/reproducibility.md). |
| `events` | Metrics of your own. See [custom metrics](../advanced/building-on-eclypse.md#custom-metrics). |
| `include_default_metrics` | Whether to collect the default metrics, such as `step_result`. See [Reporting](../advanced/reporting.md). |
| `path` | Where the outputs are written. |
| `report_format`, `report_backend` | The format and the backend of the tabular report. Parquet and the `polars` backends need `polars`, which is not installed with FLYPSE. |
| `reporters` | Additional ECLYPSE reporters, see [Building on ECLYPSE](../advanced/building-on-eclypse.md). The JSON reporter is always included. |
| `log_level`, `log_to_file` | The verbosity of the logs, and whether to keep them in a file. |

## Stopping

There are three ways to stop a process:

- `run(seconds=10)` stops it 10 seconds after the call,
- `timeout` and `max_steps`, with a plain `run()`,
- `run(steps=N)` takes N process steps and stops straight away.

!!! warning

    `run(steps=N)` does not wait for the entities: they are deployed by the first
    process step and stopped right after the last one, so they get almost no time to
    run. Use `seconds` to let a federation learn.

Stopping does not wait for a round to end: entities are stopped wherever they are.
Those waiting for a message are cancelled, so `Task was destroyed but it is
pending!` lines at the end of the log are expected.

## Where outputs go

Outputs are written under `path`, `~/eclypse-sim` by default. If that directory
already exists, a timestamp is appended to the name, so read the actual location
from `process.path`. The report format, its backend and its location can be set
with their options, for instance `report_format="json"` or
`path="runs/mean-estimation"`.

!!! note "See also"

    - [Inspect the results](../getting-started/inspect-results.md)
    - [Reporting](../advanced/reporting.md)
