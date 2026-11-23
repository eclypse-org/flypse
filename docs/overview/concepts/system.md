# System and placement

The system is where a federation runs: an *infrastructure*, a graph of devices and
the links between them, and a *placement strategy*, mapping the entities onto the
devices. A [SystemConfig][flypse.system.config.SystemConfig] keeps the two consistent.

There are two ways to choose the system:

<div class="grid cards" markdown>

-   **[Built-in systems](#built-in-systems)**

    ---

    Derive the infrastructure and the placement from the federation.

-   **[Adjust a built-in system](#adjust-a-built-in-system)**

    ---

    Set the resources of its devices and links, and change its topology.

</div>

## Built-in systems

=== "Singleton"

    A single device, with maximal resources, hosts every entity. It is the
    quickest way to check the logic of the entities:

    ```python
    from flypse.system import SystemConfig

    system = SystemConfig("singleton")
    ```

=== "Mirror"

    The default. The infrastructure copies the topology of the federation: one
    device per entity and one link per connection, with maximal resources:

    ```python
    system = SystemConfig("mirror")
    ```

!!! note

    The links of a mirror start with zero latency and maximal bandwidth, so messages
    cost next to nothing until you adjust them.

## Adjust a built-in system

`SystemConfig("mirror")` has no options for the infrastructure. To adjust a built-in system,
build it with [get_mirror][flypse.system.infrastructure.get_mirror] or
[get_singleton][flypse.system.infrastructure.get_singleton], and pass it with its strategy:

```python
from flypse.system import SystemConfig, get_mirror, get_mirror_strategy

infrastructure = get_mirror(federation, seed=0)
system = SystemConfig(infrastructure, get_mirror_strategy(federation))
```

For a singleton, the strategy is `get_singleton_strategy(federation)`. An
infrastructure given without a strategy raises `ValueError`, as there is no way to
tell where the entities go.

### Set the resources

An infrastructure is a [networkx](https://networkx.org/) graph, so its devices and
links can be read and changed like those of any graph. A link has one edge per
direction, and each direction has its own values:

```python
infrastructure.edges["client_0", "server"]["latency"] = 100.0
infrastructure.edges["client_0", "server"]["bandwidth"] = 10.0
infrastructure.nodes["client_0"]["processing_time"] = 5.0
```

To change them during a run, see [Heterogeneity](../advanced/heterogeneity.md).

### Change the topology

Devices and links can be added and removed too, as long as connected entities can
still reach each other. This infrastructure routes every client through a gateway,
so that their messages cross two links:

```python
infrastructure = get_mirror(federation, seed=0)

infrastructure.add_node("gateway")
infrastructure.add_edge("gateway", "server", symmetric=True, latency=50.0)
for client_id in ("client_0", "client_1", "client_2"):
    infrastructure.remove_edge(client_id, "server")
    infrastructure.remove_edge("server", client_id)
    infrastructure.add_edge(client_id, "gateway", symmetric=True, latency=5.0)

system = SystemConfig(infrastructure, get_mirror_strategy(federation))
```

For an infrastructure built from scratch, see
[Building on ECLYPSE](../advanced/building-on-eclypse.md).

## Devices and links { #system-resources }

Every node of the infrastructure becomes a [device](devices.md), an emulated machine hosting
the entities placed on it.

Devices and links offer resources, which entities and connections can
[require](federation.md). Each resource acts on a run in its own way:

| Resource | Of a | Effect on the run |
| --- | --- | --- |
| `cpu` | device | Decides where entities can be placed, and how long [compute](devices.md#emulate-the-speed-of-a-device) takes. |
| `ram`, `storage`, `gpu` | device | Decide where entities can be placed. They do not slow a `step` down. |
| `availability` | device | At zero, the device drops out of the paths, which [resets the placement](#placement-reset). |
| `processing_time` | device | Delays the messages crossing the device, in milliseconds. |
| `latency` | link | Delays the messages crossing the link, in milliseconds. |
| `bandwidth` | link | Sets how long a message takes to cross the link, given its size, in megabits per second. |

The delays make up the [cost of a message](messaging.md#message-cost). The computation of an
entity runs as fast as your machine, and
[compute](devices.md#emulate-the-speed-of-a-device) adds the time of a slower device.

## Exclusive placement

The built-in systems place the entities by themselves: on the only device of a
singleton, or on the device named after them in a mirror.

On an infrastructure with at least as many devices as entities, where any device will do,
[ExclusiveStrategy][flypse.system.placement.ExclusiveStrategy] places each entity on a distinct
device satisfying its requirements, as a maximum matching between entities and
devices, so an assignment is found whenever one exists:

```python
from flypse.system import ExclusiveStrategy

system = SystemConfig(infrastructure, ExclusiveStrategy())
```

!!! info "Important"

    The placement is all or nothing: if the entities cannot all get a device of
    their own, none of them is placed, and the process runs without entities.

## When the placement is reset { #placement-reset }

At every process step, the process checks that the entities can still reach each
other and that their requirements still hold. If a path between two connected
entities disappears, for instance because a device fails, or if a resource falls short
of a requirement, the placement of the whole federation is reset: **every** entity is
undeployed, and all of them are deployed again once the problem is gone. Messages in
flight are lost, and each entity runs `on_deploy` again.

It is the same for one failed device or many: in a run that took one device out of
four down for three seconds, all four entities were deployed twice, and none of them
ran in between.

!!! note "See also"

    - [Devices](devices.md)
    - [Heterogeneity](../advanced/heterogeneity.md)
    - [Building on ECLYPSE](../advanced/building-on-eclypse.md)
