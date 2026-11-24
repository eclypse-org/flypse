# Federation

In FLYPSE, a [Federation][flypse.federation.federation.Federation] is a graph whose
nodes are [entities](entities.md) and whose edges are the connections between
them. It describes who takes part in the learning process and who can talk to whom,
neither where they run nor the data they hold.

On this page, `Server`, `Client`, `Aggregator` and `Peer` stand for your own
subclasses of [Entity][flypse.federation.entity.Entity].

There are two ways to build a federation:

<div class="grid cards" markdown>

-   **[Use a topology builder](#use-a-topology-builder)**

    ---

    Connect your entities into a common topology with one call.

-   **[Build it by hand](#build-it-by-hand)**

    ---

    Add the entities and their connections one by one.

</div>

## Use a topology builder

Each builder takes the entities, already created, and connects them. Any other
argument goes to the [Federation][flypse.federation.federation.Federation], such as
`federation_id` or `seed`.

=== "Star"

    [get_star][flypse.federation.topology.get_star] connects a centre to each of
    its leaves, as in client-server federated learning:

    ```python
    from flypse.federation import get_star

    federation = get_star(
        Server("server"),
        [Client(f"client_{i}") for i in range(5)],
        federation_id="ClientServer",
    )
    ```

=== "Tree"

    [get_tree][flypse.federation.topology.get_tree] connects each entity to its
    children, as in hierarchical federated learning:

    ```python
    from flypse.federation import get_tree

    federation = get_tree(
        Server("server"),
        {
            "server": [Aggregator("group_0"), Aggregator("group_1")],
            "group_0": [Client("client_0"), Client("client_1")],
            "group_1": [Client("client_2"), Client("client_3")],
        },
    )
    ```

    The keys are the *identifiers* of the parents; the values are the entities.
    A parent not reachable from the root raises `ValueError`.

=== "Ring"

    [get_ring][flypse.federation.topology.get_ring] connects each entity to the
    next, as in decentralised federated learning. It requires at least three
    entities:

    ```python
    from flypse.federation import get_ring

    federation = get_ring([Peer(f"peer_{i}") for i in range(4)])
    ```

## Build it by hand

Any other topology is a matter of adding the entities with
[add_entity][flypse.federation.federation.Federation.add_entity] and connecting them
with [add_connection][flypse.federation.federation.Federation.add_connection], which takes
their identifiers or the entities themselves:

```python
from flypse.federation import Federation

federation = Federation(federation_id="Custom")

federation.add_entity(Server("server"))
federation.add_entity(Client("client_0"))
federation.add_entity(Client("client_1"))

federation.add_connection("server", "client_0")
federation.add_connection("server", "client_1")
```

`entities` holds the entities of the federation, keyed by identifier. Adding two
entities with the same identifier raises `ValueError`.

## Neighbours and direction

The neighbours of an entity, returned by
[get_neighbors][flypse.federation.entity.Entity.get_neighbors], are the entities it can
send messages to. A connection works in both directions by default.

!!! info "Important"

    With `add_connection(source, target, symmetric=False)`, `source` can send
    to `target` but not the other way round: `target` does not have `source`
    among its neighbours, and cannot reply.

[get_connections][flypse.federation.entity.Entity.get_connections] returns the connections to
the neighbours too, with the attributes of each.

## Parents and children

In a tree, every connection works in both directions, so the neighbours of an aggregator
include its parent and its children alike. The two directions are distinct, and each has a
`relation`, what the entity at its other end is to the one it starts from: `"parent"` going
up, `"child"` going down. An entity asks for the neighbours of a relation, which the process
gives as they are at the time of the call:

```python
async def step(self):
    parents = await self.get_neighbors(relation="parent")
    children = await self.get_neighbors(relation="child")
```

The root has no parent and a client no children, so one entity class can work at any
level of a tree, and its code needs no identifiers of the others.

Any federation has a hierarchy if its connections are added with a `relation`, what the target
is to the source. The opposite direction gets the opposite relation:

```python
federation.add_connection("client_0", "server", relation="parent")
```

Where there is no ordering to tell, as in a star or a ring, connections have no relation, and
asking for one returns no neighbours.

## Requirements

Entities and connections require nothing by default, so a federation fits any
infrastructure. Keyword arguments of `add_entity` and `add_connection` set what an entity or
a connection requires, out of the [resources of the system](system.md#system-resources):

```python
federation.add_entity(Client("client_2"), cpu=2, ram=4.0)
federation.add_connection("server", "client_2", latency=50.0, bandwidth=10.0)
```

The [placement strategy](system.md) then only uses devices and paths that
satisfy them. A federation also accepts `update_policies`, functions changing its
requirements during the run, as an infrastructure does with its
[resources](../advanced/heterogeneity.md).

!!! note "See also"

    - [Entities](entities.md)
    - [System and placement](system.md)
