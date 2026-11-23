# Examples

The examples section complements the getting-started guides with runnable
federations you can inspect and adapt.

The code for the documented examples is available in the
:fontawesome-brands-github:
[examples](https://github.com/eclypse-org/flypse/tree/main/examples/)
subdirectory of the repository. Run them from the repository root, after
`make setup`.

<div class="grid cards" markdown>

-   :octicons-broadcast-16: **[Federated averaging](fedavg.md)**

    ---

    A server and five clients learn a linear model from non-IID data.

-   :octicons-git-merge-16: **[Hierarchical federated averaging](hierarchical.md)**

    ---

    Group aggregators between the clients and the server, with three version
    levels recording where a model comes from.

-   :octicons-share-android-16: **[Decentralised federated averaging](decentralized.md)**

    ---

    Peers in a ring with no server, reaching a consensus by averaging with
    their neighbours.

-   :octicons-clock-16: **[Stragglers](stragglers.md)**

    ---

    A link slows down during the run, while a server with a deadline carries
    on without the client behind it.

</div>
