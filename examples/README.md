# FLYPSE Examples

Run examples from the repository root, after `make setup`.

| Name | Demonstrates | How to run | Docs |
| --- | --- | --- | --- |
| Federated averaging | A server and five clients learn a linear model from non-IID data. | `uv run python -m examples.fedavg.main` | `docs/source/overview/examples/fedavg.rst` |
| Hierarchical federated averaging | Group aggregators between the clients and the root, with three version levels. | `uv run python -m examples.hierarchical.main` | `docs/source/overview/examples/hierarchical.rst` |
| Decentralized federated averaging | Peers in a ring with no server, averaging with their neighbours. | `uv run python -m examples.decentralized.main` | `docs/source/overview/examples/decentralized.rst` |
