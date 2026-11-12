"""Package for the strategies of federated learning.

A strategy is a building block an entity calls inside its own ``step``: whom to
engage (:mod:`~flypse.strategies.sampling`), how to combine what it received
(:mod:`~flypse.strategies.aggregation`), and how the result changes its own
model (:mod:`~flypse.strategies.updates`). Strategies work on flat parameter
vectors, the same for a server, a group aggregator and a peer.
"""
