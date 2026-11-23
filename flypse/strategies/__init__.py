"""Package for the strategies of federated learning.

A strategy is a building block an entity calls inside its own ``step``. An
entity aggregating what others send (a server, a group aggregator, a peer)
makes a few separate decisions, one module each, in the order of a round:

- :mod:`~flypse.strategies.selection`: whom to engage;
- :mod:`~flypse.strategies.assignment`: what each engaged entity is asked to do;
- :mod:`~flypse.strategies.screening`: which contributions to use, and in which
  groups;
- :mod:`~flypse.strategies.weighting`: how much each contribution counts;
- :mod:`~flypse.strategies.aggregation`: how the contributions combine into one;
- :mod:`~flypse.strategies.optimization`: how the result changes the entity's
  own model.

Algorithms differ by the strategy they take for each decision: e.g., federated
averaging selects clients uniformly, uses every contribution, weighs each by
its number of samples, takes the mean and adopts it; FedAdam takes an Adam step
instead of adopting the mean, and is the same otherwise. Strategies work on
flat parameter vectors, the same for a server, a group aggregator and a peer.
"""
