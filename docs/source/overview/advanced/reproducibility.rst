Reproducibility
===============

A process has a single seed, its ``seed`` option, random if omitted. Components
given no seed of their own fall back to it, so one number reproduces the whole run:

.. code-block:: python

   from flypse.process import FederatedProcess

   process = FederatedProcess(federation, data, step_every_ms=100.0, seed=0)

What falls back to it
---------------------

:func:`~flypse.seed.get_seed` returns an explicit seed if there is one, otherwise
the seed of the process. These components use it:

- the splits of :class:`~flypse.data.dirichlet.DirichletDataset` and
  :class:`~flypse.data.quantity.QuantitySkewDataset`,
- :class:`~flypse.data.splitter.FractionSplitter`,
- the ``"singleton"`` and ``"mirror"`` infrastructures,
- the federation, unless it was given its own ``seed``,
- the ``rng`` of the entities.

.. important::

   An infrastructure you build yourself is not reseeded by the process, so give it
   the ``seed`` too.

Randomness in an entity
-----------------------

An entity draws from ``self.rng``, seeded from the seed of the process and its
identifier, so that entities draw different but reproducible sequences. It can
also drive a splitter:

.. code-block:: python

   from flypse.data import FractionSplitter

   weights = [self.rng.uniform(-0.1, 0.1) for _ in range(n_features)]
   splitter = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng)

.. warning::

   The global ``random`` of every device is seeded with the same value, so
   entities using it draw correlated sequences. The global random generator of
   numpy is not seeded at all. Use ``self.rng``, or a generator seeded from it.

Deterministic aggregation
-------------------------

Messages arrive in a different order at every run, and floating-point rounding
makes a sum follow the order of its terms. Sorting the messages by sender before
combining them removes the difference:

.. code-block:: python

   messages.sort(key=lambda message: message.sender)

With that, two runs of the :doc:`fedavg example <../examples/fedavg>` produce the
same model hashes, bit by bit, for every round they both reach.

What stays non-deterministic
----------------------------

The seed fixes what the entities compute, not when:

- the number of rounds in a given time depends on the machine,
- a protocol driven by a deadline or by the process steps depends on the wall
  clock, so which messages arrive in time differs between runs.

.. seealso::

   - :doc:`../concepts/process`
