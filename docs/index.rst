.. toctree::
   :maxdepth: 6
   :hidden:

   Overview <source/overview/index.rst>
   Reference <source/api/index.rst>
   Changelog <https://github.com/eclypse-org/flypse/blob/main/CHANGELOG.md>



=====================
FLYPSE documentation
=====================
**FLYPSE** is a Python library for experimenting with federated learning on
the cloud-edge continuum.

It models a federation as an `ECLYPSE <https://github.com/eclypse-org/eclypse>`_
application, whose services are the federated entities, and emulates it on an
infrastructure whose nodes are the devices holding the data. Each entity runs
as a remote ECLYPSE service and exchanges messages with the others, so both
the statistical heterogeneity of the data and the system heterogeneity of the
infrastructure shape the learning process.

Key features include:

- federations built as graphs of entities and their connections,
- federated datasets that split the data across entities, uniformly or with
  Dirichlet label or quantity skew,
- model-agnostic parameters exchanged as numpy arrays, with hierarchical
  versioning,
- ready-made singleton and mirror infrastructures, or any ECLYPSE
  infrastructure,
- the ECLYPSE reporting, update policies, and events for the emulation.

.. button-ref:: source/overview/index
   :ref-type: myst
   :outline:
   :color: secondary
   :expand:
   :align: center
   :shadow:

   :octicon:`play;1em;info` Start using FLYPSE
