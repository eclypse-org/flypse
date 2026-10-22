"""Module for the FederatedModel class.

FLYPSE is model-agnostic: how a model computes and trains is left to its
framework. A :class:`FederatedModel` exposes the parameters of a framework
model as numpy arrays, so that combining them (e.g., averages,
pseudogradients, robust aggregation) and measuring them (e.g., norms,
distances) is plain numpy math, while local training stays on the wrapped
model.

The entries the model exchanges, with their names and shapes, are selected
by the ``_read_parameters`` hook and fixed at initialization. For instance,
integer counters are never yielded, while yielding BatchNorm statistics or
not is the only difference between FedAvg and FedBN. Non-parametric models
(e.g., tree ensembles) cannot conform and fail at initialization.

Versions are hierarchical: a version carries one counter per level (by
default ``("global", "local")``; e.g., ``("global", "group", "local")`` for
three tiers). ``update_version`` is used by the entity that generates a new
model at a level (the server after a global aggregation, a client after a
local update): it increments the counter of the level and resets the finer
ones. ``load_version`` is used by the receiving entity: it adopts the
received version and timestamp as they are.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from time import time
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

    from numpy.typing import DTypeLike


@dataclass(frozen=True)
class ModelVersion:
    """A record of the version history of a federated model."""

    version: tuple[int, ...]
    """The version counters, one per level."""

    timestamp: float
    """When the version was generated."""

    parameters: np.ndarray
    """The read-only parameters of the version, as a flat vector."""


class FederatedModel(ABC):
    """A framework model whose parameters are exchanged as numpy arrays.

    Subclasses implement two hooks: ``_read_parameters`` yields the entries
    to exchange, and ``_write_parameter`` writes one entry back into the
    wrapped model, converting it to the framework format. The wrapped model
    stays available as ``model`` for local training and inference, and its
    parameters are read fresh on every access.

    The current version is in ``version``, while ``versions`` holds the
    version history, oldest first.
    """

    def __init__(
        self,
        model: Any,
        dtype: DTypeLike = np.float32,
        version_levels: Sequence[str] = ("global", "local"),
        max_versions: int | None = None,
    ) -> None:
        """Initialize the federated model.

        The entries yielded by ``_read_parameters`` are read once, fixing
        their names, order and shapes, and the initial parameters are
        recorded as the zero version.

        Args:
            model (Any): The framework model to wrap.
            dtype (DTypeLike): The dtype of the parameter arrays. Defaults
                to ``np.float32``.
            version_levels (Sequence[str]): The version levels, from the
                coarsest to the finest. Defaults to ``("global", "local")``.
            max_versions (int | None): The number of records kept in the
                history, dropping the oldest first. Defaults to None
                (unlimited).

        Raises:
            ValueError: If the wrapped model exposes no parameters, the
                read hook yields a name twice or a non-float entry, the
                levels are empty or duplicated, or ``max_versions`` is
                less than 1.
        """
        self.model = model
        self.dtype = np.dtype(dtype)
        levels = tuple(version_levels)
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("the version_levels must be non-empty and unique")
        if max_versions is not None and max_versions < 1:
            raise ValueError("max_versions must be at least 1")
        self.version_levels = levels
        entries = self._read()
        if not entries:
            raise ValueError(
                f"{type(model).__name__} exposes no parameters to exchange"
            )
        self._scheme = {name: value.shape for name, value in entries.items()}
        sizes = [value.size for value in entries.values()]
        self._offsets = np.cumsum(sizes)[:-1]
        self._total = sum(sizes)
        self.version = (0,) * len(self.version_levels)
        flat = np.concatenate([value.ravel() for value in entries.values()])
        flat.flags.writeable = False
        self.versions = deque(
            [ModelVersion(self.version, time(), flat)], maxlen=max_versions
        )

    def get_parameters(
        self, flattened: bool = True, version: Sequence[int] | None = None
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Get a copy of the parameters of the model.

        Args:
            flattened (bool): Whether to return a single flat vector rather
                than a dictionary of arrays keyed by name. Defaults to True.
            version (Sequence[int] | None): The version of the history to
                read; when several records share it, the latest is read.
                Defaults to None, meaning the current parameters of the
                wrapped model, including any local training.

        Returns:
            np.ndarray | dict[str, np.ndarray]: The parameters, in the
            requested form.

        Raises:
            ValueError: If the version is not in the history, or the
                entries read from the model differ from the ones fixed at
                initialization.
        """
        if version is None:
            entries = self._read_live()
            if flattened:
                return np.concatenate([entries[name].ravel() for name in self._scheme])
            return {name: entries[name].copy() for name in self._scheme}
        flat = self._record(version).parameters
        if flattened:
            return flat.copy()
        return {name: piece.copy() for name, piece in self._unflatten(flat).items()}

    def update_version(
        self,
        parameters: np.ndarray | Mapping[str, np.ndarray],
        level: str,
        timestamp: float | None = None,
    ) -> ModelVersion:
        """Write new parameters, incrementing the version at a level.

        It is used by the entity generating the new model (e.g., the server
        after a global aggregation, a client after a local update). The
        counter of the level is incremented, the finer ones are reset, and
        the new version is recorded in the history.

        Args:
            parameters (np.ndarray | Mapping[str, np.ndarray]): The new
                parameters, as a flat vector or a dictionary of arrays
                keyed by name.
            level (str): The level of the new model.
            timestamp (float | None): When the new model was generated.
                Defaults to None, meaning the current wall-clock time.

        Returns:
            ModelVersion: The recorded version, ready to be sent.

        Raises:
            ValueError: If the level is unknown, or the parameters do not
                match the ones of the model.
        """
        if level not in self.version_levels:
            raise ValueError(f"unknown level {level!r}")
        flat = self._validate(parameters)
        index = self.version_levels.index(level)
        counters = list(self.version)
        counters[index] += 1
        for finer in range(index + 1, len(counters)):
            counters[finer] = 0
        return self._commit(
            flat, tuple(counters), time() if timestamp is None else timestamp
        )

    def load_version(self, record: ModelVersion) -> ModelVersion:
        """Load a version generated by another entity.

        It is used by the entity receiving a model: the parameters of the
        record are written into the wrapped model, and its version and
        timestamp are adopted as they are and recorded in the history.

        Args:
            record (ModelVersion): The received version.

        Returns:
            ModelVersion: The recorded copy of the version.

        Raises:
            ValueError: If the version does not match the levels, or the
                parameters do not match the ones of the model.
        """
        flat = self._validate(record.parameters)
        return self._commit(flat, self._as_version(record.version), record.timestamp)

    def _commit(
        self,
        flat: np.ndarray,
        version: tuple[int, ...],
        timestamp: float,
    ) -> ModelVersion:
        """Write validated parameters into the model and record the version.

        The model receives its own copies and the recorded vector is made
        read-only, so neither local training nor the caller can alter the
        history.

        Args:
            flat (np.ndarray): The validated parameters.
            version (tuple[int, ...]): The version to adopt.
            timestamp (float): The timestamp of the version.

        Returns:
            ModelVersion: The recorded version.
        """
        for name, piece in self._unflatten(flat).items():
            self._write_parameter(name, piece.copy())
        flat.flags.writeable = False
        self.version = version
        record = ModelVersion(version, timestamp, flat)
        self.versions.append(record)
        return record

    def _validate(
        self, parameters: np.ndarray | Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Validate parameters against the scheme and flatten them.

        Args:
            parameters (np.ndarray | Mapping[str, np.ndarray]): The
                parameters to check, as a flat vector or a dict.

        Returns:
            np.ndarray: A new flat vector owning its data, cast to
            ``dtype``.

        Raises:
            ValueError: If the parameters do not match the scheme.
        """
        if isinstance(parameters, Mapping):
            if set(parameters) != set(self._scheme):
                raise ValueError("the parameters do not match the scheme")
            for name, value in parameters.items():
                if np.shape(value) != self._scheme[name]:
                    raise ValueError(
                        f"parameter {name!r} does not match the scheme shape"
                    )
            return np.concatenate(
                [
                    np.asarray(parameters[name], dtype=self.dtype).ravel()
                    for name in self._scheme
                ]
            )
        # np.array copies, so the vector owns its data (sized right on the wire)
        flat = np.array(np.ravel(parameters), dtype=self.dtype)
        if len(flat) != self._total:
            raise ValueError(f"expected {self._total} parameters, got {len(flat)}")
        return flat

    def _unflatten(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Split a flat vector into views shaped as the scheme.

        Args:
            flat (np.ndarray): The flat vector.

        Returns:
            dict[str, np.ndarray]: Views of the vector, keyed by name.
        """
        pieces = np.split(flat, self._offsets)
        return {
            name: piece.reshape(self._scheme[name])
            for name, piece in zip(self._scheme, pieces, strict=True)
        }

    def _read_live(self) -> dict[str, np.ndarray]:
        """Read the wrapped model, validating the hook against the scheme.

        Returns:
            dict[str, np.ndarray]: The live managed entries, cast to
            ``dtype``.

        Raises:
            ValueError: If the read hook yields different entries or
                shapes than at initialization.
        """
        entries = self._read()
        if {name: value.shape for name, value in entries.items()} != self._scheme:
            raise ValueError(
                "the read hook must yield the same entries and shapes as at "
                "initialization"
            )
        return entries

    def _read(self) -> dict[str, np.ndarray]:
        """Collect the entries yielded by the read hook.

        Returns:
            dict[str, np.ndarray]: The yielded entries, cast to ``dtype``.

        Raises:
            ValueError: If a name is yielded twice, or an entry is not
                floating point.
        """
        entries: dict[str, np.ndarray] = {}
        for name, value in self._read_parameters():
            array = np.asarray(value)
            if name in entries:
                raise ValueError(f"parameter {name!r} is yielded twice")
            if not np.issubdtype(array.dtype, np.floating):
                raise ValueError(f"parameter {name!r} is not floating point")
            entries[name] = array.astype(self.dtype, copy=False)
        return entries

    def _record(self, version: Sequence[int]) -> ModelVersion:
        """Return the latest record with the given version.

        Args:
            version (Sequence[int]): The version counters to look up.

        Returns:
            ModelVersion: The latest matching record.

        Raises:
            ValueError: If the version does not match the levels or no
                record carries it.
        """
        version = self._as_version(version)
        for record in reversed(self.versions):
            if record.version == version:
                return record
        raise ValueError(f"unknown version {version}")

    def _as_version(self, version: Sequence[int]) -> tuple[int, ...]:
        """Validate a version against the levels.

        Args:
            version (Sequence[int]): The version counters.

        Returns:
            tuple[int, ...]: The version as a tuple.

        Raises:
            ValueError: If the number of counters does not match the
                levels.
        """
        version = tuple(version)
        expected = len(self.version_levels)
        if len(version) != expected:
            raise ValueError(
                f"expected {expected} version counters, got {len(version)}"
            )
        return version

    @abstractmethod
    def _read_parameters(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield the parameters to exchange, as (name, array) pairs.

        Only the yielded entries are exchanged, and they must be the same,
        with the same shapes, on every call. Only floating-point entries are
        accepted (e.g., integer counters must be skipped), while whether to
        yield other floating-point state (e.g., BatchNorm statistics) is up
        to the subclass. The hook is first called by ``__init__``, so any
        attribute it uses must be set before calling ``super().__init__``.

        Yields:
            tuple[str, np.ndarray]: The name and the current value of a
            parameter.
        """

    @abstractmethod
    def _write_parameter(self, name: str, value: np.ndarray) -> None:
        """Write a parameter into the wrapped model.

        Any conversion to the framework format (e.g., device, dtype) is
        done here.

        Args:
            name (str): The name of the parameter, as yielded by
                ``_read_parameters``.
            value (np.ndarray): The new value of the parameter.
        """
