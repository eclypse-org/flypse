"""Module for the FederatedModel class.

FLYPSE is model-agnostic: how a model computes and trains is left to its
framework. A :class:`FederatedModel` exposes the parameters of a framework
model as numpy arrays, so that combining them (e.g., averages,
pseudogradients, robust aggregation) and measuring them (e.g., norms,
distances) is plain numpy math, while local training stays on the wrapped
model.

The entries the model exchanges, with their names and shapes, are selected
by the ``_read_parameters`` hook and fixed at initialisation. For instance,
integer counters are never yielded, while yielding BatchNorm statistics or
not is the only difference between FedAvg and FedBN. Non-parametric models
(e.g., tree ensembles) cannot conform and fail at initialisation.

The hook may also label its entries (e.g., ``"statistics"`` for BatchNorm
statistics, ``"head"`` for the layers a client keeps to itself), and ``mask``
selects the coordinates of the flat vector carrying a label: e.g., to average
the statistics without a server optimiser stepping on them, or to share only
the body of the model.

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
    Literal,
    overload,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
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

    Example:
        .. code-block:: python

            class MeanModel(FederatedModel):
                def _read_parameters(self):
                    yield "mean", self.model["mean"]

                def _write_parameter(self, name, value):
                    self.model[name] = value


            model = MeanModel({"mean": np.zeros(1)})
    """

    def __init__(
        self,
        model: Any,
        dtype: DTypeLike = np.float32,
        version_levels: Sequence[str] = ("global", "local"),
        max_versions: int | Mapping[str, int] | None = None,
    ) -> None:
        """Initialise the federated model.

        The entries yielded by ``_read_parameters`` are read once, fixing
        their names, order and shapes, and the initial parameters are
        recorded as the zero version.

        Args:
            model (Any): The framework model to wrap.
            dtype (DTypeLike): The dtype of the parameter arrays. Defaults
                to ``np.float32``.
            version_levels (Sequence[str]): The version levels, from the
                coarsest to the finest. Defaults to ``("global", "local")``.
            max_versions (int | Mapping[str, int] | None): The number of
                records kept in the history, dropping the oldest first: in
                all, or per level as a mapping (e.g., ``{"local": 1}``), where
                the levels left out keep every record. The level of a record
                is the finest one with a non-zero counter. Defaults to None
                (unlimited).

        Raises:
            ValueError: If the wrapped model exposes no parameters, the
                read hook yields a name twice or a non-float entry, the
                levels are empty or duplicated, or ``max_versions`` is
                less than 1 or names an unknown level.
        """
        self.model = model
        """The wrapped framework model."""
        self.dtype = np.dtype(dtype)
        """The dtype of the parameter arrays."""
        levels = tuple(version_levels)
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("the version_levels must be non-empty and unique")
        limits = dict(max_versions) if isinstance(max_versions, Mapping) else {}
        total = None if isinstance(max_versions, Mapping) else max_versions
        if (total is not None and total < 1) or any(n < 1 for n in limits.values()):
            raise ValueError("max_versions must be at least 1")
        if not set(limits) <= set(levels):
            raise ValueError(f"max_versions names unknown levels: {limits}")
        self._limits = limits
        self.version_levels = levels
        """The version levels, from the coarsest to the finest."""
        entries, self._labels = self._read()
        if not entries:
            raise ValueError(
                f"{type(model).__name__} exposes no parameters to exchange"
            )
        self._scheme = {name: value.shape for name, value in entries.items()}
        sizes = [value.size for value in entries.values()]
        self._offsets = np.cumsum(sizes)[:-1]
        self._total = sum(sizes)
        self.version = (0,) * len(self.version_levels)
        """The current version: one counter per level."""
        flat = np.concatenate([value.ravel() for value in entries.values()])
        flat.flags.writeable = False
        self.versions = deque([ModelVersion(self.version, time(), flat)], maxlen=total)
        """The version history, oldest first."""

    @overload
    def get_parameters(
        self,
        flattened: Literal[True] = ...,
        version: Sequence[int] | ModelVersion | None = ...,
    ) -> np.ndarray: ...

    @overload
    def get_parameters(
        self,
        flattened: Literal[False],
        version: Sequence[int] | ModelVersion | None = ...,
    ) -> dict[str, np.ndarray]: ...

    def get_parameters(
        self,
        flattened: bool = True,
        version: Sequence[int] | ModelVersion | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Get a copy of the parameters of the model.

        Args:
            flattened (bool): Whether to return a single flat vector rather
                than a dictionary of arrays keyed by name. Defaults to True.
            version (Sequence[int] | ModelVersion | None): The version of the
                history to read; when several records share it, the latest
                is read. A record is read as it is, in or out of the history
                (e.g., a received version that was not loaded). Defaults to
                None, meaning the current parameters of the wrapped model,
                including any local training.

        Returns:
            np.ndarray | dict[str, np.ndarray]: The parameters, in the
            requested form.

        Raises:
            ValueError: If the version is not in the history, or the
                entries read from the model differ from the ones fixed at
                initialisation.
        """
        if version is None:
            entries = self._read_live()
            if flattened:
                return np.concatenate([entries[name].ravel() for name in self._scheme])
            return {name: entries[name].copy() for name in self._scheme}
        record = version if isinstance(version, ModelVersion) else self._record(version)
        flat = record.parameters
        if flattened:
            return flat.copy()
        return {name: piece.copy() for name, piece in self._unflatten(flat).items()}

    def unflatten(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Split a flat vector into arrays shaped as the parameters.

        The vector can be any one over the coordinates of the parameters
        (e.g., a gradient correction, or a ``mask``), of any dtype.

        Args:
            flat (np.ndarray): The flat vector.

        Returns:
            dict[str, np.ndarray]: Views of the vector, keyed by name.

        Raises:
            ValueError: If the vector does not have one value per coordinate.
        """
        if np.shape(flat) != (self._total,):
            raise ValueError(f"expected {self._total} values, got {np.shape(flat)}")
        return self._unflatten(flat)

    def mask(self, *labels: str) -> np.ndarray:
        """Select the coordinates of the entries carrying any of the labels.

        Args:
            *labels (str): The labels, as yielded by ``_read_parameters``.

        Returns:
            np.ndarray: A boolean flat vector, True on the selected
            coordinates.

        Raises:
            ValueError: If no entry carries one of the labels.

        Example:
            .. code-block:: python

                statistics = self.model.mask("statistics")
                new = np.where(statistics, average, optimiser.step(own, delta))
        """
        carried = set().union(*self._labels.values())
        unknown = [label for label in labels if label not in carried]
        if unknown:
            raise ValueError(f"no entry carries the labels {unknown}")
        return np.concatenate(
            [
                np.full(int(np.prod(shape)), not self._labels[name].isdisjoint(labels))
                for name, shape in self._scheme.items()
            ]
        )

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

        Example:
            .. code-block:: python

                await self.bcast(self.model.update_version(average, "global"))
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

    def load_version(
        self, record: ModelVersion, mask: np.ndarray | None = None
    ) -> ModelVersion:
        """Load a version generated by another entity.

        It is used by the entity receiving a model: the parameters of the
        record are written into the wrapped model, and its version and
        timestamp are adopted as they are and recorded in the history. With
        a mask, only the selected coordinates are written, while the others
        keep their current value (e.g., a personal head), and the version is
        adopted all the same.

        Args:
            record (ModelVersion): The received version.
            mask (np.ndarray | None): The coordinates to write, as a boolean
                flat vector (see ``mask``). Defaults to None, meaning all.

        Returns:
            ModelVersion: The recorded copy of the version, with the
            parameters of the model once loaded.

        Raises:
            ValueError: If the version does not match the levels, or the
                parameters or the mask do not match the ones of the model.

        Example:
            .. code-block:: python

                shared = ~self.model.mask("personal")
                self.model.load_version(message.record, mask=shared)
        """
        flat = self._validate(record.parameters)
        if mask is not None:
            if np.shape(mask) != (self._total,):
                raise ValueError(f"expected a mask of {self._total} values")
            flat = np.where(mask, flat, self.get_parameters())
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
        level = self._level(version)
        if level in self._limits:
            same = [
                i
                for i, r in enumerate(self.versions)
                if self._level(r.version) == level
            ]
            if len(same) > self._limits[level]:
                del self.versions[same[0]]
        return record

    def _level(self, version: tuple[int, ...]) -> str:
        """Return the level of a version: its finest non-zero counter.

        Args:
            version (tuple[int, ...]): The version counters.

        Returns:
            str: The level, the coarsest one for the zero version.
        """
        nonzero = [i for i, counter in enumerate(version) if counter]
        return self.version_levels[nonzero[-1] if nonzero else 0]

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
                shapes than at initialisation.
        """
        entries, _ = self._read()
        if {name: value.shape for name, value in entries.items()} != self._scheme:
            raise ValueError(
                "the read hook must yield the same entries and shapes as at "
                "initialization"
            )
        return entries

    def _read(self) -> tuple[dict[str, np.ndarray], dict[str, frozenset[str]]]:
        """Collect the entries yielded by the read hook, with their labels.

        Returns:
            tuple[dict[str, np.ndarray], dict[str, frozenset[str]]]: The
            yielded entries, cast to ``dtype``, and their labels, keyed by
            name.

        Raises:
            ValueError: If a name is yielded twice, or an entry is not
                floating point.
        """
        entries: dict[str, np.ndarray] = {}
        labels: dict[str, frozenset[str]] = {}
        for name, value, *label in self._read_parameters():
            array = np.asarray(value)
            if name in entries:
                raise ValueError(f"parameter {name!r} is yielded twice")
            if not np.issubdtype(array.dtype, np.floating):
                raise ValueError(f"parameter {name!r} is not floating point")
            entries[name] = array.astype(self.dtype, copy=False)
            carried = label[0] if label else ()
            labels[name] = frozenset([carried] if isinstance(carried, str) else carried)
        return entries, labels

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
        # copy first: the entity may append while a remote metric looks up
        for record in reversed(list(self.versions)):
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
    def _read_parameters(
        self,
    ) -> Iterator[tuple[str, np.ndarray] | tuple[str, np.ndarray, str | Iterable[str]]]:
        """Yield the parameters to exchange, as (name, array) pairs.

        Only the yielded entries are exchanged, and they must be the same,
        with the same shapes, on every call. Only floating-point entries are
        accepted (e.g., integer counters must be skipped), while whether to
        yield other floating-point state (e.g., BatchNorm statistics) is up
        to the subclass. The hook is first called by ``__init__``, so any
        attribute it uses must be set before calling ``super().__init__``.

        An entry may carry a label, or several, as a third element: the
        labels are fixed at initialisation too, and ``mask`` selects by them.

        Yields:
            tuple[str, np.ndarray] | tuple[str, np.ndarray, str | Iterable[str]]:
            The name and the current value of a parameter, and optionally
            its labels.

        Example:
            .. code-block:: python

                def _read_parameters(self):
                    for name, value in self.model.items():
                        label = "statistics" if name.startswith("bn.") else ()
                        yield name, value, label
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
