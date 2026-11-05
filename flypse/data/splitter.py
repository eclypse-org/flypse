"""Module for the ViewSplitter and FractionSplitter classes.

A :class:`ViewSplitter` splits the data view of an entity into named parts
(e.g., train, eval and test), each a :class:`~flypse.data.view.DataView` of
its parent view: a part refers to the view it was split from, which in turn
refers to the data of the federation. How the positions of the view are
assigned to the parts is defined by subclasses, implementing ``split``.

:class:`FractionSplitter` is the default splitter: it sizes the parts by
fractions, optionally stratifying by class or following the order of the
items (e.g., by time, so that no part precedes the earlier ones).
"""

from __future__ import annotations

import math
import random
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from flypse.data.dirichlet import _allocate
from flypse.data.view import DataView
from flypse.seed import get_seed

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Mapping,
        Sequence,
    )


class ViewSplitter(ABC):
    """A splitter of a data view into named parts."""

    def apply(self, view: DataView) -> dict[str, DataView]:
        """Split a data view into named parts.

        Args:
            view (DataView): The view to split.

        Returns:
            dict[str, DataView]: The parts, keyed by name, each a view of
            the given one.
        """
        return {
            name: DataView(view, positions)
            for name, positions in self.split(view).items()
        }

    @abstractmethod
    def split(self, view: DataView) -> dict[str, Sequence[int]]:
        """Compute the positions of the view assigned to each part.

        Args:
            view (DataView): The view to split.

        Returns:
            dict[str, Sequence[int]]: The positions within the view of each
            part, keyed by name.
        """


class FractionSplitter(ViewSplitter):
    """A splitter sizing the parts of a view by fractions.

    The items are shuffled, unless ``order`` is given: then they are sorted by
    it, and the first parts get the earliest items. With ``stratify``, the
    items are split class by class, so that each part keeps the class
    proportions of the view. A part can be empty when the view is too small.

    Example:
        .. code-block:: python

            splitter = FractionSplitter({"train": 0.8, "test": 0.2}, seed=self.rng)
            self.splits = splitter.apply(self.data)
    """

    def __init__(
        self,
        parts: Mapping[str, float],
        stratify: Callable[[Any], Hashable] | None = None,
        order: Callable[[Any], Any] | None = None,
        seed: int | random.Random | None = None,
    ):
        """Initialize the splitter.

        Args:
            parts (Mapping[str, float]): The fraction of each part, keyed by
                name, in the order the parts are filled.
            stratify (Callable[[Any], Hashable] | None): A function returning
                the class of an item. Defaults to None, meaning no
                stratification.
            order (Callable[[Any], Any] | None): A function returning the
                sorting key of an item (e.g., its timestamp). Defaults to
                None, meaning the items are shuffled.
            seed (int | random.Random | None): The seed of the shuffling, or
                the generator to shuffle with (e.g., the ``rng`` of an
                entity). Defaults to None, meaning the seed of the process.

        Raises:
            ValueError: If there are no parts, or the fractions are not
                positive or do not sum to 1.
        """
        if not parts:
            raise ValueError("At least one part is required.")
        if any(fraction <= 0 for fraction in parts.values()):
            raise ValueError("The fractions of the parts must be positive.")
        if not math.isclose(sum(parts.values()), 1.0):
            raise ValueError("The fractions of the parts must sum to 1.")
        self.parts = dict(parts)
        """The fraction of each part, keyed by name."""
        self.stratify = stratify
        """The function returning the class of an item, if any."""
        self.order = order
        """The function returning the sorting key of an item, if any."""
        self.seed = seed
        """The seed of the shuffling, or the generator to shuffle with."""

    def split(self, view: DataView) -> dict[str, Sequence[int]]:
        """Assign contiguous slices of each group of positions to the parts.

        Args:
            view (DataView): The view to split.

        Returns:
            dict[str, Sequence[int]]: The positions within the view of each
            part, keyed by name.
        """
        groups: dict[Hashable, list[int]] = {}
        for position in range(len(view)):
            group = None if self.stratify is None else self.stratify(view[position])
            groups.setdefault(group, []).append(position)

        order = self.order
        keys = {} if order is None else {p: order(view[p]) for p in range(len(view))}
        rng = (
            self.seed
            if isinstance(self.seed, random.Random)
            else random.Random(get_seed(self.seed))
        )
        fractions = list(self.parts.values())
        result: dict[str, list[int]] = {name: [] for name in self.parts}
        for positions in groups.values():
            if order is None:
                rng.shuffle(positions)
            else:
                positions.sort(key=keys.__getitem__)
            start = 0
            for name, count in zip(
                self.parts, _allocate(fractions, len(positions)), strict=True
            ):
                result[name].extend(positions[start : start + count])
                start += count
        if order is not None:
            for positions in result.values():
                positions.sort(key=keys.__getitem__)
        return cast("dict[str, Sequence[int]]", result)
