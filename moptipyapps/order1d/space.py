"""
An extension of the permutation space for one-dimensional ordering.

The main difference is how the `to_str` result is noted, namely that it
contains the mapping of tags to locations.

>>> def _dist(a, b):
...     return abs(a - b)
>>> def _tags(a):
...     return f"t{a}"
>>> the_instance = Instance.from_sequence_and_distance(
...     [1, 2, 3, 3, 2, 3], _dist, 2, 10, ("x", ), _tags)
>>> the_space = OrderingSpace(the_instance)
>>> the_str = the_space.to_str(np.array([0, 2, 1]))
>>> the_str.splitlines()
['0;2;1', '', 'indexZeroBased;suggestedXin01;x', '0;0;t1', '2;1;t2', \
'2;1;t2', '1;0.5;t3', '1;0.5;t3', '1;0.5;t3']
>>> print(the_space.from_str(the_str))
[0 2 1]
"""

from typing import Final

import numpy as np
from moptipy.spaces.permutations import Permutations
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.strings import float_to_str
from pycommons.types import type_error

from moptipyapps.order1d.instance import (
    _SUGGESTED_X_IN_0_1,
    _ZERO_BASED_INDEX,
    Instance,
)


class OrderingSpace(Permutations):
    """A space for one-dimensional orderings."""

    def __init__(self, instance: Instance) -> None:
        """
        Create an ordering space from an instance.

        :param instance: the instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        super().__init__(range(instance.n))
        #: the instance
        self.instance: Final[Instance] = instance

    def to_str(self, x: np.ndarray) -> str:
        """
        Convert a solution to a string.

        :param x: the permutation
        :return: the string
        """
        tags: Final[tuple[tuple[tuple[str, ...], int], ...]] \
            = self.instance.tags

        n: Final[int] = len(self.blueprint) - 1
        text: list[str] = []

        text.extend(super().to_str(x).split("\n"))
        text.append("")  # noqa

        row: list[str] = [_ZERO_BASED_INDEX, _SUGGESTED_X_IN_0_1]
        row.extend(self.instance.tag_titles)
        text.append(f"{CSV_SEPARATOR.join(row)}")

        for tag, i in tags:
            row.clear()
            row.extend((str(x[i]), float_to_str(x[i] / n)))
            row.extend(tag)
            text.append(f"{CSV_SEPARATOR.join(row)}")
        return "\n".join(text)

    def from_str(self, text: str) -> np.ndarray:
        """
        Get the string version from the given text.

        :param text: the text
        :return: the string
        """
        text = text.lstrip()
        idx: int = text.find("\n")
        if idx > 0:
            text = text[:idx]
        return super().from_str(text.rstrip())
