"""
An extension of the permutation space for one-dimensional ordering.

The main difference is how the `to_str` result is noted.

>>> def _dist(a, b):
...     return abs(a - b)
>>> def _tags(a):
...     return f"t{a}"
>>> the_instance = Instance.from_sequence_and_distance(
...     [1, 2, 3, 3, 2, 3], _tags, _dist)
>>> the_space = OrderingSpace(the_instance)
>>> the_str = the_space.to_str(np.array([0, 2, 1]))
>>> the_str.splitlines()
['0;2;1', '', 'indexZeroBased;suggestedXin01;tag0', '0;0.25;t1', \
'2;0.75;t2', '2;0.75;t2', '1;0.5;t3', '1;0.5;t3', '1;0.5;t3']
>>> print(the_space.from_str(the_str))
[0 2 1]
"""

from io import StringIO
from typing import Final

import numpy as np
from moptipy.spaces.permutations import Permutations
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.strings import float_to_str
from moptipy.utils.types import type_error

from moptipyapps.order1d.instance import Instance


class OrderingSpace(Permutations):
    """A space for one-dimensional orderings."""

    def __init__(self, instance: Instance) -> None:
        """
        Create an ordering space from an instance.

        :param instance: the instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        super().__init__(range(len(instance)))
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

        total: Final[int] = len(x) + 1

        with StringIO() as sio:
            sio.writelines(super().to_str(x))
            sio.write("\n\n")

            row: list[str] = ["indexZeroBased", "suggestedXin01"]
            row.extend(f"tag{i}" for i in range(len(tags[0][0])))
            sio.write(CSV_SEPARATOR.join(row))
            sio.write("\n")

            for tag, i in tags:
                row.clear()
                row.append(str(x[i]))
                row.append(float_to_str((x[i] + 1) / total))
                row.extend(tag)
                sio.write(CSV_SEPARATOR.join(row))
                sio.write("\n")
            return sio.getvalue()

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
