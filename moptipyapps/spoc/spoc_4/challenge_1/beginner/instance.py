"""
The instance of the Luna Tomato Logistics beginner problem.

>>> inst1 = Instance("matching-i")
>>> inst1.n
25000
>>> inst1.shape
(25000, 4)
>>> list(map(int, inst1[0, :]))
[3390, 4454, 3664, 267]
>>> inst1[0, 1]
np.int64(4454)
>>> inst1.penalty
125526621
>>> inst1.lengths
(5000, 5000, 5000)
>>> inst1.name
'matching-i'
>>> inst1 is Instance("matching-i")
True

>>> inst2 = Instance("matching-ii")
>>> inst2.n
92103
>>> inst2.shape
(92103, 4)
>>> list(map(int, inst2[0, :]))
[5559, 5444, 3794, 9723]
>>> inst2[0, 1]
np.int64(5444)
>>> inst2.penalty
460191177
>>> inst2.lengths
(10000, 10000, 10000)
>>> inst2.name
'matching-ii'
"""

from typing import Any, Callable, Final, Generator, cast

import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import int_range_to_dtype
from pycommons.types import check_to_int_range

from moptipyapps.spoc.spoc_4.challenge_1.beginner.data import (
    open_resource_stream,
)


class Instance(np.ndarray, Component):
    """The instances of the Luna Tomato Logistics beginner problem."""

    #: set the result name
    name: str
    #: set the penalty
    penalty: int
    #: the number of orbit pairs
    n: int
    #: the number of values per column
    lengths: tuple[int, int, int]

    def __new__(cls, name: str) -> "Instance":  # noqa: PYI034
        """Create the single instance."""
        name = str.strip(name)
        if str.__len__(name) <= 0:
            raise ValueError("Invalid name.")
        meth: Final = Instance.__new__
        attr: Final[str] = f"__{name}_data"
        if hasattr(meth, attr):
            return cast("Instance", getattr(meth, attr))

        data: list[list] = []
        power: int = 0
        with open_resource_stream(f"{name}.txt") as stream:
            for oline in stream:
                line = str.strip(oline)
                if str.__len__(line) <= 0:
                    continue
                splt: list[str] = line.split()
                if list.__len__(splt) != 4:
                    raise ValueError(f"Invalid format: {oline!r}.")
                cap: str = splt[3]
                data.append([
                    check_to_int_range(splt[0], "e") - 1,
                    check_to_int_range(splt[1], "l") - 1,
                    check_to_int_range(splt[2], "d") - 1,
                    cap])
                li: int = cap.rfind(".")
                if li >= 0:
                    power = max(power, len(cap) - li)

        power -= 1
        if power != 3:
            raise ValueError(f"Power must be 3, but is {power}.")
        power = 10 ** power
        for row in data:
            row[-1] = round(float(row[-1]) * power)
            if any(r < 0 for r in row):
                raise ValueError(f"Invalid row: {row}.")

        penalty: Final[int] = sum(r[3] for r in data) + 1
        if penalty <= 1:
            raise ValueError(
                f"Maximum must be positive, but is {penalty - 1}.")
        lengths: Final[tuple[int, int, int]] = (
            max(r[0] for r in data) + 1,
            max(r[1] for r in data) + 1,
            max(r[2] for r in data) + 1)

        use_shape: tuple[int, int] = (len(data), 4)
        dt: np.dtype = int_range_to_dtype(
            0, (3 * penalty * use_shape[0]) + penalty - 1)

        result: Instance = super().__new__(Instance, use_shape, dt)

        for i, row in enumerate(data):
            for j, val in enumerate(row):
                result[i, j] = val
                if val != result[i, j]:
                    raise ValueError("Error during conversion.")

        #: set the result name
        result.name = name
        #: set the penalty
        result.penalty = penalty
        #: the number of orbit pairs
        result.n = use_shape[0]
        #: the number of values per column
        result.lengths = lengths
        setattr(meth, attr, result)
        return result

    def __str__(self) -> str:
        """
        Get the name of this instance.

        :returns: the name of this instance.
        """
        return self.name

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("penalty", self.penalty)
        logger.key_value("lengths", ";".join(map(str, self.lengths)))

    @staticmethod
    def list_resources() -> tuple[str, str]:
        """
        Get all the beginner problem instances.

        :return: the problem instance names

        >>> Instance.list_resources()
        ('matching-i', 'matching-ii')

        >>> for ix in Instance.list_resources():
        ...     print(Instance(ix).name)
        matching-i
        matching-ii
        """
        return ("matching-i", "matching-ii")

    @staticmethod
    def list_instances() -> Generator[Callable[[], Any], None, None]:
        """
        Get an iterable of all instances.

        :return: the iterable

        >>> for ix in Instance.list_instances():
        ...     print(ix().name)
        matching-i
        matching-ii
        """
        yield lambda: Instance("matching-i")
        yield lambda: Instance("matching-ii")
