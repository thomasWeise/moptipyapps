"""
An objective function for minimizing the number of bins of packings.

This function returns the number of bins.
"""
from typing import Final

from moptipy.api.objective import Objective
from moptipy.utils.types import type_error

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.packing import IDX_BIN

#: the name of the bin count objective function
BIN_COUNT_NAME: Final[str] = "binCount"


class BinCount(Objective):
    """Compute the number of bins."""

    def __init__(self, instance: Instance) -> None:  # +book
        """
        Initialize the number of bins objective function.

        :param instance: the instance to load the bounds from
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the internal instance reference
        self.__instance: Final[Instance] = instance

    def evaluate(self, x) -> int:
        """
        Get the number of bins.

        :param x: the packing
        :return: the number of bins used
        """
        return int(x[:, IDX_BIN].max())

    def lower_bound(self) -> int:
        """
        Get the lower bound of the number of bins objective.

        :return: the lower bound for the number of required bins, i.e.,
            :attr:`~moptipyapps.binpacking2d.instance.Instance.\
lower_bound_bins`

        >>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 1]])
        >>> ins.lower_bound_bins
        1
        >>> BinCount(ins).lower_bound()
        1

        >>> ins = Instance("b", 10, 50, [[10, 5, 10], [3, 3, 1], [5, 5, 1]])
        >>> ins.lower_bound_bins
        2
        >>> BinCount(ins).lower_bound()
        2

        >>> ins = Instance("c", 10, 50, [[10, 5, 20], [30, 3, 10], [5, 5, 1]])
        >>> ins.lower_bound_bins
        4
        >>> BinCount(ins).lower_bound()
        4
        """
        return self.__instance.lower_bound_bins

    def is_always_integer(self) -> bool:
        """
        Return `True` because there are only integer bins.

        :retval True: always
        """
        return True

    def upper_bound(self) -> int:
        """
        Get the upper bound of the number of bins plus emptiness.

        :return: the number of items in the instance, i.e.,
            :attr:`~moptipyapps.binpacking2d.instance.Instance.n_items`

        >>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 1]])
        >>> ins.n_items
        3
        >>> BinCount(ins).upper_bound()
        3

        >>> ins = Instance("b", 10, 50, [[10, 5, 10], [3, 3, 1], [5, 5, 1]])
        >>> ins.n_items
        12
        >>> BinCount(ins).upper_bound()
        12

        >>> ins = Instance("c", 10, 50, [[10, 5, 20], [30, 3, 10], [5, 5, 1]])
        >>> ins.n_items
        31
        >>> BinCount(ins).upper_bound()
        31
        """
        return self.__instance.n_items

    def __str__(self) -> str:
        """
        Get the name of the bins objective function.

        :return: `binCount`
        :retval "binCount": always
        """
        return BIN_COUNT_NAME
