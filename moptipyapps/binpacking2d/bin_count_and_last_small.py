"""
An objective function for minimizing the number of bins of packings.

This objective function first computes the number of bins used. Let's call it
`n_bins`. We know the area bin_area of a bin as well.
Now we return `(bin_area * (n_bins - 1)) + area_of_items_in_last_bin`,
where `area_of_items_in_last_bin` is, well, the area covered by items in the
very last bin.

The idea behind this is: If one of two packings has the smaller number of
bins, then this one will always have the smaller objective value. If two
packings have the same number of bins, but one requires less space in the very
last bin, then that one is better. With this mechanism, we drive the search
towards "emptying" the last bin. If the number of items in the last bin would
reach `0`, that last bin would disappear - and we have one bin less.
"""
from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.types import type_error

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.packing import (
    IDX_BIN,
    IDX_BOTTOM_Y,
    IDX_LEFT_X,
    IDX_RIGHT_X,
    IDX_TOP_Y,
)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def bin_count_and_last_small(y: np.ndarray, bin_area: int) -> int:
    """
    Compute the number of bins and the occupied area in the last bin.

    We compute the total number of bins minus 1 and multiply it with the
    total area of items. We then add the area of items in the last bin.

    :param y: the packing
    :param bin_area: the area of a single bin
    :return: the objective value

    >>> bin_count_and_last_small(np.array([[1, 1, 10, 10, 20, 20],
    ...                                    [1, 1, 30, 30, 40, 40],
    ...                                    [1, 1, 20, 20, 30, 30]], int),
    ...                                    50*50)
    300
    >>> bin_count_and_last_small(np.array([[1, 1, 10, 10, 20, 20],
    ...                                    [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                                    [1, 1, 20, 20, 30, 30]], int),
    ...                                    50*50)
    2600
    >>> bin_count_and_last_small(np.array([[1, 2, 10, 10, 20, 20],  # bin 2!
    ...                                    [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                                    [1, 1, 20, 20, 30, 30]], int),
    ...                                    50*50)
    2700
    >>> bin_count_and_last_small(np.array([[1, 3, 10, 10, 20, 20],  # bin 3!
    ...                                    [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                                    [1, 1, 20, 20, 30, 30]], int),
    ...                                    50*50)
    5100
    >>> bin_count_and_last_small(np.array([[1, 3, 10, 10, 50, 50],  # bin 3!
    ...                                    [1, 2, 30, 30, 60, 60],  # bin 2!
    ...                                    [1, 1, 20, 20, 30, 30]], np.int8),
    ...                                    50*50)
    6600
    """
    current_bin: int = -1  # the current idea of what the last bin is
    current_area: int = 0  # the area of items already in that bin
    n_items: Final[int] = len(y)  # the number of rows in the matrix

    for i in range(n_items):  # iterate over all packed items
        bin_idx: int = y[i, IDX_BIN]  # get the bin index of the item
        if bin_idx < current_bin:
            continue
        area: int = int(y[i, IDX_RIGHT_X] - y[i, IDX_LEFT_X]) \
            * int(y[i, IDX_TOP_Y] - y[i, IDX_BOTTOM_Y])
        if bin_idx > current_bin:  # it's a new biggest bin = new last bin?
            current_area = area  # then the current area is this
            current_bin = bin_idx  # and we remember it
        elif bin_idx == current_bin:  # did item go into the current last bin?
            current_area += area  # then increase size
    return (bin_area * (current_bin - 1)) + current_area  # return objective


class BinCountAndLastSmall(Objective):
    """Compute the number of bins and the area in the last one."""

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
        #: the bin size
        self.__bin_size: Final[int] = instance.bin_width * instance.bin_height

    def evaluate(self, x) -> int:
        """
        Evaluate the objective function.

        :param x: the solution
        :return: the bin size and last-bin-small-area factor
        """
        return bin_count_and_last_small(x, self.__bin_size)

    def lower_bound(self) -> int:
        """
        Get the lower bound of the number of bins and emptiness objective.

        We know from the instance (:attr:`~moptipyapps.binpacking2d\
.instance.Instance.lower_bound_bins`) that we require at least as many bins
        such that they can accommodate the total area of all items together.
        Let's call this number `lb`. Now if `lb` is one, then all objects could
        be in the first bin, in which case the objective value would equal to
        the total area of all items (:attr:`~moptipyapps.binpacking2d\
.instance.Instance.total_item_area`).
        If it is `lb=2`, then we know that we will need at least two bins. The
        best case would be that almost all items are in the first bin and
        only the smallest object is in the last bin. This means that we would
        get `1 * bin_area + smallest_area` as objective value. If we have
        `lb=3` bins, then we could again have all but the smallest items
        distributed over the first two bins and only the smallest one in the
        last bin, i.e., would get `(2 * bin_area) + smallest_area`. And so on.

        :return: `total_item_area` if the lower bound `lb` of the number of
            bins is `1`, else `(lb - 1) * bin_area + smallest_area`, where
            `bin_area` is the area of a bin, `total_item_area` is the area of
            all items added up, and `smallest_area` is the area of the
            smallest item

        >>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 1]])
        >>> ins.total_item_area
        84
        >>> ins.lower_bound_bins
        1
        >>> BinCountAndLastSmall(ins).lower_bound()
        84

        >>> ins = Instance("b", 10, 50, [[10, 5, 10], [3, 3, 1], [5, 5, 1]])
        >>> ins.total_item_area
        534
        >>> ins.lower_bound_bins
        2
        >>> BinCountAndLastSmall(ins).lower_bound()
        509

        >>> ins = Instance("c", 10, 50, [[10, 5, 10], [30, 3, 1], [5, 5, 1]])
        >>> ins.total_item_area
        615
        >>> ins.lower_bound_bins
        2
        >>> BinCountAndLastSmall(ins).lower_bound()
        525
        """
        if self.__instance.lower_bound_bins == 1:
            return self.__instance.total_item_area
        smallest_area: int = -1
        for row in self.__instance:
            area: int = int(row[0]) * int(row[1])
            if (smallest_area < 0) or (area < smallest_area):
                smallest_area = area
        return int(((self.__instance.lower_bound_bins - 1)
                    * self.__instance.bin_height
                    * self.__instance.bin_width) + smallest_area)

    def is_always_integer(self) -> bool:
        """
        Return `True` because there are only integer bins.

        :retval True: always
        """
        return True

    def upper_bound(self) -> int:
        """
        Get the upper bound of this objective function.

        :return: a very coarse estimate of the upper bound

        >>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 1]])
        >>> ins.n_items
        3
        >>> BinCountAndLastSmall(ins).upper_bound()
        15000

        >>> ins = Instance("b", 10, 50, [[10, 5, 10], [3, 3, 1], [5, 5, 1]])
        >>> ins.n_items
        12
        >>> BinCountAndLastSmall(ins).upper_bound()
        6000

        >>> ins = Instance("c", 10, 50, [[10, 5, 10], [30, 3, 1], [5, 5, 10]])
        >>> ins.n_items
        21
        >>> BinCountAndLastSmall(ins).upper_bound()
        10500
        """
        return self.__instance.n_items * self.__instance.bin_height \
            * self.__instance.bin_width

    def __str__(self) -> str:
        """
        Get the name of the bins objective function.

        :return: `binCountAndLastSmall`
        :retval "binCountAndLastSmall": always
        """
        return "binCountAndLastSmall"
