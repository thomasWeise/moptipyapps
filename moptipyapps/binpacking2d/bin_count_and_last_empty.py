"""
An objective function for minimizing the number of bins of packings.

This objective function first computes the number of bins used. Let's call it
`n_bins`. We know the total number of items,
:attr:`~moptipyapps.binpacking2d.instance.Instance.n_items`, as
well (because this is also the number of rows in the packing).
Now we return `(n_items * (n_bins - 1)) + number_of_items_in_last_bin`,
where `number_of_items_in_last_bin` is, well, the number of items in the
very last bin.

The idea behind this is: If one of two packings has the smaller number of
bins, then this one will always have the smaller objective value. If two
packings have the same number of bins, but one has fewer items in the very
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
from moptipyapps.binpacking2d.packing import IDX_BIN


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def bin_count_and_last_empty(y: np.ndarray) -> int:
    """
    Compute the number of bins and the emptiness of the last bin.

    We compute the total number of bins minus 1 and multiply it with the
    number of items. We then add the number of items in the last bin.

    :param y: the packing
    :return: the objective value

    >>> bin_count_and_last_empty(np.array([[1, 1, 10, 10, 20, 20],
    ...                                    [1, 1, 30, 30, 40, 40],
    ...                                    [1, 1, 20, 20, 30, 30]], int))
    3
    >>> bin_count_and_last_empty(np.array([[1, 1, 10, 10, 20, 20],
    ...                                    [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                                    [1, 1, 20, 20, 30, 30]], int))
    4
    >>> bin_count_and_last_empty(np.array([[1, 2, 10, 10, 20, 20],  # bin 2!
    ...                                    [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                                    [1, 1, 20, 20, 30, 30]], int))
    5
    >>> bin_count_and_last_empty(np.array([[1, 3, 10, 10, 20, 20],  # bin 3!
    ...                                    [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                                    [1, 1, 20, 20, 30, 30]], int))
    7
    """
    current_bin: int = -1  # the current idea of what the last bin is
    current_size: int = -1  # the number of items already in that bin
    n_items: Final[int] = len(y)  # the number of rows in the matrix

    for i in range(n_items):  # iterate over all packed items
        bin_idx: int = y[i, IDX_BIN]  # get the bin index of the item
        if bin_idx > current_bin:  # it's a new biggest bin = new last bin?
            current_size = 1  # then there is 1 object in it for now
            current_bin = bin_idx  # and we remember it
        elif bin_idx == current_bin:  # did item go into the current last bin?
            current_size = current_size + 1  # then increase size
    return (n_items * (current_bin - 1)) + current_size  # return objective


class BinCountAndLastEmpty(Objective):
    """Compute the number of bins and the emptiness of the last one."""

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
        self.evaluate = bin_count_and_last_empty  # type: ignore

    def lower_bound(self) -> int:
        """
        Get the lower bound of the number of bins and emptiness objective.

        We know from the instance (:attr:`~moptipyapps.binpacking2d\
.instance.Instance.lower_bound_bins`) that we require at least as many bins
        such that they can accommodate the total area of all items together.
        Let's call this number `lb`. Now if `lb` is one, then all objects could
        be in the first bin, in which case the objective value would equal to
        :attr:`~moptipyapps.binpacking2d.instance.Instance.n_items`,
        i.e., the total number of items in the first = last bin.
        If it is `lb=2`, then we know that we will need at least two bins. The
        best case would be that `n_items - 1` items are in the first bin and
        one is in the last bin. This means that we would get `1 * n_items + 1`
        as objective value. If we have `lb=3` bins, then we could have
        `n_items - 1` items distributed over the first two bins with one item
        left over in the last bin, i.e., would get `(2 * n_items) + 1`. And so
        on.

        :return: `max(n_items, (lb - 1) * n_items + 1)`

        >>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 1]])
        >>> ins.n_items
        3
        >>> ins.lower_bound_bins
        1
        >>> BinCountAndLastEmpty(ins).lower_bound()
        3

        >>> ins = Instance("b", 10, 50, [[10, 5, 10], [3, 3, 1], [5, 5, 1]])
        >>> ins.n_items
        12
        >>> ins.lower_bound_bins
        2
        >>> BinCountAndLastEmpty(ins).lower_bound()
        13

        >>> ins = Instance("c", 10, 50, [[10, 5, 20], [30, 3, 10], [5, 5, 1]])
        >>> ins.n_items
        31
        >>> ins.lower_bound_bins
        4
        >>> BinCountAndLastEmpty(ins).lower_bound()
        94
        """
        return max(self.__instance.n_items,
                   ((self.__instance.lower_bound_bins - 1)
                    * self.__instance.n_items) + 1)

    def is_always_integer(self) -> bool:
        """
        Return `True` because there are only integer bins.

        :retval True: always
        """
        return True

    def upper_bound(self) -> int:
        """
        Get the upper bound of the number of bins plus emptiness.

        :return: the number of items in the instance to the square

        >>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 1]])
        >>> ins.n_items
        3
        >>> BinCountAndLastEmpty(ins).upper_bound()
        9

        >>> ins = Instance("b", 10, 50, [[10, 5, 10], [3, 3, 1], [5, 5, 1]])
        >>> ins.n_items
        12
        >>> BinCountAndLastEmpty(ins).upper_bound()
        144

        >>> ins = Instance("c", 10, 50, [[10, 5, 20], [30, 3, 10], [5, 5, 1]])
        >>> ins.n_items
        31
        >>> BinCountAndLastEmpty(ins).upper_bound()
        961
        """
        return self.__instance.n_items * self.__instance.n_items

    def __str__(self) -> str:
        """
        Get the name of the bins objective function.

        :return: `binCountAndLastEmpty`
        :retval "binCountAndLastEmpty": always
        """
        return "binCountAndLastEmpty"
