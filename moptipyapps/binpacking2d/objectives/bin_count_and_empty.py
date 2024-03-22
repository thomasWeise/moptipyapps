"""
An objective function indirectly minimizing the number of bins in packings.

This objective function first computes the number of bins used. Let's call it
`n_bins`. We know the total number of items,
:attr:`~moptipyapps.binpacking2d.instance.Instance.n_items`, as
well (because this is also the number of rows in the packing).
Now we return `(n_items * (n_bins - 1)) + min_items`,
where `min_items` is the number of items in the bin with the fewest items.

This is similar to :mod:`~moptipyapps.binpacking2d.objectives.\
bin_count_and_last_small`, but instead of focussing on the very last bin, it
uses the minimum element count over all bins. It has the same lower- and upper
bound, though.
"""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.objectives.bin_count_and_last_empty import (
    BinCountAndLastEmpty,
)
from moptipyapps.binpacking2d.packing import IDX_BIN


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def bin_count_and_empty(y: np.ndarray, temp: np.ndarray) -> int:
    """
    Get the number of bins and number of elements in the emptiest one.

    We compute the total number of bins minus 1 and multiply it with the
    number of items. We then add the smallest number of elements in any
    bin.

    :param y: the packing
    :param temp: the temporary array
    :return: the objective value

    >>> tempa = np.empty(10, int)
    >>> bin_count_and_empty(np.array([[1, 1, 10, 10, 20, 20],
    ...                               [1, 1, 30, 30, 40, 40],
    ...                               [1, 1, 20, 20, 30, 30]], int), tempa)
    3
    >>> bin_count_and_empty(np.array([[1, 1, 10, 10, 20, 20],
    ...                               [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                               [1, 1, 20, 20, 30, 30]], int), tempa)
    4
    >>> bin_count_and_empty(np.array([[1, 2, 10, 10, 20, 20],  # bin 2!
    ...                               [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                               [1, 1, 20, 20, 30, 30]], int), tempa)
    4
    >>> bin_count_and_empty(np.array([[1, 3, 10, 10, 20, 20],  # bin 3!
    ...                               [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                               [1, 1, 20, 20, 30, 30]], int), tempa)
    7
    """
    total_bins: int = -1  # the current idea of what the last bin is
    temp.fill(0)  # empty all temporary values
    n_items: Final[int] = len(y)  # the number of rows in the matrix

    for i in range(n_items):  # iterate over all packed items
        bin_idx: int = int(y[i, IDX_BIN]) - 1  # get the bin index of the item
        temp[bin_idx] += 1  # increase number of items
        total_bins = max(total_bins, bin_idx)
    return (n_items * total_bins) + temp[0:total_bins + 1].min()


class BinCountAndEmpty(BinCountAndLastEmpty):
    """Get the number of bins and number of elements in the emptiest one."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the objective function.

        :param instance: the instance to load the bounds from
        """
        super().__init__(instance)
        #: the internal temporary array
        self.__temp: Final[np.ndarray] = np.empty(
            instance.n_items, instance.dtype)

    def evaluate(self, x) -> int:
        """
        Evaluate the objective function.

        :param x: the solution
        :return: the bin size and smallest-bin-area factor
        """
        return bin_count_and_empty(x, self.__temp)

    def __str__(self) -> str:
        """
        Get the name of the bins objective function.

        :return: `binCountAndEmpty`
        :retval "binCountAndEmpty": always
        """
        return "binCountAndEmpty"
