"""
An objective function indirectly minimizing the number of bins in packings.

This objective function computes the number of bins used. Let's call it
`n_bins`. We know the area `bin_area` of a bin as well. We then compute the
minimum area occupied in any bin, `min_area`. Now we return
`(bin_area * (n_bins - 1)) + min_area`.

This function always prefers a packing that has fewer bins over a packing
with more bins duw to the term `bin_area * (n_bins - 1)` and
`bin_area >= min_area` (let us ignore the case where `bin_area == min_area`,
which does not make practical sense). Since `min_area < bin_area` in all
practically relevant cases, the offset `min_area` just distinguishes
packings that have same number of bins. Amongst such packings, those whose
least-occupied bin is closer to being empty are preferred (regardless where
this bin is). The idea is that this will eventually allow us to get rid of
that least-occupied bin in subsequent optimization steps, i.e., to reduce
the number of bins.

This is similar to the :mod:`~moptipyapps.binpacking2d.objectives.\
bin_count_and_small` objective, except that we do not try to expunge the last
bin, but any bin. It has the same lower and upper bound, though.
"""
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.objectives.bin_count_and_last_small import (
    BinCountAndLastSmall,
)
from moptipyapps.binpacking2d.packing import (
    IDX_BIN,
    IDX_BOTTOM_Y,
    IDX_LEFT_X,
    IDX_RIGHT_X,
    IDX_TOP_Y,
)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def bin_count_and_small(y: np.ndarray, bin_area: int,
                        temp: np.ndarray) -> int:
    """
    Compute the number of bins and the smallest occupied area in any bin.

    We compute the total number of bins minus 1 and multiply it with the
    total area of items. We then add the area of items in the smallest bin.

    :param y: the packing
    :param bin_area: the area of a single bin
    :param temp: a temporary array to hold the current area counters
    :return: the objective value

    >>> tempa = np.empty(10, int)
    >>> bin_count_and_small(np.array([[1, 1, 10, 10, 20, 20],
    ...                               [1, 1, 30, 30, 40, 40],
    ...                               [1, 1, 20, 20, 30, 30]], int),
    ...                              50*50, tempa)
    300
    >>> bin_count_and_small(np.array([[1, 1, 10, 10, 20, 20],
    ...                               [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                               [1, 1, 20, 20, 30, 30]], int),
    ...                              50*50, tempa)
    2600
    >>> bin_count_and_small(np.array([[1, 2, 10, 10, 20, 20],  # bin 2!
    ...                               [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                               [1, 1, 20, 20, 30, 30]], int),
    ...                               50*50, tempa)
    2600
    >>> bin_count_and_small(np.array([[1, 3, 10, 10, 20, 20],  # bin 3!
    ...                               [1, 2, 30, 30, 40, 40],  # bin 2!
    ...                               [1, 1, 20, 20, 30, 30]], int),
    ...                              50*50, tempa)
    5100
    >>> bin_count_and_small(np.array([[1, 3, 10, 10, 50, 50],  # bin 3!
    ...                               [1, 2, 30, 30, 60, 60],  # bin 2!
    ...                               [1, 1, 20, 20, 30, 30]], np.int8),
    ...                              50*50, tempa)
    5100
    """
    total_bins: int = 0  # the current idea of the number of bins
    n_items: Final[int] = len(y)  # the number of rows in the matrix
    temp.fill(0)  # fill temp with zeros

    for i in range(n_items):  # iterate over all packed items
        bin_idx: int = int(y[i, IDX_BIN]) - 1  # get the bin index of the item
        temp[bin_idx] += ((y[i, IDX_RIGHT_X] - y[i, IDX_LEFT_X])
                          * (y[i, IDX_TOP_Y] - y[i, IDX_BOTTOM_Y]))
        total_bins = max(total_bins, bin_idx)
    return (bin_area * total_bins) + temp[0:total_bins + 1].min()


class BinCountAndSmall(BinCountAndLastSmall):
    """Compute the number of bins and the area in the smallest one."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the objective function.

        :param instance: the instance to load the bounds from
        """
        super().__init__(instance)
        #: the internal temporary array
        self.__temp: Final[np.ndarray] = np.empty(instance.n_items, int)

    def evaluate(self, x) -> int:
        """
        Evaluate the objective function.

        :param x: the solution
        :return: the bin size and smallest-bin-area factor
        """
        return bin_count_and_small(x, self._bin_size, self.__temp)

    def __str__(self) -> str:
        """
        Get the name of the bins objective function.

        :return: `binCountAndSmall`
        :retval "binCountAndSmall": always
        """
        return "binCountAndSmall"
