"""
An objective function indirectly minimizing the number of bins in packings.

This objective function minimizes the number of bins and maximizes the
"useable" space in any bin.

Which space is actually useful for our encodings? Let's say we have filled
a bin to a certain degree and somewhere there is a "hole" in the filled area,
but this hole is covered by another object. The area of the hole is not used,
but it also cannot be used anymore. The area that we can definitely use is the
area above the "skyline" of the objects in the bin. The skyline at any
horizontal `x` coordinate be the highest border of any object that intersects
with `x` horizontally. In other words, it is the `y` value at and above which
no other object is located at this `x` coordinate. The area below the skyline
cannot be used anymore. The area above the skyline can.

If we minimize the area below the skyline in the very last bin, then this will
a similar impact as minimizing the overall object area in the last bin (see
:mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_last_small`). We push
the skyline lower and lower and, if we are lucky, the last bin eventually
becomes empty. This is done with the objective
:mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_last_skyline`.

But we could also minimize the area under the skylines in the other bins.
Because a) if we can get any skyline in any bin to become 0, then this bin
disappears and b) if we can free up space in the bins by lowering the
skylines, then we have a better chance to move an object from the *next* bin
forward into that bin, which increases the chance to make that bin empty.

In this objective function, we therefore use the smallest skyline area over
all bins to distinguish between packings of the same number of bins.
For all intents and purposes, it has the same lower and upper bound as the
:mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_last_small`
objective.
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
    IDX_LEFT_X,
    IDX_RIGHT_X,
    IDX_TOP_Y,
)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def bin_count_and_lowest_skyline(y: np.ndarray, bin_width: int,
                                 bin_height: int) -> int:
    """
    Compute the bin count-1 times the bin size + the space below the skyline.

    :param y: the packing
    :param bin_width: the bin width
    :param bin_height: the bin height
    :return: the objective value

    >>> 10*0 + 10*20 + 10*30 + 10*40 + 10*0
    900
    >>> bin_count_and_lowest_skyline(np.array([[1, 1, 10, 10, 20, 20],
    ...                                        [1, 1, 30, 30, 40, 40],
    ...                                        [1, 1, 20, 20, 30, 30]], int),
    ...                              50, 50)
    900
    >>> 5 * 0 + 5 * 10 + 10 * 20 + 5 * 10 + 25 * 0
    300
    >>> bin_count_and_lowest_skyline(np.array([[1, 1,  5,  0, 15, 10],
    ...                                        [1, 1, 10, 10, 20, 20],
    ...                                        [1, 1, 15,  0, 25, 10]], int),
    ...                              50, 50)
    300
    >>> 50*50 + min(5*0 + 10*10 + 10*10 + 25*0, 10*0 + 10*20 + 30*0)
    2700
    >>> bin_count_and_lowest_skyline(np.array([[1, 1,  5,  0, 15, 10],
    ...                                        [1, 2, 10, 10, 20, 20],
    ...                                        [1, 1, 15,  0, 25, 10]], int),
    ...                              50, 50)
    2700
    >>> 5 * 0 + 5 * 10 + 3 * 20 + (50 - 13) * 25
    1035
    >>> bin_count_and_lowest_skyline(np.array([[1, 1,  5,  0, 15, 10],
    ...                                        [1, 1, 10, 10, 20, 20],
    ...                                        [1, 1, 15,  0, 25, 10],
    ...                                        [2, 1, 13, 20, 50, 25]], int),
    ...                              50, 50)
    1035
    >>> 2500*3 + min(10*10, 20*20, 25*10, 50*25)
    7600
    >>> bin_count_and_lowest_skyline(np.array([[1, 1, 0, 0, 10, 10],
    ...                                        [2, 2, 0, 0, 20, 20],
    ...                                        [3, 3, 0, 0, 25, 10],
    ...                                        [4, 4, 0, 0, 50, 25]], int),
    ...                              50, 50)
    7600
    """
    bins: Final[int] = int(y[:, IDX_BIN].max())
    len_y: Final[int] = len(y)
    bin_size: Final[int] = bin_height * bin_width
    min_area_under_skyline: int = bin_size

    for use_bin in range(1, bins + 1):
        cur_left: int = 0
        area_under_skyline: int = 0
        while cur_left < bin_width:
            use_right = next_left = bin_width
            use_top: int = 0
            for i in range(len_y):
                if y[i, IDX_BIN] != use_bin:
                    continue
                left: int = int(y[i, IDX_LEFT_X])
                right: int = int(y[i, IDX_RIGHT_X])
                top: int = int(y[i, IDX_TOP_Y])
                if left <= cur_left < right and top > use_top:
                    use_top = top
                    use_right = right
                if cur_left < left < next_left:
                    next_left = left

            use_right = min(use_right, next_left)
            area_under_skyline += (use_right - cur_left) * use_top
            cur_left = use_right
        min_area_under_skyline = min(min_area_under_skyline,
                                     area_under_skyline)
    return ((bins - 1) * bin_size) + min_area_under_skyline


class BinCountAndLowestSkyline(BinCountAndLastSmall):
    """Compute the number of bins and the largest useful area."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the objective function.

        :param instance: the instance to load the bounds from
        """
        super().__init__(instance)
        #: the bin width
        self.__bin_width: Final[int] = instance.bin_width
        #: the bin height
        self.__bin_height: Final[int] = instance.bin_height

    def evaluate(self, x) -> int:
        """
        Evaluate the objective function.

        :param x: the solution
        :return: the bin size and last-bin-small-area factor
        """
        return bin_count_and_lowest_skyline(
            x, self.__bin_width, self.__bin_height)

    def __str__(self) -> str:
        """
        Get the name of the bins objective function.

        :return: `binCountAndLowestSkyline`
        :retval "binCountAndLowestSkyline": always
        """
        return "binCountAndLowestSkyline"
