"""
An improved bottom left encoding by Liu and Teng extended to multiple bins.

Here we provide an implementation of the improved bottom left encoding by Liu
and Teng [1], but extended to bins with limited height. If the height of the
bin is a limiting factor, then our implementation will automatically use
multiple bins. Another variant is given in
:mod:`moptipyapps.binpacking2d.ibl_encoding_1`.

An instance :mod:`~moptipyapps.binpacking2d.instance` of the
two-dimensional bin packing problem defines a set of objects to be packed
and a bin size (width and height). Each object to be packed has itself a
width and a height as well as a repetition counter, which is `1` if the object
only occurs a single time and larger otherwise (i.e., if the repetition
counter is `5`, the object needs to be packaged five times).

The encoding receives signed permutations with repetitions as input. Each
element of the permutation identifies one object from the bin packing
instance. Each such object ID must occur exactly as often as the repetition
counter of the object in the instance data suggest. But the ID may also occur
negated, in which case the object is supposed to rotated by 90°.

Now our encoding processes such a permutation from beginning to end. It starts
with an empty bin `1`. Each object is first placed with its right end at the
right end of the first bin and with its bottom line exactly at the top of the
bin, i.e., outside of the bin. Then, in each step, we move the object as far
down as possible. Then, we move it to the left as far as possible, but we
immediately stop if there was another chance to move the object down. In
other words, downward movements are preferred over left movements. This is
repeated until no movement of the object is possible anymore.

Once the object cannot be moved anymore, we check if it is fully inside the
bin. If yes, then the object is included in the bin and we continue with the
next object. If not, it does not fit into the bin.

This is the "Improved Bottom Left" heuristic by Liu and Teng [1].

If the object does not fit into the first bin and we already "opened" a second
bin, then we try to place it into the second bin using the same procedure. And
then into the third bin if that does not work out, and so on. Until we have
tried unsuccessfully all the bins that we have opened.

In this case, we "open" the next bin and we place the object at the
bottom-left corner of a new bin. Then we continue with the next object, again
trying to put it into the first bin, then the second bin, and so on.

This is different from the first variant of this encoding implemented in file
:mod:`moptipyapps.binpacking2d.ibl_encoding_2`, which always and only
tries to put objects into the last bin that was opened (and moves to a new bin
if that does not work out). That variant of the encoding is therefore faster
than the one here, but the one here tends to yield better packings.

The problem with this is that we need to basically first try to put the object
into the first bin. For this we need to look at *all* the objects that we have
already placed, because it could be that we already have one object in the
first bin, then one in the second bin that did not fit into the first bin,
then one smaller object in the first bin again, and so on. If our new object
does not fit into the first bin, then we need to do the same with the second
bin, and so on. So for every bin we try, we need to look at all objects already
placed. And the number of bins we could try could be equal to the number of
objects that we have already placed (if each object occupies one bin alone).
So we have a worst case complexity of O(n ** 2) for placing one object. And we
do this for all objects, so we would have a O(n ** 3) overall complexity.
Well, actually, it is worse: Because we repeat the process of looking at all
the objects several times while moving our new item to the left and down and
to the left and down. So I suspect that we actually have O(n ** 4).
That is annoying.

We try to alleviate this a little bit by remembering, for each bin, the index
of the first object that we put in there and the index of the last object we
put in there. Now within these two indices, there also might be objects that
we placed into other bins. But for a very little overhead (remembering two
values per bin), we have a certain chance to speed up the process in several
situations. For instance, the worst case from above, that each object occupies
exactly one bin by itself becomes easier because we would only look at one
already placed object per bin.

This procedure has originally been developed and implemented by Mr. Rui ZHAO
(赵睿), <zr1329142665@163.com>, a Master's student at the Institute of Applied
Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School of
Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei University
(合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).

1. Dequan Liu and Hongfei Teng. An Improved BL-Algorithm for Genetic Algorithm
   of the Orthogonal Packing of Rectangles. European Journal of Operational
   Research. 112(2):413-420. January (1999).
   https://doi.org/10.1016/S0377-2217(97)00437-2.
   http://www.paper.edu.cn/scholar/showpdf/MUT2AN0IOTD0Mxxh.
"""
from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.encoding import Encoding
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error

from moptipyapps.binpacking2d.instance import (
    IDX_HEIGHT,
    IDX_WIDTH,
    Instance,
)
from moptipyapps.binpacking2d.packing import (
    IDX_BIN,
    IDX_BOTTOM_Y,
    IDX_ID,
    IDX_LEFT_X,
    IDX_RIGHT_X,
    IDX_TOP_Y,
    Packing,
)
from moptipyapps.shared import SCOPE_INSTANCE


@numba.njit(nogil=True, cache=True, inline="always", boundscheck=False)
def __move_down(packing: np.ndarray, bin_id: int,
                bin_start: int, bin_end: int, i1: int) -> bool:
    """
    Move the box at index `i1` down as far as possible in the current bin.

    `bin_start` is the index of the first object that has already been placed
    in the current bin `bin_id`. It always holds that `i1 >= bin_start`. In
    the case that `i1 == bin_start`, then we can move the object directly to
    the bottom of the bin without any issue.
    If `i1 > bin_start` we iterate over all objects at indices
    `bin_start...bin_end - 1` whose bin equals `bin`. We first set `min_down`
    to the bottom-y coordinate of the box, because this is how far down we can
    move at most.

    Then, for each of the objects already placed in the bin `bin_id`, we check
    if there is any intersection of the horizontal with the current box. If
    there is no intersection *or* if the object is not already above the
    current box, then the object will not influence the downward movement of
    our object. If there is an intersection, then we cannot move the current
    box deeper than the top-y coordinate of the other box.

    *Only* the box at index `i1` is modified and if it is modified, this
    function will return `True`.

    :param packing: the packing under construction
    :param bin_id: the bin ID
    :param bin_start: the starting index of the current bin
    :param bin_end: the exclusive end index of the current bin
    :param i1: the index of the current box
    :return: `True` if the object was moved down, `False` if the object cannot
        be moved down any further because either it has reached the bottom or
        because it would intersect with other objects

    >>> # itemID, binID, left-x, bottom-y, right-x, top-y
    >>> r = np.array([[1, 1, 10, 20, 30, 40],
    ...               [2, 1, 30, 30, 50, 60],
    ...               [3, 1, 40, 100, 60, 200]])
    >>> __move_down(r, 1, 0, 2, 2)  # try to move down the box at index 2
    True
    >>> print(r[2, :])
    [  3   1  40  60  60 160]
    >>> __move_down(r, 1, 0, 2, 2)  # try to move down the box at index 2
    False
    >>> __move_down(r, 1, 0, 1, 1)  # try to move down the box at index 1
    True
    >>> print(r[1, :])
    [ 2  1 30  0 50 30]
    >>> __move_down(r, 1, 0, 1, 1)  # try to move down the box at index 1
    False
    >>> __move_down(r, 1, 0, 2, 2)  # try to move down the box at index 2
    True
    >>> print(r[2, :])
    [  3   1  40  30  60 130]
    >>> __move_down(r, 1, 0, 2, 2)  # try to move down the box at index 2
    False
    >>> __move_down(r, 1, 0, 0, 0)
    True
    >>> print(r[0, :])
    [ 1  1 10  0 30 20]
    >>> __move_down(r, 1, 0, 0, 0)
    False
    """
    # load the coordinates of i1 into local variables to speed up computation
    packing_i1_left_x: Final[int] = packing[i1, IDX_LEFT_X]
    packing_i1_bottom_y: Final[int] = packing[i1, IDX_BOTTOM_Y]
    packing_i1_right_x: Final[int] = packing[i1, IDX_RIGHT_X]
    packing_i1_top_y: Final[int] = packing[i1, IDX_TOP_Y]
    min_down: int = packing_i1_bottom_y  # maximum move: down to bottom
    for i0 in range(bin_start, bin_end):  # iterate over items in current bin
        # An intersection exists if the right-x of an existing box is larger
        # than the left-x of the new box AND if the left-x of the existing box
        # is less than the right-x of the new box.
        # Only intersections matter and *only* with objects in the same bin
        # and only if the existing box is not already above the new box.
        if (packing[i0, IDX_BIN] == bin_id) and \
                (packing[i0, IDX_RIGHT_X] > packing_i1_left_x) and \
                (packing[i0, IDX_LEFT_X] < packing_i1_right_x) and \
                (packing[i0, IDX_BOTTOM_Y] < packing_i1_top_y):
            # The object would horizontally intersect with the current object
            diff: int = packing_i1_bottom_y - packing[i0, IDX_TOP_Y]
            if diff < min_down:  # we can move down only a shorter distance
                min_down = diff  # ok, a shorter distance was found
    if min_down > 0:  # Can we move down? If yes, update box.
        packing[i1, IDX_BOTTOM_Y] = packing_i1_bottom_y - min_down
        packing[i1, IDX_TOP_Y] = packing_i1_top_y - min_down
        return True
    return False


@numba.njit(nogil=True, cache=True, inline="always", boundscheck=False)
def __move_left(packing: np.ndarray, bin_id: int,
                bin_start: int, bin_end: int, i1: int) -> bool:
    """
    Move the box at index `i1` left as far as possible in the current bin.

    This function moves a box to the left without changing its vertical
    position. It is slightly more tricky than the downwards moving function,
    because in the improved bottom left heuristic, downward moves are
    preferred compared to left moves. This means that the box needs to be
    stopped when reaching the edge of a box on whose top it sits. Of course,
    we only consider other objects inside the current bin
    (with ID `bin_start`).

    This function is to be called *after* `__move_down` and in an alternating
    fashion.

    *Only* the box at index `i1` is modified and if it is modified, this
    function will return `True`.

    :param packing: the packing under construction
    :param bin_id: the bin ID
    :param bin_start: the starting index of the current bin
    :param bin_end: the exclusive end index of the current bin
    :param i1: the index of the current box
    :return: `True` if the object was moved down, `False` if the object cannot
        be moved down any further because either it has reached the bottom or
        because it would intersect with other objects

    >>> # itemID, binID, left-x, bottom-y, right-x, top-y
    >>> r = np.array([[1, 1,  0,  0, 30, 10],
    ...               [2, 1, 35,  0, 45, 30],
    ...               [3, 1,  0, 10, 10, 20],
    ...               [4, 1, 40, 30, 50, 40]])
    >>> __move_left(r, 1, 0, 3, 3)
    True
    >>> print(r[3, :])
    [ 4  1 25 30 35 40]
    >>> __move_left(r, 1, 0, 3, 3)
    True
    >>> print(r[3, :])
    [ 4  1  0 30 10 40]
    >>> r[3, :] = [4, 1, 25, 10, 35, 20]
    >>> __move_left(r, 1, 0, 3, 3)
    True
    >>> print(r[3, :])
    [ 4  1 10 10 20 20]
    >>> __move_left(r, 1, 0, 3, 3)
    False
    >>> # itemID, binID, left-x, bottom-y, right-x, top-y
    >>> r = np.array([[1, 1,  0,  0, 10, 2],
    ...               [2, 1, 10,  0, 20, 5],
    ...               [3, 1,  8,  2, 10, 4]])
    >>> __move_left(r, 0, 0, 3, 2)
    True
    >>> print(r[2, :])
    [3 1 0 2 2 4]
    """
    packing_i1_left_x: Final[int] = packing[i1, IDX_LEFT_X]
    packing_i1_bottom_y: Final[int] = packing[i1, IDX_BOTTOM_Y]
    packing_i1_right_x: Final[int] = packing[i1, IDX_RIGHT_X]
    packing_i1_top_y: Final[int] = packing[i1, IDX_TOP_Y]
    min_left: int = packing_i1_left_x
    for i0 in range(bin_start, bin_end):
        # consider only objects that are not yet behind us and are in
        # the same bin may intersect
        if (packing[i0, IDX_BIN] != bin_id) \
                or (packing[i0, IDX_LEFT_X] >= packing_i1_right_x):
            continue  # the object is already behind us, so it can be ignored
        if (packing[i0, IDX_RIGHT_X] > packing_i1_left_x) \
                and (packing[i0, IDX_LEFT_X] < packing_i1_right_x):
            # we have a horizontal intersection with a box below
            if packing[i0, IDX_TOP_Y] == packing_i1_bottom_y:
                # only consider those the box *directly* below and move the
                # right end of the new box to the left end of that box below
                diff = packing_i1_right_x - packing[i0, IDX_LEFT_X]
                if diff < min_left:
                    min_left = diff
        elif (packing_i1_top_y > packing[i0, IDX_BOTTOM_Y]) \
                and (packing_i1_bottom_y < packing[i0, IDX_TOP_Y]):
            diff = packing_i1_left_x - packing[i0, IDX_RIGHT_X]
            if diff < min_left:
                min_left = diff
    if min_left > 0:
        # move the box to the left
        packing[i1, IDX_LEFT_X] = packing_i1_left_x - min_left
        packing[i1, IDX_RIGHT_X] = packing_i1_right_x - min_left
        return True
    return False


@numba.njit(nogil=True, cache=True, inline="always", boundscheck=False)
def _decode(x: np.ndarray, y: np.ndarray, instance: np.ndarray,
            bin_width: int, bin_height: int, bin_starts: np.ndarray,
            bin_ends: np.ndarray) -> int:
    """
    Decode a (signed) permutation to a packing.

    The permutation is processed from the beginning to the end.
    Each element identifies one object by its ID. If the ID is negative,
    the object will be inserted rotated by 90°. If the ID is positive, the
    object will be inserted as is.

    The absolute value of the ID-1 will be used to look up the width and
    height of the object in the `instance` data. If the object needs to be
    rotated, width and height will be swapped.

    Each object is, at the beginning, placed with its right side at the right
    end of the *first* bin. The bottom line of the object is initially put on
    top of the bin, i.e., initially the object is outside of the bin.

    Then, the object is iteratively moved downward *in the bin* as far as
    possible. Once it reaches another object, we move it to the left until
    either its right side reaches the left end of the object beneath it or
    until its left side touches another object. Then we try to move the object
    down again, and so on.

    Once the object can no longer be moved down, we check if it is now fully
    inside of *the bin*. If yes, then good, the object's bin index is set to
    the ID of the current bin. If not, then we cannot place the object into
    this bin. In this case, we move to the second bin an repeat the process.
    If the object fits into the second bin, then it is placed in there and
    we are finished. If not, we try the third bin, and so on.

    If we cannot fit the object into any of the bins that already have other
    objects, then we allocate a new empty bin. We move the object to the
    bottom-left corner of this bin. In other words, the left side of the
    object touches the left side of the bin, i.e., is `0`. The bottom-line
    of the object is also the bottom of the bin, i.e., has coordinate `0` as
    well.

    :param x: a possibly signed permutation
    :param y: the packing object
    :param instance: the packing instance data
    :param bin_width: the bin width
    :param bin_height: the bin height
    :param bin_starts: a temporary index holder array for bin starts
    :param bin_ends: a temporary index holder array for bin ends
    :returns: the number of bins

    As example, we use a slightly modified version (we add more objects so we
    get to see the use of a second bin) of Figure 2 of the Liu and Teng paper
    "An Improved BL-Algorithm for Genetic Algorithm of the Orthogonal Packing
    of Rectangles."

    >>> # [width, height, repetitions]
    >>> inst = np.array([[10, 20, 5], [5, 5, 5]])
    >>> # [id = plain, -id = rotated]
    >>> xx = np.array([1, -1, 2, -2, 1, -2, -2, -1, -1, 2])
    >>> # [id, bin, left-x, bottom-y, right-x, top-y] ...
    >>> yy = np.empty((10, 6), int)
    >>> binstarts = np.empty(10, int)
    >>> binends = np.empty(10, int)
    >>> print(_decode(xx, yy, inst, 30, 30, binstarts, binends))
    2
    >>> print(yy[0, :])
    [ 1  1  0  0 10 20]
    >>> print(yy[1, :])
    [ 1  1 10  0 30 10]
    >>> print(yy[2, :])
    [ 2  1 10 10 15 15]
    >>> print(yy[3, :])
    [ 2  1 15 10 20 15]
    >>> print(yy[4, :])
    [ 1  1 20 10 30 30]
    >>> print(yy[5, :])
    [ 2  1 10 15 15 20]
    >>> print(yy[6, :])
    [ 2  1 15 15 20 20]
    >>> print(yy[7, :])
    [ 1  1  0 20 20 30]
    >>> print(yy[8, :])
    [ 1  2  0  0 20 10]
    >>> print(yy[9, :])
    [ 2  2 20  0 25  5]
    """
    w: int  # the width of the current object
    h: int  # the height of the current object
    use_id: int  # the id of the current object
    bin_id: int = 1  # the id of the current bin
    bin_starts[0] = 0
    bin_ends[0] = 0
    for i, item_id in enumerate(x):  # iterate over all objects
        if item_id < 0:  # object should be rotated
            use_id = -(item_id + 1)  # get absolute id - 1
            w = instance[use_id, IDX_HEIGHT]  # width = height (rotated!)
            h = instance[use_id, IDX_WIDTH]   # height = width (rotated!)
        else:  # the object will not be rotated
            use_id = item_id - 1   # id - 1
            w = instance[use_id, IDX_WIDTH]  # get width
            h = instance[use_id, IDX_HEIGHT]  # get height

# It could be that an object is too wide or too high for the bin in its
# current rotation even if the bin was empty entirely. In this case, we simply
# force-rotate it. A bin packing instance will not permit objects that do not
# fit into the bin in any rotation. So if the object does not fit in its
# current rotation, it must fit if we simply rotate it by 90°.
        if (w > bin_width) or (h > bin_height):
            w, h = h, w

# At first, the object's right corner is at the right corner of the bin.
# The object sits exactly at the top of the bin, i.e., its bottom line
# is the top line of the bin.
        y[i, IDX_ID] = use_id + 1  # the id of the object

        not_found: bool = True
        for item_bin in range(1, bin_id + 1):  # iterate over all bins in use
            bin_start = bin_starts[item_bin - 1]  # index of first item in bin
            bin_end = bin_ends[item_bin - 1]  # index after last item in bin
            y[i, IDX_LEFT_X] = bin_width - w  # the left end
            y[i, IDX_BOTTOM_Y] = bin_height  # object sits on top of bin
            y[i, IDX_RIGHT_X] = bin_width  # object ends at right end of bin
            y[i, IDX_TOP_Y] = bin_height + h  # top of object is outside of bin

            while __move_down(y, item_bin, bin_start, bin_end, i) \
                    or __move_left(y, item_bin, bin_start, bin_end, i):
                pass  # loop until object can no longer be moved

# Check if the object is completely inside the bin. If yes, we can put it
# there and stop searching.
            if (y[i, IDX_RIGHT_X] <= bin_width) \
                    and (y[i, IDX_TOP_Y] <= bin_height):
                not_found = False  # we found a placement, outer loop can stop
                y[i, IDX_BIN] = item_bin  # set the items bin
                bin_ends[item_bin - 1] = i + 1  # index after last item in bin
                break

# If the object is not fully inside the current bin, we move to a new bin.
        if not_found:  # we did not find a spot in any of the bins
            bin_starts[bin_id] = i  # set the starting index of the new bin
            bin_ends[bin_id] = i + 1  # set the end index of the new bin
            bin_id = bin_id + 1  # step to the next bin
            y[i, IDX_LEFT_X] = 0  # the object goes to the left end of the bin
            y[i, IDX_BOTTOM_Y] = 0  # the object goes to the bottom of the bin
            y[i, IDX_RIGHT_X] = w  # so its right end is its width
            y[i, IDX_TOP_Y] = h  # and its top end is its height
            y[i, IDX_BIN] = bin_id  # store the bin id
    return int(bin_id)  # return the total number of bins


class ImprovedBottomLeftEncoding2(Encoding):
    """An Improved Bottem Left Encoding by Liu and Teng for multiple bins."""

    def __init__(self, instance: Instance) -> None:
        """
        Instantiate the improved best first encoding.

        :param instance: the packing instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the internal instance reference
        self.__instance: Final[Instance] = instance
        #: the temporary array for holding bin start indexes
        self.__bin_starts: Final[np.ndarray] = np.empty(
            instance.n_items, instance.dtype)
        #: the temporary array for holding bin end indexes
        self.__bin_ends: Final[np.ndarray] = np.empty(
            instance.n_items, instance.dtype)

    def decode(self, x: np.ndarray, y: Packing) -> None:
        """
        Map a potentially signed permutation to a packing.

        :param x: the array
        :param y: the Gantt chart
        """
        y.n_bins = _decode(x, y, self.__instance, self.__instance.bin_width,
                           self.__instance.bin_height, self.__bin_starts,
                           self.__bin_ends)

    def __str__(self) -> str:
        """
        Get the name of this encoding.

        :return: `"ibf2"`
        :rtype: str
        """
        return "ibf2"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_INSTANCE) as kv:
            self.__instance.log_parameters_to(kv)
