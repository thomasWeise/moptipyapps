"""
Here we provide a :class:`~moptipy.api.space.Space` of bin packings.

The bin packing object is defined in module
:mod:`~moptipyapps.binpacking2d.packing`. Basically, it is a
two-dimensional numpy array holding, for each item, its ID, its bin ID, as
well as its location defined by four coordinates.

1. Dequan Liu and Hongfei Teng. An Improved BL-Algorithm for Genetic Algorithm
   of the Orthogonal Packing of Rectangles. European Journal of Operational
   Research. 112(2):413-420. January (1999).
   https://doi.org/10.1016/S0377-2217(97)00437-2.
   http://www.paper.edu.cn/scholar/showpdf/MUT2AN0IOTD0Mxxh.
"""
from collections import Counter
from math import factorial
from typing import Final

import numpy as np
from moptipy.api.space import Space
from moptipy.utils.logger import CSV_SEPARATOR, KeyValueLogSection
from moptipy.utils.types import check_int_range, type_error

from moptipyapps.binpacking2d.instance import (
    IDX_HEIGHT,
    IDX_REPETITION,
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


class PackingSpace(Space):
    """An implementation of the `Space` API of for 2D bin packings charts."""

    def __init__(self, instance: Instance) -> None:
        """
        Create a 2D packing space.

        :param instance: the 2d bin packing instance

        >>> inst = Instance.from_resource("a01")
        >>> space = PackingSpace(inst)
        >>> space.instance is inst
        True
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: The instance to which the packings apply.
        self.instance: Final[Instance] = instance
        #: fast call function forward

    def copy(self, dest: Packing, source: Packing) -> None:
        """
        Copy one packing to another one.

        :param dest: the destination packing
        :param source: the source packing
        """
        np.copyto(dest, source)
        dest.n_bins = source.n_bins

    def create(self) -> Packing:
        """
        Create a packing object without assigning items to locations.

        :return: the (empty, uninitialized) packing object

        >>> inst = Instance.from_resource("a01")
        >>> space = PackingSpace(inst)
        >>> x = space.create()
        >>> x.shape
        (24, 6)
        >>> x.instance is inst
        True
        >>> type(x)
        <class 'moptipyapps.binpacking2d.packing.Packing'>
        """
        return Packing(self.instance)

    def to_str(self, x: Packing) -> str:
        """
        Convert a packing to a string.

        :param x: the packing
        :return: a string corresponding to the flattened packing array

        >>> inst = Instance.from_resource("a01")
        >>> space = PackingSpace(inst)
        >>> y = space.create()
        >>> xx = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ...                1, 2, 2, 2, 2, 2, 2])
        >>> from moptipyapps.binpacking2d.ibl_encoding_1 import _decode
        >>> _decode(xx, y, inst, inst.bin_width, inst.bin_height)
        5
        >>> space.to_str(y)
        '1;1;0;0;463;386;1;1;463;0;926;386;1;1;926;0;1389;386;\
1;1;1389;0;1852;386;1;1;1852;0;2315;386;1;1;0;386;463;772;\
1;1;463;386;926;772;1;1;926;386;1389;772;1;1;1389;386;1852;772;\
1;1;1852;386;2315;772;1;1;0;772;463;1158;1;1;463;772;926;1158;\
1;1;926;772;1389;1158;1;1;1389;772;1852;1158;1;1;1852;772;2315;1158;\
1;2;0;0;463;386;1;2;463;0;926;386;1;2;926;0;1389;386;2;2;0;386;1680;806;\
2;3;0;0;1680;420;2;3;0;420;1680;840;2;4;0;0;1680;420;2;4;0;420;1680;840;\
2;5;0;0;1680;420'
        """
        return CSV_SEPARATOR.join([str(xx) for xx in np.nditer(x)])

    def is_equal(self, x1: Packing, x2: Packing) -> bool:
        """
        Check if two bin packings have the same contents.

        :param x1: the first chart
        :param x2: the second chart
        :return: `True` if both packings are for the same instance and have the
            same structure

        >>> inst = Instance.from_resource("a01")
        >>> space = PackingSpace(inst)
        >>> y1 = space.create()
        >>> y1.fill(0)
        >>> y2 = space.create()
        >>> y2.fill(0)
        >>> space.is_equal(y1, y2)
        True
        >>> y1 is y2
        False
        >>> y1[0, 0] = 1
        >>> space.is_equal(y1, y2)
        False
        """
        return (x1.instance is x2.instance) and np.array_equal(x1, x2)

    def from_str(self, text: str) -> Packing:
        """
        Convert a string to a packing.

        :param text: the string
        :return: the packing

        >>> inst = Instance.from_resource("a01")
        >>> space = PackingSpace(inst)
        >>> y1 = space.create()
        >>> xx = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ...                1, 2, 2, 2, 2, 2, 2])
        >>> from moptipyapps.binpacking2d.ibl_encoding_1 import _decode
        >>> y1.n_bins = _decode(xx, y1, inst, inst.bin_width, inst.bin_height)
        >>> y2 = space.from_str(space.to_str(y1))
        >>> space.is_equal(y1, y2)
        True
        >>> y1 is y2
        False
        """
        if not isinstance(text, str):
            raise type_error(text, "packing text", str)
        x: Final[Packing] = self.create()
        np.copyto(x, np.fromstring(text, dtype=x.dtype, sep=CSV_SEPARATOR)
                  .reshape(x.shape))
        x.n_bins = int(x[:, IDX_BIN].max())
        self.validate(x)
        return x

    def validate(self, x: Packing) -> None:
        """
        Check if a packing is valid and feasible.

        This method performs a comprehensive feasibility and sanity check of
        a packing. It ensures that the packing could be implemented in the
        real world exactly as it is given here and that all data are valid and
        that it matches to the bin packing instance. This includes:

        - checking that all data types, numpy `dtypes`, and matrix shapes are
          correct
        - checking that the packing belongs to the same instance as this space
        - checking that no objects in the same bin overlap
        - checking that all objects occur exactly as often as prescribed by
          the instance
        - checking that no object extends outside of its bin
        - checking that all objects have the same width/height as prescribed
          in the instance (with possible rotations)
        - check that bin ids are assigned starting at `1` and incrementing in
          steps of `1`

        :param x: the packing
        :raises TypeError: if any component of the chart is of the wrong type
        :raises ValueError: if the packing chart is not feasible
        """
        if not isinstance(x, Packing):
            raise type_error(x, "x", Packing)
        inst: Final[Instance] = self.instance
        if inst is not x.instance:
            raise ValueError(
                f"x.instance must be {inst} but is {x.instance}.")
        if inst.dtype is not x.dtype:
            raise ValueError(
                f"inst.dtype = {inst.dtype} but x.dtype={x.dtype}")
        needed_shape: Final[tuple[int, int]] = (inst.n_items, 6)
        if x.shape != needed_shape:
            raise ValueError(f"x.shape={x.shape}, but must be {needed_shape}.")

        bin_width: Final[int] = check_int_range(
            inst.bin_width, "bin_width", 1, 1_000_000_000)
        bin_height: Final[int] = check_int_range(
            inst.bin_height, "bin_height", 1, 1_000_000_000)

        bins: Final[set[int]] = set()
        items: Final[Counter[int]] = Counter()

        for i in range(inst.n_items):
            item_id: int = x[i, IDX_ID]
            if (item_id <= 0) or (item_id > inst.n_different_items):
                raise ValueError(
                    f"Encountered invalid id={item_id} for object at index "
                    f"{i}, must be in 1..{inst.n_different_items}.")

            bin_id: int = x[i, IDX_BIN]
            if (bin_id <= 0) or (bin_id > inst.n_items):
                raise ValueError(
                    f"Encountered invalid bin-id={bin_id} for object at index"
                    f" {i}, must be in 1..{inst.n_items}.")
            bins.add(bin_id)

            items[item_id] += 1
            x_left: int = x[i, IDX_LEFT_X]
            y_bottom: int = x[i, IDX_BOTTOM_Y]
            x_right: int = x[i, IDX_RIGHT_X]
            y_top: int = x[i, IDX_TOP_Y]

            if (x_left >= x_right) or (y_bottom >= y_top):
                raise ValueError(
                    f"Invalid item coordinates ({x_left}, {y_bottom}, "
                    f"{x_right}, {y_top}) for item of id {item_id} at index"
                    f" {i}.")
            if (x_left < 0) or (y_bottom < 0) or (x_right > bin_width) \
                    or (y_top > bin_height):
                raise ValueError(
                    f"Item coordinates ({x_left}, {y_bottom}, {x_right}, "
                    f"{y_top}) for item of id {item_id} at index {i} extend "
                    f"outside of the bin ({bin_width}, {bin_height}).")

            real_width: int = x_right - x_left
            real_heigth: int = y_top - y_bottom
            width: int = inst[item_id - 1, IDX_WIDTH]
            height: int = inst[item_id - 1, IDX_HEIGHT]

            if ((real_width != width) or (real_heigth != height)) \
                    and ((real_width != height) and (real_heigth != width)):
                raise ValueError(
                    f"Item coordinates ({x_left}, {y_bottom}, {x_right}, "
                    f"{y_top}) mean width={real_width} and height="
                    f"{real_heigth} for item of id {item_id} at index {i}, "
                    f"which should have width={width} and height={height}.")

            for j in range(i):
                if x[j, IDX_BIN] != bin_id:
                    continue
                x_left_2: int = x[j, IDX_LEFT_X]
                y_bottom_2: int = x[j, IDX_BOTTOM_Y]
                x_right_2: int = x[j, IDX_RIGHT_X]
                y_top_2: int = x[j, IDX_TOP_Y]
                if (x_left_2 < x_right) and (x_right_2 > x_left) \
                        and (y_bottom_2 > y_top) and (y_top_2 < y_bottom):
                    raise ValueError(
                        f"Item {x[j, IDX_ID]} in bin {bin_id} and at index "
                        f"{j} is ({x_left_2}, {y_bottom_2}, {x_right_2}, "
                        f"{y_top_2}) and thus intersects with item {item_id} "
                        f"at index {i} which is in the same bin at ({x_left},"
                        f" {y_bottom}, {x_right}, {y_top}).")

        for item_id, count in items.items():
            should: int = inst[item_id - 1, IDX_REPETITION]
            if should != count:
                raise ValueError(
                    f"Item {item_id} should occur {should} times, but occurs"
                    f" {count} times instead.")

        max_bin: Final[int] = max(bins)
        min_bin: Final[int] = min(bins)
        bin_count: Final[int] = len(bins)

        if (min_bin != 1) or ((max_bin - min_bin + 1) != bin_count):
            raise ValueError(
                f"Inconsistent use of bins. Found bins {sorted(bins)}.")

        if not isinstance(x.n_bins, int):
            raise type_error(x.n_bins, "x.n_bins", int)
        if x.n_bins != bin_count:
            raise ValueError(f"x.n_bins={x.n_bins}, but must be {bin_count}.")

    def n_points(self) -> int:
        """
        Get the number of possible packings.

        If we indeed consider that any object could be at any place, then
        there would be an incomprehensibly large number of possible packings.
        Here, we consider the bottom-left condition and the idea of encoding
        packings as signed permutations, as in the Liu and Teng paper "An
        Improved BL-Algorithm for Genetic Algorithm of the Orthogonal Packing
        of Rectangles." In this case, if `n` items are to be packed, the
        number of possible packings won't exceed `2^n * n!`.

        :return: the approximate number of packings

        >>> inst = Instance.from_resource("a01")
        >>> inst.n_items
        24
        >>> space = PackingSpace(inst)
        >>> space.n_points()
        10409396852733332453861621760000
        >>> from math import factorial
        >>> (2 ** 24) * factorial(24)
        10409396852733332453861621760000
        """
        return (2 ** self.instance.n_items) * factorial(self.instance.n_items)

    def __str__(self) -> str:
        """
        Get the name of the packing space.

        :return: the name

        >>> print(PackingSpace(Instance.from_resource("a10")))
        pack_a10
        """
        return f"pack_{self.instance}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
