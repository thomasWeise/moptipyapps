"""
An objective function counting deviations from the instance definition.

This objective function will be the smaller, the closer the structure
of an instance is to the original instance.
Due to our encoding, we create instances whose bin width, bin height,
and the number of items is the same as in an existing instance. The
lower bound for the required number of bins is also the same.

This objective function here also incorporates some additional features, such
as:

1. Is the number of different items similar to those in the original instance?
   In an existing instance, some items of same width and height could be
   grouped together. We may have 10 items to pack, but only 3 different item
   sizes exist. We here compare the number of different item sizes of a
   generated instance with the number in the instance definition.
2. In a given instance, we can observe the minimum and maximum width and
   height of any item. If an item in the generated instance is smaller than
   the minimum or larger than the maximum permitted value in one dimension,
   this will increase the error count.
3. Additionally, we want the actual minimum and maximum width and height of
   any item in the generated instance is the same as in the original instance.
4. Finally, we want that the total area covered by all items is the same in
   the generated instance as in the original instance.

All of these potential violations are counted and added up. Using this
objective function should drive the search towards generating instances that
are structurally similar to an existing instance, at least in some parameters.

We could push this arbitrarily further, trying to emulate the exact
distribution of the item sizes, etc. But this may just lead to the
reproduction of the original instance by the search and not add anything
interesting.

>>> orig = Instance.from_resource("a04")
>>> space = InstanceSpace(orig)
>>> print(f"{space.inst_name!r} with {space.n_different_items}/"
...       f"{space.n_items} items with area {space.total_item_area} "
...       f"in {space.min_bins} bins of "
...       f"size {space.bin_width}*{space.bin_height}.")
'a04n' with 2/16 items with area 7305688 in 3 bins of size 2750*1220.

>>> from moptipyapps.binpacking2d.instgen.inst_decoding import InstanceDecoder
>>> decoder = InstanceDecoder(space)
>>> import numpy as np
>>> x = np.array([ 0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, ])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 15/16 items with area 10065000 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;15;2750;1220;1101,1098;2750,244;2750,98;1101,171;1649,171;2750,976;\
441,122;1649,122;2750,10;2750,1,2;2750,3;1649,1098;2750,878;2750,58;660,122

>>> errors = Errors(space)
>>> errors.lower_bound()
0.0
>>> errors.upper_bound()
1.0
>>> errors.evaluate(y)
0.614714632631273

>>> y[0] = orig
>>> errors.evaluate(y)
0.0
"""
from math import sqrt
from typing import Final

from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from pycommons.types import type_error

from moptipyapps.binpacking2d.instance import (
    IDX_HEIGHT,
    IDX_REPETITION,
    IDX_WIDTH,
    Instance,
)
from moptipyapps.binpacking2d.instgen.instance_space import InstanceSpace


class Errors(Objective):
    """Compute the deviation of an instance from the definition."""

    def __init__(self, space: InstanceSpace) -> None:
        """
        Initialize the errors objective function.

        :param instance: the instance to load the bounds from
        """
        super().__init__()
        if not isinstance(space, InstanceSpace):
            raise type_error(space, "space", InstanceSpace)
        #: the instance description
        self.space: Final[InstanceSpace] = space

        # These errors are permitted.
        max_errors: int = max(
            space.n_different_items - 1,
            space.n_items - space.n_different_items - 1)

        goal_min_width: Final[int] = space.item_width_min
        goal_max_width: Final[int] = space.item_width_max
        goal_min_height: Final[int] = space.item_height_min
        goal_max_height: Final[int] = space.item_height_max
        bin_width: Final[int] = space.bin_width
        bin_height: Final[int] = space.bin_height

        n_items: Final[int] = space.n_items
        max_errors += n_items * max(
            goal_min_width - 1, bin_width - goal_max_width)
        max_errors += n_items * max(
            goal_min_height - 1, bin_height - goal_max_height)
        max_errors += max(goal_min_width, bin_width - goal_min_width)
        max_errors += max(goal_max_width, bin_width - goal_max_width)
        max_errors += max(goal_min_height, bin_height - goal_min_height)
        max_errors += max(goal_max_height, bin_height - goal_max_height)

        alt_area: Final[int] = (space.min_bins * bin_width * bin_height
                                - space.total_item_area)
        if alt_area < 0:
            raise ValueError("Invalid item area in space?")
        max_errors += max(space.total_item_area, alt_area)

        #: the maximum number of errors
        self.__max_errors: Final[int] = max_errors

    def evaluate(self, x: list[Instance] | Instance) -> float:
        """
        Compute the deviations from the instance definition.

        :param x: the instance
        :return: the number of deviations divided by the maximum of
            the deviations
        """
        errors: int = 0
        inst: Final[Instance] = x[0] if isinstance(x, list) else x
        space: Final[InstanceSpace] = self.space

        # Some errors are not permitted.
        errors += abs(inst.bin_width - space.bin_width)  # should be 0!
        errors += abs(inst.bin_height - space.bin_height)  # should be 0!
        errors += abs(inst.n_items - space.n_items)  # should be 0!
        if errors > 0:
            raise ValueError(
                f"Instance {inst} is inconsistent with space {space}.")

        # These errors are permitted.
        n_different: Final[int] = inst.n_different_items
        errors += abs(n_different - space.n_different_items)  # > 0
        goal_min_width: Final[int] = space.item_width_min
        goal_max_width: Final[int] = space.item_width_max
        goal_min_height: Final[int] = space.item_height_min
        goal_max_height: Final[int] = space.item_height_max

        actual_min_width: int = space.bin_width
        actual_max_width: int = 0
        actual_min_height: int = space.bin_height
        actual_max_height: int = 0
        total_area: int = 0

        for row in range(n_different):
            width: int = int(inst[row, IDX_WIDTH])
            height: int = int(inst[row, IDX_HEIGHT])
            actual_min_width = min(actual_min_width, width)
            actual_max_width = max(actual_max_width, width)
            actual_min_height = min(actual_min_height, height)
            actual_max_height = max(actual_max_height, height)

            n: int = int(inst[row, IDX_REPETITION])
            total_area += n * width * height
            if width < goal_min_width:
                errors += n * (goal_min_width - width)
            elif width > goal_max_width:
                errors += n * (width - goal_max_width)
            if height < goal_min_height:
                errors += n * (goal_min_height - height)
            elif height > goal_max_height:
                errors += n * (height - goal_max_height)

        errors += abs(actual_min_width - goal_min_width)
        errors += abs(actual_max_width - goal_max_width)
        errors += abs(actual_min_height - goal_min_height)
        errors += abs(actual_max_height - goal_max_height)

        if total_area != inst.total_item_area:
            raise ValueError("Invalid total area?")
        errors += abs(total_area - space.total_item_area)

        return max(0.0, min(1.0, sqrt(errors / self.__max_errors)))

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this instance.

        :param logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value("maxErrors", self.__max_errors)

    def lower_bound(self) -> float:
        """
        Get the lower bound of the instance template deviations.

        :returs 0.0: always
        """
        return 0.0

    def is_always_integer(self) -> bool:
        """
        Return `True` because there are only integer errors.

        :retval False: always
        """
        return False

    def upper_bound(self) -> float:
        """
        Get the upper bound of the number of deviations.

        :returs 1.0: always
        """
        return 1.0

    def __str__(self) -> str:
        """
        Get the name of the errors objective function.

        :return: `errors`
        :retval "errors": always
        """
        return "errors"
