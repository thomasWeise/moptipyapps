"""The tour length objective function."""

from typing import Final

import numba  # type: ignore  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.types import type_error

from moptipyapps.tsp.instance import Instance


@numba.njit(cache=True, inline="always")
def tour_length(instance: np.ndarray, x: np.ndarray) -> int:
    """
    Compute the length of a tour.

    :param instance: the distance matrix
    :param x: the tour
    :return: the length of the tour `x`
    """
    result: int = 0
    last: int = x[-1]
    for cur in x:
        result = result + instance[last, cur]
        last = cur
    return result


class TourLength(Objective):
    """The tour length objective function."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the tour length objective function.

        :param instance: the tsp instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the TSP instance we are trying to solve
        self.instance: Final[Instance] = instance
