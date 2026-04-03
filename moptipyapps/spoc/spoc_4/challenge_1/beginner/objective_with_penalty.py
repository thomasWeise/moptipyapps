"""
The objective function of the beginner problem.

>>> inst = Instance("matching-i")
>>> obj = BeginnerObjectiveP(inst)

>>> test_x = np.ndarray(inst.n, dtype=np.bool)
>>> test_x.fill(0)

>>> obj.evaluate(test_x)
0

>>> test_x[0] = 1
>>> obj.evaluate(test_x)
-267
>>> inst[0, -1]
np.int64(267)

The clash of two orbits incurs a penalty:

>>> test_x[2234] = 1
>>> obj.evaluate(test_x)
125518718

>>> inst.penalty - 267 - 7636
125518718
"""

from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection

from moptipyapps.spoc.spoc_4.challenge_1.beginner.base_obj import (
    BaseObjectWithArrays,
)
from moptipyapps.spoc.spoc_4.challenge_1.beginner.instance import Instance


@numba.njit(cache=True, inline="always", fastmath=False, boundscheck=False)
def _compute(x: np.ndarray, data: np.ndarray, penalty: int,
             earth: np.ndarray, lunar: np.ndarray, dest: np.ndarray) -> int:
    """
    Compute the objective function of the beginner problem.

    :param x: the candidate solution
    :param data: the orbit data
    :param penalty: the penalty for clashes
    :param earth: the earth orbits
    :param lunar: the lunar orbits
    :param dest: the destination orbits
    :return: the objective value, minus the offset
    """
    earth.fill(0)
    lunar.fill(0)
    dest.fill(0)
    result: int = 0
    for i, use in enumerate(x):
        if not use:
            continue
        ue, ul, ud, uo = data[i, :]
        if earth[ue]:
            result += penalty
        else:
            earth[ue] = 1
        if lunar[ul]:
            result += penalty
        else:
            lunar[ul] = 1
        if dest[ud]:
            result += penalty
        else:
            dest[ud] = 1
        result -= uo
    return int(result)


class BeginnerObjectiveP(BaseObjectWithArrays, Objective):
    """The objective function of the beginner problem."""

    def __init__(self, instance: Instance) -> None:
        """
        Create the objective function of the beginner problem.

        :param instance: the instance of the objective function.
        """
        super().__init__(instance)
        #: the penalty
        self.__p: Final[int] = instance.penalty

    def evaluate(self, x) -> int:
        """
        Evaluate the objective function of the beginner problem.

        :param x: the solution vector
        :return: the result
        """
        return _compute(x, self.instance, self.__p, self.earth,
                        self.lunar, self.dest)

    def lower_bound(self) -> int:
        """
        Get the lower bound.

        :return: the lower bound
        """
        return 1 - self.__p

    def upper_bound(self) -> int:
        """
        Get the upper bound.

        :return: the upper bound
        """
        return self.instance.n * 3 * self.__p

    def __str__(self) -> str:
        """
        Get the name of the objective function.

        :return: the name of the objective function
        """
        return "spoc4beginnerP"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("i") as inst:
            self.instance.log_parameters_to(inst)
