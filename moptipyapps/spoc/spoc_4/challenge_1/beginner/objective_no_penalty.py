"""
The objective function of the beginner problem.

>>> inst = Instance("matching-i")
>>> obj = BeginnerObjectiveNP(inst)

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
-7903

>>> -267 - 7636
-7903
"""

from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from pycommons.types import type_error

from moptipyapps.spoc.spoc_4.challenge_1.beginner.instance import Instance


@numba.njit(cache=True, inline="always", fastmath=False, boundscheck=False)
def _compute(x: np.ndarray, data: np.ndarray) -> int:
    """
    Compute the objective function of the beginner problem.

    :param x: the candidate solution
    :param data: the orbit data
    :return: the objective value, minus the offset
    """
    result: int = 0
    for i, use in enumerate(x):
        if use:
            result -= data[i, 3]
    return int(result)


class BeginnerObjectiveNP(Objective):
    """The objective function of the beginner problem."""

    def __init__(self, instance: Instance) -> None:
        """
        Create the objective function of the beginner problem.

        :param instance: the instance of the objective function.
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the instance
        self.instance: Final[Instance] = instance

    def evaluate(self, x) -> int:
        """
        Evaluate the objective function of the beginner problem.

        :param x: the solution vector
        :return: the result
        """
        return _compute(x, self.instance)

    def lower_bound(self) -> int:
        """
        Get the lower bound.

        :return: the lower bound
        """
        return 1 - self.instance.penalty

    def upper_bound(self) -> int:
        """
        Get the upper bound.

        :return: the upper bound
        """
        return 0

    def __str__(self) -> str:
        """
        Get the name of the objective function.

        :return: the name of the objective function
        """
        return "spoc4beginnerNP"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("i") as inst:
            self.instance.log_parameters_to(inst)
