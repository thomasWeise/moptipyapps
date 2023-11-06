"""
The objective function for the Quadratic Assignment Problem.

>>> inst = Instance.from_resource("bur26a")
>>> QAPObjective(inst).evaluate(np.array([
...     25, 14, 10, 6, 3, 11, 12, 1, 5, 17, 0, 4, 8, 20,
...     7, 13, 2, 19, 18, 24, 16, 9, 15, 23, 22, 21]))
5426670

>>> inst = Instance.from_resource("nug12")
>>> QAPObjective(inst).evaluate(np.array([
...     11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1]))
578

>>> inst = Instance.from_resource("tai12a")
>>> QAPObjective(inst).evaluate(np.array([
...     7, 0, 5, 1, 10, 9, 2, 4, 8, 6, 11, 3]))
224416
"""


from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error

from moptipyapps.qap.instance import Instance
from moptipyapps.shared import SCOPE_INSTANCE


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def _evaluate(x: np.ndarray, distances: np.ndarray, flows: np.ndarray) -> int:
    """
    Evaluate a solution to the QAP.

    :param x: the permutation representing the solution
    :param distances: the distance matrix
    :param flows: the flow matrix
    :return: the objective value
    """
    result: int = 0
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            result += flows[i, j] * distances[xi, xj]
    return int(result)


class QAPObjective(Objective):
    """An objective function for the quadratic assignment problem."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the QAP objective function.

        :param instance: the one-dimensional ordering problem.
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the instance data
        self.instance: Final[Instance] = instance

    def evaluate(self, x) -> float:
        """
        Compute the quadratic assignment problem objective value.

        :param x: the permutation of elements
        :return: the sum of flows times distances
        """
        return _evaluate(x, self.instance.distances, self.instance.flows)

    def __str__(self):
        """
        Get the name of this objective.

        :return: `"qap"`
        """
        return "qap"

    def lower_bound(self) -> float:
        """
        Get the lower bound of this objective function.

        :return: the lower bound
        """
        return self.instance.lower_bound

    def upper_bound(self) -> float:
        """
        Get the upper bound of this objective function.

        :return: the upper bound
        """
        return self.instance.upper_bound

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_INSTANCE) as scope:
            self.instance.log_parameters_to(scope)
