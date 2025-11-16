"""
The objective function for the Quadratic Assignment Problem.

The candidate solutions to the QAP are
:mod:`~moptipy.spaces.permutations` `p` of length `n` of the numbers `0`, `1`,
..., `n-1`. An :mod:`~moptipyapps.qap.instance` of the Quadratic Assignment
Problem (QAP) presents a `n*n` matrix `D` with
:attr:`~moptipyapps.qap.instance.Instance.distances` and a `n*n` matrix `F`
with flows :attr:`~moptipyapps.qap.instance.Instance.flows`. The objective
value, subject to minimization, is then the
`sum( F[i,j] * D[p[i], p[j]] for i, j in 0..n-1 )`, i.e., the sum of the
products of the flows between facilities and the distances of their assigned
locations.

1. Eliane Maria Loiola, Nair Maria Maia de Abreu, Paulo Oswaldo
   Boaventura-Netto, Peter Hahn, and Tania Querido. A survey for the
   Quadratic Assignment Problem. European Journal of Operational Research.
   176(2):657-690. January 2007. https://doi.org/10.1016/j.ejor.2005.09.032.
2. Rainer E. Burkard, Eranda Çela, Panos M. Pardalos, and
   Leonidas S. Pitsoulis. The Quadratic Assignment Problem. In Ding-Zhu Du,
   Panos M. Pardalos, eds., Handbook of Combinatorial Optimization,
   pages 1713-1809, 1998, Springer New York, NY, USA.
   https://doi.org/10.1007/978-1-4613-0303-9_27.

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
from pycommons.types import type_error

from moptipyapps.qap.instance import Instance
from moptipyapps.utils.shared import SCOPE_INSTANCE


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
