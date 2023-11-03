"""
An objective function evaluating a permutation of objects.

The goal is to find an order of objects such that the ranks of their distances
inside the permutation match to the distance ranks inside the instance data.
The objective function is symmetric with respect to the permutation, i.e.,
ordering the objects à la `[0, 1, 2, 3]` is the same as ordering them in
`[3, 2, 1, 0]`.

First, let's construct an instance.
>>> def _dist(a, b):
...     return abs(a - b)
>>> def _tags(a):
...     return f"t{a}"
>>> the_instance = Instance.from_sequence_and_distance(
...     [1, 2, 3, 4, 5], _tags, _dist)

Now let's create the objective function:
>>> the_objective = OneDimensionalDistribution(the_instance)
>>> the_x = np.array([0, 1, 2, 3, 4], int)
>>> the_objective.evaluate(the_x)
0.0

>>> the_x = np.array([1, 0, 2, 3, 4], int)
>>> the_objective.evaluate(the_x)
0.02878505920886873

>>> the_x = np.array([2, 1, 0, 3, 4], int)
>>> the_objective.evaluate(the_x)
0.04803169564121945

>>> the_x = np.array([4, 3, 2, 1, 0], int)
>>> the_objective.evaluate(the_x)
0.0
"""


from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.types import type_error
from scipy.stats import rankdata  # type: ignore

from moptipyapps.order1d.instance import Instance


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def _get_dist(x: np.ndarray, out: np.ndarray) -> None:
    """
    Fill a distance matrix.

    :param x: the permutation
    :param out: the output matrix

    >>> a = np.array([0, 1, 2, 3], int)
    >>> res = np.empty((len(a), len(a)), int)
    >>> _get_dist(a, res)
    >>> print(res)
    [[0 1 2 3]
     [1 0 1 2]
     [2 1 0 1]
     [3 2 1 0]]
    >>> a = np.array([2, 1, 0, 3], int)
    >>> _get_dist(a, res)
    >>> print(res)
    [[0 1 2 1]
     [1 0 1 2]
     [2 1 0 3]
     [1 2 3 0]]
    """
    lenx: Final[int] = len(x)
    for i1, x1 in enumerate(x):
        out[i1, i1] = 0
        for i2 in range(i1 + 1, lenx):
            x2 = x[i2]
            out[x1, x2] = out[x2, x1] = i2 - i1


class OneDimensionalDistribution(Objective):
    """A base class for figures of merit."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the one-dimensional distribution objective function.

        :param instance: the one-dimensional ordering problem.
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the instance data
        self.instance: Final[Instance] = instance
        #: the instance data by the power of minus three, to speed up stuff
        self.__inst3: Final[np.ndarray] = instance ** -2.0
        #: a temporary array
        self.__temp: Final[np.ndarray] = np.empty(
            instance.shape, instance.dtype)

    def evaluate(self, x) -> float:
        """
        Get the difference between the element ordering and the rank goals.

        :param x: the permutation of elements
        :return: the difference between the rank goals and orderings
        """
        temp: Final[np.ndarray] = self.__temp
        _get_dist(x, temp)
        rd: np.ndarray = (2.0 * rankdata(temp, axis=1, method="average")) - 1.0
        return np.multiply(np.abs(np.subtract(rd, self.instance, rd), rd),
                           self.__inst3, rd).mean()

    def __str__(self):
        """
        Get the name of this objective.

        :return: `"cubicRankDifference"` always
        """
        return "cubicRankDifference"

    def lower_bound(self) -> float:
        """
        Get the lower bound: always `0`.

        :return: `0.0`
        """
        return 0.0
