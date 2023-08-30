"""
An objective function that can be used to synthesize system models.

We consider the dynamic system to be a function `F(s, c) = ds/dt`, where `s`
is the state vector (e.g., two-dimensional for the Stuart-Landau system and
three-dimensional for the Lorenz system), `c` is the control vector
(one-dimensional in both cases), and `ds/dt` is the state differential (again
two respectively three dimensional). The
:mod:`~moptipyapps.dynamic_control.objective` `FigureOfMerit` allows us to
collect tuples of `(s,c)` and `ds/dt` vectors. So all we need to do is to
train a model `M` that receives as input a vector `x=(s,c)` (where `(s,c)` be
the concatenation of `s` and `c`) and fill as output a vector `ds/dt`.
The objective function in this module minimizes the mean square error over the
model-computed `ds/dt` vectors and the actual `ds/dt` vectors. It does so by
using the `expm1(mean(logp1(...))))` hack of the `FigureOfMeritLE` function if
`FigureOfMeritLE` was used, and otherwise minimizes the mean-square-error
directly.
"""

from typing import Callable, Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.objective import FigureOfMerit


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def _evaluate(x: np.ndarray, dim: int, pin: np.ndarray,
              pout: np.ndarray, res: np.ndarray,
              eq: Callable[[np.ndarray, float, np.ndarray,
                            np.ndarray], None]) -> np.ndarray:
    """
    Evaluate a model parameterization.

    :param x: the model parameterization
    :param dim: the dimension of state vector
    :param pin: the input vectors
    :param pout: the expected output vectors, flattened
    :param eq: the equations
    :return: the vector of squared differences
    """
    idx: int = 0

    for row in pin:
        nidx: int = idx + dim
        eq(row, 0.0, x, res[idx:nidx])
        idx = nidx
    return np.square(np.subtract(res, pout, res), res)


class ModelObjective(Objective):
    """The objective for modeling."""

    def __init__(self, real: FigureOfMerit,
                 model: Controller) -> None:
        """
        Create a model objective compatible to the given figure of merit.

        :param real: the objective used for the real optimization problem
        :param model: the model
        """
        n1 = str(real)
        n2 = "figureOfMerit"
        if n1.startswith(n2):
            n1 = n1[len(n2):]
        #: the name of this objective
        self.name: Final[str] = f"model{n1}"
        #: the result summation method
        self.__sum: Callable[[np.ndarray], float] = real.sum_up_results
        #: the equations of the model
        self.__equations: Callable[[np.ndarray, float, np.ndarray,
                                    np.ndarray], None] = model.controller
        #: the result
        self.__res: np.ndarray | None = None
        #: the input
        self.__in: np.ndarray | None = None
        #: the output
        self.__out: np.ndarray | None = None
        #: the differentials getter
        self.__get_differential: Final[Callable[[], tuple[
            np.ndarray, np.ndarray]]] = real.get_differentials
        #: the dimension of the output
        self.__dim: Final[int] = real.instance.system.state_dims

    def begin(self) -> None:
        """Begin a model optimization run."""
        self.__in, out = self.__get_differential()
        self.__out = out.flatten()
        self.__res = np.empty(len(self.__out), self.__out.dtype)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate a model parameterization.

        :param x: the model parameterization
        :return: the objective value
        """
        return self.__sum(_evaluate(x, self.__dim, self.__in, self.__out,
                                    self.__res, self.__equations))

    def end(self) -> None:
        """End a model optimization run."""
        self.__res = None
        self.__in = None
        self.__out = None

    def __str__(self) -> str:
        """
        Get the name of this objective.

        :return: the name of this objective
        """
        return self.name

    def lower_bound(self) -> float:
        """
        Get the lower bound of the model objective which is 0.

        :returns: 0.0
        """
        return 0.0
