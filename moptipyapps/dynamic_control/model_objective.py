"""
An objective function that can be used to synthesize system models.

We consider the dynamic system to be a function `D(s, c) = ds/dt`, where

- `s` is the state vector (e.g., two-dimensional for the Stuart-Landau system,
  see :mod:`~moptipyapps.dynamic_control.systems.stuart_landau`, and
  three-dimensional for the Lorenz system, see
  :mod:`~moptipyapps.dynamic_control.systems.lorenz`),
- `c` is the control vector (one-dimensional in both cases), and
- `ds/dt` is the state differential (again two respectively three-dimensional).

The :class:`~moptipyapps.dynamic_control.objective.FigureOfMerit` objective
function allows us to collect tuples of `(s,c)` and `ds/dt` vectors. So all we
need to do is to train a model `M` that receives as input a vector `x=(s,c)`
(where `(s,c)` be the concatenation of `s` and `c`) and fill as output a vector
`ds/dt`.

The objective function in this module minimizes the mean square error over the
model-computed `ds/dt` vectors and the actual `ds/dt` vectors. It does so by
using the `expm1(mean(logp1(...))))` hack of the
:class:`~moptipyapps.dynamic_control.objective.FigureOfMeritLE` function if
:class:`~moptipyapps.dynamic_control.objective.FigureOfMeritLE` was the
controller synthesis objective. Otherwise, if
:class:`~moptipyapps.dynamic_control.objective.FigureOfMerit` is the objective
for controller synthesis, it minimizes the mean-square-error directly.

The model objective function is used by the
:mod:`~moptipyapps.dynamic_control.surrogate_cma` algorithm.
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
    Compute the squared differences between expected and actual model output.

    :param x: the model parameterization
    :param dim: the dimension of state vector
    :param pin: the input vectors
    :param pout: the expected output vectors, flattened
    :param eq: the equations
    :return: the vector of squared differences
    """
    idx: int = 0
    for row in pin:  # iterate over all row=(s, c) tuples
        nidx: int = idx + dim
        eq(row, 0.0, x, res[idx:nidx])  # store the equation results
        idx = nidx
    return np.square(np.subtract(res, pout, res), res)  # compute squared diff


class ModelObjective(Objective):
    """
    The objective for modeling.

    This objective function works on sequences of tuples `(s, c)` of the
    system state `s` and controller output `c` as well as the corresponding
    `ds/dt` differentials. The goal is to train a model `M(s, c, q)` that will
    compute the `ds/dt` values reasonably accurately. The model is
    parameterized with some vector `q`, think of `M` being, e.g., an
    artificial neural network and `q` being its weight vector. To find good
    values of `q`, this objective here computes the squared differences
    between the values `M(s, c, q)` and the expected outputs `ds/dt`.

    These squared differences are then either averaged directly or the
    `expm1(mean(logp1(...))))` hack of the
    :class:`~moptipyapps.dynamic_control.objective.FigureOfMeritLE` is used to
    alleviate the impact of large differentials, depending on which "real"
    (controller-synthesis) objective is passed into the constructor).

    The tuples `(s, c)` and `ds/dt` are pulled from the real objective with
    the method :meth:`~moptipyapps.dynamic_control.model_objective.\
ModelObjective.begin`. The `ds/dt` rows are flattened for performance reasons.
    """

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
        self.__get_differentials: Final[Callable[[], tuple[
            np.ndarray, np.ndarray]]] = real.get_differentials
        #: the dimension of the output
        self.__dim: Final[int] = real.instance.system.state_dims

    def begin(self) -> None:
        """
        Begin a model optimization run.

        This function pulls the training data from the actual
        controller-synthesis objective function, which is an instance of
        :class:`~moptipyapps.dynamic_control.objective.FigureOfMerit`, via the
        method :meth:`~moptipyapps.dynamic_control.objective.FigureOfMerit.\
get_differentials` and allocates internal data structures accordingly.
        """
        self.__in, out = self.__get_differentials()
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
        """End a model optimization run and free the associated memory."""
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
        Get the lower bound of the model objective, which is `0`.

        :returns: 0.0
        """
        return 0.0
