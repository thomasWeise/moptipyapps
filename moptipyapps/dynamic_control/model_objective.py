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

The objective function in this module minimizes the root mean square error
over the model-computed `ds/dt` vectors and the actual `ds/dt` vectors.
The model objective function is used by the
:mod:`~moptipyapps.dynamic_control.surrogate_cma` algorithm.
"""
from typing import Callable, Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.objective import FigureOfMerit


@numba.njit(cache=False, inline="always", fastmath=True, boundscheck=False)
def _evaluate(x: np.ndarray, pin: np.ndarray, pout: np.ndarray,
              temp_1: np.ndarray, temp_2: np.ndarray,
              eq: Callable[[np.ndarray, float, np.ndarray,
                            np.ndarray], None]) -> float:
    """
    Compute the RMSE differences between expected and actual model output.

    :param x: the model parameterization
    :param pin: the input vectors
    :param pout: the expected output vectors, flattened
    :param temp_1: the long temporary array to receive the result values
    :param temp_2: the temporary array to receive the state output
    :param eq: the equations
    :return: the mean of the `log(x+1)` of the squared differences `x`

    >>> vx = np.array([0.5, 0.2], float)
    >>> vpin = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], float)
    >>> vpout = np.array([[10, 5], [14, 9], [18, 8]], float)
    >>> vtemp_1 = np.empty(3, float)
    >>> vtemp_2 = np.empty(2, float)
    >>> @numba.njit(cache=False, inline="always", fastmath=True)
    ... def _func(lstate: np.ndarray, _: float, lx: np.ndarray,
    ...           lout: np.ndarray) -> None:
    ...     lout[:] = (lstate[0:2] * lx[0]) - lx[1] - lstate[-1]
    >>> _evaluate(vx, vpin, vpout, vtemp_1, vtemp_2, _func)
    22.680823177337874
    >>> y_1_1 = ((0 * 0.5) - 0.2 - 3) - 10
    >>> y_1_2 = ((1 * 0.5) - 0.2 - 3) - 5
    >>> r_1 = y_1_1 ** 2 + y_1_2 ** 2
    >>> y_2_1 = ((4 * 0.5) - 0.2 - 7) - 14
    >>> y_2_2 = ((5 * 0.5) - 0.2 - 7) - 9
    >>> r_2 = y_2_1 ** 2 + y_2_2 ** 2
    >>> y_3_1 = ((8 * 0.5) - 0.2 - 11) - 18
    >>> y_3_2 = ((9 * 0.5) - 0.2 - 11) - 8
    >>> r_3 = y_3_1 ** 2 + y_3_2 ** 2
    >>> np.sqrt(np.array([r_1, r_2, r_3])).mean()
    22.680823177337874
    """
    for i, row in enumerate(pin):  # iterate over all row=(s, c) tuples
        eq(row, 0.0, x, temp_2)  # store the equation results
        temp_1[i] = np.square(np.subtract(temp_2, pout[i], temp_2),
                              temp_2).sum()
    return np.sqrt(temp_1, temp_1).mean()


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
        #: the equations of the model
        self.__equations: Callable[[np.ndarray, float, np.ndarray,
                                    np.ndarray], None] = model.controller
        #: the result
        self.__temp_1: np.ndarray | None = None
        #: the input
        self.__in: np.ndarray | None = None
        #: the output
        self.__out: np.ndarray | None = None
        #: the differentials getter
        self.__get_differentials: Final[Callable[[], tuple[
            np.ndarray, np.ndarray]]] = real.get_differentials
        #: the temporary array
        self.__temp_2: Final[np.ndarray] = np.empty(
            real.instance.system.state_dims, DEFAULT_FLOAT)
        #: the real figure of merit name
        self.__real_name: Final[str] = str(real)

    def begin(self) -> None:
        """
        Begin a model optimization run.

        This function pulls the training data from the actual
        controller-synthesis objective function, which is an instance of
        :class:`~moptipyapps.dynamic_control.objective.FigureOfMerit`, via the
        method :meth:`~moptipyapps.dynamic_control.objective.FigureOfMerit.\
get_differentials` and allocates internal data structures accordingly.
        """
        self.__in, self.__out = self.__get_differentials()
        self.__temp_1 = np.empty(len(self.__out), self.__out.dtype)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate a model parameterization.

        :param x: the model parameterization
        :return: the objective value
        """
        return _evaluate(x, self.__in, self.__out, self.__temp_1,
                         self.__temp_2, self.__equations)

    def end(self) -> None:
        """End a model optimization run and free the associated memory."""
        self.__temp_1 = None
        self.__in = None
        self.__out = None

    def __str__(self) -> str:
        """
        Get the name of this objective.

        :return: the name of this objective
        """
        return "modelRMSE"

    def lower_bound(self) -> float:
        """
        Get the lower bound of the model objective, which is `0`.

        :returns: 0.0
        """
        return 0.0

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("figureOfMeritName", self.__real_name)
        logger.key_value("nSamples",
                         0 if self.__in is None else len(self.__in))
