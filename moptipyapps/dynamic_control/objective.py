"""
An objective functions for the dynamic control problem.

The dynamic control problem means that our system starts in a given state
and we try to move it to a stable state by using control effort. Controllers
are trained over several training states, for each of which we can compute a
figure of merit.

We offer two different approaches for this:

- :class:`FigureOfMerit` computes the arithmetic mean `z` over the separate
  figures of merit `J` of the training cases.
- :class:`FigureOfMeritLE` tries to smooth out the impact of bad starting
  states by computing `exp(mean[log(J + 1)]) - 1`.

These objective functions also offer a way to collect the state+control and
corresponding differential vectors.
"""

from math import expm1
from typing import Callable, Final

import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import array_to_str
from moptipy.utils.types import type_error

from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.ode import diff_from_ode, j_from_ode, run_ode
from moptipyapps.shared import SCOPE_INSTANCE


class FigureOfMerit(Objective):
    """A base class for figures of merit."""

    def __init__(self, instance: Instance,
                 supports_model_mode: bool = False) -> None:
        """
        Create the figure-of-merit objective of the dynamic control problem.

        :param instance: the instance
        :param supports_model_mode: `True` if this objective is supposed to
            support alternating actual and model-based runs, `False` if it is
            just applied to the actual instance (see :meth:`set_model` and
            :meth:`get_differentials`).
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the dynamic control instance
        self.instance: Final[Instance] = instance
        #: the simulation steps
        self.__steps: Final[int] = instance.system.training_steps
        #: the training time
        self.__time: Final[float] = instance.system.training_time
        #: the training starting states
        self.__training: Final[np.ndarray] = \
            instance.system.training_starting_states
        #: the results
        self.__results: Final[np.ndarray] = np.empty(
            len(self.__training), float)
        #: the equations
        self.__equations: Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None] = \
            instance.system.equations
        #: the controller
        self.__controller: Final[Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]] = \
            instance.controller.controller
        #: the controller dimension
        self.__controller_dim: Final[int] = instance.controller.control_dims
        #: the collection of state+control and differential vectors
        self.__collection: Final[list[tuple[np.ndarray, np.ndarray]] | None] \
            = [] if supports_model_mode else None
        #: should we collect data?
        self.__collect: bool = self.__collection is not None
        #: the state dimensions inside the `J`
        self.__state_dims_in_j: Final[int] = instance.system.state_dims_in_j
        #: the weight of the control effort
        self.__gamma: Final[float] = instance.system.gamma

    def initialize(self) -> None:
        """Initialize the objective for use."""
        super().initialize()
        if self.__collection is not None:
            self.__collection.clear()
        self.set_raw()

    def set_raw(self) -> None:
        """
        Let this objective work on the original system equations.

        The objective function here can be used in two modi: a) based on the
        original systems model, as given in
        :attr:`~moptipyapps.dynamic_control.instance.Instance.system`, or b)
        on a learned model of the system. This function here toggles to the
        former mode, i.e., to the actual system mode. In this modus, training
        data for training the system model will be gathered if the objective
        function is configured to do so. In that case, you can toggle to model
        mode via :meth:`set_model`.
        """
        self.__equations = self.instance.system.equations
        self.__collect = self.__collection is not None

    def get_differentials(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the collected differentials.

        If `supports_model_mode` was set to `True` in the creating of this
        objective function, then the system will gather tuples `(s, c)` and
        `ds/dt` when in raw mode (see :meth:`set_raw`) and make them available
        here to train system models (see :meth:`set_model`). Notice that
        gathering training data is a very memory intense process.

        :returns: the collected differentials
        """
        cl: Final[list[tuple[np.ndarray, np.ndarray]] | None] = \
            self.__collection
        if cl is None:
            raise ValueError("Differential collection not supported.")
        if len(cl) == 1:
            return cl[0]  # pylint: disable=E1136
        result = (np.concatenate([t[0] for t in cl]),  # pylint: disable=E1133
                  np.concatenate([t[1] for t in cl]))  # pylint: disable=E1133
        cl.clear()
        cl.append(result)
        return result

    def set_model(self, equations: Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]) -> None:
        """
        Set the model-driven mode for the evaluation.

        In this modus, the internal system equations are replaced by the
        callable `equations` passed into this function and the data collection
        is stopped. The idea is that `equations` could be a model synthesized
        on the data gathered (see :meth:`get_differentials`) and thus does not
        represent the actual dynamic system but a model thereof. We could
        synthesize a controller for this model and for this purpose would use
        the exactly same objective function -- just instead of using the
        actual system equations, we use the system model. Of course, we then
        need to deactivate the data gathering mechanism (see again
        :meth:`get_differentials`), because the data would then not be real
        system data. You can toggle back to the actual system using
        :meth:`set_raw`.

        :param equations: the equations to be used instead of the actual
            system's differential equations.
        """
        if self.__collection is None:
            raise ValueError("Cannot go into model mode without gathering "
                             "model training data!")
        self.__equations = equations
        self.__collect = False

    def __str__(self) -> str:
        """
        Get the name of this objective function.

        :return: `figureOfMerit`
        """
        return "figureOfMerit"

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the parameterization of a controller.

        :param x: the controller parameters
        :return: the figure of merit
        """
        steps: Final[int] = self.__steps
        time: Final[float] = self.__time
        training: Final[np.ndarray] = self.__training
        results: Final[np.ndarray] = self.__results
        equations: Final[Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]] \
            = self.__equations
        controller: Final[
            Callable[[np.ndarray, float, np.ndarray, np.ndarray], None]] \
            = self.__controller
        controller_dim: Final[int] = self.__controller_dim
        collector: Final[Callable[[tuple[
            np.ndarray, np.ndarray]], None] | None] = \
            self.__collection.append if self.__collect else None
        state_dim: Final[int] = len(training[0])
        state_dims_in_j: Final[int] = self.__state_dims_in_j
        gamma: Final[float] = self.__gamma

        for i, start in enumerate(training):
            # The following line makes no sense at all. It creates a copy of
            # the flattened version of the (already flat) start. The copy is
            # stored nowhere, so it is immediately discarded. The value of
            # start is not changed. However, the numpy array container
            # changes, for an unclear reason. This is required and it must
            # happen exactly here, for an unclear reason. Otherwise, the
            # results of the objective function are inconsistent. For an
            # unclear reason.
            np.copy(start.flatten())  # <--- This should make no sense...
            ode = run_ode(
                start, equations, controller, x, controller_dim, steps, time)
            results[i] = z = j_from_ode(ode, state_dim, state_dims_in_j, gamma)
            if not (0.0 <= z <= 1e100):
                return 1e200
            if collector is not None:
                collector(diff_from_ode(ode, state_dim))
        z = self.sum_up_results(results)
        return z if 0.0 <= z <= 1e100 else 1e200

    def sum_up_results(self, results: np.ndarray) -> float:
        """
        Compute the final objective value from several single `J` values.

        When synthesizing controllers, we do not just apply them to a single
        simulation run. Instead, we use multiple training cases (see
        :attr:`~moptipyapps.dynamic_control.system.System.\
training_starting_states`) and perform :attr:`~moptipyapps.dynamic_control\
.system.System.training_steps` simulation steps on each of them. Each such
        training starting state will result in a single `J` value, which is
        the sum of squared state and control values. We now compute the end
        objective value from these different `J` values by using this
        function here.

        This will *destroy* the contents of `results`.

        :param results: the array of `J` values
        :return: the final result
        """
        return float(results.mean())

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("modelModeEnabled", self.__collection is not None)
        logger.key_value("dataCollecting", self.__collect)
        eq: Final = self.__equations
        logger.key_value("usingOriginalEquations",
                         eq is self.instance.system.equations)
        mp: Final[str] = "modelParameters"
        if hasattr(self.__equations, mp):
            logger.key_value(mp, array_to_str(getattr(eq, mp)))
        with logger.scope(SCOPE_INSTANCE) as scope:
            self.instance.log_parameters_to(scope)

    def lower_bound(self) -> float:
        """
        Get the lower bound of the figure of merit, which is 0.

        :returns: 0.0
        """
        return 0.0


class FigureOfMeritLE(FigureOfMerit):
    """
    Compute a `exp(mean(log(z+1)))-1` over the figures of merit `z`.

    Different from :class:`FigureOfMerit`, we compute the mean of `log(z + 1)`
    where `z` be the figures of merit of the single training cases. We then
    return `exp(mean[log(z + 1)]) - 1` as final result. The goal is to reduce
    the impact of training cases that require more control effort.

    If we solve the dynamic control problem for diverse training cases, then
    we may have some very easy cases, where the system just needs a small
    control impulse to move into a stable and cheap state. Others may have
    very far out and expensive starting states that require lots of control
    efforts to be corrected. If we simply average over all states, then these
    expensive states will dominate whatever good we are doing in the cheap
    states. Averaging over the `log(J+1)` reduces such impact. We then compute
    `exp[...]-1` of the result as cosmetics to get back into the original
    range of the figure of merits.
    """

    def __str__(self) -> str:
        """
        Get the name of this objective function.

        :return: `figureOfMeritLE`
        """
        return "figureOfMeritLE"

    def sum_up_results(self, results: np.ndarray) -> float:
        """
        Compute the final objective value from several single `J` values.

        For each training case, there is one basic figure of merit `J` and
        here we compute `exp(mean[log(J + 1)]) - 1` over all of these values.

        :param results: the array of `J` values
        :return: the final result
        """
        return float(expm1(np.log1p(results, results).mean()))
