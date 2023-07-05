"""
An objective function for the dynamic control problem.

The dynamic control problem means that our system starts in a given state
and we try to move it to a stable state by using control effort. Controllers
are trained over several training states, for each of which we can compute a
figure of merit. This objective function here just averages over these figures
of merit. Maybe :mod:`~moptipyapps.dynamic_control.objective_le`, which tries
to smooth out the impact of bad starting states leads to better results,
though.
"""

from typing import Callable, Final

import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error

from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.ode import j_from_ode, run_ode
from moptipyapps.shared import SCOPE_INSTANCE


class FigureOfMerit(Objective):
    """Compute the figure of merit for the given instance."""

    def __init__(self, instance: Instance) -> None:
        """
        Create the figure-of-merit objective of the dynamic control problem.

        :param instance: the instance
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the dynamic control instance
        self.instance: Final[Instance] = instance

        #: the simulation steps
        self.__steps: Final[int] = instance.system.training_steps
        #: the training starting states
        self.__training: Final[np.ndarray] = \
            instance.system.training_starting_states
        #: the results
        self.__results: Final[np.ndarray] = np.empty(
            len(self.__training), float)
        #: the equations
        self.__equations: Final[Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]] = \
            instance.system.equations
        #: the controller
        self.__controller: Final[Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]] = \
            instance.controller.controller
        #: the controller dimension
        self.__controller_dim: Final[int] = instance.controller.control_dims

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
        training: Final[np.ndarray] = self.__training
        results: Final[np.ndarray] = self.__results
        equations: Final[Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]] \
            = self.__equations
        controller: Final[
            Callable[[np.ndarray, float, np.ndarray, np.ndarray], None]] \
            = self.__controller
        controller_dim: Final[int] = self.__controller_dim

        for i, start in enumerate(training):
            results[i] = j_from_ode(run_ode(
                start, equations, controller, x, controller_dim, steps),
                len(start))
        return results.mean()

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_INSTANCE) as scope:
            self.instance.log_parameters_to(scope)

    def lower_bound(self) -> float:
        """
        Get the lower bound of the figure of merit, which is 0.

        :returns: 0.0
        """
        return 0.0
