"""
An alternative objective function for the dynamic control problem.

Different from :mod:`~moptipyapps.dynamic_control.objective`, we compute
the mean of `log(z + 1)` where `z` be the figures of merit of the single
training cases. We then return `exp(mean[log(z + 1)]) - 1` as final
result. The goal is to reduce the impact of training cases that require
more control effort.

If we solve the dynamic control problem for diverse training cases, then we
may have some very easy cases, where the system just needs a small control
impulse to move into a stable and cheap state. Others may have very far out
and expensive starting states that require lots of control efforts to be
corrected. If we simply average over all states, then these expensive states
will dominate whatever good we are doing in the cheap states. Averaging over
the `log(z+1)` reduces such impact. We then compute `exp[...]-1` of the result
as cosmetics to get back into the original range of the figure of merits.
"""

from math import expm1
from typing import Callable, Final

import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error

from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.ode import j_from_ode, run_ode
from moptipyapps.shared import SCOPE_INSTANCE


class FigureOfMeritLE(Objective):
    """Compute a `exp(mean(log(z+1)))-1` over the figures of merit `z`."""

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

        :return: `figureOfMeritLE`
        """
        return "figureOfMeritLE"

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
        return expm1(np.log1p(results).mean())

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
