"""
An objective functions for the dynamic control problem.

The dynamic control problem means that our system starts in a given state
and we try to move it to a stable state by using control effort. Controllers
are trained over several training states, for each of which we can compute a
figure of merit.

We offer two different approaches for this:

- :class:`FigureOfMerit` computes the arithmetic mean `z `  over the separate
  figures of merit of the training cases.
- :class:`FigureOfMeritLE` tries to smooth out the impact of bad starting
  states by computing `exp(mean[log(z + 1)]) - 1`.

These objective functions also offer a way to collect the state+control and
corresponding differential vectors.
"""

from typing import Callable, Final

import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
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
            just applied to the actual instance
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

    def initialize(self) -> None:
        """Initialize the objective for use."""
        super().initialize()
        if self.__collection is not None:
            self.__collection.clear()
        self.set_raw()

    def set_raw(self) -> None:
        """Let this objective work on the raw original equations."""
        self.__equations = self.instance.system.equations
        self.__collect = self.__collection is not None

    def get_differentials(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the collected differentials.

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

        :param equations: the equations to be used
        """
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

        for i, start in enumerate(training):
            ode = run_ode(
                start, equations, controller, x, controller_dim, steps)
            results[i] = j_from_ode(ode, state_dim)
            if collector is not None:
                collector(diff_from_ode(ode, state_dim))
        return self.sum_up_results(results)

    def sum_up_results(self, results: np.ndarray) -> float:
        """
        Compute the final objective value from several single `J` values.

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
    states. Averaging over the `log(z+1)` reduces such impact. We then compute
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

        :param results: the array of `J` values
        :return: the final result
        """
        return float(np.expm1(np.log1p(results, results).mean()))
