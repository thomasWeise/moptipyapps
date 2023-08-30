"""
A surrogate system model-based CMA-ES approach.

The idea is that we divide the computational budget into a warmup and a
model-based phase. In the warmup phase, we use CMA-ES to normally optimize
the controller based on the actual simulation of the dynamic system.
However, while doing so, for every time step in the simulation, we collect
three things: The current state vector `s`, the control vector `c`, and the
resulting state differential `ds/dt`. Now if we have such data, we can look
at the dynamic system as a function `F(s, c) = ds/dt`. If we consider the
dynamic system to be such a function and we have collected the vectors
`(s, c)` and `ds/dt`, then we may as well attempt to *learn* this system.
So after the warmup phase, our algorithm does the following: In a loop (for
the rest of the computational budget), it first tries to learn a model `M` of
`F`. Then, it replaces the actual differential equations of the system in ODE
solution approach of the objective function with `M`. In other words, we kick
out the actual system and instead use the learned system model `M`. We replace
the differential equations that describe the system using `M`. We can now
run an entire optimization process on this learned model only, with ODE
integration and all. This optimization process gives us one new solution which
we then evaluate on the real objective function (which costs 1 FE and gives us
a new heap of `(s, c)` and `ds/dt` vectors). With this new data, we again
learn a new and hopefully more accurate model `M`. This process is iterated
until the rest of the computational budget is exhausted.

This approach allows us to learn a model of a dynamic system while
synthesizing a controller for it. Since we can have infinite more time to
synthesize the controller on a learned system model compared to an actual
model, this may give us much better results.
"""


from typing import Callable, Final

import numba  # type: ignore
import numpy as np
from moptipy.algorithms.so.vector.cmaes_lib import BiPopCMAES
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.api.process import Process
from moptipy.api.subprocesses import for_fes
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_FLOAT, rand_seed_generate
from moptipy.utils.types import check_int_range, type_error
from numpy.random import Generator

from moptipyapps.dynamic_control.model_objective import ModelObjective
from moptipyapps.dynamic_control.objective import FigureOfMerit
from moptipyapps.dynamic_control.system_model import SystemModel


class SurrogateCmaEs(Algorithm):
    """A surrogate model-based CMA-ES algorithm."""

    def __init__(self,
                 system_model: SystemModel,
                 controller_space: VectorSpace,
                 objective: FigureOfMerit,
                 fes_for_warmup: int,
                 fes_for_training: int,
                 fes_per_model_run: int) -> None:
        """
        Initialize the algorithm.

        :param system_model: the system and model
        :param controller_space: the controller space
        :param objective: the figure of merit
        :param fes_for_warmup: the number of objective function evaluations
            (FEs) to be used on the initial stage on the real system for
            warming up
        :param fes_for_training: the number of FEs to be used to train the
            model
        :param fes_per_model_run: the number of FEs to be applied to each
            optimization run on the model
        """
        super().__init__()

        if not isinstance(system_model, SystemModel):
            raise type_error(system_model, "system_model", SystemModel)
        if not isinstance(controller_space, VectorSpace):
            raise type_error(
                controller_space, "controller_space", VectorSpace)
        if not isinstance(objective, FigureOfMerit):
            raise type_error(objective, "objective", FigureOfMerit)

        #: the number of objective function evaluations to be used for warmup
        self.fes_for_warmup: Final[int] = check_int_range(
            fes_for_warmup, "fes_for_warmup", 1, 1_000_000)
        #: the FEs for training the model
        self.fes_for_training: Final[int] = check_int_range(
            fes_for_training, "fes_for_training", 1, 1_000_000)
        #: the FEs for each run on the model
        self.fes_per_model_run: Final[int] = check_int_range(
            fes_per_model_run, "fes_per_model_run", 1, 1_000_000)
        #: the system model
        self.system_model: Final[SystemModel] = system_model
        #: the internal CMA-ES algorithm
        self.__control_cma: Final[BiPopCMAES] = BiPopCMAES(controller_space)
        #: the model parameter space
        self.__model_space: Final[VectorSpace] = \
            system_model.model.parameter_space()
        #: the model cma
        self.__model_cma: Final[BiPopCMAES] = BiPopCMAES(self.__model_space)
        #: the control objective function reference
        self.__control_objective: Final[FigureOfMerit] = objective
        #: the model objective
        self.__model_objective: Final[ModelObjective] = ModelObjective(
            objective, system_model.model)

    def solve(self, process: Process) -> None:
        """Solve the modelling problem."""
        should_terminate: Final[Callable[[], bool]] = process.should_terminate
        model_space: Final[VectorSpace] = self.__model_space
        model_objective: Final[ModelObjective] = self.__model_objective
        model = model_space.create()
        model_equations: Final[Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]] =\
            self.system_model.model.controller
        merged: Final[np.ndarray] = np.empty(
            self.system_model.system.state_dims
            + self.system_model.system.control_dims, DEFAULT_FLOAT)
        raw: Final[FigureOfMerit] = self.__control_objective
        random: Final[Generator] = process.get_random()
        training_execute: Final[Execution] = (
            Execution().set_solution_space(model_space)
            .set_max_fes(self.fes_for_training)
            .set_algorithm(self.__model_cma)
            .set_objective(model_objective))
        on_model_execute: Final[Execution] = \
            (Execution().set_solution_space(self.__control_cma.space)
             .set_objective(raw)
             .set_max_fes(self.fes_per_model_run)
             .set_algorithm(self.__control_cma))
        result: Final[np.ndarray] = self.__control_cma.space.create()


# First, we do the setup run that creates some basic results and
# gathers the initial information for modelling the system.
        with for_fes(process, self.fes_for_warmup) as prc:
            self.__control_cma.solve(prc)

        while not should_terminate():  # until budget exhausted

            # We now train a model on the data that was gathered.
            training_execute.set_rand_seed(rand_seed_generate(random))
            model_objective.begin()
            with training_execute.execute() as sub:
                sub.get_copy_of_best_y(model)
            model_objective.end()

# The trained model is wrapped into an equation function that can be passed to
# the ODE integrator.
            @numba.njit(cache=False, inline="always", fastmath=True,
                        boundscheck=False)
            def __new_model(state: np.ndarray, time: float,
                            control: np.ndarray, out: np.ndarray,
                            _merged=merged, _params=model,
                            _eq=model_equations) -> None:
                sd: Final[int] = len(state)
                _merged[0:sd] = state
                _merged[sd:] = control
                _eq(_merged, time, _params, out)

# OK, now that we got the model, we can perform the model optimization run.
            raw.set_model(__new_model)
            on_model_execute.set_rand_seed(rand_seed_generate(random))
            with on_model_execute.execute() as ome:
                ome.get_copy_of_best_y(result)
            raw.set_raw()

# Finally, we re-evaluate the result that we got from the model run on the
# actual objective function.
            process.evaluate(result)

    def __str__(self):
        """
        Get the name of the algorithm.

        :return: the algorithm name
        """
        return (f"surrogateCma_{self.system_model.model}_"
                f"{self.fes_for_warmup}_{self.fes_for_training}"
                f"_{self.fes_per_model_run}")

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("fesForWarmup", self.fes_for_warmup)
        logger.key_value("fesForTraining", self.fes_for_training)
        logger.key_value("fesPerModelRun", self.fes_per_model_run)
        with logger.scope("ctrlCMA") as ccma:
            self.__control_cma.log_parameters_to(ccma)
        with logger.scope("mdlSpace") as mspce:
            self.__model_space.log_parameters_to(mspce)
        with logger.scope("ctrlF") as cf:
            self.__control_objective.log_parameters_to(cf)
        with logger.scope("mdlF") as mf:
            self.__model_objective.log_parameters_to(mf)
        with logger.scope("mdlCMA") as mcma:
            self.__model_cma.log_parameters_to(mcma)
