"""
A surrogate system model-based CMA-ES approach.

In the real world, we want to synthesize a controller `c(s, p)` that can
drive a dynamic system into a good state. The controller receives as input
the current state `s`, say, from sensor readings. It can also be
parameterized by a vector `p`, imagine `c` to be, for example, an artificial
neural network and then `p` would be its weight vector. The output of `c` will
influence the system in some way. In our example experiments, this is done
by becoming part of the state differential `ds/dt`. Anyway, in the real world,
the controller may steer the rotation speed of a rotor or the angles of rotor
blades or whatever. Now you can imagine that doing real-world experiments is
costly and takes a long time. Everytime we want to test a parameterization
`p`, some form experiment, maybe in a wind tunnel, has to be done.

So it would be beneficial if we could replace the actual experiment by a
simulation. This would mean that we learn a model `M` that can compute the
state change `ds/dt` based on the current state `s` and controller output `c`
at reasonable accuracy. If we had such a computational model, then we could
run the controller optimization process on that model. Once finished, we could
apply and evaluate the best controller that we have discovered in a real
experiment. With the observed behavior of the actual controller system, we may
even be able to update and improve our system model to become more accurate.
So we could alternate between real experiments and optimization runs on the
simulation. Of course, we would always need to do some real experiments at
first to gather the data to obtain our initial model `M`. But if we can get
this to work, then we could probably get much better controllers with fewer
actual experiments.

This here is an algorithm that tries to implement the above pattern. For the
model and controller optimization, it uses the BiPop-CMA-ES offered by
`moptipy` (:class:`~moptipy.algorithms.so.vector.cmaes_lib.BiPopCMAES`). But
it uses two instances of this algorithm, namely one to optimize the controller
parameters and one that optimizes the model parameters.

The idea is that we divide the computational budget into a warmup and a
model-based phase. In the warmup phase, we use CMA-ES to normally optimize
the controller based on the actual simulation of the dynamic system.
However, while doing so, for every time step in the simulation, we collect
three things: The current state vector `s`, the control vector `c`, and the
resulting state differential `ds/dt`. Now if we have such data, we can look
at the dynamic system as a function `D(s, c) = ds/dt`. If we consider the
dynamic system to be such a function and we have collected the vectors
`(s, c)` and `ds/dt`, then we may as well attempt to *learn* this system.
So after the warmup phase, our algorithm does the following: In a loop (for
the rest of the computational budget), it first tries to learn a model `M` of
`D`. Then, it replaces the actual differential equations of the system in ODE
solution approach of the objective function with `M`. In other words, we kick
out the actual system and instead use the learned system model `M`. We replace
the differential equations that describe the system using `M`. We can now
run an entire optimization process on this learned model only, with ODE
integration and all. This optimization process gives us one new solution which
we then evaluate on the real objective function (which costs 1 FE and gives us
a new heap of `(s, c)` and `ds/dt` vectors). With this new data, we again
learn a new and hopefully more accurate model `M`. This process is iterated
until the rest of the computational budget is exhausted.

This approach hopefully allows us to learn a model of a dynamic system while
synthesizing a controller for it. Since we can have infinitely more time to
synthesize the controller on a learned system model compared to an actual
model, this may give us much better results.

The starting points of the work here were conversations with Prof. Dr. Bernd
NOACK and Guy Yoslan CORNEJO MACEDA of the Harbin Institute of Technology in
Shenzhen, China (哈尔滨工业大学(深圳)).
"""


from copy import copy
from gc import collect
from os.path import basename
from typing import Callable, Final

import numba  # type: ignore
import numpy as np
from moptipy.algorithms.so.vector.cmaes_lib import BiPopCMAES
from moptipy.api.algorithm import Algorithm
from moptipy.api.execution import Execution
from moptipy.api.logging import FILE_SUFFIX
from moptipy.api.process import Process
from moptipy.api.subprocesses import for_fes
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import rand_seed_generate
from moptipy.utils.path import Path
from moptipy.utils.types import check_int_range, type_error
from numpy.random import Generator

from moptipyapps.dynamic_control.model_objective import ModelObjective
from moptipyapps.dynamic_control.objective import FigureOfMerit
from moptipyapps.dynamic_control.system import System
from moptipyapps.dynamic_control.system_model import SystemModel


def _nop() -> None:
    """Do absolutely nothing."""


class SurrogateCmaEs(Algorithm):
    """A surrogate model-based CMA-ES algorithm."""

    def __init__(self,
                 system_model: SystemModel,
                 controller_space: VectorSpace,
                 objective: FigureOfMerit,
                 fes_for_warmup: int,
                 fes_for_training: int,
                 fes_per_model_run: int,
                 fancy_logs: bool = False) -> None:
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
        :param fancy_logs: should we perform fancy logging?
        """
        super().__init__()

        if not isinstance(system_model, SystemModel):
            raise type_error(system_model, "system_model", SystemModel)
        if not isinstance(controller_space, VectorSpace):
            raise type_error(
                controller_space, "controller_space", VectorSpace)
        if not isinstance(objective, FigureOfMerit):
            raise type_error(objective, "objective", FigureOfMerit)
        if not isinstance(fancy_logs, bool):
            raise type_error(fancy_logs, "fancy_logs", bool)

        #: should we do fancy logging?
        self.fancy_logs: Final[bool] = fancy_logs
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
        """
        Solve the modelling problem.

        This function begins by spending :attr:`fes_for_warmup` objective
        function evaluations (FEs) on the actual problem, i.e., by trying to
        synthesize controllers for the "real" system, using `process` to
        evaluate the controller performance. The objective function passed to
        the constructor (an instance of
        :class:`~moptipyapps.dynamic_control.objective.FigureOfMerit`) must be
        used by `process` as well. This way, during the warm-up phase, we can
        collect tuples of the (system state, controller output) and the
        resulting system differential for each simulated time step. This is
        done via
        :meth:`~moptipyapps.dynamic_control.objective.FigureOfMerit.set_raw`.

        After the warm-up phase, we can obtain these collected data via
        :meth:`~moptipyapps.dynamic_control.objective.FigureOfMerit.\
get_differentials`. The data is then used to train a model via the model
        objective function
        :mod:`~moptipyapps.dynamic_control.model_objective`. The system model
        is again basically a
        :class:`~moptipyapps.dynamic_control.controller.Controller` which is
        parameterized appropriately. For this, we use a CMA-ES algorithm for
        :attr:`fes_for_training` FEs.

        Once the model training is completed, we switch the objective function
        to use the model instead of the actual system for evaluating
        controllers, which is done via :meth:`~moptipyapps.dynamic_control.\
objective.FigureOfMerit.set_model`. We then train a completely new controller
        on the model objective function. Notice that now, the actual system is
        not involved at all. We do this again using a CMA-ES algorithm for
        :attr:`fes_per_model_run` FEs.

        After training the controller, we can evaluate it on the real system
        using the :meth:`~moptipy.api.process.Process.evaluate` method
        of the actual `process` (after switching back to the real model via
        :meth:`~moptipyapps.dynamic_control.objective.FigureOfMerit.set_raw`).
        This nets us a) the actual controller performance and b) a new set of
        (system state, controller output) + system state differential tuples.

        Since we now have more data, we can go back and train a new system
        model and then use this model for another model-based optimization
        run. And so on, and so on. Until the budget is exhausted.

        :param process: the original optimization process, which must use
            the `objective` function (an instance of
            :class:`~moptipyapps.dynamic_control.objective.FigureOfMerit`) as
            its objective function.
        """
        # First, we set up the local variables and fast calls.
        should_terminate: Final[Callable[[], bool]] = process.should_terminate
        model_space: Final[VectorSpace] = self.__model_space
        model_objective: Final[ModelObjective] = self.__model_objective
        model = model_space.create()
        model_equations: Final[Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]] =\
            self.system_model.model.controller
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
        orig_init: Callable = raw.initialize

# Get a log dir if logging is enabled and set up all the logging information.
        log_dir_name: str | None = process.get_log_basename() \
            if self.fancy_logs else None
        model_training_dir: Path | None = None
        model_training_log_name: str | None = None
        models_dir: Path | None = None
        models_name: str | None = None
        tempsys: System | None = None
        ctrl_dir: Path | None = None
        ctrl_log_name: str | None = None
        controllers_dir: Path | None = None
        controllers_name: str | None = None
        if log_dir_name is not None:
            log_dir: Final[Path] = Path.path(log_dir_name)
            log_dir.ensure_dir_exists()
            prefix: str = "modelTraining"
            model_training_dir = log_dir.resolve_inside(prefix)
            model_training_dir.ensure_dir_exists()
            base_name: Final[str] = basename(log_dir_name)
            model_training_log_name = f"{base_name}_{prefix}_"
            training_execute.set_log_improvements(True)
            prefix = "controllerOnModel"
            ctrl_dir = log_dir.resolve_inside(prefix)
            ctrl_dir.ensure_dir_exists()
            ctrl_log_name = f"{base_name}_{prefix}_"
            on_model_execute.set_log_improvements(True)
            prefix = "model"
            models_dir = log_dir.resolve_inside(prefix)
            models_dir.ensure_dir_exists()
            tempsys = copy(self.system_model.system)
            models_name = f"{tempsys.name}_{prefix}_"
            prefix = "controllersOnReal"
            controllers_dir = log_dir.resolve_inside(prefix)
            controllers_dir.ensure_dir_exists()
            controllers_name = f"{base_name}_{prefix}_"

# Now we do the setup run that creates some basic results and
# gathers the initial information for modelling the system.
        with for_fes(process, self.fes_for_warmup) as prc:
            self.__control_cma.solve(prc)

        while not should_terminate():  # until budget exhausted
            consumed_fes: int = process.get_consumed_fes()

            # We now train a model on the data that was gathered.
            training_execute.set_rand_seed(rand_seed_generate(random))
            if model_training_dir is not None:
                training_execute.set_log_file(
                    model_training_dir.resolve_inside(
                        f"{model_training_log_name}{consumed_fes}"
                        f"{FILE_SUFFIX}"))
            model_objective.begin()  # get the collected data
            with training_execute.execute() as sub:  # train model
                sub.get_copy_of_best_y(model)  # get best model
            model_objective.end()  # dispose the collected data

            setattr(raw, "initialize", _nop)  # prevent resetting to "raw"

# The trained model is wrapped into an equation function that can be passed to
# the ODE integrator.
            @numba.njit(cache=False, inline="always", fastmath=True,
                        boundscheck=False)
            def __new_model(state: np.ndarray, time: float,
                            control: np.ndarray, out: np.ndarray,
                            _params=model, _eq=model_equations) -> None:
                _eq(np.hstack((state, control)), time, _params, out)

            setattr(__new_model, "modelParameters", model)  # see objective

            if tempsys is not None:  # plot the model behavior
                tempsys.equations = __new_model  # type: ignore
                setattr(tempsys, "name", f"{models_name}{consumed_fes}")
                tempsys.describe_system_without_control(models_dir)

            collect()  # now we collect all garbage ... there should be much

# OK, now that we got the model, we can perform the model optimization run.
            raw.set_model(__new_model)  # switch to use the model
            on_model_execute.set_rand_seed(rand_seed_generate(random))
            if ctrl_dir is not None:
                on_model_execute.set_log_file(ctrl_dir.resolve_inside(
                    f"{ctrl_log_name}{consumed_fes}{FILE_SUFFIX}"))
            with on_model_execute.execute() as ome:
                ome.get_copy_of_best_y(result)  # get best controller
            raw.set_raw()  # switch to the actual problem and data collection
            setattr(raw, "initialize", orig_init)  # allow resetting to "raw"

            if tempsys is not None:  # plot the controller on that model
                setattr(tempsys, "name", f"{models_name}{consumed_fes}")
                tempsys.describe_system(
                    f"{models_name}{consumed_fes}",
                    self.system_model.controller.controller, result,
                    f"{models_name}{consumed_fes}_synthesized_controller",
                    models_dir)

# Finally, we re-evaluate the result that we got from the model run on the
# actual objective function.
            process.evaluate(result)  # get the real objective value

# plot the actual behavior
            if controllers_dir is not None:
                self.system_model.system.describe_system(
                    f"{self.system_model.system}_{consumed_fes}",
                    self.system_model.controller.controller,
                    result, f"{controllers_name}{consumed_fes}",
                    controllers_dir)

            collect()  # now we collect all garbage ... there should be much

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
        logger.key_value("fancyLogs", self.fancy_logs)
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
