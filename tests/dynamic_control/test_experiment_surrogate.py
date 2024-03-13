"""A test for the surrogate experiment of the dynamic control problem."""

from typing import Callable, Final, cast

from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from numpy.random import Generator, default_rng
from pycommons.io.temp import temp_dir

from moptipyapps.dynamic_control.experiment_surrogate import (
    base_setup,
    make_instances,
)
from moptipyapps.dynamic_control.surrogate_optimizer import SurrogateOptimizer
from moptipyapps.dynamic_control.system import System
from moptipyapps.dynamic_control.system_model import SystemModel


def __make_instances(random: Generator) -> list[Callable[[], SystemModel]]:
    """
    Create the instances to be used in the dynamic control experiment.

    :param random: a randomizer
    :return: the instances to be used in the dynamic control experiment.
    """
    res: list[Callable[[], SystemModel]] = []
    for cinst in make_instances():

        def __make(oc=cinst) -> SystemModel:
            inst = oc()
            orig_system = inst.system
            system = System(orig_system.name,
                            orig_system.state_dims, orig_system.control_dims,
                            orig_system.state_dim_mod,
                            orig_system.state_dims_in_j,
                            orig_system.gamma,
                            orig_system.test_starting_states,
                            orig_system.training_starting_states,
                            int(random.integers(10, 14)), 10.0,
                            int(random.integers(10, 14)), 10.0,
                            orig_system.plot_examples)
            system.equations = orig_system.equations  # type: ignore
            return SystemModel(system, inst.controller, inst.model)
        res.append(cast(Callable[[], SystemModel], __make))
    return res


def __cmaes(instance: SystemModel) -> Execution:
    """
    Create the Bi-Pop-CMA-ES setup.

    :param instance: the problem instance
    :return: the setup
    """
    execution, objective, space = base_setup(instance)
    return execution.set_solution_space(space).set_max_fes(10).set_algorithm(
        SurrogateOptimizer(
            system_model=instance,
            controller_space=space,
            objective=objective,
            fes_for_warmup=2,
            fes_for_training=4,
            fes_per_model_run=4,
            fancy_logs=False))


def test_experiment_surrogate(random: Generator = default_rng()) -> None:
    """
    Run the surrogate experiment.

    :param random: a randomizer
    """
    n_runs: Final[int] = 1
    er: list[EndResult] = []
    insts: list[Callable[[], SystemModel]] = list(__make_instances(random))
    insts = [insts[random.integers(len(insts))]]

    with temp_dir() as use_dir:
        er.clear()
        run_experiment(base_dir=use_dir,
                       instances=insts,
                       setups=[__cmaes],
                       n_runs=n_runs,
                       perform_warmup=False,
                       perform_pre_warmup=False)
        EndResult.from_logs(use_dir, er.append)
    assert len(er) == (len(insts) * n_runs)
