"""A test for the raw experiment of the dynamic control problem."""

from typing import Callable, Final, cast

from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.evaluation.end_results import EndResult
from moptipy.utils.nputils import rand_seeds_from_str
from numpy.random import Generator, default_rng
from pycommons.io.temp import temp_dir

from moptipyapps.dynamic_control.experiment_raw import cmaes, make_instances
from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.system import System


def __make_instances() -> list[Callable[[], Instance]]:
    """
    Create the instances to be used in the dynamic control experiment.

    :return: the instances to be used in the dynamic control experiment.
    """
    res: list[Callable[[], Instance]] = []
    for cinst in make_instances():

        def __make(oc=cinst) -> Instance:
            inst = oc()
            orig_system = inst.system
            system = System(orig_system.name,
                            orig_system.state_dims, orig_system.control_dims,
                            orig_system.state_dim_mod,
                            orig_system.state_dims_in_j,
                            orig_system.gamma,
                            orig_system.test_starting_states,
                            orig_system.training_starting_states,
                            10, 10.0, 32, 10.0, orig_system.plot_examples)
            return Instance(system, inst.controller)
        res.append(cast(Callable[[], Instance], __make))
    return res


def __cmaes(instance: Instance) -> Execution:
    """
    Create the Bi-Pop-CMA-ES setup.

    :param instance: the problem instance
    :return: the setup
    """
    random: Generator = default_rng(rand_seeds_from_str(str(instance), 1)[0])
    res = cmaes(instance)
    res.set_max_fes(int(random.integers(32, 64)))
    return res


def test_experiment_raw(random: Generator = default_rng()) -> None:
    """
    Run the experiment.

    :param random: a randomizer
    """
    n_runs: Final[int] = 1
    er: list[EndResult] = []
    insts: list[Callable[[], Instance]] = __make_instances()
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
