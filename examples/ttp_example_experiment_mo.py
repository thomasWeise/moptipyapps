"""
An example experiment for the multi-objective Traveling Tournament Problem.

In this experiment, we apply multi-objective methods to the TTP. The idea
is to treat the feasibility constraint :mod:`~moptipyapps.ttp.errors` as an
objective function and optimize it together with the actual
:mod:`~moptipyapps.ttp.plan_length` objective measuring the travel time.
We can do this either in a multi-objective fashion or in a single objective
fashion where we prioritize the number of errors over the length of the game
plans.
"""

import argparse
from typing import Callable, Final, Iterable

from moptipy.algorithms.mo.nsga2 import NSGA2
from moptipy.algorithms.so.rls import RLS
from moptipy.api.experiment import Parallelism, run_experiment
from moptipy.api.mo_execution import MOExecution
from moptipy.mo.problem.weighted_sum import Prioritize
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op2_gap import (
    Op2GeneralizedAlternatingPosition,
)
from moptipy.spaces.permutations import Permutations
from pycommons.io.path import Path

from moptipyapps.shared import moptipyapps_argparser
from moptipyapps.ttp.errors import Errors
from moptipyapps.ttp.game_encoding import GameEncoding
from moptipyapps.ttp.game_plan_space import GamePlanSpace
from moptipyapps.ttp.instance import Instance
from moptipyapps.ttp.plan_length import GamePlanLength


def make_instances() -> Iterable[Callable[[], Instance]]:
    """
    Create the instances to be used in the TTP experiment.

    :return: the instances to be used in the TTP experiment.
    """
    return map(lambda i: lambda _i=i: Instance.from_resource(i),
               Instance.list_resources())


def base_setup(instance: Instance) -> tuple[Permutations, MOExecution]:
    """
    Create the basic setup.

    :param instance: the instance to use
    :return: the basic execution
    """
    ge: Final[GameEncoding] = GameEncoding(instance)
    perms: Final[Permutations] = ge.search_space()
    return (perms, MOExecution().set_max_fes(512 * 1024).
            set_log_improvements(True).set_objective(
        Prioritize((Errors(instance), GamePlanLength(instance))))
        .set_search_space(perms).set_solution_space(
        GamePlanSpace(instance)).set_encoding(
        GameEncoding(instance)))


def rls(instance: Instance) -> MOExecution:
    """
    Create the priority-based RLS execution.

    :param instance: the problem instance
    :return: the setup
    """
    space, exe = base_setup(instance)
    return exe.set_algorithm(RLS(Op0Shuffle(space), Op1Swap2()))


def mo_nsga2(instance: Instance) -> MOExecution:
    """
    Create the multi-objective NSGA-2 execution.

    :param instance: the problem instance
    :return: the setup
    """
    space, exe = base_setup(instance)
    return exe.set_algorithm(NSGA2(Op0Shuffle(space), Op1Swap2(),
                                   Op2GeneralizedAlternatingPosition(space),
                                   16, 1 / 16))


def run(base_dir: str, n_runs: int = 3) -> None:
    """
    Run the experiment.

    :param base_dir: the base directory
    :param n_runs: the number of runs
    """
    use_dir: Final[Path] = Path(base_dir)
    use_dir.ensure_dir_exists()

    run_experiment(
        base_dir=use_dir,
        instances=make_instances(),
        setups=[rls, mo_nsga2],
        n_runs=n_runs,
        n_threads=Parallelism.PERFORMANCE)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "Traveling Tournament Problem (TTP)",
        "Run the Multi-Objective TTP experiment.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
