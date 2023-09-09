"""An example experiment for the TTP."""

import argparse
from typing import Callable, Final, Iterable, cast

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import Parallelism, run_experiment
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from moptipy.utils.help import argparser
from moptipy.utils.path import Path

from moptipyapps.ttp.errors import Errors
from moptipyapps.ttp.game_encoding import GameEncoding
from moptipyapps.ttp.game_plan_space import GamePlanSpace
from moptipyapps.ttp.instance import Instance


def make_instances() -> Iterable[Callable[[], Instance]]:
    """
    Create the instances to be used in the TTP experiment.

    :return: the instances to be used in the TTP experiment.
    """
    return (cast(Callable[[], Instance], lambda j=i: Instance.from_resource(
        f"circ{j}")) for i in range(4, 42, 2))


def base_setup(instance: Instance) -> tuple[Permutations, Execution]:
    """
    Create the basic setup.

    :param instance: the instance to use
    :return: the basic execution
    """
    ge: Final[GameEncoding] = GameEncoding(instance)
    perms: Final[Permutations] = ge.search_space()
    return (perms, Execution().set_max_fes(1000000000).set_log_improvements(
        True).set_objective(Errors(instance)).set_search_space(perms)
        .set_solution_space(GamePlanSpace(instance)).set_encoding(
        GameEncoding(instance)))


def rls(instance: Instance) -> Execution:
    """
    Create the RLS execution.

    :param instance: the problem instance
    :return: the setup
    """
    space, exe = base_setup(instance)
    return exe.set_algorithm(RLS(Op0Shuffle(space), Op1Swap2()))


def run(base_dir: str, n_runs: int = 5) -> None:
    """
    Run the experiment.

    :param base_dir: the base directory
    :param n_runs: the number of runs
    """
    use_dir: Final[Path] = Path.path(base_dir)
    use_dir.ensure_dir_exists()

    run_experiment(
        base_dir=use_dir,
        instances=make_instances(),
        setups=[rls],
        n_runs=n_runs,
        n_threads=Parallelism.PERFORMANCE)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = argparser(
        __file__, "Traveling Tournament Problem (TTP)",
        "Run the TTP experiment.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path.path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
