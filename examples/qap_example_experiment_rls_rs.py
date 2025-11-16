"""An example experiment for the Quadratic Assignment Problem."""

import argparse
from typing import Callable, Final, Iterable

from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from pycommons.io.path import Path

from moptipyapps.qap.instance import Instance
from moptipyapps.qap.objective import QAPObjective
from moptipyapps.utils.shared import moptipyapps_argparser


def make_instances() -> Iterable[Callable[[], Instance]]:
    """
    Create the instances to be used in the QAP experiment.

    :return: the instances to be used in the QAP experiment.
    """
    return map(lambda i: lambda _i=i: Instance.from_resource(i),
               Instance.list_resources())


def base_setup(instance: Instance) -> tuple[Permutations, Execution]:
    """
    Create the basic setup.

    :param instance: the instance to use
    :return: the basic execution
    """
    perms: Final[Permutations] = Permutations.standard(instance.n)
    return (perms, Execution().set_max_fes(32768).set_log_improvements(
        True).set_objective(QAPObjective(instance)).set_solution_space(perms))


def rls(instance: Instance) -> Execution:
    """
    Create the RLS execution.

    :param instance: the problem instance
    :return: the setup
    """
    space, exe = base_setup(instance)
    return exe.set_algorithm(RLS(Op0Shuffle(space), Op1Swap2()))


def rs(instance: Instance) -> Execution:
    """
    Create the random sampling execution.

    :param instance: the problem instance
    :return: the setup
    """
    space, exe = base_setup(instance)
    return exe.set_algorithm(RandomSampling(Op0Shuffle(space)))


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
        setups=[rls, rs],
        n_runs=n_runs)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "Quadratic Assignment Problem (QAP)",
        "Run the QAP experiment.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
