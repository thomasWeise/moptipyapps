"""
An example experiment for the Traveling Tournament Problem.

In this experiment, we apply both a random sampling algorithm (RS) and a
randomized local search (RLS) to the Traveling Tournament Problem (TTP).

Our goal is only to find feasible game plans, not feasible + short ones.
So we only use the "errors" objective function. Since we only want to find
feasible plans, we also do not need to consider the different distance
matrices that are available. Thus, we only use the 19 `circ*` instances
with numbers of teams ranging from 4 to 40. (See `make_instances`.)

We apply both algorithms on a permutation-based search spaces using the
very simple game encoding procedure (which may produce infeasible plans)
for 32768 FEs and log all improving moves under the "errors" objective
function. (See `base_setup`.)

Both algorithms use random permutation shuffling as their nullary operator.
Random sampling only applies this operator again and again and only creates
random solutions. It thus is rarely able to find a feasible solution and
does so only on the smallest problem instances. This random sampling algorithm
is set up in function `rs`.

Randomized local search - set up in function `rls` - additionally uses a unary
operator that receives an existing permutation as input and produces a
slightly modified copy as output. This enables it to try to improve the best
solution that it has encountered so far. It does so by applying the unary
operator to it, obtaining a new slightly modified copy of that best solution,
and keeping this copy as new best-so-far solution if it is *not worse*, i.e.,
better or equally good. As unary operator, it applies a simple swap-2 method
that exchanges two randomly chosen (different) elements in the permutation.

Three runs are executed for every algorithm-instance combination (see function
`run`).
"""

import argparse
from typing import Callable, Final, Iterable, cast

from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import Parallelism, run_experiment
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.spaces.permutations import Permutations
from pycommons.io.path import Path

from moptipyapps.shared import moptipyapps_argparser
from moptipyapps.ttp.errors import Errors
from moptipyapps.ttp.game_encoding import GameEncoding
from moptipyapps.ttp.game_plan_space import GamePlanSpace
from moptipyapps.ttp.instance import Instance


def make_instances() -> Iterable[Callable[[], Instance]]:
    """
    Create the instances to be used in the TTP experiment.

    Here, we simply load all the `circ*` instances. Here, all cities
    are located on a circle.

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
    return (perms, Execution().set_max_fes(32768).set_log_improvements(
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
        n_runs=n_runs,
        n_threads=Parallelism.ACCURATE_TIME_MEASUREMENTS)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "Traveling Tournament Problem (TTP)",
        "Run the TTP experiment.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
