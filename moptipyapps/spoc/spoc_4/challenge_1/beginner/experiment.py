"""A small example experiment."""

import argparse
from typing import Final

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.api.improvement_logger import FileImprovementLoggerFactory
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.operators.permutations.op1_swapn import Op1SwapN
from moptipy.spaces.bitstrings import BitStrings
from moptipy.spaces.permutations import Permutations
from moptipy.utils.help import moptipy_argparser
from pycommons.io.console import logger
from pycommons.io.path import Path

from moptipyapps.spoc.spoc_4.challenge_1.beginner.instance import Instance
from moptipyapps.spoc.spoc_4.challenge_1.beginner.objective_no_penalty import (
    BeginnerObjectiveNP,
)
from moptipyapps.spoc.spoc_4.challenge_1.beginner.permutation_encoding import (
    PermutationEncoding,
)
from moptipyapps.spoc.submission import SubmissionSpace

MAX_FES: Final[int] = 1_000  # Just a small test


def base_setup(
        instance: Instance,
        ilogger: FileImprovementLoggerFactory = FileImprovementLoggerFactory(
            max_files=10)) -> tuple[Permutations, Execution]:
    """
    Create the basic setup.

    :param instance: the instance to use
    :param title: a title
    :param desc: a description
    :param ilogger: the logger to use
    :return: the search space and the basic execution
    """
    space: Final[Permutations] = Permutations.standard(instance.n)

    return (space, Execution().set_max_fes(MAX_FES)
            .set_search_space(space)
            .set_encoding(PermutationEncoding(instance))
            .set_objective(BeginnerObjectiveNP(instance))
            .set_improvement_logger(ilogger)
            .set_solution_space(SubmissionSpace(
                space=BitStrings(instance.n),
                challenge_id="spoc-4-luna-tomato-logistics",
                problem_id=instance.name)))


def rls_n(instance: Instance) -> Execution:
    """
    Create the RLS setup.

    :param instance: the instance to use
    :return: the RLS
    """
    space, execute = base_setup(instance=instance)
    return execute.set_algorithm(RLS(
        Op0Shuffle(space), Op1SwapN()))


def rls_1(instance: Instance) -> Execution:
    """
    Create the RLS setup.

    :param instance: the instance to use
    :return: the RLS
    """
    space, execute = base_setup(instance=instance)
    return execute.set_algorithm(RLS(
        Op0Shuffle(space), Op1Swap2()))


def run(base_dir: str) -> None:
    """
    Run the experiment.

    :param base_dir: the base directory
    """
    use_dir: Final[Path] = Path(base_dir)
    use_dir.ensure_dir_exists()
    logger(f"Writing to directory {use_dir!r}.")

    run_experiment(
        base_dir=use_dir,
        instances=Instance.list_instances(),
        setups=[rls_n, rls_1],
        n_runs=(1, 2, 3, 4, 5, 6, 7, 8),
        perform_warmup=True,
        perform_pre_warmup=True)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipy_argparser(
        __file__, "SpOC 4: Challenge 1", "Run the 1st challenge of SpOC 4")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
