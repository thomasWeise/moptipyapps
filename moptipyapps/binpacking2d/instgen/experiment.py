"""An example experiment for generating bin packing instances."""

import argparse
from typing import Callable, Final, cast

from moptipy.algorithms.so.vector.cmaes_lib import BiPopCMAES
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from pycommons.io.path import Path

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.instgen.errors_and_hardness import (
    ErrorsAndHardness,
)
from moptipyapps.binpacking2d.instgen.problem import Problem
from moptipyapps.utils.shared import moptipyapps_argparser

#: the maximum number of FEs
MAX_FES: Final[int] = 10_000

#: the maximum number of FEs in for the algorithm runs inside the objective
INNER_MAX_FES: Final[int] = 225_000

#: the number of runs of the algorithms inside the objective
INNER_RUNS: Final[int] = 2


def cmaes(problem: Problem) -> Execution:
    """
    Create the basic BiPop-CMA-ES setup.

    :param problem: the problem to solve
    :return: the execution
    """
    return (Execution()
            .set_algorithm(BiPopCMAES(problem.search_space))
            .set_search_space(problem.search_space)
            .set_solution_space(problem.solution_space)
            .set_encoding(problem.encoding)
            .set_objective(ErrorsAndHardness(
                problem.solution_space, INNER_MAX_FES, INNER_RUNS))
            .set_max_fes(MAX_FES)
            .set_log_improvements(True))


#: the instances to use as blueprints
__USE_INSTANCES: Final[tuple[str, ...]] = (
    "beng01", "beng02", "beng03", "beng04", "beng05", "beng06", "beng07",
    "beng08", "beng09", "beng10", "cl01_020_01", "cl01_040_01", "cl01_060_01",
    "cl01_080_01", "cl01_100_01", "cl02_020_01", "cl02_040_01", "cl02_060_01",
    "cl02_080_01", "cl02_100_01", "cl03_020_01", "cl03_040_01", "cl03_060_01",
    "cl03_080_01", "cl03_100_01", "cl04_020_01", "cl04_040_01", "cl04_060_01",
    "cl04_080_01", "cl04_100_01", "cl05_020_01", "cl05_040_01", "cl05_060_01",
    "cl05_080_01", "cl05_100_01", "cl06_020_01", "cl06_040_01", "cl06_060_01",
    "cl06_080_01", "cl06_100_01", "cl07_020_01", "cl07_040_01", "cl07_060_01",
    "cl07_080_01", "cl07_100_01", "cl08_020_01", "cl08_040_01", "cl08_060_01",
    "cl08_080_01", "cl08_100_01", "cl09_020_01", "cl09_040_01", "cl09_060_01",
    "cl09_080_01", "cl09_100_01", "cl10_020_01", "cl10_040_01", "cl10_060_01",
    "cl10_080_01", "cl10_100_01", "cl10_100_10")


def run(base_dir: str) -> None:
    """
    Run the experiment.

    :param base_dir: the base directory
    """
    use_dir: Final[Path] = Path(base_dir)
    use_dir.ensure_dir_exists()

    inst_creators: list[Callable[[], Instance]] = [cast(
        "Callable[[], Instance]", lambda __s=_s, __t=_t: Problem(__s, __t))
        for _s in __USE_INSTANCES for _t in (0.25, 0.125)]

    run_experiment(
        base_dir=use_dir,
        instances=inst_creators,
        setups=[cmaes],
        n_runs=[1, 2, 3, 5, 7, 11, 13, 17, 20, 23, 29, 30, 31],
        perform_warmup=False,
        perform_pre_warmup=False)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "2D Bin Packing Instance Generator",
        "Run the 2D Bin Packing Instance Generator experiment.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
