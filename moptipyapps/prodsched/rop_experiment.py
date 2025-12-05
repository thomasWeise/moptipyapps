"""A small template for ROP-based experiments."""


import argparse
from typing import Final

from moptipy.algorithms.mo.nsga2 import NSGA2
from moptipy.api.experiment import run_experiment
from moptipy.api.mo_execution import MOExecution
from moptipy.mo.problem.weighted_sum import WeightedSum
from moptipy.operators.intspace.op0_random import Op0Random
from moptipy.operators.intspace.op1_mnormal import Op1MNormal
from moptipy.operators.intspace.op2_uniform import Op2Uniform
from moptipy.spaces.intspace import IntSpace
from pycommons.io.console import logger
from pycommons.io.path import Path
from pycommons.types import check_int_range

from moptipyapps.prodsched.instance import Instance
from moptipyapps.prodsched.instances import get_instances
from moptipyapps.prodsched.multistatistics import MultiStatisticsSpace
from moptipyapps.prodsched.objectives.max_stocklevel import MaxStockLevel
from moptipyapps.prodsched.objectives.worst_and_mean_fill_rate import (
    WorstAndMeanFillRate,
)
from moptipyapps.prodsched.rop_multisimulation import ROPMultiSimulation
from moptipyapps.utils.shared import moptipyapps_argparser


def run(dest: str, instances: str, n_inst: int, n_runs: int,
        max_fes: int, ps: int) -> None:
    """
    Run the experiment.

    :param dest: the destination directory
    :param instances: the directory with the instances
    :param n_inst: the number of instances
    :param n_runs: the number of runs
    :param max_fes: the maximum FEs
    :param ps: the population size
    """
    logger(f"Beginning experiment with dest={dest!r}, instances={instances!r}"
           f", n_inst={n_inst}, n_runs={n_runs}, and max_fes={max_fes}.")
    use_dest: Final[Path] = Path(dest)
    use_dest.ensure_dir_exists()
    logger(f"Destination folder is {use_dest!r}.")

    use_insts: Final[Path] = Path(instances)
    use_insts.ensure_dir_exists()
    logger(f"Instances folder is {use_insts!r}.")

    check_int_range(n_inst, "n_inst", 1, 128)
    check_int_range(max_fes, "max_fes", 10, 10 ** 10)
    check_int_range(ps, "ps", 4, 16384)

    logger(f"Loading {n_inst} instances from {use_insts!r}.")
    insts: Final[tuple[Instance, ...]] = get_instances(n_inst, instances)
    if tuple.__len__(insts) != n_inst:
        raise ValueError("Could not load required instances.")
    logger(f"Loaded {n_inst} instances from {use_insts!r}.")

    n_prod: int | None = None
    for inst in insts:
        if n_prod is None:
            n_prod = inst.n_products
        elif n_prod != inst.n_products:
            raise ValueError("Inconsistent number of products!")
    if n_prod is None:
        raise ValueError("No instances?")

    search_space: Final[IntSpace] = IntSpace(n_prod, 0, 127)
    op0: Final[Op0Random] = Op0Random(search_space)
    op1: Final[Op1MNormal] = Op1MNormal(search_space, sd=2.0)
    op2: Final = Op2Uniform()
    algo: Final[NSGA2] = NSGA2(op0, op1, op2, ps, 2 / ps)
    encoding: Final[ROPMultiSimulation] = ROPMultiSimulation(insts)
    f1: Final[WorstAndMeanFillRate] = WorstAndMeanFillRate()
    f2: Final[MaxStockLevel] = MaxStockLevel()
    ws: Final[WeightedSum] = WeightedSum((f1, f2), (
        (1 / (f1.upper_bound() - f1.lower_bound())), 1 / (2 * n_prod)))
    solution_space: Final[MultiStatisticsSpace] = MultiStatisticsSpace(insts)

    def __setup(_) -> MOExecution:
        """
        Set up the experiment.

        :return: the execution
        """
        return (MOExecution()
                .set_search_space(search_space)
                .set_algorithm(algo)
                .set_solution_space(solution_space)
                .set_objective(ws)
                .set_encoding(encoding)
                .set_max_fes(max_fes)
                .set_log_improvements(True))

    run_experiment(base_dir=use_dest, instances=(lambda: "all", ),
                   setups=(__setup, ), n_runs=n_runs,
                   pre_warmup_fes=2, perform_warmup=False,
                   perform_pre_warmup=True)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "ROP-based MFC Optimization",
        "Run a small experiment with ROP-based MFC optimization.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path, nargs="?", default="./results/")
    parser.add_argument(
        "insts", help="the directory with the instances",
        type=Path, nargs="?", default="./instances/")
    parser.add_argument(
        "n_inst", help="the number of instances",
        type=int, nargs="?", default=16)
    parser.add_argument(
        "n_runs", help="the number of runs",
        type=int, nargs="?", default=31)
    parser.add_argument(
        "max_fes", help="the number of FEs per run",
        type=int, nargs="?", default=8192)
    parser.add_argument(
        "ps", help="the population size of NSGA-II",
        type=int, nargs="?", default=64)
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest, args.insts, args.n_inst, args.n_runs, args.max_fes,
        args.ps)
