"""An example experiment for bin packing."""

import argparse
from typing import Callable, Final, cast

from moptipy.algorithms.so.ffa.fea1plus1 import FEA1plus1
from moptipy.algorithms.so.rls import RLS
from moptipy.api.encoding import Encoding
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.api.objective import Objective
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.operators.signed_permutations.op1_swap_2_or_flip import (
    Op1Swap2OrFlip,
)
from moptipy.spaces.signed_permutations import SignedPermutations
from pycommons.io.path import Path

from moptipyapps.binpacking2d.encodings.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.packing_result import DEFAULT_OBJECTIVES
from moptipyapps.binpacking2d.packing_space import PackingSpace
from moptipyapps.utils.shared import moptipyapps_argparser

#: the maximum number of FEs
MAX_FES: Final[int] = 1_000_000


def base_setup(instance: Instance,
               encoding: Callable[[Instance], Encoding],
               objective: Callable[[Instance], Objective]) \
        -> tuple[SignedPermutations, Execution]:
    """
    Create the basic setup.

    :param instance: the instance to use
    :param encoding: the encoding function
    :param objective: the objective function
    :return: the search space and the basic execution
    """
    space: Final[SignedPermutations] = SignedPermutations(
        instance.get_standard_item_sequence())

    return (space, Execution().set_max_fes(MAX_FES).set_log_improvements(True)
            .set_objective(objective(instance))
            .set_encoding(encoding(instance))
            .set_search_space(space)
            .set_solution_space(PackingSpace(instance)))


def rls(instance: Instance,
        encoding: Callable[[Instance], Encoding],
        objective: Callable[[Instance], Objective]) -> Execution:
    """
    Create the RLS setup.

    :param instance: the instance to use
    :param encoding: the encoding function
    :param objective: the objective function
    :return: the RLS execution
    """
    space, execute = base_setup(instance, encoding, objective)
    return execute.set_algorithm(
        RLS(Op0ShuffleAndFlip(space), Op1Swap2OrFlip()))


def fea(instance: Instance,
        encoding: Callable[[Instance], Encoding],
        objective: Callable[[Instance], Objective]) -> Execution:
    """
    Create the FEA setup.

    :param instance: the instance to use
    :param encoding: the encoding function
    :param objective: the objective function
    :return: the RLS execution
    """
    space, execute = base_setup(instance, encoding, objective)
    return execute.set_algorithm(
        FEA1plus1(Op0ShuffleAndFlip(space), Op1Swap2OrFlip()))


def run(base_dir: str, n_runs: int = 23) -> None:
    """
    Run the experiment.

    :param base_dir: the base directory
    :param n_runs: the number of runs, by default `23`
    """
    use_dir: Final[Path] = Path(base_dir)
    use_dir.ensure_dir_exists()

    encodings: Final[tuple[Callable[[Instance], Encoding], ...]] = (
        ImprovedBottomLeftEncoding1,
    )
    instances: list[str] = [
        inst for inst in Instance.list_resources()
        if inst.startswith(("b", "a"))]
    inst_creators: list[Callable[[], Instance]] = [cast(
        "Callable[[], Instance]", lambda __s=_s: Instance.from_resource(__s))
        for _s in instances]
    namer: Final[Instance] = Instance.from_resource(instances[0])

    for objective in DEFAULT_OBJECTIVES:
        objective_dir: Path = use_dir.resolve_inside(str(objective(namer)))
        objective_dir.ensure_dir_exists()
        for encoding in encodings:
            encoding_dir: Path = objective_dir.resolve_inside(
                str(encoding(namer)))
            encoding_dir.ensure_dir_exists()
            run_experiment(
                base_dir=encoding_dir,
                instances=inst_creators,
                setups=[
                    cast("Callable",
                         lambda ins, _e=encoding, _o=objective: rls(
                             ins, _e, _o)),
                    cast("Callable",
                         lambda ins, _e=encoding, _o=objective: fea(
                             ins, _e, _o))],
                n_runs=n_runs,
                perform_warmup=True,
                perform_pre_warmup=True)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "2D Bin Packing", "Run the 2D Bin Packing experiment.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
