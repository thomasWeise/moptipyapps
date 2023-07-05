"""An example experiment for dynamic control."""

import argparse
from os.path import basename, dirname
from typing import Any, Callable, Final, Iterable, cast

import numpy as np
from moptipy.algorithms.so.vector.cmaes_lib import BiPopCMAES
from moptipy.api.execution import Execution
from moptipy.api.experiment import Parallelism, run_experiment
from moptipy.api.process import Process
from moptipy.utils.help import argparser
from moptipy.utils.path import Path

from moptipyapps.dynamic_control.controllers.ann import anns
from moptipyapps.dynamic_control.controllers.cubic import cubic
from moptipyapps.dynamic_control.controllers.linear import linear
from moptipyapps.dynamic_control.controllers.min_ann import min_anns
from moptipyapps.dynamic_control.controllers.partially_linear import (
    partially_linear,
)
from moptipyapps.dynamic_control.controllers.peaks import peaks
from moptipyapps.dynamic_control.controllers.predefined import predefined
from moptipyapps.dynamic_control.controllers.quadratic import quadratic
from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.objective_le import FigureOfMeritLE
from moptipyapps.dynamic_control.systems.lorenz import LORENZ
from moptipyapps.dynamic_control.systems.stuart_landau import STUART_LANDAU


def make_instances() -> Iterable[Callable[[], Instance]]:
    """
    Create the instances to be used in the dynamic control experiment.

    :return: the instances to be used in the dynamic control experiment.
    """
    res: list[Callable[[], Instance]] = []
    for system in (STUART_LANDAU, LORENZ):
        controllers = [linear(system), quadratic(system), cubic(system)]
        controllers.extend(anns(system))
        controllers.extend(min_anns(system))
        controllers.extend(partially_linear(system))
        controllers.extend(predefined(system))
        controllers.extend(peaks(system))
        for controller in controllers:
            res.append(cast(
                Callable[[], Instance],
                lambda _s=system, _c=controller: Instance(_s, _c)))
    return res


def base_setup(instance: Instance) -> Execution:
    """
    Create the basic setup.

    :param instance: the instance to use
    :return: the basic execution
    """
    return Execution().set_max_fes(512).set_log_all_fes(True)\
        .set_objective(FigureOfMeritLE(instance))


def cmaes(instance: Instance) -> Execution:
    """
    Create the Bi-Pop-CMA-ES setup.

    :param instance: the problem instance
    :return: the setup
    """
    space = instance.controller.parameter_space()
    return base_setup(instance).set_solution_space(space)\
        .set_algorithm(BiPopCMAES(space))


def on_completion(instance: Any, log_file: Path, process: Process) -> None:
    """
    Plot the corresponding figures and print the objective value components.

    :param instance: the problem instance
    :param log_file: the log file
    :param process: the process
    """
    inst: Final[Instance] = cast(Instance, instance)
    dest_dir: Final[Path] = Path.directory(dirname(log_file))
    base_name: str = basename(log_file)
    base_name = base_name[:base_name.rindex(".")]
    result: np.ndarray = cast(np.ndarray, process.create())
    process.get_copy_of_best_x(result)
    j: Final[float] = process.get_best_f()
    inst.describe_parameterization(f"F = {j}", result, base_name, dest_dir)


def run(base_dir: str, n_runs: int = 5) -> None:
    """
    Run the experiment.

    :param base_dir: the base directory
    :param n_runs: the number of runs
    """
    use_dir: Final[Path] = Path.path(base_dir)
    use_dir.ensure_dir_exists()
    instances: Final[Iterable[Callable[[], Instance]]] = make_instances()
    for maker in instances:
        inst: Instance = maker()
        inst.system.describe_system_without_control(use_dir, True)
        inst.system.plot_points(use_dir, True)

    run_experiment(
        base_dir=use_dir,
        instances=instances,
        setups=[cmaes],
        n_runs=n_runs,
        n_threads=Parallelism.PERFORMANCE,
        perform_warmup=False,
        perform_pre_warmup=False,
        on_completion=on_completion)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = argparser(
        __file__, "Dynamic Control", "Run the dynamic control experiment.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path.path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
