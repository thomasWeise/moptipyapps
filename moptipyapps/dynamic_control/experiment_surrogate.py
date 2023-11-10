"""
An example experiment for dynamic control using surrogate system models.

In this experiment, we again try to synthesize (i.e., parameterize)
controllers (:mod:`~moptipyapps.dynamic_control.controller`) that steer a
dynamic system (:mod:`~moptipyapps.dynamic_control.system`) into a state by
using a figure of merit (:mod:`~moptipyapps.dynamic_control.objective`) which
minimizes both the squared system state and controller effort.

The difference compared to :mod:`~moptipyapps.dynamic_control.experiment_raw`
is that we also try to synthesize a system model at the same time. We employ
the procedure detailed in :mod:`~moptipyapps.dynamic_control.surrogate_cma`
for this purpose.

Word of advice: This experiment takes **extremely long** and needs
**a lot of memory**!

The starting points of the work here were conversations with Prof. Dr. Bernd
NOACK and Guy Yoslan CORNEJO MACEDA of the Harbin Institute of Technology in
Shenzhen, China (哈尔滨工业大学(深圳)).
"""

import argparse
from os.path import basename, dirname
from typing import Any, Callable, Final, Iterable, cast

import numpy as np
from moptipy.algorithms.so.vector.cmaes_lib import BiPopCMAES
from moptipy.api.execution import Execution
from moptipy.api.experiment import Parallelism, run_experiment
from moptipy.api.process import Process
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.help import argparser
from moptipy.utils.path import Path

from moptipyapps.dynamic_control.controllers.ann import make_ann
from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.objective import (
    FigureOfMerit,
    FigureOfMeritLE,
)
from moptipyapps.dynamic_control.surrogate_cma import SurrogateCmaEs
from moptipyapps.dynamic_control.system_model import SystemModel
from moptipyapps.dynamic_control.systems.three_coupled_oscillators import (
    THREE_COUPLED_OSCILLATORS,
)


def make_instances() -> Iterable[Callable[[], SystemModel]]:
    """
    Create the instances to be used in the dynamic control experiment.

    :return: the instances to be used in the dynamic control experiment.
    """
    res: list[Callable[[], SystemModel]] = []
    for system in [THREE_COUPLED_OSCILLATORS]:
        controllers = [
            make_ann(system.state_dims, system.control_dims, [3, 3])]
        for controller in controllers:
            for ann_model in [[2], [3, 3], [4, 4, 4]]:
                res.append(cast(
                    Callable[[], SystemModel],
                    lambda _s=system, _c=controller, _m=make_ann(
                        system.state_dims + system.control_dims,
                        system.state_dims, ann_model):
                    SystemModel(_s, _c, _m)))
    return res


#: the total objective function evaluations
MAX_FES: Final[int] = 64


def base_setup(instance: Instance, max_fes: int) -> tuple[
        Execution, FigureOfMerit, VectorSpace]:
    """
    Create the basic setup.

    :param instance: the instance to use
    :param max_fes: the maximum FEs
    :return: the basic execution
    """
    objective: Final[FigureOfMerit] = FigureOfMeritLE(
        instance, isinstance(instance, SystemModel))
    space = instance.controller.parameter_space()
    return Execution().set_max_fes(max_fes).set_log_all_fes(True)\
        .set_objective(objective).set_solution_space(space), objective, space


def cmaes_raw(instance: Instance, max_fes: int = MAX_FES) -> Execution:
    """
    Create the Bi-Pop-CMA-ES setup.

    :param instance: the problem instance
    :param max_fes: the maximum FEs
    :return: the setup
    """
    execution, _, space = base_setup(instance, max_fes)
    return execution.set_algorithm(BiPopCMAES(space))


def cmaes_surrogate(instance: SystemModel,
                    max_fes: int = MAX_FES,
                    fes_for_training: int = 128,
                    fes_per_model_run: int = 128,
                    fancy_logs: bool = True) -> Execution:
    """
    Create the Bi-Pop-CMA-ES setup.

    :param instance: the problem instance
    :param max_fes: the maximum FEs
    :param fes_for_training: the FEs for training
    :param fes_per_model_run: the FEs per model run
    :param fancy_logs: should we do fancy logging?
    :return: the setup
    """
    execution, objective, space = base_setup(instance, max_fes)
    return execution.set_solution_space(space).set_algorithm(SurrogateCmaEs(
        instance, space, objective, max_fes // 4,
        fes_for_training, fes_per_model_run, fancy_logs))


def on_completion(instance: Any, log_file: Path, process: Process) -> None:
    """
    Plot the corresponding figures and print the objective value components.

    :param instance: the problem instance
    :param log_file: the log file
    :param process: the process
    """
    inst: Final[SystemModel] = cast(SystemModel, instance)
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
    instances: Final[Iterable[Callable[[], SystemModel]]] = make_instances()
    keep_instances: Final[list[Callable[[], Instance]]] = []
    raw_names: Final[set[str]] = set()
    for maker in instances:
        inst: Instance = maker()
        raw: Instance = Instance(inst.system, inst.controller)
        name: str = str(raw)
        if name in raw_names:
            continue
        raw_names.add(name)
        raw.system.describe_system_without_control(use_dir, True)
        raw.system.plot_points(use_dir, True)
        keep_instances.append(cast(Callable[[], Instance], lambda _i=raw: _i))

    run_experiment(
        base_dir=use_dir,
        instances=keep_instances,
        setups=[cmaes_raw],
        n_runs=n_runs,
        n_threads=Parallelism.PERFORMANCE,
        perform_warmup=False,
        perform_pre_warmup=False,
        on_completion=on_completion)

    for training_fes, run_fes in ((2 ** 8, 2 ** 8),
                                  (2 ** 10, 2 ** 8),
                                  (2 ** 12, 2 ** 8)):
        run_experiment(
            base_dir=use_dir.resolve_inside(
                f"model_for_{training_fes}x{run_fes}_fes"),
            instances=instances,
            setups=[cast(Callable[[Any], Execution],
                         lambda i, __t=training_fes, __r=run_fes:
                         cmaes_surrogate(i, MAX_FES, __t, __r))],
            n_runs=n_runs,
            n_threads=Parallelism.ACCURATE_TIME_MEASUREMENTS,
            perform_warmup=False,
            perform_pre_warmup=False,
            on_completion=on_completion)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = argparser(
        __file__, "Dynamic Control",
        "Run the dynamic control surrogate model experiment.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path.path, nargs="?", default="./results/")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest)
