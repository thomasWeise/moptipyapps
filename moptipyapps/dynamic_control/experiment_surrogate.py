"""
An example experiment for dynamic control using surrogate system models.

In this experiment, we again try to synthesize (i.e., parameterize)
controllers (:mod:`~moptipyapps.dynamic_control.controller`) that steer a
dynamic system (:mod:`~moptipyapps.dynamic_control.system`) into a state by
using a figure of merit (:mod:`~moptipyapps.dynamic_control.objective`) which
minimizes both the squared system state and controller effort.

The difference compared to :mod:`~moptipyapps.dynamic_control.experiment_raw`
is that we also try to synthesize a system model at the same time. We employ
the procedure detailed in
:mod:`~moptipyapps.dynamic_control.surrogate_optimizer` for this purpose.

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
from moptipyapps.dynamic_control.surrogate_optimizer import (
    SurrogateOptimizer,
    _bpcmaes,
)
from moptipyapps.dynamic_control.system_model import SystemModel
from moptipyapps.dynamic_control.systems.lorenz import LORENZ_4
from moptipyapps.dynamic_control.systems.stuart_landau import STUART_LANDAU_4
from moptipyapps.dynamic_control.systems.three_coupled_oscillators import (
    THREE_COUPLED_OSCILLATORS,
)


def make_instances() -> Iterable[Callable[[], SystemModel]]:
    """
    Create the instances to be used in the dynamic control experiment.

    :return: the instances to be used in the dynamic control experiment.
    """
    res: list[Callable[[], SystemModel]] = []
    for system in [STUART_LANDAU_4, LORENZ_4, THREE_COUPLED_OSCILLATORS]:
        state_dims: int = system.state_dims
        ctrl_dims: int = system.control_dims
        controllers = [
            make_ann(state_dims, ctrl_dims, [state_dims, state_dims])]
        for controller in controllers:
            for ann_model in [[state_dims, state_dims],
                              [state_dims, state_dims, state_dims]]:
                res.append(cast(
                    Callable[[], SystemModel],
                    lambda _s=system, _c=controller, _m=make_ann(
                        system.state_dims + system.control_dims,
                        system.state_dims, ann_model):
                    SystemModel(_s, _c, _m)))
    return res


#: the total objective function evaluations
MAX_FES: Final[int] = 64
#: the warmup FEs
WARMUP_FES: Final[int] = MAX_FES // 16
#: the maximum milliseconds per run
MAX_MS_PER_RUN: Final[int] = 4 * 24 * 3600 * 1000


def base_setup(instance: Instance) -> tuple[
        Execution, FigureOfMerit, VectorSpace]:
    """
    Create the basic setup.

    :param instance: the instance to use
    :return: the basic execution
    """
    objective: Final[FigureOfMerit] = FigureOfMeritLE(
        instance, isinstance(instance, SystemModel))
    space = instance.controller.parameter_space()
    return Execution().set_max_fes(MAX_FES).set_log_all_fes(True)\
        .set_objective(objective).set_solution_space(space), objective, space


def cmaes_raw(instance: Instance) -> Execution:
    """
    Create the Bi-Pop-CMA-ES setup.

    :param instance: the problem instance
    :return: the setup
    """
    execution, _, space = base_setup(instance)
    return execution.set_algorithm(_bpcmaes(space))


def cmaes_surrogate(instance: SystemModel,
                    ms_for_training: int = 128,
                    ms_per_model_run: int = 128,
                    fancy_logs: bool = True) -> Execution:
    """
    Create the Bi-Pop-CMA-ES setup.

    :param instance: the problem instance
    :param ms_for_training: the milliseconds for training
    :param ms_per_model_run: the milliseconds per model run
    :param fancy_logs: should we do fancy logging?
    :return: the setup
    """
    execution, objective, space = base_setup(instance)

    return execution.set_solution_space(space).set_algorithm(
        SurrogateOptimizer(
            system_model=instance,
            controller_space=space,
            objective=objective,
            fes_for_warmup=WARMUP_FES,
            ms_for_training=ms_for_training,
            ms_per_model_run=ms_per_model_run,
            fancy_logs=fancy_logs,
            model_training_algorithm=_bpcmaes,
            controller_training_algorithm=_bpcmaes))


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

    base_ms: Final[int] = int(0.5 + (MAX_MS_PER_RUN / (
        2 * (MAX_FES - WARMUP_FES))))
    base_ms_1_3: Final[int] = int(0.5 + ((2 * base_ms) / 3))
    base_ms_2_3: Final[int] = (2 * base_ms) - base_ms_1_3
    setups: list[Callable[[Any], Execution]] = []
    for training_ms, run_ms in ((base_ms, base_ms),
                                (base_ms_1_3, base_ms_2_3),
                                (base_ms_2_3, base_ms_1_3)):
        setups.append(cast(
            Callable[[Any], Execution],
            lambda i, __t=training_ms, __r=run_ms:
            cmaes_surrogate(i, __t, __r, True)))

    for runs in range(1, n_runs + 1):
        run_experiment(
            base_dir=use_dir,
            instances=instances,
            setups=setups,
            n_runs=runs,
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
