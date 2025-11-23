"""Generate instances and simulate the production scheduling."""

import argparse
from typing import Final

from pycommons.io.console import logger
from pycommons.io.path import Path, line_writer

from moptipyapps.prodsched.instance import Instance, to_stream
from moptipyapps.prodsched.mfc_generator import sample_mfc_instance
from moptipyapps.prodsched.rop_simulation import ROPSimulation
from moptipyapps.prodsched.simulation import (
    warmup,
)
from moptipyapps.prodsched.statistics import Statistics, mean
from moptipyapps.prodsched.statistics import to_stream as stat_to_stream
from moptipyapps.prodsched.statistics_collector import StatisticsCollector
from moptipyapps.utils.shared import moptipyapps_argparser


def run(dest_dir: str | Path, n_instances: int) -> None:
    """
    Run the instance generator and simulator.

    :param dest_dir: the destination directory
    :param n_instances: the number of instances to generate
    :return: the result
    """
    dest_dir = Path(dest_dir)
    dest_dir.ensure_dir_exists()
    logger(f"Writing data to {dest_dir!r}.")
    inst_dir: Final[Path] = dest_dir.resolve_inside("instances")
    inst_dir.ensure_dir_exists()
    logger(f"Writing instances to {inst_dir!r}.")
    simulation_dir: Final[Path] = dest_dir.resolve_inside("simulations")
    simulation_dir.ensure_dir_exists()
    logger(f"Writing simulation results to {simulation_dir!r}.")
    logger("Now performing warmup.")
    all_stats: Final[list[Statistics]] = []
    warmup()
    for index in range(1, n_instances):
        logger(f"Now generating instance {index}.")

        while True:
            instance: Instance = sample_mfc_instance()
            inst_file = inst_dir.resolve_inside(f"{instance.name}.txt")
            log_file = simulation_dir.resolve_inside(
                f"{instance.name}_simulation.txt")
            if not (inst_file.exists() or log_file.exists()):
                break
        logger(f"Now writing instance {index} to {inst_file!r}.")

        with inst_file.open_for_write() as stream:
            writer = line_writer(stream)
            for s in to_stream(instance):
                writer(s)

        logger(f"Now running ROP-simulation {index}.")
        stat: Statistics = Statistics(instance.n_products)
        col: StatisticsCollector = StatisticsCollector(instance)
        col.set_dest(stat)
        simulation: ROPSimulation = ROPSimulation(instance, col)
        simulation.set_rop((4, 6, 3, 4, 4, 6, 3, 5, 6, 10))
        simulation.ctrl_run()

        logger(f"Now writing simulation data {index} to {log_file!r}.")
        with log_file.open_for_write() as stream:
            writer = line_writer(stream)
            for s in stat_to_stream(stat):
                writer(s)
        all_stats.append(stat)
        logger(f"Done generating and simulating instance {index}.")

    logger("Now writing summary statistics.")
    log_file = simulation_dir.resolve_inside("summary.txt")
    with log_file.open_for_write() as stream:
        writer = line_writer(stream)
        for s in stat_to_stream(mean(all_stats)):
            writer(s)
    logger("Done writing summary statistics.")


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "Material Flow Control Instances and Simulations",
        "Generate instances for the material flow control problem and "
        "run simulations.")
    parser.add_argument(
        "dest", help="the directory to store the data",
        type=Path, nargs="?", default="./production_scheduling/")
    parser.add_argument(
        "n_instances", help="the number of instances to generate",
        type=int, nargs="?", default=5)
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest, args.n_instances)
