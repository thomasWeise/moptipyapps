"""Generate instances and simulate the production scheduling."""

import argparse
from typing import Final

from pycommons.io.console import logger
from pycommons.io.path import Path, line_writer

from moptipyapps.prodsched.instance import Instance, from_stream, to_stream
from moptipyapps.prodsched.mfc_generator import sample_mfc_instance
from moptipyapps.prodsched.simulation import (
    PrintingListener,
    Simulation,
    warmup,
)
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

        logger("Now reading the instance back from the file.")
        with inst_file.open_for_read() as stream:
            inst_cpy = from_stream(stream)
        if inst_cpy == instance:
            logger("Confirmed that stored and loaded instance are identical.")
        else:
            logger("Instances are different!")
            raise ValueError(
                "Instance loaded from file different from original!")

        logger(f"Now writing simulation data {index} to {log_file!r}.")
        with log_file.open_for_write() as stream:
            writer = line_writer(stream)
            simulation = Simulation(instance, PrintingListener(writer))
            simulation.ctrl_run()
        logger(f"Done generating and simulating instance {index}.")


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "Material Flow Control Instances and Simulations",
        "Generate instances for the material flow control problem and "
        "run simulations.")
    parser.add_argument(
        "dest", help="the directory to store the data",
        type=Path, nargs="?", default="./instances/")
    parser.add_argument(
        "n_instances", help="the number of instances to generate",
        type=int, nargs="?", default=10)
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest, args.n_instances)
