"""A simulator for multiple instances."""
from typing import Final, Iterable

import numpy as np
from moptipy.api.encoding import Encoding
from pycommons.types import type_error

from moptipyapps.prodsched.instance import (
    Instance,
)
from moptipyapps.prodsched.multistatistics import MultiStatistics
from moptipyapps.prodsched.rop_simulation import ROPSimulation
from moptipyapps.prodsched.statistics_collector import StatisticsCollector


class ROPMultiSimulation(Encoding):
    """A multi-simulation."""

    def __init__(self, instances: Iterable[Instance]) -> None:
        """
        Instantiate the multi-statistics decoding.

        :param instances: the packing instance
        """
        if not isinstance(instances, Iterable):
            raise type_error(instances, "instances", Iterable)
        #: the statistics collectors
        col: Final[tuple[StatisticsCollector, ...]] = tuple(
            StatisticsCollector(inst) for inst in instances)
        #: the simulations and collectors
        self.__simulations: Final[tuple[tuple[
            ROPSimulation, StatisticsCollector], ...]] = tuple(
            (ROPSimulation(inst, col[i]), col[i])
            for i, inst in enumerate(instances))

    def decode(self, x: np.ndarray, y: MultiStatistics) -> None:
        """
        Map a ROP setting to a multi-statistics.

        :param x: the array
        :param y: the Gantt chart
        """
        for i, (sim, col) in enumerate(self.__simulations):
            col.set_dest(y.per_instance[i])
            sim.ctrl_reset()
            sim.set_rop(x)
            sim.ctrl_run()

    def __str__(self) -> str:
        """
        Get the name of this decoding.

        :return: `"rms"`
        :rtype: str
        """
        return "rms"
