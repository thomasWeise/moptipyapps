"""
A statistics record for multiple simulations.

:class:`~MultiStatistics` are records that hold multiple simulation
:class:`~moptipyapps.prodsched.statistics.Statistics`, each of which computed
over a separate :class:`~moptipyapps.prodsched.simulation.Simulation` based on
a separate :mod:`~moptipyapps.prodsched.instance` of the material flow
problem.
These records are filled with data by via the
:class:`moptipyapps.prodsched.rop_multisimulation.ROPMultiSimulation`
mechanism, which performs the multiple simulations.

We cn use this record as the solution space when optimizing for the MFC
scenario.
Such a record holds comprehensive statistics across several simulation runs.
This makes it suitable as source of data for objective functions
(:class:`~moptipy.api.objective.Objective`).
The objective functions can then access these statistics.

Since we use :class:`~MultiStatistics` as solution space, we also need an
implementation of moptipy's :class:`~moptipy.api.space.Space`-API to plug it
into the optimization process.
Sucha space implementation is provided as class
:class:`~MultiStatisticsSpace`.
It can create, copy, and serialize these objects to text, so that they can
appear in the log files.
"""

from dataclasses import dataclass
from typing import Final, Generator, Iterable

from moptipy.api.space import Space
from moptipy.utils.logger import (
    KeyValueLogSection,
)
from pycommons.types import type_error

from moptipyapps.prodsched.instance import Instance
from moptipyapps.prodsched.statistics import Statistics
from moptipyapps.prodsched.statistics import to_stream as stat_to_stream


@dataclass(order=False, frozen=True)
class MultiStatistics:
    """A set of statistics gathered over multiple instances."""

    #: the per-instance statistics
    per_instance: tuple[Statistics, ...]
    #: the instance names
    inst_names: tuple[str, ...]

    def __init__(self, instances: Iterable[Instance]) -> None:
        """
        Create the multi-statistics object.

        :param instances: the instances for which we create the statistics
        """
        object.__setattr__(self, "per_instance", tuple(
            Statistics(inst.n_products) for inst in instances))
        object.__setattr__(self, "inst_names", tuple(
            inst.name for inst in instances))


def to_stream(multi: MultiStatistics) -> Generator[str, None, None]:
    """
    Convert a multi-statistics object to a stream.

    :param multi: the multi-statistics object
    :return: the stream of strings
    """
    if not isinstance(multi, MultiStatistics):
        raise type_error(multi, "multi", MultiStatistics)
    for i, ss in enumerate(multi.per_instance):
        yield f"-------- Instance {i}: {multi.inst_names[i]!r} -------"
        yield from stat_to_stream(ss)


class MultiStatisticsSpace(Space):
    """An implementation of the `Space` API of for multiple statistics."""

    def __init__(self, instances: tuple[Instance, ...]) -> None:
        """
        Create a multi-statistics space.

        :param instances: the instances
        """
        if not isinstance(instances, tuple):
            raise type_error(instances, "instances", tuple)
        for inst in instances:
            if not isinstance(inst, Instance):
                raise type_error(inst, "instance", Instance)
        #: The instance to which the packings apply.
        self.instances: Final[tuple[Instance, ...]] = instances

    def copy(self, dest: MultiStatistics, source: MultiStatistics) -> None:
        """
        Copy one multi-statistics to another one.

        :param dest: the destination multi-statistics
        :param source: the source multi-statistics
        """
        for i, d in enumerate(dest.per_instance):
            d.copy_from(source.per_instance[i])

    def create(self) -> MultiStatistics:
        """
        Create an empty multi-statistics record.

        :return: the empty multi-statistics record
        """
        return MultiStatistics(self.instances)

    def to_str(self, x: MultiStatistics) -> str:
        """
        Convert a multi-statistics to a string.

        :param x: the packing
        :return: a string corresponding to the multi-statistics
        """
        return "\n".join(to_stream(x))

    def is_equal(self, x1: MultiStatistics, x2: MultiStatistics) -> bool:
        """
        Check if two multi-statistics have the same contents.

        :param x1: the first multi-statistics
        :param x2: the second multi-statistics
        :return: `True` if both multi-statistics have the same content
        """
        return (x1 is x2) or (x1.per_instance == x2.per_instance)

    def from_str(self, text: str) -> MultiStatistics:
        """
        Convert a string to a multi-statistics.

        :param text: the string
        :return: the multi-statistics
        """
        if not isinstance(text, str):
            raise type_error(text, "text", str)
        raise NotImplementedError

    def validate(self, x: MultiStatistics) -> None:
        """
        Check if a multi-statistics is valid.

        :param x: the multi-statistics
        :raises TypeError: if any component of the multi-statistics is of the
            wrong type
        :raises ValueError: if the multi-statistics is not feasible
        """
        if not isinstance(x, MultiStatistics):
            raise type_error(x, "x", MultiStatistics)
        if not isinstance(x.per_instance, tuple):
            raise type_error(x.per_instance, "x.per_instance", tuple)
        for s in x.per_instance:
            if not isinstance(s, Statistics):
                raise type_error(s, "x.per_instance[i]", Statistics)

    def n_points(self) -> int:
        """
        Get the number of possible multi-statistics.

        :return: just some arbitrary very large number
        """
        return 100 ** tuple.__len__(self.instances)

    def __str__(self) -> str:
        """
        Get the name of the multi-statistics space.

        :return: the name
        """
        return f"multistats_{tuple.__len__(self.instances)}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        for i, inst in enumerate(self.instances):
            logger.key_value(f"inst_{i}", inst.name)
