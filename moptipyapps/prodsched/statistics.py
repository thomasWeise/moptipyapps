"""
A statistics record for the simulation.

This module provides a record with statistics derived from one single
MFC simulation. It can store values such as the mean fill rate or the
mean stock level.
Such statistics records are filled in by instances of the
:class:`~moptipyapps.prodsched.statistics_collector.StatisticsCollector`
plugged into the
:class:`~moptipyapps.prodsched.simulation.Simulation`.
"""

from itertools import chain
from typing import Callable, Final, Generator

from moptipy.utils.logger import KEY_VALUE_SEPARATOR
from pycommons.io.csv import CSV_SEPARATOR, SCOPE_SEPARATOR
from pycommons.math.stream_statistics import (
    KEY_MAXIMUM,
    KEY_MEAN_ARITH,
    KEY_MINIMUM,
    KEY_STDDEV,
    StreamStatistics,
)
from pycommons.strings.string_conv import num_or_none_to_str
from pycommons.types import check_int_range, type_error

#: the name of the statistics key
COL_STAT: Final[str] = "stat"
#: the total column name
COL_TOTAL: Final[str] = "total"
#: the statistics rate
KEY_RATE: Final[str] = "rate"
#: the product column prefix
COL_PRODUCT_PREFIX: Final[str] = "product_"
#: the mean TRP row
ROW_TRP: Final[str] = "trp"
#: the fill rate row
ROW_FILL_RATE: Final[str] = f"fill{SCOPE_SEPARATOR}{KEY_RATE}"
#: the CWT row
ROW_CWT: Final[str] = "cwt"
#: the mean stock level row
ROW_STOCK_LEVEL_MEAN: Final[str] = \
    f"stocklevel{SCOPE_SEPARATOR}{KEY_MEAN_ARITH}"
#: the fulfilled rate
ROW_FULFILLED_RATE: Final[str] = f"fulfilled{SCOPE_SEPARATOR}{KEY_RATE}"
#: the simulation time getter
ROW_SIMULATION_TIME: Final[str] = \
    f"time{SCOPE_SEPARATOR}s{KEY_VALUE_SEPARATOR}"

#: the statistics that we will print
__STATS: tuple[tuple[str, Callable[[
    StreamStatistics], int | float | None]], ...] = (
    (KEY_MINIMUM, StreamStatistics.getter_or_none(KEY_MINIMUM)),
    (KEY_MEAN_ARITH, StreamStatistics.getter_or_none(KEY_MEAN_ARITH)),
    (KEY_MAXIMUM, StreamStatistics.getter_or_none(KEY_MAXIMUM)),
    (KEY_STDDEV, StreamStatistics.getter_or_none(KEY_STDDEV)))


class Statistics:
    """
    A statistics record based on production scheduling.

    It provides the following statistics:

    - :attr:`~immediate_rates`: The per-product-fillrate, i.e., the fraction
      of demands of a given product that were immediately fulfilled when
      arriving in the system (i.e., that were fulfilled by using product that
      was available in the warehouse/in stock).
      Higher values are good.
    - :attr:`~immediate_rate`: The overall fillrate, i.e., the total fraction
      of demands that were immediately fulfilled upon arrival in the system
      over all demands. That is, this is the fraction of demands that were
      fulfilled by using product that was available in the warehouse/in stock.
      Higher values are good.
    - :attr:`~waiting_times`: The per-product waiting times ("CWT") for the
      demands that came in but could *not* immediately be fulfilled. These are
      the demands for a given product that were, so to say, not covered by the
      fillrate/:attr:`~immediate_rate`.  If all demands of a product could
      immediately be satisfied, then this is `None`.
      Otherwise, smaller values are good.
    - :attr:`~waiting_time`: The overall waiting times ("CWT") for the demands
      that came in but could *not* immediately be fulfilled. These are all the
      demands for a given product that were, so to say, not covered by the
      fillrate/:attr:`~immediate_rate`. If all demands could immediately be
      satisfied, then this is `None`.
      Otherwise, smaller values are good.
    - :attr:`~production_times`: The per-product times that producing one unit
      of the product takes from the moment that a production job is created
      until it is completed. Smaller values of this "TRP" are better.
    - :attr:`~production_time`: The overall statistics on the times that
      producing one unit of any product takes from the moment that a
      production job is created until it is completed. Smaller values this
      "TRP" are better.
    - :attr:`~fulfilled_rates`: The per-product fraction of demands that were
      satisfied. Demands for a product may remain unsatisfied if they have not
      been satisfied by the end of the simulation period. Larger values are
      better.
    - :attr:`~fulfilled_rate`: The fraction of demands that were satisfied.
      Demands may remain unsatisfied if they have not been satisfied by the
      end of the simulation period. Larger values are better.
    - :attr:`~stock_levels`: The average amount of a given product in the
      warehouse averaged over the simulation time. Smaller values are better.
    - :attr:`~stock_level`: The total average amount units of any product in
      the warehouse averaged over the simulation time. Smaller values are
      better.
    - :attr:`~simulation_time_nanos`: The total time that the simulation took,
      measured in nanoseconds.

    Instances of this class are filled by
    :class:`~moptipyapps.prodsched.statistics_collector.StatisticsCollector`
    objects plugged into the
    :class:`~moptipyapps.prodsched.simulation.Simulation`.
    """

    def __init__(self, n_products: int) -> None:
        """
        Create the statistics record for a given number of products.

        :param n_products: the number of products
        """
        check_int_range(n_products, "n_products", 1, 1_000_000_000)
        #: the production time (TRP) statistics per-product
        self.production_times: Final[list[
            StreamStatistics | None]] = [None] * n_products
        #: the overall production time (TRP) statistics
        self.production_time: StreamStatistics | None = None
        #: the fraction of demands that were immediately satisfied,
        #: on a per-product basis, i.e., the fillrate
        self.immediate_rates: Final[list[int | float | None]] = (
            [None] * n_products)
        #: the overall fraction of immediately satisfied demands, i.e.,
        #: the fillrate
        self.immediate_rate: int | float | None = None
        #: the average waiting time for all demands that were not immediately
        #: satisfied -- only counting demands that were actually satisfied,
        #: i.e., the CWT
        self.waiting_times: Final[list[
            StreamStatistics | None]] = [None] * n_products
        #: the overall waiting time for all demands that were not immediately
        #: satisfied -- only counting demands that were actually satisfied,
        #: i.e., the CWT
        self.waiting_time: StreamStatistics | None = None
        #: the fraction of demands that were fulfilled, on a per-product basis
        self.fulfilled_rates: Final[list[
            int | float | None]] = [None] * n_products
        #: the fraction of demands that were fulfilled overall
        self.fulfilled_rate: int | float | None = None
        #: the average stock level, on a per-product basis
        self.stock_levels: Final[list[
            int | float | None]] = [None] * n_products
        #: the overall average stock level
        self.stock_level: int | float | None = None
        #: the nanoseconds used by the simulation
        self.simulation_time_nanos: int | float | None = None

    def __str__(self) -> str:
        """Convert this object to a string."""
        return "\n".join(to_stream(self))

    def copy_from(self, stat: "Statistics") -> None:
        """
        Copy the contents of another statistics record.

        :param stat: the other statistics record
        """
        if not isinstance(stat, Statistics):
            raise type_error(stat, "stat", Statistics)
        self.production_times[:] = stat.production_times
        self.production_time = stat.production_time
        self.immediate_rates[:] = stat.immediate_rates
        self.immediate_rate = stat.immediate_rate
        self.waiting_times[:] = stat.waiting_times
        self.waiting_time = stat.waiting_time
        self.fulfilled_rates[:] = stat.fulfilled_rates
        self.fulfilled_rate = stat.fulfilled_rate
        self.stock_levels[:] = stat.stock_levels
        self.stock_level = stat.stock_level
        self.simulation_time_nanos = stat.simulation_time_nanos


def to_stream(stats: Statistics) -> Generator[str, None, None]:
    """
    Write a statistics record to a stream.

    :param stats: the statistics record
    :return: the stream of data
    """
    n_products: Final[int] = list.__len__(stats.production_times)
    nts: Final[Callable[[int | float | None], str]] = num_or_none_to_str

    yield str.join(CSV_SEPARATOR, chain((
        COL_STAT, COL_TOTAL), (f"{COL_PRODUCT_PREFIX}{i}" for i in range(
            n_products))))

    for key, alle, single in (
            (ROW_TRP, stats.production_times, stats.production_time),
            (ROW_CWT, stats.waiting_times, stats.waiting_time)):
        for stat, call in __STATS:
            yield str.join(CSV_SEPARATOR, chain((
                f"{key}{SCOPE_SEPARATOR}{stat}", nts(call(single))), (
                map(nts, map(call, alle)))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_FILL_RATE, nts(stats.immediate_rate)), (
        map(nts, stats.immediate_rates))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_STOCK_LEVEL_MEAN, nts(stats.stock_level)), (
        map(nts, stats.stock_levels))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_FULFILLED_RATE, nts(stats.fulfilled_rate)), (
        map(nts, stats.fulfilled_rates))))
    yield f"{ROW_SIMULATION_TIME}{nts(
        stats.simulation_time_nanos / 1_000_000_000)}"
