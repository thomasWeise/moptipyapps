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
from typing import Callable, Final, Generator, Iterable, Self

from moptipy.utils.logger import KEY_VALUE_SEPARATOR
from pycommons.io.csv import CSV_SEPARATOR, SCOPE_SEPARATOR
from pycommons.math.stream_statistics import (
    KEY_MAXIMUM,
    KEY_MEAN_ARITH,
    KEY_MINIMUM,
    KEY_STDDEV,
    StreamStatistics,
)
from pycommons.strings.string_conv import (
    num_or_none_to_str,
    str_to_num_or_none,
)
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
_STATS: tuple[tuple[str, Callable[[
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

    def clear(self) -> None:
        """Clear all the data."""
        n: Final[int] = list.__len__(self.production_times)
        if n <= 0:
            raise ValueError("Huh?")
        for i in range(n):
            self.production_times[i] = None
            self.immediate_rates[i] = None
            self.waiting_times[i] = None
            self.fulfilled_rates[i] = None
            self.stock_levels[i] = None

        self.production_time = None
        self.immediate_rate = None
        self.waiting_time = None
        self.fulfilled_rate = None
        self.stock_level = None
        self.simulation_time_nanos = None

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

    def from_stream(self, stream: Iterable[str]) -> Self:
        """
        Load the data from a stream.

        Notice: The `n` values of the statistics records cannot be loaded.
        They will be lost and just set to some more or less random number.

        :param stream: the stream of data
        :return: this object
        """
        self.clear()

        n: Final[int] = list.__len__(self.production_times)
        if n <= 0:
            raise ValueError("Huh?")

        keys: Final[set[str]] = {
            f"{key}{SCOPE_SEPARATOR}{the_stat[0]}"
            for key in (ROW_TRP, ROW_CWT) for the_stat in _STATS}
        keys.update((ROW_FILL_RATE, ROW_FULFILLED_RATE,
                     ROW_STOCK_LEVEL_MEAN))
        sim_time_key: Final[str] = ROW_SIMULATION_TIME

        data: dict[str, list[int | float | None]] = {}
        sim_time: int | None = None
        for srow in stream:
            row = str.strip(srow)
            if row.startswith(sim_time_key):
                sim_time = check_int_range(
                    round(float(row[str.__len__(
                        sim_time_key):]) * 1_000_000_000),
                    sim_time_key, 0, 1_000_000_000_000_000_000_000_000)
                if set.__len__(keys) <= 0:
                    break
                continue

            cols: list[str] = str.split(srow, CSV_SEPARATOR)
            key: str = cols[0]
            if (list.__len__(cols) <= (n + 1)) or (key not in keys):
                continue
            if key in data:
                raise ValueError(f"Duplicate key '{key}'.")
            data[key] = [str_to_num_or_none(cols[i]) for i in range(1, n + 2)]
            keys.remove(key)
            if (set.__len__(keys) <= 0) and (sim_time is not None):
                break

        if set.__len__(keys) > 0:
            raise ValueError(f"Missing keys: {keys}")
        if sim_time is None:
            raise ValueError(f"Did not find key '{sim_time_key}'.")

        self.simulation_time_nanos = sim_time
        self.production_time = _split_data_stat(
            data, ROW_TRP, self.production_times)
        self.waiting_time = _split_data_stat(
            data, ROW_CWT, self.waiting_times)

        vals: list[int | float | None] = data[ROW_FILL_RATE]
        self.immediate_rate = vals[0]
        self.immediate_rates[:] = vals[1:]

        vals = data[ROW_STOCK_LEVEL_MEAN]
        self.stock_level = vals[0]
        self.stock_levels[:] = vals[1:]

        vals = data[ROW_FULFILLED_RATE]
        self.fulfilled_rate = vals[0]
        self.fulfilled_rates[:] = vals[1:]

        return self


def _split_data_stat(data: dict[str, list[int | float | None]],
                     key: str,
                     dest: list[StreamStatistics | None]) \
        -> StreamStatistics | None:
    """
    Split a data set.

    :param data: the data set
    :param dest: the destination list
    :return: the main statistics, if any
    """
    key_min: Final[str] = f"{key}{SCOPE_SEPARATOR}{KEY_MINIMUM}"
    key_mean: Final[str] = f"{key}{SCOPE_SEPARATOR}{KEY_MEAN_ARITH}"
    key_max: Final[str] = f"{key}{SCOPE_SEPARATOR}{KEY_MAXIMUM}"
    key_sd: Final[str] = f"{key}{SCOPE_SEPARATOR}{KEY_STDDEV}"
    for i in range(1, list.__len__(dest) + 1):
        dest[i - 1] = __stream_stats(data, key_min, key_mean, key_max,
                                     key_sd, i)
    return __stream_stats(data, key_min, key_mean, key_max, key_sd, 0)


def __stream_stats(data: dict[str, list[int | float | None]],
                   key_min: str, key_mean: str, key_max: str, key_sd: str,
                   i: int) -> StreamStatistics | None:
    """
    Get a stream statistics.

    :param data: the data array
    :param key_min: the minimum key
    :param key_mean: the mean key
    :param key_max: the maximum key
    :param key_sd: the standard deviation key
    :param i: the index
    :return: the statistics or `None`
    """
    the_min = data[key_min][i]
    the_mean = data[key_mean][i]
    the_max = data[key_max][i]
    the_sd = data[key_sd][i]
    if (the_min is None) or (the_max is None):
        return None
    if the_mean is None:
        raise ValueError(
            f"Invalid mean {the_mean} for min={the_min}, max={the_max}!")
    return StreamStatistics(n=1 if the_sd is None else 100, minimum=the_min,
                            mean_arith=the_mean,
                            maximum=the_max, stddev=the_sd)


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
        for stat, call in _STATS:
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
