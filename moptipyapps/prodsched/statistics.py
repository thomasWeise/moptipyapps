"""A statistics record for the simulation."""

from dataclasses import dataclass
from itertools import chain
from typing import Callable, Final, Generator, Iterable

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

from moptipyapps.prodsched.instance import Instance

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
    """A statistics record based on production scheduling."""

    def __init__(self, n_products: int) -> None:
        """
        Create the statistics record.

        :param n_products: the number of products
        """
        check_int_range(n_products, "n_products", 1, 1_000_000_000)
        #: the production time statistics per-product
        self.production_times: Final[list[
            StreamStatistics | None]] = [None] * n_products
        #: the overall production time statistics
        self.production_time: StreamStatistics | None = None
        #: the fraction of demands that were immediately satisfied,
        #: on a per-product basis
        self.immediate_rates: Final[list[int | float | None]] = (
            [None] * n_products)
        #: the overall fraction of immediately satisfied demands
        self.immediate_rate: int | float | None = None
        #: the average waiting time for all demands that were not immediately
        #: satisfied -- only counting demands that were actually satisfied
        self.waiting_times: Final[list[
            StreamStatistics | None]] = [None] * n_products
        #: the overall waiting time for all demands that were not immediately
        #: satisfied -- only counting demands that were actually satisfied
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
        #: the nano seconds used by the simulation
        self.simulation_time_nanos: int | float | None = None

    def __str__(self) -> str:
        """Convert this object to a string."""
        return "\n".join(to_stream(self))


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


@dataclass(order=False, frozen=True)
class MultiStatistics:
    """A set of statistics gathered over multiple instances."""

    #: the per-instance statistics
    per_instance: tuple[Statistics, ...]

    def __init__(self, instances: Iterable[Instance]) -> None:
        """
        Create the multi-statistics object.

        :param instances: the instances for which we create the statistics
        """
        object.__setattr__(self, "per_instance", tuple(
            Statistics(inst.n_products) for inst in instances))


def multi_to_stream(multi: MultiStatistics) -> Generator[str, None, None]:
    """
    Convert a multi-statistics object to a stream.

    :param multi: the multi-statistics object
    :return: the stream of strings
    """
    if not isinstance(multi, MultiStatistics):
        raise type_error(multi, "multi", MultiStatistics)
    for i, ss in enumerate(multi.per_instance):
        yield f"-------- Instance {i} -------"
        yield from to_stream(ss)
