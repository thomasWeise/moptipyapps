"""A statistics record for the simulation."""

from itertools import chain
from math import isfinite, isnan, nan
from typing import Final, Generator, Iterable

from moptipy.utils.logger import KEY_VALUE_SEPARATOR
from pycommons.io.csv import CSV_SEPARATOR
from pycommons.math.stream_statistics import StreamStats
from pycommons.strings.string_conv import num_to_str
from pycommons.types import check_int_range

#: the name of the statistics key
COL_STAT: Final[str] = "stat"
#: the total column name
COL_TOTAL: Final[str] = "total"
#: the product column prefix
COL_PRODUCT_PREFIX: Final[str] = "product_"
#: the mean TRP row
ROW_TRP_MEAN: Final[str] = "trp.mean"
#: the TRP standard deviation row
ROW_TRP_SD: Final[str] = "trp.sd"
#: the fill rate row
ROW_FILL_RATE: Final[str] = "fill.rate"
#: the mean CWT row
ROW_CWT_MEAN: Final[str] = "cwt.mean"
#: the CWT standard deviation row
ROW_CWT_SD: Final[str] = "cwt.sd"
#: the mean stock level row
ROW_STOCK_LEVEL_MEAN: Final[str] = "stocklevel.mean"
#: the fulfilled rate
ROW_FULFILLED_RATE: Final[str] = "fulfilled.rate"
ROW_SIMULATION_TIME: Final[str] = f"time.s{KEY_VALUE_SEPARATOR}"


class Statistics:
    """A statistics record based on production scheduling."""

    def __init__(self, n_products: int) -> None:
        """
        Create the statistics record.

        :param n_products: the number of products
        """
        check_int_range(n_products, "n_products", 1, 1_000_000_000)
        #: the production time means per-product
        self.production_time_means: Final[list[float]] = [nan] * n_products
        #: the overall production time mean
        self.production_time_mean: float = nan
        #: the production time standard deviations per-product
        self.production_time_sds: Final[list[float]] = [nan] * n_products
        #: the overall production time standard deviations
        self.production_time_sd: float = nan
        #: the fraction of demands that were immediately satisfied,
        #: on a per-product basis
        self.immediately_satisfied_rates: Final[list[float]] = (
            [nan] * n_products)
        #: the overall fraction of immediately satisfied demands
        self.immediately_satisfied_rate: float = nan
        #: the average waiting time for all demands that were not immediately
        #: satisfied -- only counting demands that were actually satisfied
        self.waiting_time_for_none_immediates_means: Final[list[float]] = (
            [nan] * n_products)
        #: the standard deviation of the waiting time for all demands that
        #: were not immediately satisfied -- only counting demands that were
        #: actually satisfied
        self.waiting_time_for_none_immediates_sds: Final[list[float]] = (
            [nan] * n_products)
        #: the overall average waiting time for all demands that were not
        #: immediately satisfied -- only counting demands that were actually
        #: satisfied
        self.waiting_time_for_none_immediates_mean: float = nan
        #: the overall standard deviation of the waiting time for all demands
        #: that were not immediately satisfied -- only counting demands that
        #: were actually  satisfied
        self.waiting_time_for_none_immediates_sd: float = nan
        #: the fraction of demands that were fulfilled, on a per-product basis
        self.fulfilled_rates: Final[list[float]] = [nan] * n_products
        #: the fraction of demands that were fulfilled overall
        self.fulfilled_rate: float = nan
        #: the average stock level, on a per-product basis
        self.stock_level_means: Final[list[float]] = [nan] * n_products
        #: the overall average stock level
        self.stock_level_mean: float = nan
        #: the nano seconds used by the simulation
        self.simulation_time_nanos: int | float = nan

    def __str__(self) -> str:
        """Convert this object to a string."""
        return "\n".join(to_stream(self))


def __nts(num: int | float) -> str:
    """
    Convert a number to a string, taking care of `nan`.

    :param num: the number
    :return: the string
    """
    return "nan" if isnan(num) else num_to_str(num)


def to_stream(stats: Statistics) -> Generator[str, None, None]:
    """
    Write a statistics record to a stream.

    :param stats: the statistics record
    :return: the stream of data
    """
    n_products: Final[int] = list.__len__(stats.production_time_means)
    yield str.join(CSV_SEPARATOR, chain((
        COL_STAT, COL_TOTAL), (f"{COL_PRODUCT_PREFIX}{i}" for i in range(
            n_products))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_TRP_MEAN, __nts(stats.production_time_mean)), (
        map(__nts, stats.production_time_means))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_TRP_SD, __nts(stats.production_time_sd)), (
        map(__nts, stats.production_time_sds))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_FILL_RATE, __nts(stats.immediately_satisfied_rate)), (
        map(__nts, stats.immediately_satisfied_rates))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_CWT_MEAN, __nts(stats.waiting_time_for_none_immediates_mean)), (
        map(__nts, stats.waiting_time_for_none_immediates_means))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_CWT_SD, __nts(stats.waiting_time_for_none_immediates_sd)), (
        map(__nts, stats.waiting_time_for_none_immediates_sds))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_STOCK_LEVEL_MEAN, __nts(stats.stock_level_mean)), (
        map(__nts, stats.stock_level_means))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_FULFILLED_RATE, __nts(stats.fulfilled_rate)), (
        map(__nts, stats.fulfilled_rates))))
    yield f"{ROW_SIMULATION_TIME}{__nts(
        stats.simulation_time_nanos / 1_000_000_000)}"


def mean(stats: Iterable[Statistics]) -> Statistics:
    """
    Compute the mean over a given statistic.

    :param stats: the statistic
    :return: the mean statistics
    """
    stats = tuple(stats)
    n_products = list.__len__(stats[0].production_time_means)
    for stat in stats:
        if list.__len__(stat.production_time_means) != n_products:
            raise ValueError("Inconsistent number of products.")

    result: Final[Statistics] = Statistics(n_products)
    ss: Final[StreamStats] = StreamStats()

    for i in range(n_products):
        ss.reset()
        ss.update(stat.stock_level_means[i] for stat in stats if isfinite(
            stat.stock_level_means[i]))
        result.stock_level_means[i] = ss.mean()
    ss.reset()
    ss.update(stat.stock_level_mean for stat in stats if isfinite(
        stat.stock_level_mean))
    result.stock_level_mean = ss.mean()

    for i in range(n_products):
        ss.reset()
        ss.update(stat.production_time_means[i] for stat in stats if isfinite(
            stat.production_time_means[i]))
        result.production_time_means[i] = ss.mean()
    ss.reset()
    ss.update(stat.production_time_mean for stat in stats if isfinite(
        stat.production_time_mean))
    result.production_time_mean = ss.mean()

    for i in range(n_products):
        ss.reset()
        ss.update(stat.production_time_sds[i] for stat in stats if isfinite(
            stat.production_time_sds[i]))
        result.production_time_sds[i] = ss.mean()
    ss.reset()
    ss.update(stat.production_time_sd for stat in stats if isfinite(
        stat.production_time_sd))
    result.production_time_sd = ss.mean()

    for i in range(n_products):
        ss.reset()
        ss.update(stat.immediately_satisfied_rates[i]
                  for stat in stats if isfinite(
            stat.immediately_satisfied_rates[i]))
        result.immediately_satisfied_rates[i] = ss.mean()
    ss.reset()
    ss.update(stat.immediately_satisfied_rate for stat in stats if isfinite(
        stat.immediately_satisfied_rate))
    result.immediately_satisfied_rate = ss.mean()

    for i in range(n_products):
        ss.reset()
        ss.update(stat.waiting_time_for_none_immediates_means[i]
                  for stat in stats if isfinite(
            stat.waiting_time_for_none_immediates_means[i]))
        result.waiting_time_for_none_immediates_means[i] = ss.mean()
    ss.reset()
    ss.update(stat.waiting_time_for_none_immediates_mean
              for stat in stats if isfinite(
                  stat.waiting_time_for_none_immediates_mean))
    result.waiting_time_for_none_immediates_mean = ss.mean()

    for i in range(n_products):
        ss.reset()
        ss.update(stat.waiting_time_for_none_immediates_sds[i]
                  for stat in stats if isfinite(
            stat.waiting_time_for_none_immediates_sds[i]))
        result.waiting_time_for_none_immediates_sds[i] = ss.mean()
    ss.reset()
    ss.update(stat.waiting_time_for_none_immediates_sd
              for stat in stats if isfinite(
                  stat.waiting_time_for_none_immediates_sd))
    result.waiting_time_for_none_immediates_sd = ss.mean()

    for i in range(n_products):
        ss.reset()
        ss.update(stat.fulfilled_rates[i] for stat in stats if isfinite(
            stat.fulfilled_rates[i]))
        result.fulfilled_rates[i] = ss.mean()
    ss.reset()
    ss.update(stat.fulfilled_rate for stat in stats if isfinite(
        stat.fulfilled_rate))
    result.fulfilled_rate = ss.mean()

    # get the mean simulation time: most likely a sum of integers
    count: int = 0
    overall: int | float = 0
    for stat in stats:
        val: int | float = stat.simulation_time_nanos
        if isfinite(val):
            count += 1
            overall += val
    result.simulation_time_nanos = nan if count <= 0 else overall / count

    return result
