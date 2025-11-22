"""A statistics record for the simulation."""

from itertools import chain
from math import isnan, nan
from typing import Callable, Final, Generator

from pycommons.io.csv import CSV_SEPARATOR
from pycommons.strings.string_conv import num_to_str
from pycommons.types import type_error

from moptipyapps.prodsched.instance import (
    Instance,
)

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


class Statistics:
    """A statistics record based on production scheduling."""

    def __init__(self, instance: Instance) -> None:
        """
        Create the statistics record.

        :param instance: the instance for which we create the record
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        n_products: Final[int] = instance.n_products
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
    nts: Final[Callable[[int | float], str]] = num_to_str
    yield str.join(CSV_SEPARATOR, chain((
        COL_STAT, COL_TOTAL), (f"{COL_PRODUCT_PREFIX}{i}" for i in range(
            n_products))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_TRP_MEAN, nts(stats.production_time_mean)), (
        map(nts, stats.production_time_means))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_TRP_SD, nts(stats.production_time_sd)), (
        map(__nts, stats.production_time_sds))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_FILL_RATE, nts(stats.fulfilled_rate)), (
        map(nts, stats.fulfilled_rates))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_CWT_MEAN, nts(stats.waiting_time_for_none_immediates_mean)), (
        map(nts, stats.waiting_time_for_none_immediates_means))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_CWT_SD, nts(stats.waiting_time_for_none_immediates_sd)), (
        map(__nts, stats.waiting_time_for_none_immediates_sds))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_STOCK_LEVEL_MEAN, nts(stats.stock_level_mean)), (
        map(nts, stats.stock_level_means))))
    yield str.join(CSV_SEPARATOR, chain((
        ROW_FULFILLED_RATE, nts(stats.fulfilled_rate)), (
        map(nts, stats.fulfilled_rates))))
