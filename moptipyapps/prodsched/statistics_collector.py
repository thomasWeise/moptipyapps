"""
A tool for collecting statistics.

>>> instance = Instance(
...     name="test2", n_products=2, n_customers=1, n_stations=2, n_demands=5,
...     time_end_warmup=21, time_end_measure=10000,
...     routes=[[0, 1], [1, 0]],
...     demands=[[0, 0, 1, 10, 20, 90], [1, 0, 0, 5, 22, 200],
...              [2, 0, 1, 7, 30, 200], [3, 0, 1, 6, 60, 200],
...              [4, 0, 0, 125, 5, 2000]],
...     warehous_at_t0=[2, 1],
...     station_product_unit_times=[[[10.0, 50.0, 15.0, 100.0],
...                                  [ 5.0, 20.0,  7.0,  35.0, 4.0, 50.0]],
...                                 [[ 5.0, 24.0,  7.0,  80.0],
...                                  [ 3.0, 21.0,  6.0,  50.0,]]])

>>> instance.name
'test2'
>>> from moptipyapps.prodsched.simulation import Simulation
>>> statistics = Statistics(instance)
>>> collector = StatisticsCollector(instance)
>>> collector.set_dest(statistics)
>>> simulation = Simulation(instance, collector)
>>> simulation.ctrl_run()
>>> print(statistics)
stat;total;product_0;product_1
trp.mean;301.17460317460325;451.1000000000001;243.51098901098905
trp.sd;97.09947158455819;0;19.72076158183606
fill.rate;1;1;1
cwt.mean;1807.0476190476195;2255.5000000000005;1582.821428571429
cwt.sd;388.3725353194742;nan;1.4647211896006136
stocklevel.mean;5.010522096402445e-5;0;0.15610281591341824
fulfilled.rate;1;1;1
"""

from typing import Final

from pycommons.math.stream_statistics import StreamStats, StreamSum
from pycommons.types import type_error

from moptipyapps.prodsched.instance import (
    Demand,
    Instance,
)
from moptipyapps.prodsched.simulation import Job, Listener
from moptipyapps.prodsched.statistics import Statistics


class StatisticsCollector(Listener):
    """A listener for simulation events."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the listener.

        :param instance: the instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the destination to write to.
        self.__dest: Statistics | None = None
        #: the end of the warmup period
        self.__warmup: Final[float] = instance.time_end_warmup
        #: the total time window length
        self.__total: Final[float] = instance.time_end_measure
        #: the number of products
        self.__n_products: Final[int] = instance.n_products
        #: the measurable demands per product
        self.__n_mdpb: Final[tuple[int, ...]] = (
            instance.n_measurable_demands_per_product)

        n_products: Final[int] = instance.n_products
        #: the internal per-product production time records
        self.__production_times: Final[tuple[StreamStats, ...]] = tuple(
            StreamStats() for _ in range(n_products))
        #: the internal total production time
        self.__production_time: Final[StreamStats] = StreamStats()
        #: the immediately satisfied products: satisfied vs. total
        self.__immediately_satisfied: Final[list[int]] = [0] * n_products
        #: the statistics of the waiting time for not-immediately satisfied
        #: products, per product
        self.__waiting_time_for_none_immediates: Final[tuple[
            StreamStats, ...]] = tuple(StreamStats() for _ in range(
                n_products))
        #: the total waiting time for non-immediately satisfied products
        self.__waiting_time_for_none_immediate: Final[StreamStats] = (
            StreamStats())
        #: the number of fulfilled jobs, per-product
        self.__fulfilled: Final[list[int]] = [0] * n_products
        #: the stock levels on a per-product basis
        self.__stock_levels: Final[tuple[StreamSum, ...]] = tuple(
            StreamSum() for _ in range(n_products))
        #: the total stock level sum
        self.__stock_level: Final[StreamSum] = StreamSum()
        #: the number of products currently in warehouse and since when
        #: they were there
        self.__in_warehouse: Final[list[list[float]]] = [[
            0.0, 0.0]] * n_products

    def set_dest(self, dest: Statistics) -> None:
        """
        Set the statistics record to fill.

        :param dest: the destination
        """
        if not isinstance(dest, Statistics):
            raise type_error(dest, "dest", Statistics)
        self.__dest = dest

    def start(self) -> None:
        """Clear all the data of the collector."""
        if self.__dest is None:
            raise ValueError("Need destination statistics!")
        n_products: Final[int] = self.__n_products
        for i in range(n_products):
            self.__production_times[i].reset()
            self.__immediately_satisfied[i] = 0
            self.__waiting_time_for_none_immediates[i].reset()
            self.__fulfilled[i] = 0
            self.__stock_levels[i].reset()
            wh = self.__in_warehouse[i]
            wh[0] = 0.0
            wh[1] = 0.0
        self.__production_time.reset()
        self.__waiting_time_for_none_immediate.reset()
        self.__stock_level.reset()

    def product_in_warehouse(
            self, time: float, product_id: int, amount: int,
            is_in_measure_period: bool) -> None:
        """
        Report a change of the amount of products in the warehouse.

        :param time: the current time
        :param product_id: the product ID
        :param amount: the new absolute total amount of that product in the
            warehouse
        :param is_in_measure_period: is this event inside the measurement
            period?
        """
        iwh: Final[list[float]] = self.__in_warehouse[product_id]
        if is_in_measure_period:
            self.__stock_levels[product_id].add(
                (time - max(iwh[1], self.__warmup)) * iwh[0])
            self.__stock_level.add(product_id)
        iwh[0] = amount
        iwh[1] = time

    def produce_at_end(
            self, time: float, station_id: int, job: Job) -> None:
        """
        Report the completion of the production of a product at a station.

        :param time: the current time
        :param station_id: the station ID
        :param job: the production job
        """
        if job.measure and job.completed:
            am: Final[int] = job.amount
            tt: float = time - job.arrival
            if am <= 1:
                self.__production_times[job.product_id].add(tt)
                self.__production_time.add(tt)
            else:  # deal with amounts > 1
                tt /= am
                for _ in range(am):
                    self.__production_times[job.product_id].add(tt)
                    self.__production_time.add(tt)

    def demand_satisfied(
            self, time: float, demand: Demand) -> None:
        """
        Report that a given demand has been satisfied.

        :param time: the time index when the demand was satisfied
        :param demand: the demand that was satisfied
        :param is_in_measure_period: is this event inside the measurement
            period?
        """
        if demand.measure:
            at: Final[float] = demand.arrival
            pid: Final[int] = demand.product_id
            if at >= time:
                self.__immediately_satisfied[pid] += 1
            else:
                tt: Final[float] = time - at
                self.__waiting_time_for_none_immediates[pid].add(tt)
                self.__waiting_time_for_none_immediate.add(tt)
            self.__fulfilled[pid] += 1

    def finished(self, time: float) -> None:
        """
        Fill the collected statistics into the statistics record.

        :param time: the time when we are finished
        """
        dest: Final[Statistics | None] = self.__dest
        if dest is None:
            raise ValueError("Lost destination statistics record?")
        total: Final[tuple[int, ...]] = self.__n_mdpb

        for i, stat in enumerate(self.__production_times):
            dest.production_time_means[i] = stat.mean()
            dest.production_time_sds[i] = stat.sd()
        dest.production_time_mean = self.__production_time.mean()
        dest.production_time_sd = self.__production_time.sd()

        sn: int = 0
        sf: int = 0
        st: int = 0
        for i, n in enumerate(self.__immediately_satisfied):
            t = total[i]
            dest.immediately_satisfied_rates[i] = n / t
            sn += n
            f = self.__fulfilled[i]
            dest.fulfilled_rates[i] = f / t
            sf += f
            st += t
        dest.immediately_satisfied_rate = sn / st
        dest.fulfilled_rate = sf / st

        for i, stat in enumerate(self.__waiting_time_for_none_immediates):
            dest.waiting_time_for_none_immediates_means[i] = stat.mean()
            dest.waiting_time_for_none_immediates_sds[i] = stat.sd()
        dest.waiting_time_for_none_immediates_mean = (
            self.__waiting_time_for_none_immediate.mean())
        dest.waiting_time_for_none_immediates_sd = (
            self.__waiting_time_for_none_immediate.sd())

        slm: Final[StreamSum] = self.__stock_level
        twl: Final[float] = self.__total - self.__warmup
        for i, sm in enumerate(self.__stock_levels):
            wh = self.__in_warehouse[i]
            v: float = (self.__total - wh[1]) * wh[0]
            sm.add(v)
            slm.add(v)
            dest.stock_level_means[i] = sm.result() / twl
        dest.stock_level_mean = slm.result() / (self.__n_products * twl)
