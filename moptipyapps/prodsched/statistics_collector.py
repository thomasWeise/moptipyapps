r"""
A tool for collecting statistics from an MFC simulation.

A statistics collector :class:`~StatisticsCollector` is a special
:class:`~moptipyapps.prodsched.simulation.Listener`
that can be plugged into a
:class:`~moptipyapps.prodsched.simulation.Simulation`.
During the execution of the simulation, it gathers statistics about what is
going on. It finally stores these into a
:class:`~moptipyapps.prodsched.statistics.Statistics` record.
Such a record can then be used to understand the key characteristics of the
behavior of the simulation in on a given
:class:`~moptipyapps.prodsched.instance.Instance`.

The simulation listeners (:class:`~moptipyapps.prodsched.simulation.Listener`)
offer a method to pipe out data from arbitrary subclasses of
:class:`~moptipyapps.prodsched.simulation.Simulation`.
This means that they allow us to access data in a unified way, regardless of
which manufacturing logic or production scheduling we actually implement.

In the case of the :class:`~StatisticsCollector` implemented here, we
implement the :class:`~moptipyapps.prodsched.simulation.Listener`-API to
fill a :class:`~moptipyapps.prodsched.statistics.Statistics` record with data.
Such records offer the standard statistics that Thürer et al. used in their
works.
In other words, we can make such statistics available and accessible,
regardless of how our simulation schedules the production.

Moreover, multiple such :class:`~moptipyapps.prodsched.statistics.Statistics`
records, filled with data from simulations over multiple different instances
(:mod:`~moptipyapps.prodsched.instance`) can be combined in a
:class:`~moptipyapps.prodsched.multistatistics.MultiStatistics` record.
This record, which will comprehensively represent performance over several
independent instances, then can be used as basis for objective functions
(:class:`~moptipy.api.objective.Objective`) such as those given in
:mod:`~moptipyapps.prodsched.objectives`.

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


>>> from moptipyapps.prodsched.simulation import Simulation
>>> statistics = Statistics(instance.n_products)
>>> collector = StatisticsCollector(instance)
>>> collector.set_dest(statistics)
>>> simulation = Simulation(instance, collector)
>>> simulation.ctrl_run()
>>> print("\n".join(str(statistics).split("\n")[:-1]))
stat;total;product_0;product_1
trp.min;229.57142857142858;455.6;229.57142857142858
trp.mean;304.8888888888889;455.6;246.92307692307693
trp.max;455.6;455.6;267.1666666666667
trp.sd;97.56326157594913;0;19.50721118346466
cwt.min;1603;2278;1603
cwt.mean;1829.3333333333333;2278;1605
cwt.max;2278;2278;1607
cwt.sd;388.56187838403986;;2.8284271247461903
fill.rate;0;0;0
stocklevel.mean;0.6078765407355446;0.4497444633730835;0.15813207736246118
fulfilled.rate;1;1;1
"""

from time import time_ns
from typing import Final

from pycommons.math.stream_statistics import (
    StreamStatistics,
    StreamStatisticsAggregate,
)
from pycommons.math.streams import StreamSum
from pycommons.types import type_error

from moptipyapps.prodsched.instance import (
    Demand,
    Instance,
)
from moptipyapps.prodsched.simulation import Job, Listener
from moptipyapps.prodsched.statistics import Statistics


class StatisticsCollector(Listener):
    """A listener for simulation events that collects basic statistics."""

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
        self.__production_times: Final[tuple[
            StreamStatisticsAggregate[StreamStatistics], ...]] = tuple(
            StreamStatistics.aggregate() for _ in range(n_products))
        #: the internal total production time
        self.__production_time: Final[StreamStatisticsAggregate[
            StreamStatistics]] = StreamStatistics.aggregate()
        #: the immediately satisfied products: satisfied vs. total
        self.__immediately_satisfied: Final[list[int]] = [0] * n_products
        #: the statistics of the waiting time for not-immediately satisfied
        #: products, per product
        self.__waiting_times: Final[tuple[
            StreamStatisticsAggregate[StreamStatistics], ...]] = tuple(
            StreamStatistics.aggregate() for _ in range(n_products))
        #: the total waiting time for non-immediately satisfied products
        self.__waiting_time: Final[StreamStatisticsAggregate[
            StreamStatistics]] = StreamStatistics.aggregate()
        #: the number of fulfilled jobs, per-product
        self.__fulfilled: Final[list[int]] = [0] * n_products
        #: the stock levels on a per-product basis
        self.__stock_levels: Final[tuple[StreamSum, ...]] = tuple(
            StreamSum() for _ in range(n_products))
        #: the total stock level sum
        self.__stock_level: Final[StreamSum] = StreamSum()
        #: the number of products currently in warehouse and since when
        #: they were there
        self.__in_warehouse: Final[tuple[list[float], ...]] = tuple([
            0.0, 0.0] for _ in range(n_products))
        #: the stat time of the simulation
        self.__start: int = -1

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
        self.__start = time_ns()
        n_products: Final[int] = self.__n_products
        for i in range(n_products):
            self.__production_times[i].reset()
            self.__immediately_satisfied[i] = 0
            self.__waiting_times[i].reset()
            self.__fulfilled[i] = 0
            self.__stock_levels[i].reset()
            wh = self.__in_warehouse[i]
            wh[0] = 0.0
            wh[1] = 0.0
        self.__production_time.reset()
        self.__waiting_time.reset()
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
            value: float = (time - max(iwh[1], self.__warmup)) * iwh[0]
            self.__stock_levels[product_id].add(value)
            self.__stock_level.add(value)
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
        """
        if demand.measure:
            at: Final[float] = demand.arrival
            pid: Final[int] = demand.product_id
            if at >= time:
                self.__immediately_satisfied[pid] += 1
            else:
                tt: Final[float] = time - at
                self.__waiting_times[pid].add(tt)
                self.__waiting_time.add(tt)
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
            dest.production_times[i] = stat.result_or_none()
        dest.production_time = self.__production_time.result_or_none()

        sn: int = 0
        sf: int = 0
        st: int = 0
        for i, n in enumerate(self.__immediately_satisfied):
            t = total[i]
            dest.immediate_rates[i] = n / t
            sn += n
            f = self.__fulfilled[i]
            dest.fulfilled_rates[i] = f / t
            sf += f
            st += t
        dest.immediate_rate = sn / st
        dest.fulfilled_rate = sf / st

        for i, stat in enumerate(self.__waiting_times):
            dest.waiting_times[i] = stat.result_or_none()
        dest.waiting_time = self.__waiting_time.result_or_none()

        slm: Final[StreamSum] = self.__stock_level
        twl: Final[float] = self.__total - self.__warmup
        wu: Final[float] = self.__warmup
        for i, sm in enumerate(self.__stock_levels):
            wh = self.__in_warehouse[i]
            v: float = (self.__total - max(wh[1], wu)) * wh[0]
            sm.add(v)
            slm.add(v)
            dest.stock_levels[i] = sm.result() / twl
        dest.stock_level = slm.result() / twl
        dest.simulation_time_nanos = time_ns() - self.__start
