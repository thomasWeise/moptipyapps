"""
A simulator for production scheduling that is more "randomized".

>>> ri = RandomInstance(seed=324)
>>> ri.n_products
10
>>> ri.n_demands
9998
>>> ri.station_product_unit_times
(((nan, nan), (nan, nan), (nan, nan), (nan, nan), (nan, nan), (nan, nan), \
(nan, nan), (nan, nan), (nan, nan), (nan, nan)), ((nan, nan), (nan, nan), \
(nan, nan), (nan, nan), (nan, nan), (nan, nan), (nan, nan), (nan, nan), \
(nan, nan), (nan, nan)), ((), (), (), (), (), (), (), (nan, nan), \
(nan, nan), (nan, nan)), ((nan, nan), (), (nan, nan), (nan, nan), \
(nan, nan), (), (), (nan, nan), (nan, nan), (nan, nan)), ((), (nan, nan), \
(), (), (), (nan, nan), (), (), (nan, nan), (nan, nan)), ((), (), \
(nan, nan), (), (), (), (nan, nan), (nan, nan), (nan, nan), (nan, nan)), \
((), (), (), (nan, nan), (), (nan, nan), (), (nan, nan), (), (nan, nan)), \
((), (nan, nan), (), (), (), (), (nan, nan), (nan, nan), (nan, nan), ()), \
((nan, nan), (nan, nan), (nan, nan), (nan, nan), (nan, nan), (nan, nan), \
(), (nan, nan), (), (nan, nan)), ((nan, nan), (nan, nan), (), (nan, nan), \
(), (), (), (), (nan, nan), ()), ((nan, nan), (nan, nan), (nan, nan), \
(nan, nan), (), (), (), (), (), (nan, nan)), ((), (), (nan, nan), (), \
(nan, nan), (nan, nan), (nan, nan), (nan, nan), (nan, nan), (nan, nan)), \
((), (), (), (), (nan, nan), (nan, nan), (nan, nan), (nan, nan), \
(nan, nan), (nan, nan)))

>>> collector: list[str] = []
>>> from moptipyapps.prodsched.simulation import PrintingListener
>>> simulation = RandomSimulation(ri, PrintingListener(output=collector.append,
...                                              print_time=False))
>>> simulation.ctrl_run()

>>> for i in range(20):
...     print(collector[i])
start
T=0.0: product=0, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=2, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=6, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=5, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=1, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=4, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=9, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=3, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=8, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=7, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=3.424208875424501: product=5, amount=0, in_warehouse=0, in_production=0, \
1 pending demands
T=3.424208875424501: station=0, 1 jobs queued
T=3.424208875424501: start j(id: 0, p: 5, am: 1, ar: 3.424208875424501, \
me: F, c: F, st: 3.424208875424501, sp: 0) at station 0
T=3.6567696275343033: finished j(id: 0, p: 5, am: 1, ar: 3.424208875424501, \
me: F, c: F, st: 3.424208875424501, sp: 0) at station 0
T=3.6567696275343033: station=1, 1 jobs queued
T=3.6567696275343033: start j(id: 0, p: 5, am: 1, ar: 3.424208875424501, \
me: F, c: F, st: 3.6567696275343033, sp: 1) at station 1
T=4.0981763278696475: finished j(id: 0, p: 5, am: 1, ar: 3.424208875424501, \
me: F, c: F, st: 3.6567696275343033, sp: 1) at station 1
T=4.0981763278696475: station=4, 1 jobs queued
T=4.0981763278696475: start j(id: 0, p: 5, am: 1, ar: 3.424208875424501, \
me: F, c: F, st: 4.0981763278696475, sp: 2) at station 4

>>> for i in range(len(collector) - 20, len(collector)):
...     print(collector[i])
T=9999.428331523672! finished j(id: 9995, p: 0, am: 1, ar: \
9996.460708942874, me: T, c: F, st: 9997.801768977704, sp: 1) at station 1
T=9999.428331523672! station=1, 3 jobs queued
T=9999.428331523672! start j(id: 9996, p: 3, am: 1, ar: 9997.478248967976, \
me: T, c: F, st: 9998.587385267516, sp: 1) at station 1
T=9999.428331523672! station=3, 1 jobs queued
T=9999.428331523672! start j(id: 9995, p: 0, am: 1, ar: 9996.460708942874, \
me: T, c: F, st: 9999.428331523672, sp: 2) at station 3
T=9999.58444837528! finished j(id: 9983, p: 6, am: 1, ar: 9983.484802449133, \
me: T, c: T, st: 9998.124429217616, sp: 7) at station 12
T=9999.58444837528! station=12, 1 jobs queued
T=9999.58444837528! start j(id: 9959, p: 9, am: 1, ar: 9956.252508364129, \
me: T, c: F, st: 9998.430253328681, sp: 12) at station 12
T=9999.58444837528! product=6, amount=1, in_warehouse=0, in_production=0, \
1 pending demands
T=9999.58444837528! d(id: 6984, p: 6, c: 6984, am: 1, ar: 9983.484802449133, \
dl: 9983.484802449133, me: T) statisfied
T=9999.869861971298! finished j(id: 9984, p: 9, am: 1, ar: 9984.207389040752, \
me: T, c: F, st: 9996.357685721181, sp: 7) at station 6
T=9999.869861971298! station=6, 1 jobs queued
T=9999.869861971298! start j(id: 9990, p: 7, am: 1, ar: 9992.63213872734, \
me: T, c: F, st: 9998.56285992803, sp: 3) at station 6
T=9999.920430233462! finished j(id: 9996, p: 3, am: 1, ar: 9997.478248967976, \
me: T, c: F, st: 9998.587385267516, sp: 1) at station 1
T=9999.920430233462! station=1, 3 jobs queued
T=9999.920430233462! start j(id: 9982, p: 7, am: 1, ar: 9982.32594565836, \
me: T, c: F, st: 9999.143844794991, sp: 6) at station 1
T=9999.998742667354! product=7, amount=0, in_warehouse=0, in_production=3, \
4 pending demands
T=9999.998742667354! station=0, 1 jobs queued
T=9999.998742667354! start j(id: 9997, p: 7, am: 1, ar: 9999.998742667354, \
me: T, c: F, st: 9999.998742667354, sp: 0) at station 0
T=9999.998742667354 -- finished
"""

from dataclasses import dataclass
from math import nan
from typing import Final, Iterable, cast

from moptipy.utils.nputils import rand_generator, rand_seeds_from_str
from numpy.random import Generator

from moptipyapps.prodsched.instance import (
    Instance,
)
from moptipyapps.prodsched.mfc_generator import (
    Product,
    Station,
    _sample_demands,
    default_products,
    default_stations,
)
from moptipyapps.prodsched.simulation import (
    Job,
    Listener,
    Simulation,
)


@dataclass(order=True, frozen=True)
class TimeJob(Job):
    """The record for a production job."""

    #: the work times
    worktimes: tuple[float, ...] = ()


class RandomInstance(Instance):
    """An instance of the Production Scheduling Problem."""

    def __init__(
            self,
            seed: int,
            products: Iterable[Product] = default_products(),
            stations: Iterable[Station] = default_stations(),
            time_end_warmup: int | float = 3000,
            time_end_measure: int | float = 10000) -> None:
        """
        Create an instance of the production scheduling time.

        :param seed: the instance seed
        :param products: the list of products
        :param stations: the list of stations
        :param time_end_warmup: the time unit when the warmup time ends and the
            actual measurement begins
        :param time_end_measure: the time unit when the actual measure time
            ends
        """
        name: Final[str] = f"randomInstance_0x{seed:x}"

        prods: Final[list[Product]] = sorted(products)
        n_products: Final[int] = list.__len__(prods)
        gen: Final[Generator] = rand_generator(rand_seeds_from_str(name, 1)[0])
        demands, _ = _sample_demands(
            gen, prods, time_end_warmup, time_end_measure)
        n_demands: Final[int] = tuple.__len__(demands)
        n_customers: Final[int] = set.__len__({
            dem.customer_id for dem in demands})

        #: the stations
        self.stations: Final[tuple[Station, ...]] = tuple(sorted(stations))
        n_stations: Final[int] = tuple.__len__(self.stations)
        routes: Final[tuple[tuple[int, ...], ...]] = tuple(
            prod.routing for prod in prods)

        # the work times, will later be replaced
        worktimes: Final[tuple[tuple[tuple[float, ...], ...], ...]] = tuple(
            tuple((1.0, 1.0) if sid in prd.routing else () for prd in prods)
            for sid in range(n_stations))

        #: the random seed to use
        self.seed: Final[int] = seed

        super().__init__(name=name,
                         n_products=n_products,
                         n_customers=n_customers,
                         n_stations=n_stations,
                         n_demands=n_demands,
                         time_end_warmup=time_end_warmup,
                         time_end_measure=time_end_measure,
                         routes=routes,
                         demands=demands,
                         warehous_at_t0=(0, ) * n_products,
                         station_product_unit_times=worktimes)

        # the invalid work times, will later be replaced
        nan_worktimes: Final[tuple[tuple[tuple[
            float, ...], ...], ...]] = tuple(
            tuple((nan, nan) if sid in prd.routing else () for prd in prods)
            for sid in range(n_stations))

        # reset the production times NAN to cause trouble if they are
        # accidentally used
        object.__setattr__(self, "station_product_unit_times", nan_worktimes)


class RandomSimulation(Simulation):
    """A randomized simulation."""

    def __init__(self, instance: RandomInstance, listener: Listener) -> None:
        """
        Create the randomized simulation.

        :param instance: the instance
        :param listener: the listener
        """
        super().__init__(instance, listener)

        #: the random seed
        self.seed: Final[int] = instance.seed
        #: the stations
        self.__stations: Final[tuple[Station, ...]] = instance.stations
        #: the internal generator
        self.__rnd: Generator = rand_generator(self.seed)

    def ctrl_run(self) -> None:
        """Begin the simulation."""
        self.__rnd = rand_generator(self.seed)
        super().ctrl_run()

    def _create_job(self, jid: int, product_id: int, amount: int,
                    time: float, measure: bool) -> Job:
        """
        Create the job.

        :param jid: the job id
        :param product_id: the product id
        :param amount: the amount of product
        :param time: the start time
        :param measure: will this be measured
        :return: the job
        """
        gen: Final[Generator] = self.__rnd
        stat: Final[tuple[Station, ...]] = self.__stations
        return TimeJob(jid, product_id, amount, time, self.warmup <= time,
                       worktimes=tuple(stat[i].processing_time.sample(gen)
                                       for i in self.routes[product_id]))

    def _get_job_finish_time(
            self, time: float, job: Job,
            step: int, station_id: int, product_id: int) -> float:
        """
        Get the finish time of a job.

        :param time: the start time
        :param job: the job
        :param step: the job step
        :param station_id: the station ID
        :param product_id: the product ID
        :return: the finish time
        """
        return time + cast("TimeJob", job).worktimes[step]
