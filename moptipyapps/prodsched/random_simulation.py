"""
A simulator for production scheduling that is more "randomized".

>>> ri = RandomInstance(seed=324)
>>> ri.n_products
10
>>> ri.n_demands
10038
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

>>> from moptipyapps.prodsched.mfc_generator import sample_mfc_instance
>>> di = sample_mfc_instance(seed=324)

>>> di.demands == ri.demands
True

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
T=2.981548103971826: product=0, amount=0, in_warehouse=0, in_production=0, \
1 pending demands
T=2.981548103971826: station=0, 1 jobs queued
T=2.981548103971826: start j(id: 0, p: 0, am: 1, ar: 2.981548103971826, \
me: F, c: F, st: 2.981548103971826, sp: 0) at station 0
T=3.4193569376148965: finished j(id: 0, p: 0, am: 1, ar: 2.981548103971826, \
me: F, c: F, st: 2.981548103971826, sp: 0) at station 0
T=3.4193569376148965: station=1, 1 jobs queued
T=3.4193569376148965: start j(id: 0, p: 0, am: 1, ar: 2.981548103971826, \
me: F, c: F, st: 3.4193569376148965, sp: 1) at station 1
T=3.55779441043288: product=1, amount=0, in_warehouse=0, in_production=0, \
1 pending demands
T=3.55779441043288: station=0, 1 jobs queued
T=3.55779441043288: start j(id: 1, p: 1, am: 1, ar: 3.55779441043288, me: F, \
c: F, st: 3.55779441043288, sp: 0) at station 0

>>> for i in range(len(collector) - 20, len(collector)):
...     print(collector[i])
T=9999.323989064784! start j(id: 10035, p: 4, am: 1, ar: 9997.792977742796, \
me: T, c: F, st: 9998.976626430733, sp: 1) at station 1
T=9999.360407487817! finished j(id: 9983, p: 7, am: 1, ar: 9949.406339628928, \
me: T, c: T, st: 9989.970820097158, sp: 11) at station 12
T=9999.360407487817! station=12, 7 jobs queued
T=9999.360407487817! start j(id: 10003, p: 4, am: 1, ar: 9969.66567880894, \
me: T, c: F, st: 9991.124064608839, sp: 7) at station 12
T=9999.360407487817! product=7, amount=1, in_warehouse=0, in_production=4, \
5 pending demands
T=9999.360407487817! d(id: 8000, p: 7, c: 8000, am: 1, ar: \
9949.406339628928, dl: 9949.406339628928, me: T) satisfied
T=9999.504763898161! finished j(id: 10035, p: 4, am: 1, ar: \
9997.792977742796, me: T, c: F, st: 9998.976626430733, sp: 1) \
at station 1
T=9999.79811329168! product=6, amount=0, in_warehouse=0, \
in_production=3, 4 pending demands
T=9999.953134938109! finished j(id: 10029, p: 6, am: 1, \
ar: 9993.680935722501, me: T, c: F, st: 9995.014424332754, sp: 2) at station 5
T=9999.953134938109! station=5, 1 jobs queued
T=9999.953134938109! start j(id: 10030, p: 2, am: 1, ar: 9993.858569427597, \
me: T, c: F, st: 9995.741612058786, sp: 2) at station 5
T=9999.984813470339! finished j(id: 10003, p: 4, am: 1, ar: \
9969.66567880894, me: T, c: T, st: 9991.124064608839, sp: 7) at station 12
T=9999.984813470339! station=12, 6 jobs queued
T=9999.984813470339! start j(id: 10002, p: 5, am: 1, ar: \
9964.392348080759, me: T, c: F, st: 9992.164222251737, sp: 7) at station 12
T=9999.984813470339! product=4, amount=1, in_warehouse=0, \
in_production=3, 4 pending demands
T=9999.984813470339! d(id: 4992, p: 4, c: 4992, am: 1, ar: \
9969.66567880894, dl: 9969.66567880894, me: T) satisfied
T=9999.992789452692! finished j(id: 10018, p: 0, am: 1, ar: \
9978.69384408688, me: T, c: F, st: 9993.208961813833, sp: 4) at station 8
T=9999.992789452692! station=8, 4 jobs queued
T=9999.992789452692! start j(id: 10012, p: 2, am: 1, ar: \
9975.265252124665, me: T, c: F, st: 9993.724592052344, sp: 5) at station 8
T=9999.992789452692 -- finished
"""

from dataclasses import dataclass
from math import nan
from typing import Final, Iterable, cast

from moptipy.utils.nputils import rand_generator
from numpy.random import Generator
from pycommons.ds.immutable_map import immutable_mapping

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
        gen: Final[Generator] = rand_generator(seed)
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
        #: the internal state where a random generator should pick up
        self.state: Final = immutable_mapping(dict(gen.bit_generator.state))

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
        self.__rnd: Final[Generator] = rand_generator(self.seed)

    def ctrl_run(self) -> None:
        """Begin the simulation."""
        self.__rnd.bit_generator.state = (  # type: ignore
            dict(self.instance.state))  # type: ignore
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
                       worktimes=tuple(
                           sum(stat[i].processing_time.sample(gen)
                               for _ in range(amount))
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
