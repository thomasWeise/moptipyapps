"""
A re-order-point-based simulation.

Re-Order-Point (ROP) scenarios are such that for each product, a value `X` is
provided. Once there are no more than `X` elements of that product in the
warehouse, one new unit is ordered to be produced.
Therefore, we have `n_products` such `X` values.

ROP-based simulations extend the basic behavior of the class
:class:`~moptipyapps.prodsched.simulation.Simulation` to re-order production
based on ROPs.

>>> from moptipyapps.prodsched.simulation import PrintingListener
>>> instance = Instance(
...     name="test1", n_products=2, n_customers=4, n_stations=2, n_demands=4,
...     time_end_warmup=3000, time_end_measure=10000,
...     routes=[[0, 1], [1, 0]],
...     demands=[Demand(arrival=100, deadline=100, demand_id=0,
...                     customer_id=0, product_id=0, amount=1, measure=False),
...              Demand(arrival=3100, deadline=3100, demand_id=1,
...                     customer_id=1, product_id=0, amount=1, measure=True),
...              Demand(arrival=500, deadline=500, demand_id=2,
...                     customer_id=2, product_id=1, amount=1, measure=False),
...              Demand(arrival=4000, deadline=5000, demand_id=3,
...                     customer_id=3, product_id=1, amount=1, measure=True)],
...     warehous_at_t0=[0, 0],
...     station_product_unit_times=[[[10.0, 50.0, 15.0, 100.0],
...                                  [ 5.0, 20.0,  7.0,  35.0, 4.0, 50.0]],
...                                 [[ 5.0, 24.0,  7.0,  80.0],
...                                  [ 3.0, 21.0,  6.0,  50.0,]]])
>>> rop_sim = ROPSimulation(instance, PrintingListener(print_time=False))
>>> rop_sim.set_rop((10, 20))
>>> rop_sim.ctrl_run()
start
T=0.0: product=0, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: station=0, 1 jobs queued
T=0.0: start j(id: 0, p: 0, am: 11, ar: 0, me: F, c: F, st: 0, sp: 0)\
 at station 0
T=0.0: product=1, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: station=1, 1 jobs queued
T=0.0: start j(id: 1, p: 1, am: 21, ar: 0, me: F, c: F, st: 0, sp: 0)\
 at station 1
T=84.0: finished j(id: 1, p: 1, am: 21, ar: 0, me: F, c: F, st: 0, sp: 0)\
 at station 1
T=100.0: product=0, amount=0, in_warehouse=0, in_production=11,\
 1 pending demands
T=130.0: finished j(id: 0, p: 0, am: 11, ar: 0, me: F, c: F, st: 0,\
 sp: 0) at station 0
T=130.0: station=0, 2 jobs queued
T=130.0: start j(id: 1, p: 1, am: 21, ar: 0, me: F, c: F, st: 84,\
 sp: 1) at station 0
T=130.0: station=1, 1 jobs queued
T=130.0: start j(id: 0, p: 0, am: 11, ar: 0, me: F, c: F, st: 130,\
 sp: 1) at station 1
T=197.0: finished j(id: 0, p: 0, am: 11, ar: 0, me: F, c: T, st: 130,\
 sp: 1) at station 1
T=197.0: product=0, amount=11, in_warehouse=0, in_production=1,\
 1 pending demands
T=197.0: d(id: 0, p: 0, c: 0, am: 1, ar: 100, dl: 100, me: F) statisfied
T=197.0: 10 units of product 0 in warehouse
T=240.0: finished j(id: 1, p: 1, am: 21, ar: 0, me: F, c: T, st: 84,\
 sp: 1) at station 0
T=240.0: station=0, 1 jobs queued
T=240.0: start j(id: 2, p: 0, am: 1, ar: 100, me: F, c: F, st: 100,\
 sp: 0) at station 0
T=240.0: product=1, amount=21, in_warehouse=0, in_production=0,\
 0 pending demands
T=240.0: 21 units of product 1 in warehouse
T=250.0: finished j(id: 2, p: 0, am: 1, ar: 100, me: F, c: F, st: 100,\
 sp: 0) at station 0
T=250.0: station=1, 1 jobs queued
T=250.0: start j(id: 2, p: 0, am: 1, ar: 100, me: F, c: F, st: 250, sp: 1)\
 at station 1
T=255.0: finished j(id: 2, p: 0, am: 1, ar: 100, me: F, c: T, st: 250, sp: 1)\
 at station 1
T=255.0: product=0, amount=1, in_warehouse=10, in_production=0,\
 0 pending demands
T=255.0: 11 units of product 0 in warehouse
T=500.0: product=1, amount=0, in_warehouse=21, in_production=0,\
 1 pending demands
T=500.0: d(id: 2, p: 1, c: 2, am: 1, ar: 500, dl: 500, me: F) statisfied
T=500.0: 20 units of product 1 in warehouse
T=500.0: station=1, 1 jobs queued
T=500.0: start j(id: 3, p: 1, am: 1, ar: 500, me: F, c: F, st: 500,\
 sp: 0) at station 1
T=503.0: finished j(id: 3, p: 1, am: 1, ar: 500, me: F, c: F, st: 500,\
 sp: 0) at station 1
T=503.0: station=0, 1 jobs queued
T=503.0: start j(id: 3, p: 1, am: 1, ar: 500, me: F, c: F, st: 503,\
 sp: 1) at station 0
T=508.0: finished j(id: 3, p: 1, am: 1, ar: 500, me: F, c: T, st: 503,\
 sp: 1) at station 0
T=508.0: product=1, amount=1, in_warehouse=20, in_production=0,\
 0 pending demands
T=508.0: 21 units of product 1 in warehouse
T=3100.0! product=0, amount=0, in_warehouse=11, in_production=0,\
 1 pending demands
T=3100.0! d(id: 1, p: 0, c: 1, am: 1, ar: 3100, dl: 3100, me: T) statisfied
T=3100.0! 10 units of product 0 in warehouse
T=3100.0! station=0, 1 jobs queued
T=3100.0! start j(id: 4, p: 0, am: 1, ar: 3100, me: T, c: F, st: 3100,\
 sp: 0) at station 0
T=3110.0! finished j(id: 4, p: 0, am: 1, ar: 3100, me: T, c: F, st: 3100,\
 sp: 0) at station 0
T=3110.0! station=1, 1 jobs queued
T=3110.0! start j(id: 4, p: 0, am: 1, ar: 3100, me: T, c: F, st: 3110,\
 sp: 1) at station 1
T=3117.0! finished j(id: 4, p: 0, am: 1, ar: 3100, me: T, c: T, st: 3110,\
 sp: 1) at station 1
T=3117.0! product=0, amount=1, in_warehouse=10, in_production=0,\
 0 pending demands
T=3117.0! 11 units of product 0 in warehouse
T=4000.0! product=1, amount=0, in_warehouse=21, in_production=0,\
 1 pending demands
T=4000.0! d(id: 3, p: 1, c: 3, am: 1, ar: 4000, dl: 5000, me: T) statisfied
T=4000.0! 20 units of product 1 in warehouse
T=4000.0! station=1, 1 jobs queued
T=4000.0! start j(id: 5, p: 1, am: 1, ar: 4000, me: T, c: F, st: 4000,\
 sp: 0) at station 1
T=4003.0! finished j(id: 5, p: 1, am: 1, ar: 4000, me: T, c: F, st: 4000,\
 sp: 0) at station 1
T=4003.0! station=0, 1 jobs queued
T=4003.0! start j(id: 5, p: 1, am: 1, ar: 4000, me: T, c: F, st: 4003,\
 sp: 1) at station 0
T=4008.0! finished j(id: 5, p: 1, am: 1, ar: 4000, me: T, c: T, st: 4003,\
 sp: 1) at station 0
T=4008.0! product=1, amount=1, in_warehouse=20, in_production=0,\
 0 pending demands
T=4008.0! 21 units of product 1 in warehouse
T=4008.0 -- finished

>>> instance = Instance(
...     name="test2", n_products=2, n_customers=1, n_stations=2, n_demands=5,
...     time_end_warmup=21, time_end_measure=10000,
...     routes=[[0, 1], [1, 0]],
...     demands=[[0, 0, 1, 4, 20, 90], [1, 0, 0, 5, 22, 200],
...              [2, 0, 1, 4, 30, 200], [3, 0, 1, 6, 60, 200],
...              [4, 0, 0, 3, 5, 2000]],
...     warehous_at_t0=[2, 1],
...     station_product_unit_times=[[[10.0, 50.0, 15.0, 100.0],
...                                  [ 5.0, 20.0,  7.0,  35.0, 4.0, 50.0]],
...                                 [[ 5.0, 24.0,  7.0,  80.0],
...                                  [ 3.0, 21.0,  6.0,  50.0,]]])
>>> rop_sim = ROPSimulation(instance, PrintingListener(print_time=False))
>>> rop_sim.set_rop((10, 10))
>>> rop_sim.ctrl_run()
start
T=0.0: product=0, amount=2, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: 2 units of product 0 in warehouse
T=0.0: station=0, 1 jobs queued
T=0.0: start j(id: 0, p: 0, am: 9, ar: 0, me: F, c: F, st: 0, sp: 0)\
 at station 0
T=0.0: product=1, amount=1, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: 1 units of product 1 in warehouse
T=0.0: station=1, 1 jobs queued
T=0.0: start j(id: 1, p: 1, am: 10, ar: 0, me: F, c: F, st: 0, sp: 0)\
 at station 1
T=5.0: product=0, amount=0, in_warehouse=2, in_production=9, 1 pending demands
T=20.0: product=1, amount=0, in_warehouse=1, in_production=10,\
 1 pending demands
T=22.0! product=0, amount=0, in_warehouse=2, in_production=12,\
 2 pending demands
T=30.0! product=1, amount=0, in_warehouse=1, in_production=14,\
 2 pending demands
T=39.0: finished j(id: 1, p: 1, am: 10, ar: 0, me: F, c: F, st: 0, sp: 0)\
 at station 1
T=39.0! station=1, 2 jobs queued
T=39.0: start j(id: 3, p: 1, am: 4, ar: 20, me: F, c: F, st: 20, sp: 0)\
 at station 1
T=57.0: finished j(id: 3, p: 1, am: 4, ar: 20, me: F, c: F, st: 20, sp: 0)\
 at station 1
T=57.0! station=1, 1 jobs queued
T=57.0! start j(id: 5, p: 1, am: 4, ar: 30, me: T, c: F, st: 30, sp: 0)\
 at station 1
T=60.0! product=1, amount=0, in_warehouse=1, in_production=18,\
 3 pending demands
T=69.0! finished j(id: 5, p: 1, am: 4, ar: 30, me: T, c: F, st: 30, sp: 0)\
 at station 1
T=69.0! station=1, 1 jobs queued
T=69.0! start j(id: 6, p: 1, am: 6, ar: 60, me: T, c: F, st: 60, sp: 0)\
 at station 1
T=102.0! finished j(id: 6, p: 1, am: 6, ar: 60, me: T, c: F, st: 60, sp: 0)\
 at station 1
T=110.0: finished j(id: 0, p: 0, am: 9, ar: 0, me: F, c: F, st: 0, sp: 0)\
 at station 0
T=110.0! station=0, 6 jobs queued
T=110.0: start j(id: 2, p: 0, am: 3, ar: 5, me: F, c: F, st: 5, sp: 0)\
 at station 0
T=110.0! station=1, 1 jobs queued
T=110.0: start j(id: 0, p: 0, am: 9, ar: 0, me: F, c: F, st: 110, sp: 1)\
 at station 1
T=140.0: finished j(id: 2, p: 0, am: 3, ar: 5, me: F, c: F, st: 5, sp: 0)\
 at station 0
T=140.0! station=0, 5 jobs queued
T=140.0! start j(id: 4, p: 0, am: 5, ar: 22, me: T, c: F, st: 22, sp: 0)\
 at station 0
T=171.0: finished j(id: 0, p: 0, am: 9, ar: 0, me: F, c: T, st: 110, sp: 1)\
 at station 1
T=171.0! station=1, 1 jobs queued
T=171.0: start j(id: 2, p: 0, am: 3, ar: 5, me: F, c: F, st: 140, sp: 1)\
 at station 1
T=171.0! product=0, amount=9, in_warehouse=2, in_production=8,\
 2 pending demands
T=171.0: d(id: 4, p: 0, c: 0, am: 3, ar: 5, dl: 2000, me: F) statisfied
T=171.0! d(id: 1, p: 0, c: 0, am: 5, ar: 22, dl: 200, me: T) statisfied
T=171.0! 3 units of product 0 in warehouse
T=186.0: finished j(id: 2, p: 0, am: 3, ar: 5, me: F, c: T, st: 140, sp: 1)\
 at station 1
T=186.0! product=0, amount=3, in_warehouse=3, in_production=5,\
 0 pending demands
T=186.0! 6 units of product 0 in warehouse
T=210.0! finished j(id: 4, p: 0, am: 5, ar: 22, me: T, c: F, st: 22, sp: 0)\
 at station 0
T=210.0! station=0, 4 jobs queued
T=210.0: start j(id: 1, p: 1, am: 10, ar: 0, me: F, c: F, st: 39, sp: 1)\
 at station 0
T=210.0! station=1, 1 jobs queued
T=210.0! start j(id: 4, p: 0, am: 5, ar: 22, me: T, c: F, st: 210, sp: 1)\
 at station 1
T=245.0! finished j(id: 4, p: 0, am: 5, ar: 22, me: T, c: T, st: 210, sp: 1)\
 at station 1
T=245.0! product=0, amount=5, in_warehouse=6, in_production=0,\
 0 pending demands
T=245.0! 11 units of product 0 in warehouse
T=263.0: finished j(id: 1, p: 1, am: 10, ar: 0, me: F, c: T, st: 39, sp: 1)\
 at station 0
T=263.0! station=0, 3 jobs queued
T=263.0: start j(id: 3, p: 1, am: 4, ar: 20, me: F, c: F, st: 57, sp: 1)\
 at station 0
T=263.0! product=1, amount=10, in_warehouse=1, in_production=14,\
 3 pending demands
T=263.0: d(id: 0, p: 1, c: 0, am: 4, ar: 20, dl: 90, me: F) statisfied
T=263.0! d(id: 2, p: 1, c: 0, am: 4, ar: 30, dl: 200, me: T) statisfied
T=263.0! 3 units of product 1 in warehouse
T=287.0: finished j(id: 3, p: 1, am: 4, ar: 20, me: F, c: T, st: 57, sp: 1)\
 at station 0
T=287.0! station=0, 2 jobs queued
T=287.0! start j(id: 5, p: 1, am: 4, ar: 30, me: T, c: F, st: 69, sp: 1)\
 at station 0
T=287.0! product=1, amount=4, in_warehouse=3, in_production=10,\
 1 pending demands
T=287.0! d(id: 3, p: 1, c: 0, am: 6, ar: 60, dl: 200, me: T) statisfied
T=287.0! 1 units of product 1 in warehouse
T=303.0! finished j(id: 5, p: 1, am: 4, ar: 30, me: T, c: T, st: 69, sp: 1)\
 at station 0
T=303.0! station=0, 1 jobs queued
T=303.0! start j(id: 6, p: 1, am: 6, ar: 60, me: T, c: F, st: 102, sp: 1)\
 at station 0
T=303.0! product=1, amount=4, in_warehouse=1, in_production=6,\
 0 pending demands
T=303.0! 5 units of product 1 in warehouse
T=337.0! finished j(id: 6, p: 1, am: 6, ar: 60, me: T, c: T, st: 102, sp: 1)\
 at station 0
T=337.0! product=1, amount=6, in_warehouse=5, in_production=0,\
 0 pending demands
T=337.0! 11 units of product 1 in warehouse
T=337.0 -- finished
"""

from typing import Final, Iterable

import numpy as np
from pycommons.types import check_int_range, type_error

from moptipyapps.prodsched.instance import (
    Demand,
    Instance,
)
from moptipyapps.prodsched.simulation import Listener, Simulation


class ROPSimulation(Simulation):
    """Create the re-order point-based simulation."""

    def __init__(self, instance: Instance, listener: Listener) -> None:
        """
        Initialize this simulation.

        :param instance: the instance
        :param listener: the listener
        """
        super().__init__(instance, listener)
        #: the re-order point
        self.__rop: Final[list[int]] = [0] * instance.n_products

    def set_rop(self, rop: Iterable[int] | np.ndarray) -> None:
        """
        Set the re-order point.

        :param rop: the re-order point
        """
        if isinstance(rop, np.ndarray):
            rop = map(int, rop)  # type: ignore
        if not isinstance(rop, Iterable):
            raise type_error(rop, "rop", Iterable)
        i: int = -1
        for i, rv in enumerate(rop):
            self.__rop[i] = check_int_range(
                rv, "rop-value", 0, 1_000_000_000)
        i += 1
        n_prod: Final[int] = list.__len__(self.__rop)
        if i != n_prod:
            raise ValueError(f"Invalid rop length {i}, must be {n_prod}!")

    def event_product(self, time: float,  # pylint: disable=W0613,R0913,R0917
                      product_id: int, amount: int,
                      in_warehouse: int,
                      in_production: int,  # pylint: disable=W0613
                      pending_demands: tuple[Demand, ...]) -> None:
        """Perform a simulation step based on a product."""
        # satisfy demands
        available: int = in_warehouse + amount
        unsatisfied: int = 0
        for demand in pending_demands:
            request: int = demand.amount
            if request <= available:
                self.act_demand_satisfied(demand)
                available -= request
            else:
                unsatisfied += request

        # update warehouse
        if available < in_warehouse:
            self.act_take_from_warehouse(product_id, in_warehouse - available)
        elif available > in_warehouse:
            self.act_store_in_warehouse(product_id, available - in_warehouse)

        # check if re-order point needs to be triggered
        rop: Final[int] = self.__rop[product_id]
        available = available - unsatisfied + in_production
        if available <= rop:
            self.act_produce(product_id, rop - available + 1)
