"""
A randomized re-order-point-based simulation.

>>> ri = RandomInstance(seed=324)
>>> collector: list[str] = []
>>> from moptipyapps.prodsched.simulation import PrintingListener
>>> simulation = RandomROPSimulation(ri, PrintingListener(
...                 output=collector.append, print_time=False))
>>> simulation.set_rop([1] * ri.n_products)
>>> simulation.ctrl_run()

>>> for i in range(20):
...     print(collector[i])
start
T=0.0: product=0, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: station=0, 1 jobs queued
T=0.0: start j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: 0, sp: 0) at \
station 0
T=0.0: product=2, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=6, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=5, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=1, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=4, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=9, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=3, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=8, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=7, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.4378088336430705: finished j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
0, sp: 0) at station 0
T=0.4378088336430705: station=0, 9 jobs queued
T=0.4378088336430705: start j(id: 1, p: 2, am: 2, ar: 0, me: F, c: F, st: 0, \
sp: 0) at station 0
T=0.4378088336430705: station=1, 1 jobs queued
T=0.4378088336430705: start j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
0.4378088336430705, sp: 1) at station 1
T=0.9836355774755805: finished j(id: 1, p: 2, am: 2, ar: 0, me: F, c: F, \
st: 0, sp: 0) at station 0
T=0.9836355774755805: station=0, 8 jobs queued

>>> for i in range(len(collector) - 20, len(collector)):
...     print(collector[i])
T=9999.230289432131! finished j(id: 10023, p: 9, am: 1, ar: \
9975.333182144175, me: T, c: F, st: 9998.414533693287, sp: 7) \
at station 6
T=9999.26408720406! finished j(id: 10033, p: 2, am: 1, ar: \
9983.827600477664, me: T, c: F, st: 9998.821067949524, sp: 4) \
at station 1
T=9999.26408720406! station=1, 2 jobs queued
T=9999.26408720406! start j(id: 10042, p: 1, am: 1, ar: \
9995.016744835068, me: T, c: F, st: 9998.889388697426, sp: 1) at station 1
T=9999.667600796907! finished j(id: 10042, p: 1, am: 1, \
ar: 9995.016744835068, me: T, c: F, st: 9998.889388697426, sp: 1) \
at station 1
T=9999.667600796907! station=1, 1 jobs queued
T=9999.667600796907! start j(id: 10023, p: 9, am: 1, ar: \
9975.333182144175, me: T, c: F, st: 9999.230289432131, sp: 8) at station 1
T=9999.667600796907! station=4, 1 jobs queued
T=9999.667600796907! start j(id: 10042, p: 1, am: 1, ar: \
9995.016744835068, me: T, c: F, st: 9999.667600796907, sp: 2) at station 4
T=9999.69022300424! finished j(id: 9995, p: 8, am: 1, ar: \
9950.739209359475, me: T, c: F, st: 9993.99398437373, sp: 11) at station 5
T=9999.69022300424! station=5, 2 jobs queued
T=9999.69022300424! start j(id: 10039, p: 6, am: 1, ar: \
9993.680935722501, me: T, c: F, st: 9996.015895757433, sp: 2) at station 5
T=9999.79811329168! product=6, amount=0, in_warehouse=0, \
in_production=2, 1 pending demands
T=9999.935223781173! finished j(id: 10043, p: 7, am: 1, \
ar: 9996.127589696003, me: T, c: F, st: 9996.127589696003, sp: 0) at station 0
T=9999.935223781173! station=0, 4 jobs queued
T=9999.935223781173! start j(id: 10044, p: 5, am: 1, ar: \
9997.291327122433, me: T, c: F, st: 9997.291327122433, sp: 0) at station 0
T=9999.971791214713! finished j(id: 10023, p: 9, am: 1, \
ar: 9975.333182144175, me: T, c: F, st: 9999.230289432131, sp: 8) at station 1
T=9999.971791214713! station=1, 1 jobs queued
T=9999.971791214713! start j(id: 10043, p: 7, am: 1, \
ar: 9996.127589696003, me: T, c: F, st: 9999.935223781173, sp: 1) at station 1
T=9999.971791214713 -- finished

>>> collector.clear()
>>> simulation.ctrl_reset()
>>> simulation.ctrl_run()

>>> for i in range(20):
...     print(collector[i])
start
T=0.0: product=0, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: station=0, 1 jobs queued
T=0.0: start j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: 0, sp: 0) at \
station 0
T=0.0: product=2, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=6, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=5, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=1, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=4, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=9, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=3, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=8, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.0: product=7, amount=0, in_warehouse=0, in_production=0, 0 pending demands
T=0.4378088336430705: finished j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
0, sp: 0) at station 0
T=0.4378088336430705: station=0, 9 jobs queued
T=0.4378088336430705: start j(id: 1, p: 2, am: 2, ar: 0, me: F, c: F, st: 0, \
sp: 0) at station 0
T=0.4378088336430705: station=1, 1 jobs queued
T=0.4378088336430705: start j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
0.4378088336430705, sp: 1) at station 1
T=0.9836355774755805: finished j(id: 1, p: 2, am: 2, ar: 0, me: F, c: F, \
st: 0, sp: 0) at station 0
T=0.9836355774755805: station=0, 8 jobs queued

>>> for i in range(len(collector) - 20, len(collector)):
...     print(collector[i])
T=9999.230289432131! finished j(id: 10023, p: 9, am: 1, ar: \
9975.333182144175, me: T, c: F, st: 9998.414533693287, sp: 7) \
at station 6
T=9999.26408720406! finished j(id: 10033, p: 2, am: 1, ar: \
9983.827600477664, me: T, c: F, st: 9998.821067949524, sp: 4) \
at station 1
T=9999.26408720406! station=1, 2 jobs queued
T=9999.26408720406! start j(id: 10042, p: 1, am: 1, ar: \
9995.016744835068, me: T, c: F, st: 9998.889388697426, sp: 1) at station 1
T=9999.667600796907! finished j(id: 10042, p: 1, am: 1, \
ar: 9995.016744835068, me: T, c: F, st: 9998.889388697426, sp: 1) \
at station 1
T=9999.667600796907! station=1, 1 jobs queued
T=9999.667600796907! start j(id: 10023, p: 9, am: 1, ar: \
9975.333182144175, me: T, c: F, st: 9999.230289432131, sp: 8) at station 1
T=9999.667600796907! station=4, 1 jobs queued
T=9999.667600796907! start j(id: 10042, p: 1, am: 1, ar: \
9995.016744835068, me: T, c: F, st: 9999.667600796907, sp: 2) at station 4
T=9999.69022300424! finished j(id: 9995, p: 8, am: 1, ar: \
9950.739209359475, me: T, c: F, st: 9993.99398437373, sp: 11) at station 5
T=9999.69022300424! station=5, 2 jobs queued
T=9999.69022300424! start j(id: 10039, p: 6, am: 1, ar: \
9993.680935722501, me: T, c: F, st: 9996.015895757433, sp: 2) at station 5
T=9999.79811329168! product=6, amount=0, in_warehouse=0, \
in_production=2, 1 pending demands
T=9999.935223781173! finished j(id: 10043, p: 7, am: 1, \
ar: 9996.127589696003, me: T, c: F, st: 9996.127589696003, sp: 0) at station 0
T=9999.935223781173! station=0, 4 jobs queued
T=9999.935223781173! start j(id: 10044, p: 5, am: 1, ar: \
9997.291327122433, me: T, c: F, st: 9997.291327122433, sp: 0) at station 0
T=9999.971791214713! finished j(id: 10023, p: 9, am: 1, \
ar: 9975.333182144175, me: T, c: F, st: 9999.230289432131, sp: 8) at station 1
T=9999.971791214713! station=1, 1 jobs queued
T=9999.971791214713! start j(id: 10043, p: 7, am: 1, \
ar: 9996.127589696003, me: T, c: F, st: 9999.935223781173, sp: 1) at station 1
T=9999.971791214713 -- finished
"""

from typing import Final, Iterable

import numpy as np
from pycommons.types import check_int_range, type_error

from moptipyapps.prodsched.instance import (
    Demand,
)
from moptipyapps.prodsched.random_simulation import (
    RandomInstance,
    RandomSimulation,
)
from moptipyapps.prodsched.simulation import Listener


class RandomROPSimulation(RandomSimulation):
    """Create the randomized re-order point-based simulation."""

    def __init__(self, instance: RandomInstance, listener: Listener) -> None:
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
