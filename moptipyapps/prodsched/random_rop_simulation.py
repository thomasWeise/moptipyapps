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
T=0.23256075210980243: finished j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, \
st: 0, sp: 0) at station 0
T=0.23256075210980243: station=0, 9 jobs queued
T=0.23256075210980243: start j(id: 1, p: 2, am: 2, ar: 0, me: F, c: F, st: 0, \
sp: 0) at station 0
T=0.23256075210980243: station=1, 1 jobs queued
T=0.23256075210980243: start j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
0.23256075210980243, sp: 1) at station 1
T=0.6739674524451462: finished j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
0.23256075210980243, sp: 1) at station 1
T=0.6739674524451462: station=3, 1 jobs queued

>>> for i in range(len(collector) - 20, len(collector)):
...     print(collector[i])
T=9999.071078667212! station=1, 1 jobs queued
T=9999.071078667212! start j(id: 10006, p: 3, am: 1, ar: 9997.478248967976, \
me: T, c: F, st: 9999.047688155826, sp: 1) at station 1
T=9999.374319435468! finished j(id: 9973, p: 9, am: 1, ar: 9961.480560480835, \
me: T, c: F, st: 9979.538561053007, sp: 9) at station 8
T=9999.374319435468! station=8, 12 jobs queued
T=9999.374319435468! start j(id: 9985, p: 5, am: 1, ar: 9975.80936360626, \
me: T, c: F, st: 9980.276754034685, sp: 5) at station 8
T=9999.495833576984! finished j(id: 9989, p: 8, am: 1, ar: 9980.63490662843, \
me: T, c: F, st: 9992.691023635964, sp: 3) at station 4
T=9999.495833576984! station=4, 3 jobs queued
T=9999.495833576984! start j(id: 10001, p: 1, am: 1, ar: 9992.88475372726, \
me: T, c: F, st: 9994.188584640297, sp: 2) at station 4
T=9999.633994264326! finished j(id: 9992, p: 7, am: 1, ar: 9982.32594565836, \
me: T, c: F, st: 9995.278219527183, sp: 4) at station 3
T=9999.633994264326! station=3, 3 jobs queued
T=9999.633994264326! start j(id: 10003, p: 2, am: 1, ar: 9995.709243058642, \
me: T, c: F, st: 9998.013009569017, sp: 3) at station 3
T=9999.633994264326! station=11, 1 jobs queued
T=9999.633994264326! start j(id: 9992, p: 7, am: 1, ar: 9982.32594565836, \
me: T, c: F, st: 9999.633994264326, sp: 5) at station 11
T=9999.649149366647! finished j(id: 10006, p: 3, am: 1, \
ar: 9997.478248967976, me: T, c: F, st: 9999.047688155826, sp: 1) at station 1
T=9999.649149366647! station=6, 1 jobs queued
T=9999.649149366647! start j(id: 10006, p: 3, am: 1, ar: 9997.478248967976, \
me: T, c: F, st: 9999.649149366647, sp: 2) at station 6
T=9999.998742667354! product=7, amount=0, in_warehouse=0, in_production=3, \
2 pending demands
T=9999.998742667354! station=0, 1 jobs queued
T=9999.998742667354! start j(id: 10007, p: 7, am: 1, ar: 9999.998742667354, \
me: T, c: F, st: 9999.998742667354, sp: 0) at station 0
T=9999.998742667354 -- finished
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
