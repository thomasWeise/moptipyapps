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
T=1.700981296962859: finished j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
0, sp: 0) at station 0
T=1.700981296962859: station=0, 9 jobs queued
T=1.700981296962859: start j(id: 1, p: 2, am: 2, ar: 0, me: F, c: F, st: 0, \
sp: 0) at station 0
T=1.700981296962859: station=1, 1 jobs queued
T=1.700981296962859: start j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
1.700981296962859, sp: 1) at station 1
T=2.217634752949551: finished j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, \
st: 1.700981296962859, sp: 1) at station 1
T=2.217634752949551: station=3, 1 jobs queued

>>> for i in range(len(collector) - 20, len(collector)):
...     print(collector[i])
T=9999.33328342021! start j(id: 10014, p: 2, am: 1, ar: 9970.157310086975, \
me: T, c: F, st: 9996.158106819432, sp: 7) at station 10
T=9999.33328342021! product=3, amount=1, in_warehouse=0, in_production=2, \
1 pending demands
T=9999.33328342021! d(id: 4008, p: 3, c: 4008, am: 1, ar: 9989.806382347304, \
dl: 9989.806382347304, me: T) satisfied
T=9999.522691001024! finished j(id: 10021, p: 4, am: 1, ar: 9974.764243825606\
, me: T, c: F, st: 9995.187765657156, sp: 6) at station 1
T=9999.522691001024! station=1, 9 jobs queued
T=9999.522691001024! start j(id: 10017, p: 6, am: 1, ar: 9971.454782810877, \
me: T, c: F, st: 9995.510736597318, sp: 6) at station 1
T=9999.66344376004! finished j(id: 10044, p: 5, am: 1, ar: 9997.291327122433\
, me: T, c: F, st: 9997.291327122433, sp: 0) at station 0
T=9999.66344376004! station=0, 2 jobs queued
T=9999.66344376004! start j(id: 10045, p: 4, am: 1, ar: 9997.792977742796, \
me: T, c: F, st: 9997.792977742796, sp: 0) at station 0
T=9999.669678312202! finished j(id: 10039, p: 6, am: 1, ar: \
9993.680935722501, me: T, c: F, st: 9998.800471480217, sp: 2) at station 5
T=9999.79811329168! product=6, amount=0, in_warehouse=0, in_production=4, \
3 pending demands
T=9999.93543391247! finished j(id: 10014, p: 2, am: 1, ar: 9970.157310086975,\
 me: T, c: T, st: 9996.158106819432, sp: 7) at station 10
T=9999.93543391247! station=10, 2 jobs queued
T=9999.93543391247! start j(id: 9982, p: 9, am: 1, ar: 9933.380445487133, \
me: T, c: F, st: 9997.379811520846, sp: 10) at station 10
T=9999.93543391247! product=2, amount=1, in_warehouse=0, in_production=3, \
2 pending demands
T=9999.93543391247! d(id: 2989, p: 2, c: 2989, am: 1, ar: 9983.827600477664\
, dl: 9983.827600477664, me: T) satisfied
T=9999.995308785708! finished j(id: 10024, p: 8, am: 1, ar: 9976.199007067482\
, me: T, c: F, st: 9997.64169250468, sp: 3) at station 4
T=9999.995308785708! station=4, 1 jobs queued
T=9999.995308785708! start j(id: 10038, p: 5, am: 1, ar: 9993.649532104819\
, me: T, c: F, st: 9998.100880797963, sp: 2) at station 4
T=9999.995308785708 -- finished

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
T=1.700981296962859: finished j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
0, sp: 0) at station 0
T=1.700981296962859: station=0, 9 jobs queued
T=1.700981296962859: start j(id: 1, p: 2, am: 2, ar: 0, me: F, c: F, st: 0, \
sp: 0) at station 0
T=1.700981296962859: station=1, 1 jobs queued
T=1.700981296962859: start j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, st: \
1.700981296962859, sp: 1) at station 1
T=2.217634752949551: finished j(id: 0, p: 0, am: 2, ar: 0, me: F, c: F, \
st: 1.700981296962859, sp: 1) at station 1
T=2.217634752949551: station=3, 1 jobs queued

>>> for i in range(len(collector) - 20, len(collector)):
...     print(collector[i])
T=9999.33328342021! start j(id: 10014, p: 2, am: 1, ar: 9970.157310086975, \
me: T, c: F, st: 9996.158106819432, sp: 7) at station 10
T=9999.33328342021! product=3, amount=1, in_warehouse=0, in_production=2, \
1 pending demands
T=9999.33328342021! d(id: 4008, p: 3, c: 4008, am: 1, ar: 9989.806382347304, \
dl: 9989.806382347304, me: T) satisfied
T=9999.522691001024! finished j(id: 10021, p: 4, am: 1, ar: 9974.764243825606\
, me: T, c: F, st: 9995.187765657156, sp: 6) at station 1
T=9999.522691001024! station=1, 9 jobs queued
T=9999.522691001024! start j(id: 10017, p: 6, am: 1, ar: 9971.454782810877, \
me: T, c: F, st: 9995.510736597318, sp: 6) at station 1
T=9999.66344376004! finished j(id: 10044, p: 5, am: 1, ar: 9997.291327122433\
, me: T, c: F, st: 9997.291327122433, sp: 0) at station 0
T=9999.66344376004! station=0, 2 jobs queued
T=9999.66344376004! start j(id: 10045, p: 4, am: 1, ar: 9997.792977742796, \
me: T, c: F, st: 9997.792977742796, sp: 0) at station 0
T=9999.669678312202! finished j(id: 10039, p: 6, am: 1, ar: \
9993.680935722501, me: T, c: F, st: 9998.800471480217, sp: 2) at station 5
T=9999.79811329168! product=6, amount=0, in_warehouse=0, in_production=4, \
3 pending demands
T=9999.93543391247! finished j(id: 10014, p: 2, am: 1, ar: 9970.157310086975,\
 me: T, c: T, st: 9996.158106819432, sp: 7) at station 10
T=9999.93543391247! station=10, 2 jobs queued
T=9999.93543391247! start j(id: 9982, p: 9, am: 1, ar: 9933.380445487133, \
me: T, c: F, st: 9997.379811520846, sp: 10) at station 10
T=9999.93543391247! product=2, amount=1, in_warehouse=0, in_production=3, \
2 pending demands
T=9999.93543391247! d(id: 2989, p: 2, c: 2989, am: 1, ar: 9983.827600477664\
, dl: 9983.827600477664, me: T) satisfied
T=9999.995308785708! finished j(id: 10024, p: 8, am: 1, ar: 9976.199007067482\
, me: T, c: F, st: 9997.64169250468, sp: 3) at station 4
T=9999.995308785708! station=4, 1 jobs queued
T=9999.995308785708! start j(id: 10038, p: 5, am: 1, ar: 9993.649532104819\
, me: T, c: F, st: 9998.100880797963, sp: 2) at station 4
T=9999.995308785708 -- finished
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
