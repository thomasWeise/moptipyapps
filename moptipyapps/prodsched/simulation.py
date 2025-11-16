"""
A simulator for production scheduling.

For simulation a production system, we can build on the class
:class:`~Simulation`. This base class offers the support to implement almost
arbitrarily complex production system scheduling logic. There are three groups
of methods:

- Methods starting with `ctrl_*` are for starting and resetting the
  simulation so that it can be started again. You may override them if you
  have additional need for initialization or clean-up.
- Methods that start with `event_*` are methods that are invoked by the
  simulator to notify you about an event in the simulation. You can overwrite
  these methods to implement the logic of your production scheduling method.
- Methods that start with `act_*` are actions that you can invoke inside the
  `event_*` methods. The tell the simulator or stations what to do.

## We have the following `ctrl_*` methods:**

- :meth:`~Simulation.ctrl_run` runs the simulation.
- :meth:`~Simulation.ctrl_reset` resets the simulator so that we can start it
  again. If you want to re-use a simulation, you need to first invoke
  :meth:`~Simulation.ctrl_reset` to clear the internal state.

## We have the following `event_*` methods:**

- :meth:`~Simulation.event_product` is invoked by the simulation if one of the
  following three things happened:

    1. An amount of a product has been produced (`amount > 0`).
    2. An amount of a product has been made available at the start of the
       simulation to form the initial amount in the warehouse
       (`amount > 0`).
    3. A customer demand for the product has appeared in the system.

  In this method, you can store product into the warehouse, remove product
  from the warehouse, and/or mark a demand as completed.


## Examples

Here we have a very easy production scheduling instance.
There is 1 product that passes through 2 stations.
First it passes through station 0, then through station 1.
The per-unit production time is always 10 time units on station 0 and 30 time
units on station 2.
There is one customer demand, for 10 units of this product, which enters the
system at time unit 20.
The warehouse is initially empty.
>>> instance = Instance(
...     name="test1", n_products=1, n_customers=1, n_stations=2, n_demands=1,
...     routes=[[0, 1]],
...     demands=[[0, 0, 0, 10, 20, 100]],
...     warehous_at_t0=[0],
...     station_product_unit_times=[[[10.0, 10000.0]],
...                                 [[30.0, 10000.0]]])

The simulation will see that the customer demand for 10 units of product 0
appears at time unit 20.
It will issue a production order for these 10 units at station 0.
Since station 0 is not occupied, it can immediately begin with the production.
It will finish the production after 10*10 time units, i.e., at time unit 120.
The product is then routed to station 1, which is also idle and can
immediately begin producing.
It needs 10*30 time units, meaning that it finishes after 300 time units.
The demanded product amount is completed after 420 time units and the demand 0
can be fulfilled.
>>> simulation = Simulation(instance, PrintingListener(print_time=False))
>>> simulation.ctrl_run()
start
T=20.0: EVENT product=0, amount=0, in_warehouse=0, in_production=0, \
1 pending demands
T=20.0: EVENT station=0, 1 jobs queued
T=20.0: beginning to produce 10 units of product 0 on station 0
T=120.0: finished producing 10 units of product 0 on station 0
T=120.0: EVENT station=1, 1 jobs queued
T=120.0: beginning to produce 10 units of product 0 on station 1
T=420.0: finished producing 10 units of product 0 on station 1
T=420.0: EVENT product=0, amount=10, in_warehouse=0, in_production=0, \
1 pending demands
T=420.0: demand 0 for 10 units of product 0 satisfied
T=420.0: finished


>>> instance = Instance(
...     name="test2", n_products=2, n_customers=1, n_stations=2, n_demands=2,
...     routes=[[0, 1], [1, 0]],
...     demands=[[0, 0, 1, 10, 20, 90], [1, 0, 0, 5, 22, 200]],
...     warehous_at_t0=[2, 1],
...     station_product_unit_times=[[[10.0, 50.0, 15.0, 100.0],
...                                  [ 5.0, 20.0,  7.0,  35.0, 4.0, 50.0]],
...                                 [[ 5.0, 24.0,  7.0,  80.0],
...                                  [ 3.0, 21.0,  6.0,  50.0,]]])

>>> instance.name
'test2'

>>> simulation = Simulation(instance, PrintingListener(print_time=False))
>>> simulation.ctrl_run()
start
T=0.0: EVENT product=0, amount=2, in_warehouse=0, in_production=0, \
0 pending demands
T=0.0: 2 units of product 0 in warehouse
T=0.0: EVENT product=1, amount=1, in_warehouse=0, in_production=0, \
0 pending demands
T=0.0: 1 units of product 1 in warehouse
T=20.0: EVENT product=1, amount=0, in_warehouse=1, in_production=0, \
1 pending demands
T=20.0: EVENT station=1, 1 jobs queued
T=20.0: beginning to produce 9 units of product 1 on station 1
T=22.0: EVENT product=0, amount=0, in_warehouse=2, in_production=0, \
1 pending demands
T=22.0: EVENT station=0, 1 jobs queued
T=22.0: beginning to produce 3 units of product 0 on station 0
T=53.0: finished producing 3 units of product 0 on station 0
T=61.5: finished producing 9 units of product 1 on station 1
T=61.5: EVENT station=1, 1 jobs queued
T=61.5: beginning to produce 3 units of product 0 on station 1
T=61.5: EVENT station=0, 1 jobs queued
T=61.5: beginning to produce 9 units of product 1 on station 0
T=81.78571428571429: finished producing 3 units of product 0 on station 1
T=81.78571428571429: EVENT product=0, amount=3, in_warehouse=2, \
in_production=0, 1 pending demands
T=81.78571428571429: demand 1 for 5 units of product 0 satisfied
T=81.78571428571429: 0 units of product 0 in warehouse
T=107.03571428571428: finished producing 9 units of product 1 on station 0
T=107.03571428571428: EVENT product=1, amount=9, in_warehouse=1, \
in_production=0, 1 pending demands
T=107.03571428571428: demand 0 for 10 units of product 1 satisfied
T=107.03571428571428: 0 units of product 1 in warehouse
T=107.03571428571428: finished


>>> simulation.ctrl_reset()
>>> simulation.ctrl_run()
start
T=0.0: EVENT product=0, amount=2, in_warehouse=0, in_production=0, \
0 pending demands
T=0.0: 2 units of product 0 in warehouse
T=0.0: EVENT product=1, amount=1, in_warehouse=0, in_production=0, \
0 pending demands
T=0.0: 1 units of product 1 in warehouse
T=20.0: EVENT product=1, amount=0, in_warehouse=1, in_production=0, \
1 pending demands
T=20.0: EVENT station=1, 1 jobs queued
T=20.0: beginning to produce 9 units of product 1 on station 1
T=22.0: EVENT product=0, amount=0, in_warehouse=2, in_production=0, \
1 pending demands
T=22.0: EVENT station=0, 1 jobs queued
T=22.0: beginning to produce 3 units of product 0 on station 0
T=53.0: finished producing 3 units of product 0 on station 0
T=61.5: finished producing 9 units of product 1 on station 1
T=61.5: EVENT station=1, 1 jobs queued
T=61.5: beginning to produce 3 units of product 0 on station 1
T=61.5: EVENT station=0, 1 jobs queued
T=61.5: beginning to produce 9 units of product 1 on station 0
T=81.78571428571429: finished producing 3 units of product 0 on station 1
T=81.78571428571429: EVENT product=0, amount=3, in_warehouse=2, \
in_production=0, 1 pending demands
T=81.78571428571429: demand 1 for 5 units of product 0 satisfied
T=81.78571428571429: 0 units of product 0 in warehouse
T=107.03571428571428: finished producing 9 units of product 1 on station 0
T=107.03571428571428: EVENT product=1, amount=9, in_warehouse=1, \
in_production=0, 1 pending demands
T=107.03571428571428: demand 0 for 10 units of product 1 satisfied
T=107.03571428571428: 0 units of product 1 in warehouse
T=107.03571428571428: finished
"""

from dataclasses import dataclass, field
from heapq import heappop, heappush
from time import time_ns
from typing import Any, Callable, Final

import numpy as np
from pycommons.types import type_error

from moptipyapps.prodsched.instance import (
    Demand,
    Instance,
    compute_finish_time,
)


@dataclass(order=True, frozen=True)
class _Event:
    """The internal record for events in the simulation."""

    #: When does the event happen?
    when: float
    #: Which function to call?
    call: Callable = field(compare=False)
    #: The arguments to pass to the function
    args: tuple = field(compare=False)


@dataclass(order=True, frozen=True)
class Job:
    """The record for a production job."""

    #: the ID of the product to be produced.
    product_id: int
    #: the amount to produce
    amount: int
    #: the time when the job was issued
    issue_time: float = -1.0
    #: the time when the job arrived at the queue of the current station.
    station_time: float = -1.0
    #: the current job step, starts at 0.
    step: int = -1


class Listener:
    """A listener for simulation events."""

    def start(self) -> None:
        """Get notification that the simulation is starting."""

    def product_in_warehouse(
            self, time: float, product_id: int, amount: int) -> None:
        """
        Report a change of the amount of products in the warehouse.

        :param time: the current time
        :param product_id: the product ID
        :param amount: the new absolute total amount of that product in the
            warehouse
        """

    def produce_at_begin(
            self, time: float, station_id: int, job: Job) -> None:
        """
        Report the start of the production of a certain product at a station.

        :param time: the current time
        :param station_id: the station ID
        :param job: the production job
        """

    def produce_at_end(
            self, time: float, station_id: int, job: Job) -> None:
        """
        Report the completion of the production of a product at a station.

        :param time: the current time
        :param station_id: the station ID
        :param job: the production job
        """

    def demand_satisfied(
            self, time: float, demand: Demand) -> None:
        """
        Report that a given demand has been satisfied.

        :param time: the time index when the demand was satisfied
        :param demand: the demand that was satisfied
        """

    def event_product(self, time: float,
                      product_id: int, amount: int,
                      in_warehouse: int, in_production: int,
                      pending_demands: tuple[Demand, ...]) -> None:
        """
        Get notified right before :meth:`Simulation.event_product`.

        :param time: the current system time
        :param product_id: the id of the product
        :param amount: the amount of the product that appears
        :param in_warehouse: the amount of the product currently in the
            warehouse
        :param in_production: the amounf of product currently under production
        :param pending_demands: the pending orders for the product
        """

    def event_station(self, time: float, station_id: int,
                      queue: tuple[Job, ...]) -> None:
        """
        Get notified right before :meth:`Simulation.event_station`.

        If this event happens, the station is not busy. It could process a job
        and there is at least one job that it could process. You can now
        select the job to be executed from the `queue` and pass it to
        :meth:`~Simulation.act_exec_job`.

        :param time: the current time
        :param station_id: the station ID
        :param queue: the job queue for this station
        """

    def finished(self, time: float) -> None:
        """
        Be notified that the simulation has been finished.

        :param time: the time when we are finished
        """


class Simulation:
    """A simulator for production scheduling."""

    def __init__(self, instance: Instance, listener: Listener) -> None:
        """
        Initialize the simulator.

        :param instance: the instance
        :param listener: the listener
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        if not isinstance(listener, Listener):
            raise type_error(listener, "listener", Listener)

        #: the instance whose data is simulated
        self.instance: Final[Instance] = instance
        #: the product routes
        self.__routes: Final[tuple[tuple[int, ...], ...]] = instance.routes
        #: the station-product-unit-times
        self.__mput: Final[tuple[tuple[np.ndarray, ...], ...]] = (
            instance.station_product_unit_times)

        #: the start event function
        self.__l_start: Final[Callable[[], None]] = listener.start
        #: the product-level-in-warehouse-changed event function
        self.__l_product_in_warehouse: Final[Callable[[
            float, int, int], None]] = listener.product_in_warehouse
        #: the demand satisfied event function
        self.__l_demand_satisfied: Final[Callable[[float, Demand], None]] \
            = listener.demand_satisfied
        #: the listener to be notified if the production of a certain
        #: product begins at a certain station.
        self.__l_produce_at_begin: Final[Callable[[
            float, int, Job], None]] = listener.produce_at_begin
        #: the listener to be notified if the production of a certain
        #: product end at a certain station.
        self.__l_produce_at_end: Final[Callable[[
            float, int, Job], None]] = listener.produce_at_end
        #: the listener to notify about simulation end
        self.__l_finished: Final[Callable[[float], None]] = listener.finished
        #: the listener to notify about product events
        self.__l_event_product: Final[Callable[[
            float, int, int, int, int, tuple[Demand, ...]], None]] = \
            listener.event_product
        #: the listener to notify about station events
        self.__l_event_station: Final[Callable[[
            float, int, tuple[Job, ...]], None]] = listener.event_station

        #: the current time
        self.__time: float = 0.0
        #: the internal event queue
        self.__queue: Final[list[_Event]] = []
        #: the internal list of pending demands
        self.__pending_demands: Final[list[list[Demand]]] = [
            [] for _ in range(instance.n_products)]
        #: the internal list of the amount of product currently in production
        self.__in_production: Final[list[int]] = [
            0 for _ in range(instance.n_products)]
        #: the internal warehouse
        self.__warehouse: Final[list[int]] = [0] * instance.n_products
        #: the station queues.
        self.__mq: Final[list[list[Job]]] = [
            [] for _ in range(instance.n_stations)]
        #: whether the stations are busy
        self.__mbusy: Final[list[bool]] = [False] * instance.n_stations

    def ctrl_reset(self) -> None:
        """
        Reset the simulation.

        This function sets the time to 0, clears the event queue, clears
        the pending orders list, clears the warehouse.
        """
        self.__time = 0.0
        self.__queue.clear()
        for i in range(self.instance.n_products):
            self.__warehouse[i] = 0
            self.__pending_demands[i].clear()
            self.__in_production[i] = 0
        for mq in self.__mq:
            mq.clear()
        for i in range(self.instance.n_stations):
            self.__mbusy[i] = False

    def ctrl_run(self) -> None:
        """
        Run the simulation.

        This function executes the main loop of the simulation. It runs the
        central event pump, which is a priority queue. It processes the
        simulation events one by one.
        """
        self.__l_start()
        queue: Final[list[_Event]] = self.__queue

        #: fill the warehouse at time index 0
        for product_id, amount in enumerate(self.instance.warehous_at_t0):
            if amount > 0:
                heappush(queue, _Event(0.0, self.__product_available, (
                    product_id, amount)))
        #: fill in the customer demands/orders
        for demand in self.instance.demands:
            heappush(queue, _Event(
                demand.arrival, self.__demand_issued, (demand, )))

        while list.__len__(queue):
            event: _Event = heappop(queue)
            time: float = event.when
            if time < self.__time:
                raise ValueError(f"Event for {time} at time {self.__time}?")
            self.__time = time
            event.call(*event.args)

        self.__l_finished(self.__time)

    def act_demand_satisfied(self, demand: Demand) -> None:
        """
        Notify the system that a given demand has been satisfied.

        :param demand: the demand that was satisfied
        """
        pd: Final[list[Demand]] = self.__pending_demands[demand.product_id]
        del pd[pd.index(demand)]  # force exception if not contained
        self.__l_demand_satisfied(self.__time, demand)

    def event_product(self, time: float,  # pylint: disable=W0613
                      product_id: int, amount: int,
                      in_warehouse: int,
                      in_production: int,  # pylint: disable=W0613
                      pending_demands: tuple[Demand, ...]) -> None:
        """
        Take actions when an event regarding a product or demand occurred.

        The following events may have occurred:

        1. An amount of a product has been produced (`amount > 0`).
        2. An amount of a product has been made available at the start of the
           simulation to form the initial amount in the warehouse
           (`amount > 0`).
        3. A customer demand for the product has appeared in the system. If
           there is any demand to be fulfilled, then `pending_demands` is not
           empty.

        You can choose to execute one or multiple of the following actions:

        1. :meth:`~Simulation.act_store_in_warehouse` to store a positive
           amount of product in the warehouse.
        2. :meth:`~Simulation.act_take_from_warehouse` to take a positive
           amount of product out of the warehouse (must be `<= in_warehouse`.
        3. :meth:`~Simulation.act_produce` to order the production of a
           positive amount of the product.
        4. :meth:`~Simulation.act_demand_satisfied` to mark one of the demands
           from `queue` as satisfied. Notice that in this case, you must make
           sure to remove the corresponding amount of product units from the
           system. If sufficient units are in `amount`, you would simply not
           store these in the warehouse. You could also simply take some units
           out of the warehouse with :meth:`~act_take_from_warehouse`.

        :param time: the current system time
        :param product_id: the id of the product
        :param amount: the amount of the product that appears
        :param in_warehouse: the amount of the product currently in the
            warehouse
        :param in_production: the amounf of product currently under production
        :param pending_demands: the pending orders for the product
        """
        dem_len: int = tuple.__len__(pending_demands)
        if dem_len <= 0:
            self.act_store_in_warehouse(product_id, amount)
            return

        # Go through the list of demands and satisfy them on a first-come-
        # first-serve basis.
        total: int = in_warehouse + amount  # The available units.
        product_needed: int = 0  # The amount needed to satisfy the demands.
        for demand in pending_demands:
            demand_needed: int = demand.amount
            if demand_needed <= total:
                total -= demand_needed
                self.act_demand_satisfied(demand)
                continue
            product_needed += demand_needed

        #: Update the warehouse.
        if total > in_warehouse:
            self.act_store_in_warehouse(product_id, total - in_warehouse)
        elif total < in_warehouse:
            self.act_take_from_warehouse(product_id, in_warehouse - total)

        # Order the production of the product units required to satisfy all
        # demands.
        product_needed -= total + in_production
        if product_needed > 0:
            self.act_produce(product_id, product_needed)

    def event_station(self,
                      time: float,  # pylint: disable=W0613
                      station_id: int,  # pylint: disable=W0613
                      queue: tuple[Job, ...]) -> None:
        """
        Process an event for a given station.

        If this event happens, the station is not busy. It could process a job
        and there is at least one job that it could process. You can now
        select the job to be executed from the `queue` and pass it to
        :meth:`~Simulation.act_exec_job`.

        :param time: the current time
        :param station_id: the station ID
        :param queue: the job queue for this station
        """
        self.act_exec_job(queue[0])

    def act_exec_job(self, job: Job) -> None:
        """
        Execute the job on its current station.

        :param job: the job to be executed
        """
        product_id: Final[int] = job.product_id
        station_id: Final[int] = self.__routes[product_id][job.step]
        time: Final[float] = self.__time
        queue: list[Job] = self.__mq[station_id]
        del queue[queue.index(job)]  # force exception if job is not there

        if self.__mbusy[station_id]:
            raise ValueError("Cannot execute job on busy station.")
        self.__mbusy[station_id] = True
        self.__l_produce_at_begin(time, station_id, job)
        end_time: float = compute_finish_time(
            time, job.amount, self.__mput[station_id][product_id])
        heappush(self.__queue, _Event(end_time, self.__job_step, (job, )))

    def act_store_in_warehouse(self, product_id: int, amount: int) -> None:
        """
        Add a certain amount of product to the warehouse.

        :param product_id: the product ID
        :param amount: the amount
        """
        if amount <= 0:
            raise ValueError(
                f"Cannot add amount {amount} of product {product_id}!")
        wh: int = self.__warehouse[product_id] + amount
        self.__warehouse[product_id] = wh
        self.__l_product_in_warehouse(self.__time, product_id, wh)

    def act_take_from_warehouse(self, product_id: int, amount: int) -> None:
        """
        Remove a certain amount of product to the warehouse.

        :param product_id: the product ID
        :param amount: the amount
        """
        if amount <= 0:
            raise ValueError(
                f"Cannot remove amount {amount} of product {product_id}!")
        wh: int = self.__warehouse[product_id] - amount
        if wh < 0:
            raise ValueError(
                f"Cannot remove {amount} of product {product_id} from "
                "warehouse if there are only "
                f"{self.__warehouse[product_id]} units in it.")
        self.__warehouse[product_id] = wh
        self.__l_product_in_warehouse(self.__time, product_id, wh)

    def act_produce(self, product_id: int, amount: int) -> None:
        """
        Order the production of `amount` units of product.

        :param product_id: the product ID
        :param amount: the amount that needs to be produced
        """
        if amount <= 0:
            raise ValueError(
                f"Cannot produce {amount} units of product {product_id}.")
        self.__job_step(Job(product_id, amount, self.__time))

    def __product_available(
            self, product_id: int, amount: int) -> None:
        """
        Process that an amount of a product enters the warehouse.

        :param time: the time when it enters the warehouse
        :param product_id: the product ID
        :param amount: the amount of the product that enters the warehouse
        """
        lst: Final[list[Demand]] = self.__pending_demands[product_id]
        tp: Final[tuple] = tuple(lst) if list.__len__(lst) > 0 else ()
        wh: Final[int] = self.__warehouse[product_id]
        ip: Final[int] = self.__in_production[product_id]
        time: Final[float] = self.__time
        self.__l_event_product(time, product_id, amount, wh, ip, tp)
        self.event_product(time, product_id, amount, wh, ip, tp)

    def __demand_issued(self, demand: Demand) -> None:
        """
        Process that a demand was issued by a customer.

        :param demand: the demand record
        """
        time: float = self.__time
        if demand.arrival != time:
            raise ValueError(
                f"Demand time {demand.arrival} != system time {time}")
        product_id: int = demand.product_id
        lst: list[Demand] = self.__pending_demands[product_id]
        lst.append(demand)
        tp: Final[tuple[Demand, ...]] = tuple(lst)
        ip: Final[int] = self.__in_production[product_id]
        iw: Final[int] = self.__warehouse[product_id]
        self.__l_event_product(time, product_id, 0, iw, ip, tp)
        self.event_product(time, product_id, 0, iw, ip, tp)

    def __job_step(self, job: Job) -> None:
        """
        Move a job a step forward.

        If this job just enters the system, it gets enqueued at its first
        station. If it was already running on a station, then that station
        becomes idle and can process the next job. Our job now either moves to
        the next station and enters the queue of that station OR, if it has
        been completed, its produced product amount can enter the warehouse.

        :param job: the job
        """
        product_id: Final[int] = job.product_id
        routes: Final[tuple[int, ...]] = self.__routes[product_id]
        time: Final[float] = self.__time

        job_step: int = job.step
        if job_step >= 0:  # The job was running on a station.
            old_station_id: Final[int] = routes[job_step]
            self.__l_produce_at_end(time, old_station_id, job)
            self.__mbusy[old_station_id] = False
            old_mq: Final[list[Job]] = self.__mq[old_station_id]
            if list.__len__(old_mq) > 0:
                tupo: Final[tuple[Job, ...]] = tuple(old_mq)
                self.__l_event_station(time, old_station_id, tupo)
                self.event_station(time, old_station_id, tupo)
        else:
            self.__in_production[product_id] += job.amount

        job_step += 1
        max_route: Final[int] = tuple.__len__(routes)
        if job_step >= max_route:
            self.__in_production[product_id] -= job.amount
            self.__product_available(product_id, job.amount)
            return

        object.__setattr__(job, "step", job_step)
        object.__setattr__(job, "station_time", time)

        new_station_id: Final[int] = routes[job_step]
        queue: list[Job] = self.__mq[new_station_id]
        queue.append(job)
        if not self.__mbusy[new_station_id]:
            tupq: Final[tuple[Job, ...]] = tuple(queue)
            self.__l_event_station(time, new_station_id, tupq)
            self.event_station(time, new_station_id, tupq)


class PrintingListener(Listener):
    """A listener that just prints simulation events."""

    def __init__(self, output: Callable[[str], Any] = print,
                 print_time: bool = True) -> None:
        """
        Initialize the printing listener.

        :param output: the output callable
        :param print_time: shall we print the time?
        """
        if not callable(output):
            raise type_error(output, "output", call=True)
        if not isinstance(print_time, bool):
            raise type_error(print_time, "print_time", bool)
        #: the output callable
        self.__output: Final[Callable[[str], Any]] = output
        #: shall we print the time at the end?
        self.__print_time: Final[bool] = print_time
        #: the internal start time
        self.__start_time_ns: int | None = None

    def start(self) -> None:
        """Print that the simulation begins."""
        self.__start_time_ns = time_ns()
        self.__output("start")

    def product_in_warehouse(
            self, time: float, product_id: int, amount: int) -> None:
        """Print the product amount in the warehouse."""
        self.__output(f"T={time}: {amount} units of product "
                      f"{product_id} in warehouse")

    def produce_at_begin(
            self, time: float, station_id: int, job: Job) -> None:
        """Print that the production at a given station begun."""
        self.__output(f"T={time}: beginning to produce {job.amount} units "
                      f"of product {job.product_id} on station {station_id}")

    def produce_at_end(self, time: float, station_id: int, job: Job) -> None:
        """Print that the production at a given station ended."""
        self.__output(f"T={time}: finished producing {job.amount} units of "
                      f"product {job.product_id} on station {station_id}")

    def demand_satisfied(self, time: float, demand: Demand) -> None:
        """Print that a demand was satisfied."""
        self.__output(f"T={time}: demand {demand.demand_id} for "
                      f"{demand.amount} units of product {demand.product_id} "
                      "satisfied")

    def event_product(self, time: float,
                      product_id: int, amount: int,
                      in_warehouse: int, in_production: int,
                      pending_demands: tuple[Demand, ...]) -> None:
        """Print the prouct event."""
        self.__output(f"T={time}: EVENT product={product_id}, amount={amount}"
                      f", in_warehouse={in_warehouse}, in_production="
                      f"{in_production}, {tuple.__len__(pending_demands)} "
                      "pending demands")

    def event_station(self, time: float, station_id: int,
                      queue: tuple[Job, ...]) -> None:
        """Print the station event."""
        self.__output(f"T={time}: EVENT station={station_id}, "
                      f"{tuple.__len__(queue)} jobs queued")

    def finished(self, time: float) -> None:
        """Print that the simulation has finished."""
        end: Final[int] = time_ns()
        self.__output(f"T={time}: finished")
        if self.__print_time and self.__start_time_ns is not None:
            required: float = (end - self.__start_time_ns) / 1_000_000_000
            self.__output(f"Simulation time: {required}s")


def warmup() -> None:
    """
    Perform a warm-up for our simulator.

    The simulator uses some code implemented in numba etc., which may need to
    be jitted before the actual execution.

    >>> warmup()
    """
    instance = Instance(
        name="warmup", n_products=2, n_customers=1, n_stations=2, n_demands=2,
        routes=[[0, 1], [1, 0]],
        demands=[[0, 0, 1, 10, 20, 90], [1, 0, 0, 5, 22, 200]],
        warehous_at_t0=[2, 1],
        station_product_unit_times=[
            [[10.0, 50.0, 15.0, 100.0], [5.0, 20.0, 7.0, 35.0, 4.0, 50.0]],
            [[5.0, 24.0, 7.0, 80.0], [3.0, 21.0, 6.0, 50.0]]])
    Simulation(instance, Listener()).ctrl_run()
