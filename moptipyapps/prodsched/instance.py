"""
A production scheduling instance.

Production instances have names :attr:`Instance.name`.

Notice that production times are used in a cycling fashion.
The time when a certain product is finished can be computed via
:func:`~compute_finish_time` in an efficient way.

>>> name = "my_instance"

The number of products be 3.
>>> n_products = 3

The number of customers be 5.
>>> n_customers = 5

The number of machines be 4.
>>> n_machines = 4

There will be 6 customer demands.
>>> n_demands = 6

Each product may take a different route through different machines.
>>> route_p0 = [0, 3, 2]
>>> route_p1 = [0, 2, 1, 3]
>>> route_p2 = [1, 2, 3]
>>> routes = [route_p0, route_p1, route_p2]

Each demand is a tuple of demand_id, customer_id, product_id, amount,
release time, and deadline.
>>> d0 = [0, 0, 1, 20, 1240,  3000]
>>> d1 = [1, 1, 0, 10, 2300,  4000]
>>> d2 = [2, 2, 2,  7, 8300, 11000]
>>> d3 = [3, 3, 1, 12, 7300,  9000]
>>> d4 = [4, 4, 2, 23, 5410, 16720]
>>> d5 = [5, 3, 0, 19, 4234, 27080]
>>> demands = [d0, d1, d2, d3, d4, d5]

There is a fixed amount of each product in the warehouse at time step 0.
>>> warehous_at_t0 = [10, 0, 6]

Each machine requires a certain working time for each unit of each product.
This production time may vary over time.
For example, maybe machine 0 needs 10 time units for 1 unit of product 0 from
time step 0 to time step 19, then 11 time units from time step 20 to 39, then
8 time units from time step 40 to 59.
These times are cyclic, meaning that at time step 60 to 79, it will again need
10 time units, and so on.
Of course, production times are only specified for machines that a product is
actually routed through.
>>> m0_p0 = [10, 20, 11, 40,  8, 60]
>>> m0_p1 = [12, 20,  7, 40, 11, 70]
>>> m0_p2 = []
>>> m1_p0 = []
>>> m1_p1 = [20, 50, 30, 120,  7, 200]
>>> m1_p2 = [21, 50, 29, 130,  8, 190]
>>> m2_p0 = [ 8, 20,  9, 60]
>>> m2_p1 = [10, 90]
>>> m2_p2 = [12, 70,  30, 120]
>>> m3_p0 = [70, 200,  3, 220]
>>> m3_p1 = [60, 220,  5, 260]
>>> m3_p2 = [30, 210, 10, 300]
>>> machine_product_unit_times = [[m0_p0, m0_p1, m0_p2],
...                               [m1_p0, m1_p1, m1_p2],
...                               [m2_p0, m2_p1, m2_p2],
...                               [m3_p0, m3_p1, m3_p2]]

We can (but do not need to) provide additional information as key-value pairs.
>>> infos = {"source": "manually created",
...          "creation_date": "2025-11-09"}

From all of this data, we can create the instance.
>>> instance = Instance(name, n_products, n_customers, n_machines, n_demands,
...                     routes, demands, warehous_at_t0,
...                     machine_product_unit_times, infos)
>>> instance.name
'my_instance'

>>> instance.n_customers
5

>>> instance.n_machines
4

>>> instance.n_demands
6

>>> instance.n_products
3

>>> instance.routes
((0, 3, 2), (0, 2, 1, 3), (1, 2, 3))

>>> instance.demands
(Demand(release_time=1240, deadline=3000, demand_id=0, customer_id=0, \
product_id=1, amount=20), Demand(release_time=2300, deadline=4000, \
demand_id=1, customer_id=1, product_id=0, amount=10), \
Demand(release_time=4234, deadline=27080, demand_id=5, customer_id=3, \
product_id=0, amount=19), Demand(release_time=5410, deadline=16720, \
demand_id=4, customer_id=4, product_id=2, amount=23), \
Demand(release_time=7300, deadline=9000, demand_id=3, customer_id=3, \
product_id=1, amount=12), Demand(release_time=8300, deadline=11000, \
demand_id=2, customer_id=2, product_id=2, amount=7))

>>> instance.warehous_at_t0
(10, 0, 6)

>>> instance.machine_product_unit_times
(((10, 20, 11, 40, 8, 60), (12, 20, 7, 40, 11, 70), ()),\
 ((), (20, 50, 30, 120, 7, 200), (21, 50, 29, 130, 8, 190)),\
 ((8, 20, 9, 60), (10, 90), (12, 70, 30, 120)),\
 ((70, 200, 3, 220), (60, 220, 5, 260), (30, 210, 10, 300)))

>>> dict(instance.infos)
{'source': 'manually created', 'creation_date': '2025-11-09'}

We can serialize instances to a stream of strings and also load them back
from a stream of strings.
Here, we store `instance` to a stream.
We then load the independent instance `i2` from that stream.
>>> i2 = from_stream(to_stream(instance))
>>> i2 is instance
False

You can see that the loaded instance has the same data as the stored one.
>>> i2.name == instance.name
True
>>> i2.n_customers == instance.n_customers
True
>>> i2.n_machines == instance.n_machines
True
>>> i2.n_demands == instance.n_demands
True
>>> i2.n_products == instance.n_products
True
>>> i2.routes == instance.routes
True
>>> i2.demands == instance.demands
True
>>> i2.warehous_at_t0 == instance.warehous_at_t0
True
>>> i2.machine_product_unit_times == instance.machine_product_unit_times
True
>>> i2.infos == instance.infos
True
"""

from dataclasses import dataclass
from itertools import batched
from string import ascii_letters, digits
from typing import (
    Callable,
    Final,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    cast,
)

import numba  # type: ignore
from moptipy.api.component import Component
from moptipy.utils.logger import (
    COMMENT_START,
    KEY_VALUE_SEPARATOR,
)
from moptipy.utils.strings import sanitize_name
from pycommons.ds.immutable_map import immutable_mapping
from pycommons.io.csv import CSV_SEPARATOR
from pycommons.types import check_int_range, check_to_int_range, type_error

#: The maximum for the number of machines, products, or customers.
_MAX_DIM: Final[int] = 1_000_000

#: No value bigger than this is permitted in any tuple anywhere.
_INNER_MAX_DIM: Final[int] = min(366 * 24 * 3600 * _MAX_DIM, 2 ** 31 - 1)

#: the index of the demand ID
DEMAND_ID: Final[int] = 0
#: the index of the customer ID
DEMAND_CUSTOMER: Final[int] = 1
#: the index of the product ID
DEMAND_PRODUCT: Final[int] = 2
#: the index of the demanded amount
DEMAND_AMOUNT: Final[int] = 3
#: the index of the demand release time
DEMAND_TIME: Final[int] = 4
#: the index of the demand deadline
DEMAND_DEADLINE: Final[int] = 5


@dataclass(order=True, frozen=True)
class Demand(Iterable[int]):
    """The record for demands."""

    #: the release time, i.e., when the demand enters the system
    release_time: int
    #: the deadline, i.e., when the customer expects the result
    deadline: int
    #: the ID of the demand
    demand_id: int
    #: the customer ID
    customer_id: int
    #: the ID of the product
    product_id: int
    #: the amount
    amount: int

    def __post_init__(self) -> None:
        """Perform some basic sanity checks."""
        check_int_range(self.release_time, "release_time", 0, _INNER_MAX_DIM)
        check_int_range(self.deadline, "deadline", 0, _INNER_MAX_DIM)
        check_int_range(self.demand_id, "demand_id", 0, _MAX_DIM)
        check_int_range(self.customer_id, "customer_id", 0, _MAX_DIM)
        check_int_range(self.product_id, "product_id", 0, _MAX_DIM)
        check_int_range(self.amount, "amount", 0, _MAX_DIM)

    def __getitem__(self, item: int) -> int:
        """
        Access an element of this demand via an index.

        :param item: the index
        :return: the demand value at that index
        """
        if item == DEMAND_ID:
            return self.demand_id
        if item == DEMAND_CUSTOMER:
            return self.customer_id
        if item == DEMAND_PRODUCT:
            return self.product_id
        if item == DEMAND_AMOUNT:
            return self.amount
        if item == DEMAND_TIME:
            return self.release_time
        if item == DEMAND_DEADLINE:
            return self.deadline
        raise IndexError(
            f"index {item} out of bounds [0,{DEMAND_DEADLINE}].")

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the values in this demand.

        :return: the demand iterable
        """
        yield self.demand_id  # DEMAND_ID
        yield self.customer_id  # DEMAND_CUSTOMER
        yield self.product_id  # DEMAND_PRODUCT:
        yield self.amount  # DEMAND_AMOUNT:
        yield self.release_time  # DEMAND_TIME
        yield self.deadline  # DEMAND_DEADLINE

    def __len__(self) -> int:
        """
        Get the length of the demand record.

        :returns `6`: always
        """
        return 6


def __to_tuple(source: Iterable[int],
               pool: dict,
               empty_ok: bool = False) -> tuple[int, ...]:
    """
    Convert an iterable of type integer to a tuple.

    :param source: the data source
    :param empty_ok: are empty tuples OK?
    :param pool: the tuple pool
    :return: the tuple

    >>> ppl = {}
    >>> k1 = __to_tuple([1, 2, 3], ppl)
    >>> print(k1)
    (1, 2, 3)

    >>> __to_tuple({2}, ppl)
    (2,)

    >>> k2 = __to_tuple([1, 2, 3], ppl)
    >>> print(k2)
    (1, 2, 3)
    >>> k1 is k2
    True
    """
    use_row = source if isinstance(source, tuple) else tuple(source)
    if (tuple.__len__(use_row) <= 0) and (not empty_ok):
        raise ValueError("row has length 0.")
    for j, v in enumerate(use_row):
        if not isinstance(v, int):
            raise type_error(v, f"row[{j}]", int)
        if not 0 <= v <= _INNER_MAX_DIM:
            raise ValueError(f"row[{j}]={v} not in 0..{_INNER_MAX_DIM}")
    if use_row in pool:
        return pool[use_row]
    pool[use_row] = use_row
    return use_row


def __to_nested_tuples(source: Iterable,
                       pool: dict,
                       empty_ok: bool = False,
                       inner: Callable = __to_tuple) -> tuple:
    """
    Turn nested iterables of ints into nested tuples.

    :param source: the source list
    :param pool: the tuple pool
    :param empty_ok: are empty tuples OK?
    :param inner: the inner function
    :return: the tuple or array

    >>> ppl = {}
    >>> k1 = __to_nested_tuples([(1, 2), [3, 2]], ppl)
    >>> print(k1)
    ((1, 2), (3, 2))

    >>> k2 = __to_nested_tuples([(1, 2), (1, 2, 4),  (1, 2)], ppl)
    >>> print(k2)
    ((1, 2), (1, 2, 4), (1, 2))

    >>> k2[0] is k2[2]
    True

    >>> k1[0] is k2[0]
    True
    >>> k1[0] is k2[2]
    True

    >>> __to_nested_tuples([(1, 2), (1, 2, 4),  (1, 2), []], ppl, True)
    ((1, 2), (1, 2, 4), (1, 2), ())

    >>> __to_nested_tuples([(), {}, []], ppl, True)
    ()
    """
    if not isinstance(source, Iterable):
        raise type_error(source, "source", Iterable)
    dest: list[tuple] = []
    ins: int = 0
    for i, row in enumerate(source):
        use_row: tuple = inner(row, pool, empty_ok)
        ins += len(use_row)
        for j in range(i):
            if dest[j] == use_row:
                use_row = dest[j]
                break
        dest.append(use_row)

    if (ins <= 0) and empty_ok:  # if all inner tuples are empty,
        dest.clear()             # clear the tuple source

    n_rows: Final[int] = list.__len__(dest)
    if (n_rows <= 0) and (not empty_ok):
        raise ValueError("Got empty set of rows!")

    ret_val: tuple[tuple, ...] = tuple(dest)
    ret_val = cast("tuple[tuple, ...]", source) if source == ret_val \
        else ret_val
    if ret_val in pool:
        return pool[ret_val]
    pool[ret_val] = ret_val
    return ret_val


def __to_tuples(source: Iterable[Iterable[int]],
                pool: dict, empty_ok: bool = False) \
        -> tuple[tuple[int, ...], ...]:
    """
    Turn 2D nested iterables into 2D nested tuples.

    :param source: the source
    :param pool: the tuple pool
    :param empty_ok: are empty tuples OK?
    :return: the nested tuples

    >>> ppl = {}
    >>> k1 = __to_tuples([(1, 2), [3, 2]], ppl)
    >>> print(k1)
    ((1, 2), (3, 2))

    >>> k2 = __to_tuples([(1, 2), (1, 2, 4),  (1, 2)], ppl)
    >>> print(k2)
    ((1, 2), (1, 2, 4), (1, 2))

    >>> k2[0] is k2[2]
    True
    >>> k1[0] is k2[0]
    True
    >>> k1[0] is k2[2]
    True
    """
    return __to_nested_tuples(source, pool, empty_ok, __to_tuple)


def __to_tuples3(source: Iterable[Iterable[Iterable[int]]],
                 pool: dict, empty_ok: bool) \
        -> tuple[tuple[tuple[int, ...], ...], ...]:
    """
    Turn 3D nested iterables into 3D nested tuples.

    :param source: the source
    :param pool: the tuple pool
    :param empty_ok: are empty tuples OK?
    :param key: the sort key
    :return: the nested tuples

    >>> ppl = {}
    >>> k1 = __to_tuples3([[[3, 2], [44, 5], [2]], [[2], [5, 7]]], ppl, False)
    >>> print(k1)
    (((3, 2), (44, 5), (2,)), ((2,), (5, 7)))
    >>> k1[0][2] is k1[1][0]
    True
    """
    return __to_nested_tuples(source, pool, empty_ok, __to_tuples)


def _make_routes(
        n_products: int, n_machines: int,
        source: Iterable[Iterable[int]],
        pool: dict) -> tuple[tuple[int, ...], ...]:
    """
    Create the routes through machines for the products.

    Each product passes through a set of machines. It can pass through each
    machine at most once. It can only pass through valid machines.

    :param n_products: the number of products
    :param n_machines: the number of machines
    :param source: the source data
    :param pool: the tuple pool
    :return: the routes, a tuple of tuples

    >>> ppl = {}
    >>> _make_routes(2, 3, ((1, 2), (1, 0)), ppl)
    ((1, 2), (1, 0))

    >>> _make_routes(3, 3, ((1, 2), (1, 0), (0, 1, 2)), ppl)
    ((1, 2), (1, 0), (0, 1, 2))

    >>> k = _make_routes(3, 3, ((1, 2), (1, 2), (0, 1, 2)), ppl)
    >>> k[0] is k[1]
    True
    """
    check_int_range(n_products, "n_products", 1, _MAX_DIM)
    check_int_range(n_machines, "n_machines", 1, _MAX_DIM)
    dest: tuple[tuple[int, ...], ...] = __to_tuples(source, pool)

    n_rows: Final[int] = tuple.__len__(dest)
    if n_rows != n_products:
        raise ValueError(f"{n_products} products, but {n_rows} routes.")
    for i, route in enumerate(dest):
        stations: int = tuple.__len__(route)
        if not (0 < stations <= n_machines):
            raise ValueError(
                f"len(row[{i}])={stations}, but n_machines={n_machines}")
        if set.__len__(set(route)) != stations:
            raise ValueError(f"{route} contains duplicates.")
        for j, v in enumerate(route):
            if not (0 <= v < n_machines):
                raise ValueError(
                    f"row[{i},{j}]={v}, but n_machines={n_machines}")
    return dest


def __to_demand(
        source: Iterable[int], pool: dict, _) -> Demand:
    """
    Convert an integer source to a tuple or a demand.

    :param source: the source
    :param pool: the pool
    :return: the Demand
    """
    if isinstance(source, Demand):
        return cast("Demand", source)
    tup: tuple[int, ...] = __to_tuple(source, pool, False)
    dl: int = tuple.__len__(tup)
    if dl != 6:
        raise ValueError(f"Expected 6 values, got {dl}.")
    return Demand(
        demand_id=tup[DEMAND_ID], customer_id=tup[DEMAND_CUSTOMER],
        product_id=tup[DEMAND_PRODUCT], amount=tup[DEMAND_AMOUNT],
        release_time=tup[DEMAND_TIME],
        deadline=tup[DEMAND_DEADLINE])


def _make_demands(n_products: int, n_customers: int, n_demands: int,
                  source: Iterable[Iterable[int]], pool: dict) \
        -> tuple[Demand, ...]:
    """
    Create the demand records, sorted by release time.

    Each demand is a tuple of demand_id, customer_id, product_id, amount,
    release time, and deadline.

    :param n_products: the number of products
    :param n_customers: the number of customers
    :param n_demands: the number of demands
    :param source: the source data
    :param pool: the tuple pool
    :return: the demand tuples

    >>> ppl = {}
    >>> _make_demands(10, 10, 4, [[0, 2, 1, 4, 20, 21],
    ...     [2, 5, 2, 6, 17, 27],
    ...     [1, 6, 7, 12, 17, 21],
    ...     [3, 7, 3, 23, 5, 21]], ppl)
    (Demand(release_time=5, deadline=21, demand_id=3, customer_id=7, \
product_id=3, amount=23), Demand(release_time=17, deadline=21, demand_id=1, \
customer_id=6, product_id=7, amount=12), Demand(release_time=17, \
deadline=27, demand_id=2, customer_id=5, product_id=2, amount=6), \
Demand(release_time=20, deadline=21, demand_id=0, customer_id=2, \
product_id=1, amount=4))
    """
    check_int_range(n_products, "n_products", 1, _MAX_DIM)
    check_int_range(n_customers, "n_customers", 1, _MAX_DIM)
    check_int_range(n_demands, "n_demands", 1, _MAX_DIM)

    temp: tuple[Demand, ...] = __to_nested_tuples(
        source, pool, False, __to_demand)
    n_dem: int = tuple.__len__(temp)
    if n_dem != n_demands:
        raise ValueError(f"Expected {n_demands} demands, got {n_dem}?")

    used_ids: set[int] = set()
    min_id: int = 1000 * _MAX_DIM
    max_id: int = -1000 * _MAX_DIM
    dest: list[Demand] = []

    for i, demand in enumerate(temp):
        d_id: int = demand.demand_id
        if not (0 <= d_id < n_demands):
            raise ValueError(f"demand[{i}].id = {d_id}")
        if d_id in used_ids:
            raise ValueError(f"demand[{i}].id {d_id} appears twice!")
        used_ids.add(d_id)
        min_id = min(min_id, d_id)
        max_id = max(max_id, d_id)

        c_id: int = demand.customer_id
        if not (0 <= c_id < n_customers):
            raise ValueError(f"demand[{i}].customer = {c_id}, "
                             f"but n_customers={n_customers}")

        p_id: int = demand.product_id
        if not (0 <= p_id < n_products):
            raise ValueError(f"demand[{i}].product = {p_id}, "
                             f"but n_products={n_products}")

        amount: int = demand.amount
        if not (0 < amount < _MAX_DIM):
            raise ValueError(f"demand[{i}].amount = {amount}.")

        release_time: int = demand.release_time
        if not (0 < release_time < _MAX_DIM):
            raise ValueError(f"demand[{i}].release_time = {release_time}.")

        deadline: int = demand.deadline
        if not (release_time <= deadline < _MAX_DIM):
            raise ValueError(f"demand[{i}].deadline = {deadline}.")
        dest.append(demand)

    sl: int = set.__len__(used_ids)
    if sl != n_demands:
        raise ValueError(f"Got {n_demands} demands, but {sl} ids???")
    if ((max_id - min_id + 1) != n_demands) or (min_id != 0):
        raise ValueError(f"Invalid demand id range [{min_id}, {max_id}].")
    dest.sort()
    return tuple(dest)


def _make_in_warehouse(n_products: int, source: Iterable[int],
                       pool: dict) \
        -> tuple[int, ...]:
    """
    Make the amount of product in the warehouse at time 0.

    :param n_products: the total number of products
    :param source: the data source
    :param pool: the tuple pool
    :return: the amount of products in the warehouse

    >>> _make_in_warehouse(3, [1, 2, 3], {})
    (1, 2, 3)
    """
    ret: tuple[int, ...] = __to_tuple(source, pool)
    rl: Final[int] = tuple.__len__(ret)
    if rl != n_products:
        raise ValueError(f"We have {n_products} products, "
                         f"but the warehouse list length is {rl}.")
    for p, v in enumerate(ret):
        if not (0 <= v <= _MAX_DIM):
            raise ValueError(f"Got {v} units of product {p} in warehouse?")
    return ret


def _make_machine_product_unit_times(
        n_products: int, n_machines: int,
        routes: tuple[tuple[int, ...], ...],
        source: Iterable[Iterable[Iterable[int]]],
        pool: dict) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """
    Create the structure for the work times per product unit per machine.

    Here we have for each machine, for each product, a sequence of per-unit
    production settings. Each such "production settings" is a tuple with a
    per-unit production time and an end time index until which it is valid.
    Production times cycle, so if we produce something after the last end
    time index, we begin again at production time index 0.

    :param n_products: the number of products
    :param n_machines: the number of machines
    :param routes: the routes of the products through the machines
    :param source: the source array
    :param pool: the tuple pool
    :return: the machine unit times

    >>> ppl = {}
    >>> rts = _make_routes(3, 2, [[0, 1], [0], [1, 0]], ppl)
    >>> print(rts)
    ((0, 1), (0,), (1, 0))

    >>> mpt = _make_machine_product_unit_times(3, 2, rts, [
    ...     [[1, 2, 3, 5], [2, 4, 7, 18, 4, 52], [1, 10, 2, 30]],
    ...     [[2, 20, 3, 40], [], [4, 56, 34, 444]]], ppl)
    >>> print(mpt)
    (((1, 2, 3, 5), (2, 4, 7, 18, 4, 52), (1, 10, 2, 30)), \
((2, 20, 3, 40), (), (4, 56, 34, 444)))

    >>> mpt = _make_machine_product_unit_times(3, 2, rts, [
    ...     [[1, 2, 3, 5], [1, 2, 3, 5], [1, 10, 2, 30]],
    ...     [[2, 20, 3, 40], [], [4, 56, 34, 444]]], ppl)
    >>> print(mpt)
    (((1, 2, 3, 5), (1, 2, 3, 5), (1, 10, 2, 30)), ((2, 20, 3, 40), \
(), (4, 56, 34, 444)))
    >>> mpt[0][0] is mpt[0][1]
    True
    """
    ret: tuple[tuple[tuple[int, ...], ...], ...] = __to_tuples3(
        source, pool, True)

    if tuple.__len__(routes) != n_products:
        raise ValueError("invalid routes!")

    d1: int = tuple.__len__(ret)
    if d1 != n_machines:
        raise ValueError(
            f"Got {d1} machine-times, but {n_machines} machines.")
    for mid, machine in enumerate(ret):
        d2: int = tuple.__len__(machine)
        if d2 <= 0:
            for pid, r in enumerate(routes):
                if mid in r:
                    raise ValueError(
                        f"Machine {mid} in route for product {pid}, "
                        "but has no production time")
            continue
        if d2 != n_products:
            raise ValueError(f"got {d2} products for machine {mid}, "
                             f"but have {n_products} products")
        for pid, product in enumerate(machine):
            needs_times: bool = mid in routes[pid]
            d3: int = tuple.__len__(product)
            if (not needs_times) and (d3 > 0):
                raise ValueError(
                    f"product {pid} does not pass through machine {mid}, "
                    "so there must not be production times!")
            if needs_times and (d3 <= 0):
                raise ValueError(
                    f"product {pid} does pass through machine {mid}, "
                    "so there must be production times!")
            if (d3 % 2) != 0:
                raise ValueError(
                    f"production times for {pid} does pass through machine "
                    f"{mid}, must be of even length, but got length {d3}.")
            last_end = 0
            for pt, time in enumerate(batched(product, 2)):
                if tuple.__len__(time) != 2:
                    raise ValueError(f"production times must be 2-tuples, "
                                     f"but got {time} for product {pid} on "
                                     f"machine {mid} at position {pt}")
                unit_time, end = time
                duration = end - last_end
                if not (0 < unit_time <= duration < _MAX_DIM):
                    raise ValueError(
                        f"Invalid unit time {unit_time} and duration "
                        f"{duration} for product {pid} on machine {mid}")
                last_end = end

    return ret


def _make_infos(source: Iterable[tuple[str, str]] | Mapping[str, str] | None)\
        -> Mapping[str, str]:
    """
    Make the additional information record.

    :param source: the information to represent
    :return: the information record
    """
    use_source: Iterable[tuple[str, str]] = () if source is None else (
        source.items() if isinstance(source, Mapping) else source)
    if not isinstance(use_source, Iterable):
        raise type_error(source, "infos", Iterable)
    dst: dict[str, str] = {}
    for i, tup in enumerate(use_source):
        if tuple.__len__(tup) != 2:
            raise ValueError(f"Invalid tuple {tup} at index {i} in infos.")
        k: str = str.strip(tup[0])
        v: str = str.strip(tup[1])
        if (str.__len__(k) <= 0) or (str.__len__(v) <= 0):
            raise ValueError(f"Invalid key/values {k!r}/{v!r} in tuple "
                             f"{tup} at index {i} in infos.")
        if __FORBIDDEN_INFO_KEYS(str.lower(k)):
            raise ValueError(
                f"Info key {k!r} in tuple {tup} forbidden at index {i}.")
        if not all(map(__ALLOWED_INFO_KEY_CHARS, k)):
            raise ValueError(
                f"Malformed info key {k!r} in tuple {tup} at index {i}.")
        if k in dst:
            raise ValueError(f"Duplicate key {k!r} found in tuple {tup} "
                             f"at index {i} in infos.")
        dst[k] = v
    return immutable_mapping(dst)


class Instance(Component):
    """An instance of the Production Scheduling Problem."""

    def __init__(
            self, name: str,
            n_products: int, n_customers: int, n_machines: int,
            n_demands: int,
            routes: Iterable[Iterable[int]],
            demands: Iterable[Iterable[int]],
            warehous_at_t0: Iterable[int],
            machine_product_unit_times: Iterable[Iterable[Iterable[int]]],
            infos: Iterable[tuple[str, str]] | Mapping[
                str, str] | None = None) \
            -> None:
        """
        Create an instance of the production scheduling time.

        :param name: the instance name
        :param n_products: the number of products
        :param n_customers: the number of customers
        :param n_machines: the number of machines
        :param n_demands: the number of demand records
        :param routes: for each product, the sequence of machines that it has
            to pass
        :param demands: a sequences of demands of the form (
            customer_id, product_id, product_amount, release_time) OR a
            sequence of :class:`Demand` records.
        :param warehous_at_t0: the amount of products in the warehouse at time
            0 for each product
        :param machine_product_unit_times: for each machine and each product
            the per-unit-production time schedule, in the form of
            "per_unit_time, duration", where duration is the number of time
            units for which the per_unit_time is value
        :param machine_product_unit_times: the cycling unit times for each
            product on each machine, each with a validity duration
        :param infos: additional infos to be stored with the instance.
            These are key-value pairs with keys that are not used by the
            instance. They have no impact on the instance performance, but may
            explain settings of an instance generator.
        """
        use_name: Final[str] = sanitize_name(name)
        if name != use_name:
            raise ValueError(f"Name {name!r} is not a valid name.")
        if not all(map(_ALLOWED_NAME_CHARS, name)):
            raise ValueError(f"Name {name!r} contains invalid characters.")
        #: the name of this instance
        self.name: Final[str] = name

        #: the number of products in the scenario
        self.n_products: Final[int] = check_int_range(
            n_products, "n_products", 1, _MAX_DIM)
        #: the number of customers in the scenario
        self.n_customers: Final[int] = check_int_range(
            n_customers, "n_customers", 1, _MAX_DIM)
        #: the number of machines or workstations in the scenario
        self.n_machines: Final[int] = check_int_range(
            n_machines, "n_machines", 1, _MAX_DIM)
        #: the number of demands in the scenario
        self.n_demands: Final[int] = check_int_range(
            n_demands, "n_demands", 1, _MAX_DIM)

        pool: Final[dict] = {}

        #: the product routes, i.e., the machines through which each product
        #: must pass
        self.routes: Final[tuple[tuple[int, ...], ...]] = _make_routes(
            n_products, n_machines, routes, pool)
        #: The demands: Each demand is a tuple of demand_id, customer_id,
        #: product_id, amount, release_time, and deadline.
        #: The customer makes their order at time step release_time.
        #: They expect to receive their product by the deadline.
        #: The demands are sorted by release time and then deadline.
        #: The release time is always > 0.
        #: The deadline is always >= release time.
        #: Demand ids are unique.
        self.demands: Final[tuple[Demand, ...]] = _make_demands(
            n_products, n_customers, n_demands, demands, pool)

        #: The units of product in the warehouse at time step 0.
        #: For each product, we have either 0 or a positive amount of product.
        self.warehous_at_t0: Final[tuple[int, ...]] = _make_in_warehouse(
            n_products, warehous_at_t0, pool)

        #: The per-machine unit production times for each product.
        #: Each machine can have different production times per product.
        #: Let's say that this is tuple `A`.
        #: For each product, it has a tuple `B` at the index of the product
        #: id.
        #: If the product does not pass through the machine, `B` is empty.
        #: Otherwise, it holds one or multiple tuples `C`.
        #: Each tuple `C` consists of two numbers:
        #: A per-unit-production time for the product.
        #: An end time index for this production time.
        #: Once the real time surpasses the end time of the last of these
        #: production specs, the production specs are recycled and begin
        #: again.
        self.machine_product_unit_times: Final[tuple[tuple[tuple[
            int, ...], ...], ...]] = _make_machine_product_unit_times(
            n_products, n_machines, self.routes, machine_product_unit_times,
            pool)

        #: Additional information about the nature of the instance can be
        #: stored here. This has no impact on the behavior of the instance,
        #: but it may explain, e.g., settings of an instance generator.
        self.infos: Final[Mapping[str, str]] = _make_infos(infos)

    def __str__(self):
        """
        Get the name of this instance.

        :return: the name of this instance
        """
        return self.name


#: the instance name key
KEY_NAME: Final[str] = "name"
#: the key for the number of products
KEY_N_PRODUCTS: Final[str] = "n_products"
#: the key for the number of customers
KEY_N_CUSTOMERS: Final[str] = "n_customers"
#: the key for the number of machines
KEY_N_MACHINES: Final[str] = "n_machines"
#: the number of demands in the scenario
KEY_N_DEMANDS: Final[str] = "n_demands"
#: the start of a key index
KEY_IDX_START: Final[str] = "["
#: the end of a key index
KEY_IDX_END: Final[str] = "]"
#: the first part of the product route key
KEY_ROUTE: Final[str] = "product_route"
#: the first part of the demand key
KEY_DEMAND: Final[str] = "demand"
#: The amount of products in the warehouse at time step 0.
KEY_IN_WAREHOUSE: Final[str] = "products_in_warehouse_at_t0"
#: the first part of the production time
KEY_PRODUCTION_TIME: Final[str] = "production_time"

#: the key value split string
_KEY_VALUE_SPLIT: Final[str] = str.strip(KEY_VALUE_SEPARATOR)

#: the forbidden keys
__FORBIDDEN_INFO_KEYS: Final[Callable[[str], bool]] = {
    KEY_NAME, KEY_N_PRODUCTS, KEY_N_CUSTOMERS, KEY_N_MACHINES,
    KEY_N_DEMANDS, KEY_ROUTE, KEY_DEMAND, KEY_IN_WAREHOUSE,
    KEY_PRODUCTION_TIME}.__contains__

#: the allowed information key characters
__ALLOWED_INFO_KEY_CHARS: Final[Callable[[str], bool]] = set(
    ascii_letters + digits + "_." + KEY_IDX_START + KEY_IDX_END).__contains__

#: the allowed characters in names
_ALLOWED_NAME_CHARS: Final[Callable[[str], bool]] = set(
    ascii_letters + digits + "_").__contains__


def to_stream(instance: Instance) -> Generator[str, None, None]:
    """
    Convert an instance to a stream of data.

    :param instance: the instance to convert to a stream
    :return: the stream of data
    """
    if not isinstance(instance, Instance):
        raise type_error(instance, "instance", Instance)

    yield f"{COMMENT_START} --- the data of instance {instance.name!r} ---"
    yield COMMENT_START
    yield (f"{COMMENT_START} Lines beginning with {COMMENT_START!r} are "
           f"comments.")
    yield COMMENT_START
    yield f"{COMMENT_START} the unique identifying name of the instance"
    yield f"{KEY_NAME}{KEY_VALUE_SEPARATOR}{instance.name}"
    yield COMMENT_START
    yield f"{COMMENT_START} the number of products in the instance, > 0"
    yield f"{KEY_N_PRODUCTS}{KEY_VALUE_SEPARATOR}{instance.n_products}"
    yield (f"{COMMENT_START} (Valid product indices are in 0.."
           f"{instance.n_products - 1}.)")
    yield COMMENT_START
    yield f"{COMMENT_START} the number of customers in the instance, > 0"
    yield f"{KEY_N_CUSTOMERS}{KEY_VALUE_SEPARATOR}{instance.n_customers}"
    yield (f"{COMMENT_START} (Valid customer indices are in 0.."
           f"{instance.n_customers - 1}.)")
    yield COMMENT_START
    yield f"{COMMENT_START} the number of machines in the instance, > 0"
    yield f"{KEY_N_MACHINES}{KEY_VALUE_SEPARATOR}{instance.n_machines}"
    yield (f"{COMMENT_START} (Valid machine indices are in 0.."
           f"{instance.n_machines - 1}.)")
    yield COMMENT_START
    yield (f"{COMMENT_START} the number of orders (demands) issued by the "
           f"customers, > 0")
    yield f"{KEY_N_DEMANDS}{KEY_VALUE_SEPARATOR}{instance.n_demands}"
    yield (f"{COMMENT_START} (Valid demand/order indices are in 0.."
           f"{instance.n_demands - 1}.)")

    yield COMMENT_START
    yield (f"{COMMENT_START} For each machine, we now specify the indices of "
           f"the machines by which it will be processed, in the order in "
           f"which it will be processed by them.")
    yield (f"{COMMENT_START} {KEY_ROUTE}{KEY_IDX_START}0"
           f"{KEY_IDX_END} is the production route by the first product, "
           "which has index 0.")
    route_0: tuple[int, ...] = instance.routes[0]
    yield (f"{COMMENT_START} This product is processed by "
           f"{tuple.__len__(route_0)} machines, namely first by the "
           f"machine with index {int(route_0[0])} and last by the machine "
           f"with index {int(route_0[-1])}.")
    for p, route in enumerate(instance.routes):
        yield (f"{KEY_ROUTE}{KEY_IDX_START}{p}{KEY_IDX_END}"
               f"{KEY_VALUE_SEPARATOR}"
               f"{CSV_SEPARATOR.join(map(str, route))}")

    yield COMMENT_START
    yield (f"{COMMENT_START} For each customer order/demand, we now "
           f"specify the following values:")
    yield f"{COMMENT_START} the demand ID in square brackets"
    yield f"{COMMENT_START} the ID of the customer who made the order"
    yield f"{COMMENT_START} the ID of the product that the customer ordered"
    yield (f"{COMMENT_START} the amount of the product that the customer"
           "ordered")
    yield f"{COMMENT_START} the release time of the order, > 0"
    yield (f"{COMMENT_START} the deadline, i.e., when the customer expects "
           f"the product, >= release_time")
    srt: list[Demand] = sorted(instance.demands, key=lambda d: d.demand_id)
    fd: Demand = srt[0]
    yield (f"{COMMENT_START} for example, the demand with ID {fd.demand_id} "
           f"was issued by the customer with ID {fd.customer_id} for "
           f"{fd.amount} units of the product with ID "
           f"{fd.product_id}. The order comes into the "
           f"system at time unit {fd.release_time} and the customer expects "
           f"the product to be ready at time unit {fd.demand_id}.")
    for demand in srt:
        it = iter(demand)
        next(it)  # pylint: disable=R1708
        row: str = CSV_SEPARATOR.join(map(str, it))
        yield (f"{KEY_DEMAND}{KEY_IDX_START}{demand.demand_id}{KEY_IDX_END}"
               f"{KEY_VALUE_SEPARATOR}"
               f"{row}")

    yield COMMENT_START
    yield (f"{COMMENT_START} For each product, we now specify the amount "
           f"that is in the warehouse at time step 0.")
    yield (f"{COMMENT_START} For example, there are "
           f"{instance.warehous_at_t0[0]} units of product 0 in the "
           f"warehouse at the beginning of the simulation.")
    yield (f"{KEY_IN_WAREHOUSE}{KEY_VALUE_SEPARATOR}"
           f"{CSV_SEPARATOR.join(map(str, instance.warehous_at_t0))}")

    yield COMMENT_START
    yield (f"{COMMENT_START} For each machine, we now specify the production "
           f"times for each product that passes through the machine.")
    empty_pdx: tuple[int, int] | None = None
    filled_pdx: tuple[int, int, tuple[int, ...]] | None = None
    need: int = 2
    for mid, machine in enumerate(instance.machine_product_unit_times):
        for pid, product in enumerate(machine):
            pdl: int = tuple.__len__(product)
            if (pdl <= 0) and (empty_pdx is None):
                empty_pdx = mid, pid
                need -= 1
                if need <= 0:
                    break
            elif (pdl > 0) and (filled_pdx is None):
                filled_pdx = mid, pid, product
                need -= 1
                if need <= 0:
                    break
        if need <= 0:
            break
    if empty_pdx is not None:
        yield (f"{COMMENT_START} For example, product {empty_pdx[1]} does "
               f"not pass through machine {empty_pdx[0]}, so it is not "
               "listed here.")
    if filled_pdx is not None:
        yield (f"{COMMENT_START} For example, one unit of product "
               f"{filled_pdx[1]} passes through machine {filled_pdx[0]}.")
        yield (f"{COMMENT_START} There, it needs {filled_pdx[2][0]} time "
               f"units per product unit from t=0 to t={filled_pdx[2][1]}.")
        if tuple.__len__(filled_pdx[2]) > 2:
            yield (f"{COMMENT_START} After that, it needs {filled_pdx[2][2]}"
                   " time units per product unit until t="
                   f"{filled_pdx[2][3]}.")
    for mid, machine in enumerate(instance.machine_product_unit_times):
        for pid, product in enumerate(machine):
            if tuple.__len__(product) <= 0:
                continue
            yield (f"{KEY_PRODUCTION_TIME}{KEY_IDX_START}{mid}{CSV_SEPARATOR}"
                   f"{pid}{KEY_IDX_END}{KEY_VALUE_SEPARATOR}"
                   f"{CSV_SEPARATOR.join(map(str, product))}")

    n_infos: Final[int] = len(instance.infos)
    if n_infos > 0:
        yield COMMENT_START
        yield (f"{COMMENT_START} The following {n_infos} key/value pairs "
               f"denote additional information about the instance.")
        yield (f"{COMMENT_START} They have no impact whatsoever on the"
               f"instance behavior.")
        for k, v in instance.infos.items():
            yield f"{k}{KEY_VALUE_SEPARATOR}{v}"


def __get_key_index(full_key: str) -> str:
    """
    Extract the key index from a key.

    :param full_key: the full key
    :return: the key index

    >>> __get_key_index("s[12 ]")
    '12'
    """
    start: int = str.index(full_key, KEY_IDX_START)
    end: int = str.index(full_key, KEY_IDX_END, start)
    if not (0 < start < end < str.__len__(full_key)):
        raise ValueError(f"Invalid key {full_key!r}.")
    idx: str = str.strip(full_key[start + 1:end])
    if str.__len__(idx) <= 0:
        raise ValueError(f"Invalid index in key {full_key!r}.")
    return idx


def __pe(message: str, oline: str, line_idx: int) -> ValueError:
    """
    Create a value error to be raised inside the parser.

    :param message: the message
    :param oline: the original line
    :param line_idx: the line index
    :return: the error
    """
    return ValueError(f"{message} at line {line_idx + 1} ({oline!r})")


def from_stream(stream: Iterable[str]) -> Instance:
    """
    Read an instance from a data stream.

    :param stream: the data stream
    :return: the instance
    """
    if not isinstance(stream, Iterable):
        raise type_error(stream, "stream", Iterable)
    name: str | None = None
    n_products: int | None = None
    n_customers: int | None = None
    n_machines: int | None = None
    n_demands: int | None = None
    routes: list[list[int]] | None = None
    demands: list[list[int]] | None = None
    in_warehouse: list[int] | None = None
    machine_product_times: list[list[list[int]]] | None = None
    infos: dict[str, str] = {}

    for line_idx, oline in enumerate(stream):
        line = str.strip(oline)
        if str.startswith(line, COMMENT_START):
            continue

        split_idx: int = str.find(line, _KEY_VALUE_SPLIT)
        if split_idx > -1:
            key: str = str.lower(str.strip(line[:split_idx]))
            value: str = str.strip(
                line[split_idx + str.__len__(_KEY_VALUE_SPLIT):])
            if (str.__len__(key) <= 0) or (str.__len__(value) <= 0):
                raise __pe(f"Invalid key/value pair {key!r}/{value!r}",
                           oline, line_idx)

            if key == KEY_NAME:
                if name is not None:
                    raise __pe(f"{KEY_NAME} already defined as {name!r}, "
                               f"cannot be set to {value!r}", oline, line_idx)
                name = value

            elif key == KEY_N_MACHINES:
                if n_machines is not None:
                    raise __pe(
                        f"{KEY_N_MACHINES} already defined as {n_machines!r},"
                        f" cannot be set to {value!r}", oline, line_idx)
                n_machines = check_to_int_range(
                    value, KEY_N_MACHINES, 1, 1_000_000)

            elif key == KEY_N_PRODUCTS:
                if n_products is not None:
                    raise __pe(
                        f"{KEY_N_PRODUCTS} already defined as {n_products!r},"
                        f" cannot be set to {value!r}", oline, line_idx)
                n_products = check_to_int_range(
                    value, KEY_N_PRODUCTS, 1, 1_000_000)

            elif key == KEY_N_CUSTOMERS:
                if n_customers is not None:
                    raise __pe(
                        f"{KEY_N_CUSTOMERS} already defined as "
                        f"{n_customers!r}, cannot be set to {value!r}",
                        oline, line_idx)
                n_customers = check_to_int_range(
                    value, KEY_N_CUSTOMERS, 1, 1_000_000)

            elif key == KEY_N_DEMANDS:
                if n_demands is not None:
                    raise __pe(
                        f"{KEY_N_DEMANDS} already defined as {n_demands!r}, "
                        f"cannot be set to {value!r}", oline, line_idx)
                n_demands = check_to_int_range(
                    value, KEY_N_DEMANDS, 1, 1_000_000)

            elif key == KEY_IN_WAREHOUSE:
                if in_warehouse is not None:
                    raise __pe(f"{KEY_IN_WAREHOUSE} already defined",
                               oline, line_idx)
                if n_products is None:
                    raise __pe(f"Must define {KEY_N_PRODUCTS} before "
                               f"{KEY_IN_WAREHOUSE}.", oline, line_idx)
                in_warehouse = list(map(int, str.split(
                    value, CSV_SEPARATOR)))
                if list.__len__(in_warehouse) != n_products:
                    raise __pe(
                        f"Expected {n_products} products in warehouse, got "
                        f"{in_warehouse}.", oline, line_idx)

            elif str.startswith(key, KEY_ROUTE):
                if n_products is None:
                    raise __pe(f"Must define {KEY_N_PRODUCTS} before "
                               f"{KEY_ROUTE}.", oline, line_idx)
                if n_machines is None:
                    raise ValueError(f"Must define {KEY_N_MACHINES} before "
                                     f"{KEY_ROUTE}.", oline, line_idx)
                if routes is None:
                    routes = [[] for _ in range(n_products)]

                product_id: int = check_to_int_range(
                    __get_key_index(key), KEY_ROUTE, 0, n_products - 1)
                rlst: list[int] = routes[product_id]
                if list.__len__(rlst) != 0:
                    raise __pe(
                        f"Already gave {KEY_ROUTE}{KEY_IDX_START}{product_id}"
                        f"{KEY_IDX_END}", oline, line_idx)
                rlst.extend(map(int, str.split(value, CSV_SEPARATOR)))
                if list.__len__(rlst) <= 0:
                    raise __pe(f"Route for product {product_id} is empty",
                               oline, line_idx)

            elif str.startswith(key, KEY_DEMAND):
                if n_customers is None:
                    raise __pe(f"Must define {KEY_N_CUSTOMERS} before "
                               f"{KEY_DEMAND}", oline, line_idx)
                if n_products is None:
                    raise __pe(f"Must define {KEY_N_PRODUCTS} before "
                               f"{KEY_DEMAND}.", oline, line_idx)
                if n_demands is None:
                    raise __pe(f"Must define {KEY_N_DEMANDS} before "
                               f"{KEY_DEMAND}", oline, line_idx)
                if demands is None:
                    demands = [[i] for i in range(n_demands)]

                demand_id: int = check_to_int_range(
                    __get_key_index(key), KEY_DEMAND, 0, n_demands - 1)
                dlst: list[int] = demands[demand_id]
                if list.__len__(dlst) != 1:
                    raise __pe(f"Already gave {KEY_DEMAND}{KEY_IDX_START}"
                               f"{demand_id}{KEY_IDX_END}", oline, line_idx)
                dlst.extend(map(int, str.split(value, CSV_SEPARATOR)))
                if list.__len__(dlst) <= 5:
                    raise __pe(f"Demand {demand_id} is too short: {dlst}",
                               oline, line_idx)

            elif str.startswith(key, KEY_PRODUCTION_TIME):
                if n_products is None:
                    raise __pe(f"Must define {KEY_N_PRODUCTS} before "
                               f"{KEY_PRODUCTION_TIME}", oline, line_idx)
                if n_machines is None:
                    raise __pe(f"Must define {KEY_N_MACHINES} before"
                               f" {KEY_PRODUCTION_TIME}", oline, line_idx)
                machine, product = str.split(
                    __get_key_index(key), CSV_SEPARATOR)
                machine_id: int = check_to_int_range(
                    machine, "machine", 0, n_machines - 1)
                product_id = check_to_int_range(
                    product, "product", 0, n_products - 1)

                if machine_product_times is None:
                    machine_product_times = \
                        [[[] for _ in range(n_products)]
                         for __ in range(n_machines)]

                mpd: list[int] = machine_product_times[
                    machine_id][product_id]
                if list.__len__(mpd) > 0:
                    raise __pe(f"Already gave {KEY_PRODUCTION_TIME}"
                               f"{KEY_IDX_START}{machine_id}{CSV_SEPARATOR}"
                               f"{product_id}{KEY_IDX_END}", oline, line_idx)
                mpd.extend(
                    map(int, str.split(value, CSV_SEPARATOR)))
            else:
                infos[key] = value

    if name is None:
        raise ValueError(f"Did not specify instance name ({KEY_NAME}).")
    if n_products is None:
        raise ValueError("Did not specify instance n_products"
                         f" ({KEY_N_PRODUCTS}).")
    if n_customers is None:
        raise ValueError("Did not specify instance n_customers"
                         f" ({KEY_N_CUSTOMERS}).")
    if n_machines is None:
        raise ValueError("Did not specify instance n_machines"
                         f" ({KEY_N_MACHINES}).")
    if n_demands is None:
        raise ValueError("Did not specify instance n_demands"
                         f" ({KEY_N_DEMANDS}).")
    if routes is None:
        raise ValueError(f"Did not specify instance routes ({KEY_ROUTE}).")
    if demands is None:
        raise ValueError(f"Did not specify instance demands ({KEY_DEMAND}).")
    if in_warehouse is None:
        raise ValueError("Did not specify instance warehouse values"
                         f" ({KEY_IN_WAREHOUSE}).")
    if machine_product_times is None:
        raise ValueError("Did not specify per-machine product production"
                         f"times ({KEY_PRODUCTION_TIME}).")

    return Instance(name, n_products, n_customers, n_machines, n_demands,
                    routes, demands, in_warehouse, machine_product_times,
                    infos)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __round_time(value: int | float) -> int:
    """
    Round a production time.

    Production times are always rounded up (ceil), except if they are very
    very close to the nearest integer. "Very very close" here means that,
    if a predicted time differs only by 1/16834th (= 2**⁻14) from the
    next-smaller integer, it is rounded down. Production times can never be
    zero.

    If the original value be `v` and the result of this function is `r`, then
    it is guaranteed that `0 < r` and that `floor(v) <= r <= `floor(v) + 1`
    for any `v > 0`. For `v <= 1`, `r = 1`.

    :param value: the value `v`
    :return: the rounded value `r`.

    >>> __round_time(10.0)
    10
    >>> __round_time(10)
    10
    >>> __round_time(10.000001)
    10
    >>> __round_time(10.00006103515625)
    11
    >>> __round_time(10.0000610351562)
    10
    >>> __round_time(10.02)
    11
    >>> __round_time(10.9999)
    11
    >>> __round_time(0)
    1
    >>> __round_time(0.0)
    1
    >>> __round_time(0.00006103515625)
    1
    >>> __round_time(0.0000610351562)
    1
    >>> __round_time(0.9)
    1
    >>> __round_time(1.01)
    2
    """
    fl: int = int(value)
    if fl < 1:
        return 1
    if (fl >= value) or ((value - fl) < 0.00006103515625):
        return fl
    return fl + 1


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def compute_finish_time(start_time: int, amount: int,
                        production_times: tuple[int, ...]) -> int:
    """
    Compute the time when one job is finished.

    The production times are cyclic intervals of unit production times and
    interval ends.

    :param start_time: the starting time of the job
    :param amount: the number of units to be produced
    :param production_times: the production times array
    :return: the end time


    Here, the production time is 10 time units / 1 product unit, valid until
    end time 100.
    >>> compute_finish_time(0, 1, (10, 100))
    10

    Here, the production time is 10 time units / 1 product unit, valid until
    end time 100. We begin producing at time unit 250. Since the production
    periods are cyclic, this is OK: we would be halfway through the third
    production period when the request comes in. It will consume 10 time units
    and be done at time unit 260.
    >>> compute_finish_time(250, 1, (10, 100))
    260

    Here, the end time of the production time validity is at time unit 100.
    However, we begin producing 1 product unit at time step 90. This unit will
    use 10 time units, meaning that its production is exactly finished when
    the production time validity ends.
    It will be finished at time step 100.
    >>> compute_finish_time(90, 1, (10, 100))
    100

    Here, the end time of the production time validity is at time unit 100.
    However, we begin producing 1 product unit at time step 95. This unit would
    use 10 time units. It cannot be completed until the end of the period. So
    only 0.5 units are produced, needing 5 time units and finishing at period
    100. Then the production time window begins again. Since there is only 1
    production time, in the second production cycle there will still be 10
    time units per production unit. The remaining 0.5 units will be produced
    in 5 time units, meaning that the overall production is finished at time
    step 105.
    >>> compute_finish_time(95, 1, (10, 100))
    105

    Now we have two production periods. The production begins again at time
    step 95. 0.5 units are finished until the production period ends at time
    step 100. Then, the second production period begins, requiring 20 time
    units per product unit. We thus need 10 time units to finish the remaining
    0.5 product unit, completing the job at time unit 110.
    >>> compute_finish_time(95, 1, (10, 100, 20, 200))
    110

    Now things get more complex. We want to do 10 units of product.
    We can finish 0.5 units until the first period ends after 5 time steps.
    Then, in the second period, we can do exactly 2 units of product until
    the period ends at time 140. This leaves 7.5 units of product to be done
    in the third period, where each unit needs 50 time units. Thus, we end
    up needing 5 + 40 + 7.5*50 time units, which add to the starting time.
    >>> compute_finish_time(95, 10, (10, 100, 20, 140, 50, 5000))
    515
    >>> 95 + (0.5*10 + 2*20 + 7.5*50)
    515.0

    This is the same as the example before, except that the third period now
    ends at time step 200. Thus, we can only do 1.2=(200-140)/50 units of
    product in the 200 - 140 time units this period lasts. Then the production
    cycle wraps over. Luckily, we can complete the remaining 6.3 units of
    product in the first period, where each product unit needs only 10 time
    units.
    >>> compute_finish_time(95, 10, (10, 100, 20, 140, 50, 200))
    263
    >>> 95 + (0.5*10 + 2*20 + 1.2*50 + 6.3*10)
    263.0

    Here we illustrate that production times are rounded to the nearest larger
    integer. If we encounter a situation such that the production would need
    246.6666 time units, we return 247. The only exception for this "rounding
    up" strategy is if the result is very very close to the nearest integer.
    For example, 10.00006103515625 would be rounded to 11, whereas
    10.0000610351562 will be rounded to 10. (0.00006103515625 = 1/16384
    = 2**⁻14).
    >>> compute_finish_time(95, 10, (10, 100, 20, 140, 50, 200, 3, 207))
    247
    >>> 95 + (0.5*10 + 2*20 + 1.2*50 + (7/3)*3 + (6.3-(7/3))*10)
    246.66666666666666
    """
    time_mod: Final[int] = production_times[-1]
    low_end: Final[int] = len(production_times)
    total: Final[int] = low_end // 2

    # First, we need to find the segment in the production cycle
    # where the production begins. We use a binary search for that.
    remaining: int | float = amount
    seg_start: int = start_time % time_mod
    low: int = 0
    high: int = total
    while low < high:
        mid: int = ((low + high) // 2)
        th: int = production_times[mid * 2 + 1]
        if th <= seg_start:
            low = mid + 1
        else:
            high = mid - 1
    low *= 2
    max_time: int = production_times[low + 1]
    if max_time <= seg_start:
        low += 2

    # Now we can cycle through the production cycle until the product has
    # been produced.
    while True:
        max_time = production_times[low + 1]
        unit_time: int = production_times[low]
        seg_end = seg_start + (unit_time * remaining)
        if seg_end <= max_time:
            return __round_time(start_time + seg_end - seg_start)
        duration = max_time - seg_start
        start_time += duration
        remaining -= (duration / unit_time)
        if remaining <= 0:
            return __round_time(start_time)
        low += 2
        if low >= low_end:
            low = 0
            seg_start = 0
            continue
        seg_start = max_time
