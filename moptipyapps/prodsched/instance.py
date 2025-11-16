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

The number of stations be 4.
>>> n_stations = 4

There will be 6 customer demands.
>>> n_demands = 6

Each product may take a different route through different stations.
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

Each station requires a certain working time for each unit of each product.
This production time may vary over time.
For example, maybe station 0 needs 10 time units for 1 unit of product 0 from
time step 0 to time step 19, then 11 time units from time step 20 to 39, then
8 time units from time step 40 to 59.
These times are cyclic, meaning that at time step 60 to 79, it will again need
10 time units, and so on.
Of course, production times are only specified for stations that a product is
actually routed through.
>>> m0_p0 = [10.0, 20.0, 11.0, 40.0,  8.0, 60.0]
>>> m0_p1 = [12.0, 20.0,  7.0, 40.0, 11.0, 70.0]
>>> m0_p2 = []
>>> m1_p0 = []
>>> m1_p1 = [20.0, 50.0, 30.0, 120.0,  7.0, 200.0]
>>> m1_p2 = [21.0, 50.0, 29.0, 130.0,  8.0, 190.0]
>>> m2_p0 = [ 8.0, 20.0,  9.0, 60.0]
>>> m2_p1 = [10.0, 90.0]
>>> m2_p2 = [12.0, 70.0,  30.0, 120.0]
>>> m3_p0 = [70.0, 200.0,  3.0, 220.0]
>>> m3_p1 = [60.0, 220.0,  5.0, 260.0]
>>> m3_p2 = [30.0, 210.0, 10.0, 300.0]
>>> station_product_unit_times = [[m0_p0, m0_p1, m0_p2],
...                               [m1_p0, m1_p1, m1_p2],
...                               [m2_p0, m2_p1, m2_p2],
...                               [m3_p0, m3_p1, m3_p2]]

We can (but do not need to) provide additional information as key-value pairs.
>>> infos = {"source": "manually created",
...          "creation_date": "2025-11-09"}

From all of this data, we can create the instance.
>>> instance = Instance(name, n_products, n_customers, n_stations, n_demands,
...                     routes, demands, warehous_at_t0,
...                     station_product_unit_times, infos)
>>> instance.name
'my_instance'

>>> instance.n_customers
5

>>> instance.n_stations
4

>>> instance.n_demands
6

>>> instance.n_products
3

>>> instance.routes
((0, 3, 2), (0, 2, 1, 3), (1, 2, 3))

>>> instance.demands
(Demand(arrival=1240.0, deadline=3000.0, demand_id=0, customer_id=0, \
product_id=1, amount=20), Demand(arrival=2300.0, deadline=4000.0, \
demand_id=1, customer_id=1, product_id=0, amount=10), \
Demand(arrival=4234.0, deadline=27080.0, demand_id=5, customer_id=3, \
product_id=0, amount=19), Demand(arrival=5410.0, deadline=16720.0, \
demand_id=4, customer_id=4, product_id=2, amount=23), \
Demand(arrival=7300.0, deadline=9000.0, demand_id=3, customer_id=3, \
product_id=1, amount=12), Demand(arrival=8300.0, deadline=11000.0, \
demand_id=2, customer_id=2, product_id=2, amount=7))

>>> instance.warehous_at_t0
(10, 0, 6)

>>> instance.station_product_unit_times
((array([10., 20., 11., 40.,  8., 60.]), \
array([12., 20.,  7., 40., 11., 70.]), array([], dtype=float64)), (\
array([], dtype=float64), array([ 20.,  50.,  30., 120.,   7., 200.]), \
array([ 21.,  50.,  29., 130.,   8., 190.])), (array([ 8., 20.,  9., 60.]), \
array([10., 90.]), array([ 12.,  70.,  30., 120.])), (\
array([ 70., 200.,   3., 220.]), array([ 60., 220.,   5., 260.]), array(\
[ 30., 210.,  10., 300.])))

>>> dict(instance.infos)
{'source': 'manually created', 'creation_date': '2025-11-09'}

We can serialize instances to a stream of strings and also load them back
from a stream of strings.
Here, we store `instance` to a stream.
We then load the independent instance `i2` from that stream.
>>> i2 = from_stream(to_stream(instance))
>>> i2 is instance
False
>>> i2 == instance
True

You can see that the loaded instance has the same data as the stored one.
>>> i2.name == instance.name
True
>>> i2.n_customers == instance.n_customers
True
>>> i2.n_stations == instance.n_stations
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
>>> eq: bool = True
>>> for i in range(i2.n_stations):
...     ma1 = i2.station_product_unit_times[i]
...     ma2 = instance.station_product_unit_times[i]
...     for j in range(i2.n_products):
...         pr1 = ma1[j]
...         pr2 = ma2[j]
...         if not np.array_equal(pr1, pr2):
...             eq = False
>>> eq
True

True
>>> i2.infos == instance.infos
True
"""

from dataclasses import dataclass
from itertools import batched
from math import isfinite
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
import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import (
    COMMENT_START,
    KEY_VALUE_SEPARATOR,
)
from moptipy.utils.strings import sanitize_name
from pycommons.ds.immutable_map import immutable_mapping
from pycommons.io.csv import CSV_SEPARATOR
from pycommons.math.int_math import try_int
from pycommons.types import check_int_range, check_to_int_range, type_error

#: The maximum for the number of stations, products, or customers.
MAX_ID: Final[int] = 1_000_000_000

#: No value bigger than this is permitted in any tuple anywhere.
MAX_VALUE: Final[int] = 2_147_483_647

#: the index of the demand ID
DEMAND_ID: Final[int] = 0
#: the index of the customer ID
DEMAND_CUSTOMER: Final[int] = 1
#: the index of the product ID
DEMAND_PRODUCT: Final[int] = 2
#: the index of the demanded amount
DEMAND_AMOUNT: Final[int] = 3
#: the index of the demand release time
DEMAND_ARRIVAL: Final[int] = 4
#: the index of the demand deadline
DEMAND_DEADLINE: Final[int] = 5


@dataclass(order=True, frozen=True)
class Demand(Iterable[int | float]):
    """
    The record for demands.

    >>> Demand(arrival=0.6, deadline=0.8, demand_id=1,
    ...        customer_id=2, product_id=6, amount=12)
    Demand(arrival=0.6, deadline=0.8, demand_id=1, customer_id=2,\
 product_id=6, amount=12)
    >>> Demand(arrival=16, deadline=28, demand_id=1,
    ...        customer_id=2, product_id=6, amount=12)
    Demand(arrival=16.0, deadline=28.0, demand_id=1, customer_id=2,\
 product_id=6, amount=12)
    """

    #: the arrival time, i.e., when the demand enters the system
    arrival: float
    #: the deadline, i.e., when the customer expects the result
    deadline: float
    #: the ID of the demand
    demand_id: int
    #: the customer ID
    customer_id: int
    #: the ID of the product
    product_id: int
    #: the amount
    amount: int

    def __init__(self, arrival: int | float,
                 deadline: int | float, demand_id: int,
                 customer_id: int, product_id: int, amount: int) -> None:
        """
        Initialize the record.

        :param arrival: the arrival time
        :param deadline: the deadline
        :param demand_id: the demand id
        :param customer_id: the customer id
        :param product_id: the product id
        :param amount: the amount
        """
        if isinstance(arrival, int):
            t: float = float(arrival)
            if t != arrival:
                raise ValueError(f"invalid arrival time {arrival}")
            arrival = t
        if not isinstance(arrival, float):
            raise type_error(arrival, "arrival", float)
        if not (isfinite(arrival) and (
                0 < arrival < MAX_VALUE)):
            raise ValueError(f"invalid arrival={arrival}")

        if isinstance(deadline, int):
            t = float(deadline)
            if t != deadline:
                raise ValueError(f"invalid deadline time {deadline}")
            deadline = t
        if not isinstance(deadline, float):
            raise type_error(deadline, "deadline", float)
        if not (isfinite(deadline) and (0 < deadline < MAX_VALUE)):
            raise ValueError(f"invalid deadline={deadline}")

        if deadline < arrival:
            raise ValueError(
                f"arrival={arrival} and deadline={deadline}")
        object.__setattr__(self, "arrival", arrival)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "demand_id", check_int_range(
            demand_id, "demand_id", 0, MAX_ID))
        object.__setattr__(self, "customer_id", check_int_range(
            customer_id, "customer_id", 0, MAX_ID))
        object.__setattr__(self, "product_id", check_int_range(
            product_id, "product_id", 0, MAX_ID))
        object.__setattr__(self, "amount", check_int_range(
            amount, "amount", 1, MAX_ID))

    def __getitem__(self, item: int) -> int | float:
        """
        Access an element of this demand via an index.

        :param item: the index
        :return: the demand value at that index

        >>> d = Demand(arrival=16, deadline=28, demand_id=1,
        ...        customer_id=2, product_id=6, amount=12)
        >>> d[0]
        1
        >>> d[1]
        2
        >>> d[2]
        6
        >>> d[3]
        12
        >>> d[4]
        16
        >>> d[5]
        28
        """
        if item == DEMAND_ID:
            return self.demand_id
        if item == DEMAND_CUSTOMER:
            return self.customer_id
        if item == DEMAND_PRODUCT:
            return self.product_id
        if item == DEMAND_AMOUNT:
            return self.amount
        if item == DEMAND_ARRIVAL:
            return try_int(self.arrival)
        if item == DEMAND_DEADLINE:
            return try_int(self.deadline)
        raise IndexError(
            f"index {item} out of bounds [0,{DEMAND_DEADLINE}].")

    def __iter__(self) -> Iterator[int | float]:
        """
        Iterate over the values in this demand.

        :return: the demand iterable

        >>> d = Demand(arrival=16, deadline=28, demand_id=1,
        ...        customer_id=2, product_id=6, amount=12)
        >>> list(d)
        [1, 2, 6, 12, 16, 28]
        """
        yield self.demand_id  # DEMAND_ID
        yield self.customer_id  # DEMAND_CUSTOMER
        yield self.product_id  # DEMAND_PRODUCT:
        yield self.amount  # DEMAND_AMOUNT:
        yield try_int(self.arrival)  # DEMAND_ARRIVAL
        yield try_int(self.deadline)  # DEMAND_DEADLINE

    def __len__(self) -> int:
        """
        Get the length of the demand record.

        :returns `6`: always

        >>> len(Demand(arrival=16, deadline=28, demand_id=1,
        ...        customer_id=2, product_id=6, amount=12))
        6
        """
        return 6


def __to_tuple(source: Iterable[int | float],
               pool: dict, empty_ok: bool = False,
               type_var: type = int) -> tuple:
    """
    Convert an iterable of type integer to a tuple.

    :param source: the data source
    :param pool: the tuple pool
    :param empty_ok: are empty tuples OK?
    :param type_var: the type variable
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

    >>> k3 = __to_tuple([3.4, 2.3, 3.1], ppl, type_var=float)
    >>> print(k3)
    (3.4, 2.3, 3.1)
    >>> k1 is k3
    False

    >>> k4 = __to_tuple([3.4, 2.3, 3.1], ppl, type_var=float)
    >>> print(k4)
    (3.4, 2.3, 3.1)
    >>> k3 is k4
    True

    >>> try:
    ...     __to_tuple([], ppl)
    ... except Exception as e:
    ...     print(e)
    row has length 0.

    >>> __to_tuple([], ppl, empty_ok=True)
    ()

    >>> try:
    ...     __to_tuple([1, 2.0], ppl)
    ... except Exception as e:
    ...     print(e)
    row[1] should be an instance of int but is float, namely 2.0.

    >>> try:
    ...     __to_tuple([1.1, 2.0, 4, 3.4], ppl, type_var=float)
    ... except Exception as e:
    ...     print(e)
    row[2] should be an instance of float but is int, namely 4.
    """
    use_row = source if isinstance(source, tuple) else tuple(source)
    if (tuple.__len__(use_row) <= 0) and (not empty_ok):
        raise ValueError("row has length 0.")
    for j, v in enumerate(use_row):
        if not isinstance(v, type_var):
            raise type_error(v, f"row[{j}]", type_var)
        if not (isfinite(v) and (0 <= v <= MAX_VALUE)):  # type: ignore
            raise ValueError(f"row[{j}]={v} not in 0..{MAX_VALUE}")

    key: Final[str] = repr(use_row)
    if key in pool:
        return pool[key]
    pool[key] = use_row
    return use_row


def __to_npfloats(source: Iterable[int | float],  # pylint: disable=W1113
                  pool: dict, empty_ok: bool = False,
                  *_) -> np.ndarray:  # pylint: disable=W1113
    """
    Convert to numpy floats.

    :param source: the source data
    :param pool: the pool
    :param empty_ok: are empty arrays OK?
    :return: the arrays

    >>> ppl = {}
    >>> a = __to_npfloats([3.4, 2.3, 3.1], ppl)
    >>> a
    array([3.4, 2.3, 3.1])
    >>> b = __to_npfloats([3.4, 2.3, 3.1], ppl)
    >>> b is a
    True

    >>> c = __to_npfloats([], ppl, empty_ok=True)
    >>> c
    array([], dtype=float64)

    >>> d = __to_npfloats([], ppl, empty_ok=True)
    >>> d is c
    True
    """
    res: np.ndarray = np.array(__to_tuple(
        source, pool, empty_ok, float), np.float64)
    key: Final[str] = repr(res)
    if key in pool:
        return pool[key]
    pool[key] = res
    return res


def __to_nested_tuples(source: Iterable,
                       pool: dict, empty_ok: bool = False,
                       type_var: type = int,
                       inner: Callable = __to_tuple) -> tuple:
    """
    Turn nested iterables of ints into nested tuples.

    :param source: the source list
    :param pool: the tuple pool
    :param empty_ok: are empty tuples OK?
    :param type_var: the type variable
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

    >>> __to_nested_tuples([(1.0, 2.4), (1.0, 2.2)], ppl, True, float)
    ((1.0, 2.4), (1.0, 2.2))
    """
    if not isinstance(source, Iterable):
        raise type_error(source, "source", Iterable)
    dest: list = []
    ins: int = 0
    for row in source:
        use_row = inner(row, pool, empty_ok, type_var)
        ins += len(use_row)
        dest.append(use_row)

    if (ins <= 0) and empty_ok:  # if all inner tuples are empty,
        dest.clear()             # clear the tuple source

    n_rows: Final[int] = list.__len__(dest)
    if (n_rows <= 0) and (not empty_ok):
        raise ValueError("Got empty set of rows!")

    ret_val: tuple = tuple(dest)
    key: Final[str] = repr(ret_val)
    if key in pool:
        return pool[key]
    pool[key] = ret_val
    return ret_val


def __to_tuples(source: Iterable[Iterable],
                pool: dict, empty_ok: bool = False, type_var=int,
                inner: Callable = __to_tuple) \
        -> tuple[tuple, ...]:
    """
    Turn 2D nested iterables into 2D nested tuples.

    :param source: the source
    :param pool: the tuple pool
    :param empty_ok: are empty tuples OK?
    :param type_var: the type variable
    :param inner: the inner callable
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
    return __to_nested_tuples(source, pool, empty_ok, type_var, inner)


def __to_2d_npfloat(source: Iterable[Iterable],  # pylint: disable=W1113
                    pool: dict, empty_ok: bool = False,
                    *_) -> tuple[np.ndarray, ...]:  # pylint: disable=W1113
    """
    Turn 2D nested iterables into 2D nested tuples.

    :param source: the source
    :param pool: the tuple pool
    :param empty_ok: are empty tuples OK?
    :param inner: the inner callable
    :return: the nested tuples

    >>> ppl = {}
    >>> k2 = __to_2d_npfloat([(1.0, 2.0), (1.0, 2.0, 4.0),  (1.0, 0.2)], ppl)
    >>> print(k2)
    (array([1., 2.]), array([1., 2., 4.]), array([1. , 0.2]))
    """
    return __to_nested_tuples(source, pool, empty_ok, float, __to_npfloats)


def __to_3d_npfloat(source: Iterable[Iterable[Iterable]],
                    pool: dict, empty_ok: bool) \
        -> tuple[tuple[np.ndarray, ...], ...]:
    """
    Turn 3D nested iterables into 3D nested tuples.

    :param source: the source
    :param pool: the tuple pool
    :param empty_ok: are empty tuples OK?
    :return: the nested tuples

    >>> ppl = {}
    >>> k1 = __to_3d_npfloat([[[3.0, 2.0], [44.0, 5.0], [2.0]],
    ...                        [[2.0], [5.0, 7.0]]], ppl, False)
    >>> print(k1)
    ((array([3., 2.]), array([44.,  5.]), array([2.])), \
(array([2.]), array([5., 7.])))
    >>> k1[0][2] is k1[1][0]
    True
    """
    return __to_nested_tuples(source, pool, empty_ok, float, __to_2d_npfloat)


def _make_routes(
        n_products: int, n_stations: int,
        source: Iterable[Iterable[int]],
        pool: dict) -> tuple[tuple[int, ...], ...]:
    """
    Create the routes through stations for the products.

    Each product passes through a set of stations. It can pass through each
    station at most once. It can only pass through valid stations.

    :param n_products: the number of products
    :param n_stations: the number of stations
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
    check_int_range(n_products, "n_products", 1, MAX_ID)
    check_int_range(n_stations, "n_stations", 1, MAX_ID)
    dest: tuple[tuple[int, ...], ...] = __to_tuples(source, pool)

    n_rows: Final[int] = tuple.__len__(dest)
    if n_rows != n_products:
        raise ValueError(f"{n_products} products, but {n_rows} routes.")
    for i, route in enumerate(dest):
        stations: int = tuple.__len__(route)
        if stations <= 0:
            raise ValueError(
                f"len(row[{i}])={stations} but n_stations={n_stations}")
        for j, v in enumerate(route):
            if not (0 <= v < n_stations):
                raise ValueError(
                    f"row[{i},{j}]={v}, but n_stations={n_stations}")
    return dest


def __to_demand(
        source: Iterable[int | float],
        pool: dict, *_) -> Demand:
    """
    Convert an integer source to a tuple or a demand.

    :param source: the source
    :param pool: the pool
    :return: the Demand

    >>> ppl = {}
    >>> d1 = __to_demand([1, 2, 3, 20, 10, 100], ppl)
    >>> d1
    Demand(arrival=10.0, deadline=100.0, demand_id=1, \
customer_id=2, product_id=3, amount=20)
    >>> d2 = __to_demand([1, 2, 3, 20, 10, 100], ppl)
    >>> d1 is d2
    True
    """
    if isinstance(source, Demand):
        return cast("Demand", source)
    tup: tuple[int | float, ...] = tuple(source)
    dl: int = tuple.__len__(tup)
    if dl != 6:
        raise ValueError(f"Expected 6 values, got {dl}.")
    ret: Demand = Demand(
        demand_id=cast("int", tup[DEMAND_ID]),
        customer_id=cast("int", tup[DEMAND_CUSTOMER]),
        product_id=cast("int", tup[DEMAND_PRODUCT]),
        amount=cast("int", tup[DEMAND_AMOUNT]),
        arrival=tup[DEMAND_ARRIVAL], deadline=tup[DEMAND_DEADLINE])

    key: Final[str] = repr(ret)
    if key in pool:
        return pool[key]
    pool[key] = ret
    return ret


def _make_demands(n_products: int, n_customers: int, n_demands: int,
                  source: Iterable[Iterable[int | float]], pool: dict) \
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
    (Demand(arrival=5.0, deadline=21.0, demand_id=3, customer_id=7,\
 product_id=3, amount=23), Demand(arrival=17.0, deadline=21.0, demand_id=1,\
 customer_id=6, product_id=7, amount=12), Demand(arrival=17.0, deadline=27.0,\
 demand_id=2, customer_id=5, product_id=2, amount=6), Demand(arrival=20.0,\
 deadline=21.0, demand_id=0, customer_id=2, product_id=1, amount=4))
    """
    check_int_range(n_products, "n_products", 1, MAX_ID)
    check_int_range(n_customers, "n_customers", 1, MAX_ID)
    check_int_range(n_demands, "n_demands", 1, MAX_ID)

    temp: tuple[Demand, ...] = __to_nested_tuples(
        source, pool, False, inner=__to_demand)
    n_dem: int = tuple.__len__(temp)
    if n_dem != n_demands:
        raise ValueError(f"Expected {n_demands} demands, got {n_dem}?")

    used_ids: set[int] = set()
    min_id: int = 1000 * MAX_ID
    max_id: int = -1000 * MAX_ID
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
        if not (0 < amount < MAX_ID):
            raise ValueError(f"demand[{i}].amount = {amount}.")

        arrival: float = demand.arrival
        if not (isfinite(arrival) and (0 < arrival < MAX_ID)):
            raise ValueError(f"demand[{i}].arrival = {arrival}.")

        deadline: float = demand.deadline
        if not (isfinite(deadline) and arrival <= deadline < MAX_ID):
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
        if not (0 <= v <= MAX_ID):
            raise ValueError(f"Got {v} units of product {p} in warehouse?")
    return ret


def _make_station_product_unit_times(
        n_products: int, n_stations: int,
        routes: tuple[tuple[float, ...], ...],
        source: Iterable[Iterable[Iterable[float]]],
        pool: dict) -> tuple[tuple[np.ndarray, ...], ...]:
    """
    Create the structure for the work times per product unit per station.

    Here we have for each station, for each product, a sequence of per-unit
    production settings. Each such "production settings" is a tuple with a
    per-unit production time and an end time index until which it is valid.
    Production times cycle, so if we produce something after the last end
    time index, we begin again at production time index 0.

    :param n_products: the number of products
    :param n_stations: the number of stations
    :param routes: the routes of the products through the stations
    :param source: the source array
    :param pool: the tuple pool
    :return: the station unit times

    >>> ppl = {}
    >>> rts = _make_routes(3, 2, [[0, 1], [0], [1, 0]], ppl)
    >>> print(rts)
    ((0, 1), (0,), (1, 0))

    >>> mpt1 = _make_station_product_unit_times(3, 2, rts, [
    ...     [[1.0, 2.0, 3.0, 5.0], [1.0, 2.0, 3.0, 5.0],
    ...      [1.0, 10.0, 2.0, 30.0]],
    ...     [[2.0, 20.0, 3.0, 40.0], [], [4.0, 56.0, 34.0, 444.0]]], ppl)
    >>> print(mpt1)
    ((array([1., 2., 3., 5.]), array([1., 2., 3., 5.]), \
array([ 1., 10.,  2., 30.])), (array([ 2., 20.,  3., 40.]), \
array([], dtype=float64), array([  4.,  56.,  34., 444.])))
    >>> mpt1[0][0] is mpt1[0][1]
    True

    >>> mpt2 = _make_station_product_unit_times(3, 2, rts, [
    ...     [[1.0, 2.0, 3.0, 5.0], [1.0, 2.0, 3.0, 5.0],
    ...      [1.0, 10.0, 2.0, 30.0]],
    ...     [[2.0, 20.0, 3.0, 40.0], [], [4.0, 56.0, 34.0, 444.0]]], ppl)
    >>> print(mpt2)
    ((array([1., 2., 3., 5.]), array([1., 2., 3., 5.]), \
array([ 1., 10.,  2., 30.])), (array([ 2., 20.,  3., 40.]), \
array([], dtype=float64), array([  4.,  56.,  34., 444.])))
    >>> mpt1 is mpt2
    True
    """
    ret: tuple[tuple[np.ndarray, ...], ...] = __to_3d_npfloat(
        source, pool, True)

    if tuple.__len__(routes) != n_products:
        raise ValueError("invalid routes!")

    d1: int = tuple.__len__(ret)
    if d1 != n_stations:
        raise ValueError(
            f"Got {d1} station-times, but {n_stations} stations.")
    for mid, station in enumerate(ret):
        d2: int = tuple.__len__(station)
        if d2 <= 0:
            for pid, r in enumerate(routes):
                if mid in r:
                    raise ValueError(
                        f"Station {mid} in route for product {pid}, "
                        "but has no production time")
            continue
        if d2 != n_products:
            raise ValueError(f"got {d2} products for station {mid}, "
                             f"but have {n_products} products")
        for pid, product in enumerate(station):
            needs_times: bool = mid in routes[pid]
            d3: int = np.ndarray.__len__(product)
            if (not needs_times) and (d3 > 0):
                raise ValueError(
                    f"product {pid} does not pass through station {mid}, "
                    "so there must not be production times!")
            if needs_times and (d3 <= 0):
                raise ValueError(
                    f"product {pid} does pass through station {mid}, "
                    "so there must be production times!")
            if (d3 % 2) != 0:
                raise ValueError(
                    f"production times for {pid} does pass through station "
                    f"{mid}, must be of even length, but got length {d3}.")
            last_end = 0
            for pt, time in enumerate(batched(product, 2)):
                if tuple.__len__(time) != 2:
                    raise ValueError(f"production times must be 2-tuples, "
                                     f"but got {time} for product {pid} on "
                                     f"station {mid} at position {pt}")
                unit_time, end = time
                if not ((unit_time > 0) and (last_end < end < MAX_ID)):
                    raise ValueError(
                        f"Invalid unit time {unit_time} and end time "
                        f"{end} for product {pid} on station {mid}")
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
            n_products: int, n_customers: int, n_stations: int,
            n_demands: int,
            routes: Iterable[Iterable[int]],
            demands: Iterable[Iterable[int | float]],
            warehous_at_t0: Iterable[int],
            station_product_unit_times: Iterable[Iterable[Iterable[float]]],
            infos: Iterable[tuple[str, str]] | Mapping[
                str, str] | None = None) \
            -> None:
        """
        Create an instance of the production scheduling time.

        :param name: the instance name
        :param n_products: the number of products
        :param n_customers: the number of customers
        :param n_stations: the number of stations
        :param n_demands: the number of demand records
        :param routes: for each product, the sequence of stations that it has
            to pass
        :param demands: a sequences of demands of the form (
            customer_id, product_id, product_amount, release_time) OR a
            sequence of :class:`Demand` records.
        :param warehous_at_t0: the amount of products in the warehouse at time
            0 for each product
        :param station_product_unit_times: for each station and each product
            the per-unit-production time schedule, in the form of
            "per_unit_time, duration", where duration is the number of time
            units for which the per_unit_time is value
        :param station_product_unit_times: the cycling unit times for each
            product on each station, each with a validity duration
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
            n_products, "n_products", 1, MAX_ID)
        #: the number of customers in the scenario
        self.n_customers: Final[int] = check_int_range(
            n_customers, "n_customers", 1, MAX_ID)
        #: the number of stations or workstations in the scenario
        self.n_stations: Final[int] = check_int_range(
            n_stations, "n_stations", 1, MAX_ID)
        #: the number of demands in the scenario
        self.n_demands: Final[int] = check_int_range(
            n_demands, "n_demands", 1, MAX_ID)

        pool: dict = {}  # the pool for resolving tuples

        #: the product routes, i.e., the stations through which each product
        #: must pass
        self.routes: Final[tuple[tuple[int, ...], ...]] = _make_routes(
            n_products, n_stations, routes, pool)
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

        #: The per-station unit production times for each product.
        #: Each station can have different production times per product.
        #: Let's say that this is tuple `A`.
        #: For each product, it has a tuple `B` at the index of the product
        #: id.
        #: If the product does not pass through the station, `B` is empty.
        #: Otherwise, it holds one or multiple tuples `C`.
        #: Each tuple `C` consists of two numbers:
        #: A per-unit-production time for the product.
        #: An end time index for this production time.
        #: Once the real time surpasses the end time of the last of these
        #: production specs, the production specs are recycled and begin
        #: again.
        self.station_product_unit_times: Final[tuple[tuple[
            np.ndarray, ...], ...]] = _make_station_product_unit_times(
            n_products, n_stations, self.routes, station_product_unit_times,
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

    def _tuple(self) -> tuple:
        """
        Convert this object to a tuple.

        :return: the tuple

        >>> Instance(name="test1", n_products=1, n_customers=1, n_stations=2,
        ...         n_demands=1, routes=[[0, 1]],
        ...         demands=[[0, 0, 0, 10, 20, 100]],
        ...         warehous_at_t0=[0],
        ...     station_product_unit_times=[[[10.0, 10000.0]],
        ...                                 [[30.0, 10000.0]]])._tuple()
        ('test1', 2, 1, 1, 1, (Demand(arrival=20.0, deadline=100.0,\
 demand_id=0, customer_id=0, product_id=0, amount=10),), ((0, 1),),\
 (0,), (), ((10.0, 10000.0), (30.0, 10000.0)))
        """
        return (self.name, self.n_stations, self.n_products,
                self.n_demands, self.n_customers, self.demands,
                self.routes, self.warehous_at_t0, tuple(self.infos.items()),
                tuple(tuple(float(x) for x in a2) for a1 in
                      self.station_product_unit_times for a2 in a1))

    def __eq__(self, other):
        """
        Compare this object with another object.

        :param other: the other object
        :return: `NotImplemented` if the other object is not an `Instance`,
            otherwise the equality comparison result.

        >>> i1 = Instance(name="test1", n_products=1, n_customers=1,
        ...         n_stations=2, n_demands=1, routes=[[0, 1]],
        ...         demands=[[0, 0, 0, 10, 20, 100]],
        ...         warehous_at_t0=[0],
        ...     station_product_unit_times=[[[10.0, 10000.0]],
        ...                                 [[30.0, 10000.0]]])
        >>> i2 = Instance(name="test1", n_products=1, n_customers=1,
        ...         n_stations=2, n_demands=1, routes=[[0, 1]],
        ...         demands=[[0, 0, 0, 10, 20, 100]],
        ...         warehous_at_t0=[0],
        ...     station_product_unit_times=[[[10.0, 10000.0]],
        ...                                 [[30.0, 10000.0]]])
        >>> i1 == i2
        True
        >>> i3 = Instance(name="test1", n_products=1, n_customers=1,
        ...          n_stations=2, n_demands=1, routes=[[0, 1]],
        ...         demands=[[0, 0, 0, 10, 20, 100]],
        ...         warehous_at_t0=[0],
        ...     station_product_unit_times=[[[10.0, 10000.1]],
        ...                                 [[30.0, 10000.0]]])
        >>> i1 == i3
        False
        """
        if other is None:
            return False
        if not isinstance(other, Instance):
            return NotImplemented
        return self._tuple() == cast("Instance", other)._tuple()

    def __hash__(self) -> int:
        """
        Get the hash code of this object.

        :return: the hash code of this object
        """
        return hash(self._tuple())


#: the instance name key
KEY_NAME: Final[str] = "name"
#: the key for the number of products
KEY_N_PRODUCTS: Final[str] = "n_products"
#: the key for the number of customers
KEY_N_CUSTOMERS: Final[str] = "n_customers"
#: the key for the number of stations
KEY_N_STATIONS: Final[str] = "n_stations"
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
    KEY_NAME, KEY_N_PRODUCTS, KEY_N_CUSTOMERS, KEY_N_STATIONS,
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
    yield f"{COMMENT_START} the number of stations in the instance, > 0"
    yield f"{KEY_N_STATIONS}{KEY_VALUE_SEPARATOR}{instance.n_stations}"
    yield (f"{COMMENT_START} (Valid station indices are in 0.."
           f"{instance.n_stations - 1}.)")
    yield COMMENT_START
    yield (f"{COMMENT_START} the number of orders (demands) issued by the "
           f"customers, > 0")
    yield f"{KEY_N_DEMANDS}{KEY_VALUE_SEPARATOR}{instance.n_demands}"
    yield (f"{COMMENT_START} (Valid demand/order indices are in 0.."
           f"{instance.n_demands - 1}.)")

    yield COMMENT_START
    yield (f"{COMMENT_START} For each station, we now specify the indices of "
           f"the stations by which it will be processed, in the order in "
           f"which it will be processed by them.")
    yield (f"{COMMENT_START} {KEY_ROUTE}{KEY_IDX_START}0"
           f"{KEY_IDX_END} is the production route by the first product, "
           "which has index 0.")
    route_0: tuple[int, ...] = instance.routes[0]
    yield (f"{COMMENT_START} This product is processed by "
           f"{tuple.__len__(route_0)} stations, namely first by the "
           f"station with index {int(route_0[0])} and last by the station "
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
           f"system at time unit {fd.arrival} and the customer expects "
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
    yield (f"{COMMENT_START} For each station, we now specify the production "
           f"times for each product that passes through the station.")
    empty_pdx: tuple[int, int] | None = None
    filled_pdx: tuple[int, int, np.ndarray] | None = None
    need: int = 2
    for mid, station in enumerate(instance.station_product_unit_times):
        for pid, product in enumerate(station):
            pdl: int = np.ndarray.__len__(product)
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
               f"not pass through station {empty_pdx[0]}, so it is not "
               "listed here.")
    if filled_pdx is not None:
        yield (f"{COMMENT_START} For example, one unit of product "
               f"{filled_pdx[1]} passes through station {filled_pdx[0]}.")
        yield (f"{COMMENT_START} There, it needs {filled_pdx[2][0]} time "
               f"units per product unit from t=0 to t={filled_pdx[2][1]}.")
        if np.ndarray.__len__(filled_pdx[2]) > 2:
            yield (f"{COMMENT_START} After that, it needs {filled_pdx[2][2]}"
                   " time units per product unit until t="
                   f"{filled_pdx[2][3]}.")
    for mid, station in enumerate(instance.station_product_unit_times):
        for pid, product in enumerate(station):
            if np.ndarray.__len__(product) <= 0:
                continue
            flat = map(str, map(try_int, map(float, product)))
            yield (f"{KEY_PRODUCTION_TIME}{KEY_IDX_START}{mid}{CSV_SEPARATOR}"
                   f"{pid}{KEY_IDX_END}{KEY_VALUE_SEPARATOR}"
                   f"{CSV_SEPARATOR.join(flat)}")

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
    n_stations: int | None = None
    n_demands: int | None = None
    routes: list[list[int]] | None = None
    demands: list[list[int | float]] | None = None
    in_warehouse: list[int] | None = None
    station_product_times: list[list[list[float]]] | None = None
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

            elif key == KEY_N_STATIONS:
                if n_stations is not None:
                    raise __pe(
                        f"{KEY_N_STATIONS} already defined as {n_stations!r},"
                        f" cannot be set to {value!r}", oline, line_idx)
                n_stations = check_to_int_range(
                    value, KEY_N_STATIONS, 1, 1_000_000)

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
                if n_stations is None:
                    raise ValueError(f"Must define {KEY_N_STATIONS} before "
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
                dlst: list[int | float] = demands[demand_id]
                if list.__len__(dlst) != 1:
                    raise __pe(f"Already gave {KEY_DEMAND}{KEY_IDX_START}"
                               f"{demand_id}{KEY_IDX_END}", oline, line_idx)
                str_lst = str.split(value, CSV_SEPARATOR)
                if list.__len__(str_lst) != 5:
                    raise __pe(
                        f"Demand {demand_id} must have 5 entries, but got: "
                        f"{str_lst!r}", oline, line_idx)
                dlst.extend((int(str_lst[0]), int(str_lst[1]),
                             int(str_lst[2]), float(str_lst[3]),
                             float(str_lst[4])))

            elif str.startswith(key, KEY_PRODUCTION_TIME):
                if n_products is None:
                    raise __pe(f"Must define {KEY_N_PRODUCTS} before "
                               f"{KEY_PRODUCTION_TIME}", oline, line_idx)
                if n_stations is None:
                    raise __pe(f"Must define {KEY_N_STATIONS} before"
                               f" {KEY_PRODUCTION_TIME}", oline, line_idx)
                station, product = str.split(
                    __get_key_index(key), CSV_SEPARATOR)
                station_id: int = check_to_int_range(
                    station, "station", 0, n_stations - 1)
                product_id = check_to_int_range(
                    product, "product", 0, n_products - 1)

                if station_product_times is None:
                    station_product_times = \
                        [[[] for _ in range(n_products)]
                         for __ in range(n_stations)]

                mpd: list[float] = station_product_times[
                    station_id][product_id]
                if list.__len__(mpd) > 0:
                    raise __pe(f"Already gave {KEY_PRODUCTION_TIME}"
                               f"{KEY_IDX_START}{station_id}{CSV_SEPARATOR}"
                               f"{product_id}{KEY_IDX_END}", oline, line_idx)
                mpd.extend(
                    map(float, str.split(value, CSV_SEPARATOR)))
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
    if n_stations is None:
        raise ValueError("Did not specify instance n_stations"
                         f" ({KEY_N_STATIONS}).")
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
    if station_product_times is None:
        raise ValueError("Did not specify per-station product production"
                         f"times ({KEY_PRODUCTION_TIME}).")

    return Instance(name, n_products, n_customers, n_stations, n_demands,
                    routes, demands, in_warehouse, station_product_times,
                    infos)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def compute_finish_time(start_time: float, amount: int,
                        production_times: np.ndarray) -> float:
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
    >>> compute_finish_time(0.0, 1, np.array((10.0, 100.0), np.float64))
    10.0

    Here, the production time is 10 time units / 1 product unit, valid until
    end time 100. We begin producing at time unit 250. Since the production
    periods are cyclic, this is OK: we would be halfway through the third
    production period when the request comes in. It will consume 10 time units
    and be done at time unit 260.
    >>> compute_finish_time(250.0, 1, np.array((10.0, 100.0)))
    260.0

    Here, the end time of the production time validity is at time unit 100.
    However, we begin producing 1 product unit at time step 90. This unit will
    use 10 time units, meaning that its production is exactly finished when
    the production time validity ends.
    It will be finished at time step 100.
    >>> compute_finish_time(90.0, 1, np.array((10.0, 100.0)))
    100.0

    Here, the end time of the production time validity is at time unit 100.
    However, we begin producing 1 product unit at time step 95. This unit would
    use 10 time units. It cannot be completed until the end of the period. So
    only 0.5 units are produced, needing 5 time units and finishing at period
    100. Then the production time window begins again. Since there is only 1
    production time, in the second production cycle there will still be 10
    time units per production unit. The remaining 0.5 units will be produced
    in 5 time units, meaning that the overall production is finished at time
    step 105.
    >>> compute_finish_time(95.0, 1, np.array((10.0, 100.0)))
    105.0

    Now we have two production periods. The production begins again at time
    step 95. 0.5 units are finished until the production period ends at time
    step 100. Then, the second production period begins, requiring 20 time
    units per product unit. We thus need 10 time units to finish the remaining
    0.5 product unit, completing the job at time unit 110.
    >>> compute_finish_time(95.0, 1, np.array((10.0, 100.0, 20.0, 200.0)))
    110.0

    Now things get more complex. We want to do 10 units of product.
    We can finish 0.5 units until the first period ends after 5 time steps.
    Then, in the second period, we can do exactly 2 units of product until
    the period ends at time 140. This leaves 7.5 units of product to be done
    in the third period, where each unit needs 50 time units. Thus, we end
    up needing 5 + 40 + 7.5*50 time units, which add to the starting time.
    >>> compute_finish_time(95.0, 10, np.array((
    ...     10.0, 100.0, 20.0, 140.0, 50.0, 5000.0)))
    515.0
    >>> 95 + (0.5*10 + 2*20 + 7.5*50)
    515.0

    This is the same as the example before, except that the third period now
    ends at time step 200. Thus, we can only do 1.2=(200-140)/50 units of
    product in the 200 - 140 time units this period lasts. Then the production
    cycle wraps over. Luckily, we can complete the remaining 6.3 units of
    product in the first period, where each product unit needs only 10 time
    units.
    >>> compute_finish_time(95.0, 10, np.array((
    ...     10.0, 100.0, 20.0, 140.0, 50.0, 200.0)))
    263.0
    >>> 95 + (0.5*10 + 2*20 + 1.2*50 + 6.3*10)
    263.0

    It is also possible to yield fractional production times.
    >>> compute_finish_time(95.0, 10, np.array((
    ...     10.0, 100.0, 20.0, 140.0, 50.0, 200.0,  3.0, 207.0)))
    246.66666666666666
    >>> 95 + (0.5*10 + 2*20 + 1.2*50 + (7/3)*3 + (6.3-(7/3))*10)
    246.66666666666666

    Production unit times may extend beyond the intervals.
    >>> compute_finish_time(0.0, 5, np.array((1000.0, 100.0, 10.0, 110.0)))
    545.0
    >>> (0.1*1000.0 + 1*10 + 0.1*1000.0 + 1*10 + 0.1*1000.0 + 1*10 +
    ...     0.1*1000.0 + 1*10 + 0.1*1000.0 + 0.5*10)
    545.0
    """
    time_mod: Final[float] = production_times[-1]
    low_end: Final[int] = len(production_times)
    total: Final[int] = low_end // 2

    # First, we need to find the segment in the production cycle
    # where the production begins. We use a binary search for that.
    remaining: int | float = amount
    seg_start: float = start_time % time_mod
    low: int = 0
    high: int = total
    while low < high:
        mid: int = ((low + high) // 2)
        th: float = production_times[mid * 2 + 1]
        if th <= seg_start:
            low = mid + 1
        else:
            high = mid - 1
    low *= 2
    max_time: float = production_times[low + 1]
    if max_time <= seg_start:
        low += 2

    # Now we can cycle through the production cycle until the product has
    # been produced.
    while True:
        max_time = production_times[low + 1]
        unit_time: float = production_times[low]
        seg_end = seg_start + (unit_time * remaining)
        if seg_end <= max_time:
            return float(start_time + seg_end - seg_start)
        duration = max_time - seg_start
        start_time += duration
        remaining -= (duration / unit_time)
        if remaining <= 0:
            return float(start_time)
        low += 2
        if low >= low_end:
            low = 0
            seg_start = 0
            continue
        seg_start = max_time
