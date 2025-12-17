"""
Methods for generating MFC instances.

In the module :mod:`~moptipyapps.prodsched.instance`, we provide the class
:class:`~moptipyapps.prodsched.instance.Instance`. Objects of this class
represent a fully deterministic production scheduling scenario. They prescribe
demands (:class:`~moptipyapps.prodsched.instance.Demand`) arriving at fixed
points in time in the system and work stations that need fixed amounts of
work times per product during certain time periods.
This allows us to create fully reproducible simulations
(:mod:`~moptipyapps.prodsched.simulation`) that show what a factory would do
to satisfy the demands.

But where does such instance data come from?

In the existing research on material flow control, no such fixed instances
exist. We invented them. Instead, the existing research [1] uses fixed numbers
of products and machines, fixed routes of products through machines, and
random distributions to generate demands and work times.

So we create the function :func:`~default_stations` that creates the standard
work time distributions for the standard work stations. We also create the
function :func:`~default_products` that creates the default distributions
for the default products.

The function :func:`sample_mfc_instance` then creates a material flow instance
following these distributions based on a given random seed.
This allows us to create scenarios that follow the same structure and random
distributions as prescribed in the paper [1] by Thürer et al.
However, our instances are fully deterministic.

Once could not create a certain number of such instances and average
performance metrics over simulations on them. This would likely yield metrics
of reasonable accuracy, while allowing us to reproduce, analyze, and trace
every single production decision if need be.

>>> from moptipyapps.utils.sampling import Gamma
>>> inst = sample_mfc_instance([
...  Product(0, (0, 1), Gamma.from_alpha_beta(3, 0.26))], [
...  Station(0, Gamma.from_k_and_mean(3, 10)),
...  Station(1, Gamma.from_k_and_mean(2, 10))],
...  time_end_measure=100, seed=123)

>>> inst.name
'mfc_1_2_100_0x7b'

>>> inst.n_demands
7

>>> inst.demands
(Demand(arrival=5.213885878801001, deadline=5.213885878801001, demand_id=0,\
 customer_id=0, product_id=0, amount=1, measure=False),\
 Demand(arrival=25.872387132411287, deadline=25.872387132411287, demand_id=1,\
 customer_id=1, product_id=0, amount=1, measure=False),\
 Demand(arrival=43.062182155666896, deadline=43.062182155666896, demand_id=2,\
 customer_id=2, product_id=0, amount=1, measure=True),\
 Demand(arrival=49.817978344678004, deadline=49.817978344678004, demand_id=3,\
 customer_id=3, product_id=0, amount=1, measure=True),\
 Demand(arrival=58.21166922638016, deadline=58.21166922638016, demand_id=4,\
 customer_id=4, product_id=0, amount=1, measure=True),\
 Demand(arrival=69.09054693162531, deadline=69.09054693162531, demand_id=5,\
 customer_id=5, product_id=0, amount=1, measure=True),\
 Demand(arrival=88.804579148131, deadline=88.804579148131, demand_id=6,\
 customer_id=6, product_id=0, amount=1, measure=True))

>>> len(inst.station_product_unit_times[0][0])
1600

>>> len(inst.station_product_unit_times[1][0])
1600

>>> inst.time_end_measure
100.0

>>> inst.time_end_warmup
30.0

>>> d = dict(inst.infos)
>>> del d["info_generated_on"]
>>> del d["info_generator_version"]
>>> d
{'info_generator': 'moptipyapps.prodsched.mfc_generator',\
 'info_rand_seed_src': 'USER_PROVIDED',\
 'info_rand_seed': '0x7b',\
 'info_time_end_measure_src': 'USER_PROVIDED',\
 'info_time_end_measure': '100',\
 'info_time_end_warmup_src': 'SAMPLED',\
 'info_time_end_warmup': '30',\
 'info_name_src': 'SAMPLED',\
 'info_product_interarrival_times[0]':\
 'Erlang(k=3, theta=3.846153846153846)',\
 'info_product_route[0]': 'USER_PROVIDED',\
 'info_station_processing_time[0]':\
 'Erlang(k=3, theta=3.3333333333333335)',\
 'info_station_processing_time_window_length[0]': 'Const(v=0.125)',\
 'info_station_processing_time[1]': 'Erlang(k=2, theta=5)',\
 'info_station_processing_time_window_length[1]': 'Const(v=0.125)'}

>>> inst = sample_mfc_instance(seed=23445)
>>> inst.name
'mfc_10_13_10000_0x5b95'

>>> inst.n_demands
9922

>>> len([dem for dem in inst.demands if dem.product_id == 0])
959

>>> len([dem for dem in inst.demands if dem.product_id == 1])
1055

>>> [len(k[0]) for k in inst.station_product_unit_times]
[160000, 160000, 0, 160000, 0, 0, 0, 0, 160000, 160000, 160000, 0, 0]

1. Matthias Thürer, Nuno O. Fernandes, Hermann Lödding, and Mark Stevenson.
   Material Flow Control in Make-to-Stock Production Systems: An Assessment of
   Order Generation, Order Release and Production Authorization by Simulation
   Flexible Services and Manufacturing Journal. 37(1):1-37. March 2025.
   doi:<https://doi.org/10.1007/s10696-024-09532-2>
"""
import datetime
from dataclasses import dataclass
from typing import Callable, Final, Iterable

from moptipy.utils.nputils import (
    rand_generator,
    rand_seed_generate,
)
from moptipy.utils.strings import sanitize_name
from numpy.random import Generator
from pycommons.math.int_math import try_int
from pycommons.strings.string_conv import num_to_str
from pycommons.types import check_int_range, type_error

from moptipyapps.prodsched.instance import (
    KEY_IDX_END,
    KEY_IDX_START,
    MAX_VALUE,
    Demand,
    Instance,
)
from moptipyapps.utils.sampling import (
    AtLeast,
    Const,
    Distribution,
    Erlang,
    Gamma,
    Uniform,
)
from moptipyapps.version import __version__

#: the "now" function
__DTN: Final[Callable[[], datetime.datetime]] = datetime.datetime.now


#: The information key for interarrival times
INFO_PRODUCT_INTERARRIVAL_TIME_DIST: Final[str] = \
    "info_product_interarrival_times"

#: The generator key
INFO_GENERATOR: Final[str] = "info_generator"
#: The generator version
INFO_GENERATOR_VERSION: Final[str] = "info_generator_version"
#: When was the instance generated?
INFO_GENERATED_ON: Final[str] = "info_generated_on"
#: The information key for interarrival times
INFO_PRODUCT_ROUTE: Final[str] = "info_product_route"
#: a fixed structure
INFO_USER_PROVIDED: Final[str] = "USER_PROVIDED"
#: a sampled structure
INFO_SAMPLED: Final[str] = "SAMPLED"
#: The information key for the processing time distribution
INFO_STATION_PROCESSING_TIME: Final[str] = "info_station_processing_time"
#: The information key for the processing time window length distribution
INFO_STATION_PROCESSING_WINDOW_LENGTH: Final[str] = \
    "info_station_processing_time_window_length"
#: the random seed
INFO_RAND_SEED: Final[str] = "info_rand_seed"
#: the random seed source
INFO_RAND_SEED_SRC: Final[str] = f"{INFO_RAND_SEED}_src"
#: the name source
INFO_NAME_SRC: Final[str] = "info_name_src"
#: the warmup time end
INFO_TIME_END_WARMUP: Final[str] = "info_time_end_warmup"
#: the source of the warmup time end
INFO_TIME_END_WARMUP_SRC: Final[str] = f"{INFO_TIME_END_WARMUP}_src"
#: the measurement time end
INFO_TIME_END_MEASURE: Final[str] = "info_time_end_measure"
#: the source of the measurement time end
INFO_TIME_END_MEASURE_SRC: Final[str] = f"{INFO_TIME_END_MEASURE}_src"


@dataclass(order=True, frozen=True)
class Product:
    """The product sampling definition."""

    #: the product ID
    product_id: int
    #: the routing of the product
    routing: tuple[int, ...]
    #: the interarrival distribution
    interarrival_times: Distribution

    def __init__(self, product_id: int, routing: Iterable[int],
                 interarrival_times: int | float | Distribution) -> None:
        """
        Create the product sampling instruction.

        :param product_id: the product id
        :param routing: the routing information
        :param interarrival_times: the interarrival time distribution
        """
        object.__setattr__(self, "product_id", check_int_range(
            product_id, "product_id", 0, 1_000_000))
        route: tuple[int, ...] = tuple(routing)
        n_route: int = tuple.__len__(route)
        if n_route <= 0:
            raise ValueError("Route cannot be empty!")
        for k in route:
            check_int_range(k, "station", 0, 1_000_000)
        object.__setattr__(self, "routing", route)
        object.__setattr__(
            self, "interarrival_times", AtLeast.greater_than_zero(
                interarrival_times).simplify())

    def log_info(self, infos: dict[str, str]) -> None:
        """
        Log the sampling information of this product to the infos `dict`.

        :param infos: the information dictionary
        """
        key: Final[str] = f"{KEY_IDX_START}{self.product_id}{KEY_IDX_END}"
        infos[f"{INFO_PRODUCT_INTERARRIVAL_TIME_DIST}{key}"] = repr(
            self.interarrival_times)
        infos[f"{INFO_PRODUCT_ROUTE}{key}"] = INFO_USER_PROVIDED


@dataclass(order=True, frozen=True)
class Station:
    """The station sampling definition."""

    #: the product ID
    station_id: int
    #: the processing time distribution
    processing_time: Distribution
    #: the processing window distribution
    processing_windows: Distribution

    def __init__(
            self, station_id: int,
            processing_time: int | float | Distribution,
            processing_windows: int | float | Distribution | None = None) \
            -> None:
        """
        Create the station sampling instruction.

        :param station_id: the station id
        :param processing_time: the processing time distribution
        :param processing_windows: the processing time window length
            distribution
        """
        object.__setattr__(self, "station_id", check_int_range(
            station_id, "station_id", 0, 1_000_000))
        object.__setattr__(
            self, "processing_time",
            AtLeast.greater_than_zero(processing_time).simplify())

        if processing_windows is None:
            processing_windows = Const(1 / 8)
        object.__setattr__(
            self, "processing_windows",
            AtLeast.greater_than_zero(processing_windows).simplify())

    def log_info(self, infos: dict[str, str]) -> None:
        """
        Log the sampling information of this product to the infos `dict`.

        :param infos: the information dictionary
        """
        key: Final[str] = f"{KEY_IDX_START}{self.station_id}{KEY_IDX_END}"
        infos[f"{INFO_STATION_PROCESSING_TIME}{key}"] = repr(
            self.processing_time)
        infos[f"{INFO_STATION_PROCESSING_WINDOW_LENGTH}{key}"] = repr(
            self.processing_windows)


# pylint: disable=R0914,R0912,R0915
def sample_mfc_instance(products: Iterable[Product] | None = None,
                        stations: Iterable[Station] | None = None,
                        time_end_warmup: int | float | None = None,
                        time_end_measure: int | float | None = None,
                        name: str | None = None,
                        seed: int | None = None) -> Instance:
    """
    Sample an MFC instance.

    :param products: the products
    :param stations: the work stations
    :param time_end_warmup: the end of the warmup period
    :param time_end_measure: the end of the measurement period
    :param name: the instance name
    :param seed: the random seed, if any
    :return: the instance
    """
    generator: str = str(__file__)
    idx: int = str.rfind(generator, "moptipyapps")
    if idx >= 0:
        generator = str.removesuffix(generator[idx:].replace("/", "."), ".py")
    else:
        generator = "mfc_generator"
    infos: Final[dict[str, str]] = {
        INFO_GENERATOR: generator,
        INFO_GENERATOR_VERSION: __version__,
        INFO_GENERATED_ON: str(__DTN()),
    }

    if products is None:
        products = default_products()
    products = sorted(products)
    n_products: Final[int] = list.__len__(products)
    if n_products <= 0:
        raise ValueError(f"Cannot have {n_products} products.")

    ids: Final[set[int]] = set()
    used_stations: Final[set[int]] = set()
    for product in products:
        if not isinstance(product, Product):
            raise type_error(product, "product", Product)
        ids.add(product.product_id)
        used_stations.update(product.routing)
    if (set.__len__(ids) != n_products) or (
            max(ids) - min(ids) + 1 != n_products):
        raise ValueError("Inconsistent product ids.")

    n_stations: Final[int] = set.__len__(used_stations)
    if not 0 < n_stations < 1_000_000:
        raise ValueError(f"Invalid number {n_stations} of stations.")
    if stations is None:
        if n_stations == 13:
            stations = default_stations()
        else:
            raise ValueError(
                "Can only use default settings with 13 stations, "
                f"but got {n_stations}.")

    stations = sorted(stations)
    n_stations_real: int = list.__len__(stations)
    if n_stations_real != n_stations:
        raise ValueError(
            f"Products use {n_stations} stations,"
            f" but {n_stations_real} are provided.")

    ids.clear()
    for station in stations:
        if not isinstance(station, Station):
            raise type_error(station, "station", Station)
        ids.add(station.station_id)
    min_id: int = min(ids)
    max_id: int = max(ids)
    if (set.__len__(ids) != n_stations) or (
            max_id - min_id + 1 != n_stations):
        raise ValueError("Inconsistent station ids.")
    if ids != used_stations:
        raise ValueError(
            f"Station ids are {min_id}...{max_id}, but products use "
            f"stations {sorted(used_stations)}.")

    if seed is None:
        infos[INFO_RAND_SEED_SRC] = INFO_SAMPLED
        seed = rand_seed_generate()
    else:
        infos[INFO_RAND_SEED_SRC] = INFO_USER_PROVIDED
    if not isinstance(seed, int):
        raise type_error(seed, "seed", int)
    infos[INFO_RAND_SEED] = hex(seed)

    if time_end_measure is None:
        time_end_measure = 10_000 if (time_end_warmup is None) or (
            time_end_warmup <= 0) else max(
            time_end_warmup + 1, (10 * time_end_warmup) / 3)
        infos[INFO_TIME_END_MEASURE_SRC] = INFO_SAMPLED
    else:
        infos[INFO_TIME_END_MEASURE_SRC] = INFO_USER_PROVIDED
    time_end_measure = try_int(time_end_measure)
    if not 0 < time_end_measure < MAX_VALUE:
        raise ValueError(
            f"Invalid time_end_measure={time_end_measure}.")
    infos[INFO_TIME_END_MEASURE] = num_to_str(time_end_measure)

    if time_end_warmup is None:
        time_end_warmup = (3 * time_end_measure) / 10
        infos[INFO_TIME_END_WARMUP_SRC] = INFO_SAMPLED
    else:
        infos[INFO_TIME_END_WARMUP_SRC] = INFO_USER_PROVIDED
    time_end_warmup = try_int(time_end_warmup)
    if not 0 <= time_end_warmup < time_end_measure:
        raise ValueError(f"Invalid time_end_warmup={time_end_warmup} "
                         f"for time_end_measure={time_end_measure}.")
    infos[INFO_TIME_END_WARMUP] = num_to_str(time_end_warmup)

    if name is None:
        infos[INFO_NAME_SRC] = INFO_SAMPLED
        s: str = num_to_str(time_end_measure).replace(".", "d")
        name = f"mfc_{n_products}_{n_stations}_{s}_{seed:#x}"
    else:
        infos[INFO_NAME_SRC] = INFO_USER_PROVIDED
    uname: str = sanitize_name(name)
    if uname != name:
        raise ValueError(f"Invalid name {name!r}.")

    random: Final[Generator] = rand_generator(seed)

    # sample the demands
    demands: Final[list[Demand]] = []
    current_id: int = 0
    for product in products:
        time: float = 0.0
        while True:
            until = product.interarrival_times.sample(random)
            time += until
            if time >= time_end_measure:
                break
            demands.append(Demand(
                arrival=time, deadline=time, demand_id=current_id,
                customer_id=current_id, product_id=product.product_id,
                amount=1, measure=time_end_warmup <= time))
            current_id += 1

    #: sample the working times
    production_times: Final[list[list[list[float]]]] = []
    for station in stations:
        times: list[float] = []
        time = 0.0
        while True:
            processing = station.processing_time.sample(random)
            window = station.processing_windows.sample(random)
            time += window
            times.extend((processing, time))
            if time >= time_end_measure:
                break
        production_times.append([
            times if station.station_id in product.routing else []
            for product in products])

    # log the information
    for product in products:
        product.log_info(infos)
    for station in stations:
        station.log_info(infos)

    return Instance(
        name=name, n_products=n_products,
        n_customers=current_id, n_stations=n_stations,
        n_demands=current_id,
        time_end_warmup=time_end_warmup, time_end_measure=time_end_measure,
        routes=(product.routing for product in products),
        demands=demands, warehous_at_t0=[0] * n_products,
        station_product_unit_times=production_times,
        infos=infos)


def __s1t0(s: Iterable[int]) -> tuple[int, ...]:
    """
    Convert stations from 1 to 0-based index.

    :param s: the stations
    :return: the index

    >>> __s1t0((1, 2, 3))
    (0, 1, 2)
    """
    return tuple(x - 1 for x in s)


def default_products() -> tuple[Product, ...]:
    """
    Create the default product sequence as used in [1].

    :return: the default product sequence

    >>> default_products()
    (Product(product_id=0, routing=(0, 1, 3, 1, 8, 9, 10), \
interarrival_times=Erlang(k=3, theta=3.3333333333333335)), \
Product(product_id=1, routing=(0, 1, 4, 1, 7, 8, 9, 10), \
interarrival_times=Erlang(k=2, theta=5)), \
Product(product_id=2, routing=(0, 1, 5, 3, 1, 8, 11, 10), \
interarrival_times=Uniform(low=5, high=15)), \
Product(product_id=3, routing=(0, 1, 6, 3, 1, 8, 9, 10), \
interarrival_times=Erlang(k=3, theta=3.3333333333333335)), \
Product(product_id=4, routing=(0, 1, 3, 11, 1, 8, 1, 12), \
interarrival_times=Erlang(k=4, theta=2.5)), \
Product(product_id=5, routing=(0, 1, 4, 11, 1, 8, 6, 12), \
interarrival_times=Erlang(k=2, theta=5)), \
Product(product_id=6, routing=(0, 1, 5, 11, 1, 7, 1, 12), \
interarrival_times=Erlang(k=4, theta=2.5)), \
Product(product_id=7, routing=(0, 1, 2, 6, 3, 11, 1, 7, 5, 8, 1, 12), \
interarrival_times=Uniform(low=5, high=15)), \
Product(product_id=8, routing=(0, 1, 2, 4, 3, 5, 11, 1, 7, 1, 9, 5, 12), \
interarrival_times=Erlang(k=4, theta=2.5)), \
Product(product_id=9, routing=(0, 1, 2, 5, 1, 3, 11, 6, 1, 8, 10, 4, 12), \
interarrival_times=Erlang(k=2, theta=5)))

    1. Matthias Thürer, Nuno O. Fernandes, Hermann Lödding, and Mark
       Stevenson. Material Flow Control in Make-to-Stock Production Systems:
       An Assessment of Order Generation, Order Release and Production
       Authorization by Simulation Flexible Services and Manufacturing
       Journal. 37(1):1-37. March 2025.
       doi: https://doi.org/10.1007/s10696-024-09532-2
    """
    return (
        Product(0, __s1t0((1, 2, 4, 2, 9, 10, 11)),
                Erlang.from_k_and_mean(3, 10)),
        Product(1, __s1t0((1, 2, 5, 2, 8, 9, 10, 11)),
                Erlang.from_k_and_mean(2, 10)),
        Product(2, __s1t0((1, 2, 6, 4, 2, 9, 12, 11)),
                Uniform(5, 15)),
        Product(3, __s1t0((1, 2, 7, 4, 2, 9, 10, 11)),
                Erlang.from_k_and_mean(3, 10)),
        Product(4, __s1t0((1, 2, 4, 12, 2, 9, 2, 13)),
                Erlang.from_k_and_mean(4, 10)),
        Product(5, __s1t0((1, 2, 5, 12, 2, 9, 7, 13)),
                Erlang.from_k_and_mean(2, 10)),
        Product(6, __s1t0((1, 2, 6, 12, 2, 8, 2, 13)),
                Erlang.from_k_and_mean(4, 10)),
        Product(7, __s1t0((1, 2, 3, 7, 4, 12, 2, 8, 6, 9, 2, 13)),
                Uniform(5, 15)),
        Product(8, __s1t0((1, 2, 3, 5, 4, 6, 12, 2, 8, 2, 10, 6, 13)),
                Erlang.from_k_and_mean(4, 10)),
        Product(9, __s1t0((1, 2, 3, 6, 2, 4, 12, 7, 2, 9, 11, 5, 13)),
                Erlang.from_k_and_mean(2, 10)))


def default_stations() -> tuple[Station, ...]:
    """
    Create the default station sequence as used in [1].

    :return: the default product station

    >>> default_stations()
    (Station(station_id=0, processing_time=Erlang(k=3, theta=0.26),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=1, processing_time=Erlang(k=3, theta=0.12),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=2, processing_time=Erlang(k=2, theta=1.33),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=3,\
 processing_time=AtLeast(lb=5e-324, d=Exponential(eta=1)),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=4, processing_time=Erlang(k=3, theta=0.67),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=5, processing_time=Erlang(k=4, theta=0.35),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=6, processing_time=Erlang(k=3, theta=0.59),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=7, processing_time=Erlang(k=3, theta=0.63),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=8, processing_time=Erlang(k=2, theta=0.59),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=9, processing_time=Erlang(k=3, theta=0.6),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=10,\
 processing_time=AtLeast(lb=5e-324, d=Exponential(eta=1)),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=11, processing_time=Erlang(k=4, theta=0.29),\
 processing_windows=Const(v=0.125)),\
 Station(station_id=12, processing_time=Erlang(k=3, theta=0.48),\
 processing_windows=Const(v=0.125)))

    1. Matthias Thürer, Nuno O. Fernandes, Hermann Lödding, and Mark
       Stevenson. Material Flow Control in Make-to-Stock Production Systems:
       An Assessment of Order Generation, Order Release and Production
       Authorization by Simulation Flexible Services and Manufacturing
       Journal. 37(1):1-37. March 2025.
       doi: https://doi.org/10.1007/s10696-024-09532-2
    """
    return (
        Station(0, Gamma(3, 0.26)),
        Station(1, Gamma(3, 0.12)),
        Station(2, Gamma(2, 1.33)),
        Station(3, Gamma(1, 1.06)),
        Station(4, Gamma(3, 0.67)),
        Station(5, Gamma(4, 0.35)),
        Station(6, Gamma(3, 0.59)),
        Station(7, Gamma(3, 0.63)),
        Station(8, Gamma(2, 0.59)),
        Station(9, Gamma(3, 0.6)),
        Station(10, Gamma(1, 1.44)),
        Station(11, Gamma(4, 0.29)),
        Station(12, Gamma(3, 0.48)))
