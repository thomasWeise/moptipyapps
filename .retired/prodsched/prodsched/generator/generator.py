"""
Methods for generating instances.

>>> from moptipyapps.prodsched.instance import to_stream

>>> def receive(inst: Instance) -> None:
...     for z in to_stream(inst):
...         print(z)


>>> with InstanceGenerator(receive) as ig:
...     ig.set_seed(11323)
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil, isqrt
from operator import itemgetter
from typing import Any, Callable, Final, Iterable

from moptipy.utils.nputils import (
    rand_generator,
    rand_seed_generate,
)
from moptipy.utils.strings import sanitize_name
from numpy.random import Generator
from pycommons.types import check_int_range, type_error

from moptipyapps.prodsched.instance import (
    DEMAND_AMOUNT,
    DEMAND_CUSTOMER,
    DEMAND_DEADLINE,
    DEMAND_PRODUCT,
    DEMAND_TIME,
    Demand,
    Instance,
)
from moptipyapps.utils.sampling import (
    AtLeast,
    Choice,
    Const,
    In,
    IntDistribution,
    Mul,
    Normal,
    distribution,
)

#: the random seed of the generator
INFO_RAND_SEED: Final[str] = "generator_rand_seed"
#: the source for the number of products
INFO_N_PRODUCTS: Final[str] = "src_n_products"
#: the source for the number of customers
INFO_N_CUSTOMERS: Final[str] = "src_n_customers"
#: the source for the number of machines
INFO_N_MACHINES: Final[str] = "src_n_machines"
#: the source for the number of demands
INFO_N_DEMANDS: Final[str] = "src_n_demands"
#: the source for the routes
INFO_ROUTES: Final[str] = "src_routes"
#: the machines per product, if any
INFO_MACHINES_PER_PRODUCT: Final[str] = "src_routes_machines_per_product"
#: the machines order offset
INFO_MACHINE_ORDER_OFFSET: Final[str] = "src_routes_machine_order_offset"
#: the key for the day time unit distribution
INFO_DAY_TIME_UNIS: Final[str] = "src_day_time_units"
#: the key for the day time unit distribution generator
INFO_DAY_TIME_UNIS_GEN: Final[str] = "src_day_time_units_gen"
#: the key for the production time unit distribution
INFO_PRODUCTION_TIME_UNITS: Final[str] = "src_production_time_units"
#: the key for the production time unit distribution generator
INFO_PRODUCTION_TIME_UNITS_GEN: Final[str] = "src_production_time_units_gen"

#: the source for the product base demands
INFO_PRODUCT_BASE_DEMANDS: Final[str] = "src_product_base_demands"
#: the distribution of the amount that is usually ordered per product
INFO_PRODUCT_BASE_DEMAND: Final[str] = "src_product_base_demand"

#: the source for the customers
INFO_CUSTOMERS: Final[str] = "src_customers"
#: the time offset of the customer
INFO_CUSTOMER_TIME_OFFSET: Final[str] = "src_customer_time_offset"
#: the size of the custmer
INFO_CUSTOMER_SIZE: Final[str] = "src_customer_size"
#: the distribution for the number of demands per customer
INFO_DEMANDS_PER_CUSTOMER: Final[str] = "src_demands_per_customer"
#: the distribution for the number of products per customer
INFO_PRODUCTS_PER_CUSTOMER: Final[str] = "src_products_per_customer"


#: the source for the routes
INFO_DEMANDS: Final[str] = "src_demands"

#: the distribution for the number of days between orders per customer
INFO_CUSTOMER_BETWEEN_ORDER_DAYS: Final[str] = \
    "src_customer_between_order_days"
#: the time jitter added to the days between orders
INFO_BETWEEN_ORDER_JITTER: Final[str] = "src_between_order_jitter"

#: the base amount that the customer usually orders
INFO_CUSTOMER_BASE_AMOUNT: Final[str] = "src_customer_base_amount"
#: the jitter applied to the orders
INFO_ORDER_AMOUNT_JITTER: Final[str] = "src_order_amount_jitter"
#: the normal time that the customer provides until the deadline
INFO_CUSTOMER_TO_DEADLINE_DAYS: Final[str] = "src_customer_to_deadline_days"
#: the jitter applied until the deadline
INFO_TO_DEADLINE_JITTER: Final[str] = "src_to_deadline_jitter"

#: a fixed structure
INFO_FIXED: Final[str] = "USER_PROVIDED"
#: a deterministic structure
INFO_DETERMINISTIC: Final[str] = "ONLY_ONE_CHOICE"
#: a sampled structure
INFO_SAMPLED: Final[str] = "SAMPLED"
#: an aggregation of the giving sampling method
INFO_AGGREGATED: Final[str] = "AGGREGATED "

#: the maximum attempts at sampling before giving up
_MAX_TRIALS: Final[int] = 1_000_000


@dataclass(order=True, frozen=True)
class Customer:
    """A customer represents purchase orders sequences."""

    #: the customer ID
    id: int
    #: the number of demands that the customer will issue
    n_demands: int
    #: the time when the customer begins doing their orders
    time_offset: int
    #: the products that the customer orders and the corresponding amount
    #: distributions
    demands_products: tuple[tuple[int, IntDistribution], ...]
    #: the time between orders
    between_order_times: IntDistribution
    #: the base time given until the deadline
    until_deadline: IntDistribution

    def __post_init__(self) -> None:
        """Perform sanity checks."""
        check_int_range(self.id, "id", 0, 1_000_000_000_000)
        check_int_range(self.n_demands, "n_demands", 0, 1_000_000_000_000)
        check_int_range(self.time_offset, "time_offset", 0, 1_000_000_000_000)
        object.__setattr__(self, "between_order_times", distribution(
            self.between_order_times))
        object.__setattr__(self, "until_deadline", distribution(
            self.until_deadline))


# pylint: disable=R0904
class InstanceGenerator(AbstractContextManager):
    """The class for generating instances."""

    def __init__(self, consumer: Callable[[Instance], Any]) -> None:
        """
        Initialize the instance generator.

        :param consumer: the consumer for the generated instances
        """
        if not callable(consumer):
            raise type_error(consumer, "consumer", call=True)

        #: the internal consumer
        self.__consumer: Final[Callable[[Instance], Any]] = consumer
        #: the information collection
        self.__infos: Final[dict[str, str]] = {}
        #: the random seed
        self.__seed: int | None = None
        #: the random number generator
        self.__random: Generator | None = None
        #: the name of the instance
        self.__name: str | None = None
        #: the number of products
        self.__n_products: int | None = None
        #: the number of customers
        self.__n_customers: int | None = None
        #: the number of machines
        self.__n_machines: int | None = None
        #: the number of demands
        self.__n_demands: int | None = None
        #: the routes that the products take through the machines
        self.__routes: tuple[tuple[int, ...], ...] | None = None
        #: the time units per day
        self.__day_time_unis: IntDistribution | None = None
        #: the production-step base time units
        self.__production_time_units: IntDistribution | None = None
        #: the normal amount of order per product
        self.__product_base_demands: tuple[IntDistribution, ...] | None = None
        #: the customer records
        self.__customers: tuple[Customer, ...] | None = None
        #: the demand records
        self.__demands: tuple[Demand, ...] | None = None

    def __enter__(self) -> "InstanceGenerator":
        """Begin the random instance generation process."""
        self.__infos.clear()
        self.__name = None
        self.__seed = None
        self.__random = None
        self.__n_products = None
        self.__n_customers = None
        self.__n_machines = None
        self.__n_demands = None
        self.__routes = None
        self.__day_time_unis = None
        self.__production_time_units = None
        self.__customers = None
        self.__product_base_demands = None
        self.__demands = None
        return self

    def __exit__(self, exception_type, _, __) -> bool:
        """
        Finalize the instance generation process.

        :param exception_type: the exception type
        :return: `True` to suppress an exception, `False` to rethrow it
        """
        if exception_type is not None:
            return False

        k: dict[str, str] = self.__infos
        p = str  # print
        p(f"name: {self.name()}")
        p(f"seed: {self.seed()}")
        p(f"products: {self.n_products()} / {k.pop(INFO_N_PRODUCTS)}")
        p(f"customers: {self.n_customers()} / {k.pop(INFO_N_CUSTOMERS)}")
        p(f"demands: {self.n_demands()} / {k.pop(INFO_N_DEMANDS)}")
        p(f"machines: {self.n_machines()} / {k.pop(INFO_N_MACHINES)}")
        p(f"routes: {self.routes()} / {k.pop(INFO_ROUTES)}")
        p(f"tu/day: {self.day_time_units()}")
        p(f"tu/prod: {self.production_time_units()}")
        p(f"customers: {self.customers()}")
        p(f"demands: {self.demands()}")
        p(f"remaining: {k}")

        return True

    def set_name(self, name: str) -> None:
        """
        Set the name of the instance.

        :param name: the name of the instance.
        """
        if self.__name is not None:
            raise ValueError(f"Name already set to {self.__name!r}.")
        self.__name = sanitize_name(name)

    def name(self) -> str:
        """
        Get the name.

        :return: the name
        """
        if self.__name is None:
            self.set_name(f"inst_{self.seed():x}")
        return self.__name

    def set_seed(self, seed: int | None = None) -> None:
        """
        Set the random seed either explicitly or automatically.

        :param seed: the optional random seed
        """
        if (self.__random is not None) or (self.__seed is not None):
            raise ValueError("Random number generator already initialized.")
        self.__seed = rand_seed_generate() if seed is None else seed
        self.__infos[INFO_RAND_SEED] = f"{seed:x}"

    def seed(self) -> int:
        """
        Get the seed.

        :return: the seed
        """
        if self.__seed is None:
            self.set_seed(None)
        return self.__seed

    def __rnd(self) -> Generator:
        """
        Get the random number generator.

        :return: the random number generator
        """
        if self.__random is None:
            self.__random = rand_generator(self.seed())
        return self.__random

    def set_n_products(self, n_products: int | IntDistribution) -> None:
        """
        Set the number of products.

        :param n_products: the number of products, either explicitly or
            implicitly set.
        """
        if self.__n_products is not None:
            raise ValueError("Number of products already set.")

        n_products = distribution(n_products)
        nx: int | None = self.__n_demands
        if nx is not None:
            n_products = In(1, nx + 1, n_products).simplify()
        else:
            n_products = AtLeast(1, n_products).simplify()
        products: Final[int] = n_products.sample(self.__rnd())

        if self.__routes is not None:
            nx = tuple.__len__(self.__routes)
            if n_products != nx:
                raise ValueError(f"Invalid value {products} for n_products "
                                 f"for {nx} routes")

        self.__n_products = products
        self.__infos[INFO_N_PRODUCTS] = repr(n_products)

    def n_products(self) -> int:
        """
        Get the number of products.

        :return: the number of products
        """
        if self.__n_products is None:
            if self.__routes is not None:
                self.set_n_products(tuple.__len__(self.__routes))
            else:
                dist: IntDistribution = Choice(
                    (1, 2, Normal(5, 1), Normal(10, 3), Normal(20, 5)))
                at_most: int | None = self.__n_demands
                self.set_n_products(AtLeast(1, dist) if at_most is None
                                    else In(1, at_most + 1, dist))
        return self.__n_products

    def set_n_customers(self, n_customers: int | IntDistribution) -> None:
        """
        Set the number of customers.

        :param n_customers: the number of customers, either explicitly or
            implicitly set.
        """
        if self.__n_customers is not None:
            raise ValueError("Number of customers already defined.")

        n_customers = distribution(n_customers)
        nx: int | None = self.__n_demands
        if nx is not None:
            n_customers = In(1, nx + 1, n_customers).simplify()
        else:
            n_customers = AtLeast(1, n_customers).simplify()
        customers: Final[int] = n_customers.sample(self.__rnd())

        self.__n_customers = customers
        self.__infos[INFO_N_CUSTOMERS] = repr(n_customers)

    def n_customers(self) -> int:
        """
        Get the number of customers.

        :return: the number of customers
        """
        if self.__n_customers is None:
            dist: IntDistribution = Choice((
                Normal(8, 2), Normal(32, 4), Normal(64, 16)))
            at_most: int | None = self.__n_demands
            self.set_n_customers(AtLeast(1, dist) if at_most is None
                                 else In(1, at_most + 1, dist))
        return self.__n_customers

    def set_n_machines(self, n_machines: int | IntDistribution) -> None:
        """
        Set the number of machines.

        :param n_machines: the number of machines, either explicitly or
            implicitly set.
        """
        if self.__n_machines is not None:
            raise ValueError("Number of machines already set.")
        n_machines = AtLeast(1, distribution(n_machines)).simplify()
        machines: Final[int] = n_machines.sample(self.__rnd())
        self.__n_machines = machines
        self.__infos[INFO_N_MACHINES] = repr(n_machines)

    def n_machines(self) -> int:
        """
        Get the number of machines.

        :return: the number of machines
        """
        if self.__n_machines is None:
            self.set_n_machines(AtLeast(1, Choice((
                Normal(4, 1), Normal(16, 4)))))
        return self.__n_machines

    def set_n_demands(self, n_demands: int | IntDistribution) -> None:
        """
        Set the number of demands.

        :param n_demands: the number of demands, either explicitly or
            implicitly set.
        """
        if self.__n_demands is not None:
            raise ValueError("Number of demands already set.")

        min_v: int = 1
        nx: int | None = self.__n_customers
        if (nx is not None) and (nx > min_v):
            min_v = nx
        nx = self.__n_products
        if (nx is not None) and (nx > min_v):
            min_v = nx

        n_demands = AtLeast(min_v, distribution(n_demands)).simplify()
        demands: Final[int] = n_demands.sample(self.__rnd())
        self.__n_demands = demands
        self.__infos[INFO_N_DEMANDS] = repr(n_demands)

    def n_demands(self) -> int:
        """
        Get the number of demands.

        :return: the number of demands
        """
        if self.__n_demands is None:
            products: int = self.n_products()
            customers: int = self.n_customers()
            at_least: int = max(products, customers)

            choices: list[IntDistribution] = [
                Normal(at_least, ceil(0.2 * at_least)),
                Normal(2 * at_least, ceil(0.5 * at_least)),
                Normal(8 * at_least, ceil(2 * at_least)),
            ]
            if at_least < 100:
                choices.append(Normal(100, 10))
            if at_least < 1000:
                choices.append(Normal(1000, 100))
            self.set_n_demands(AtLeast(at_least, Choice(tuple(choices))))
        return self.__n_demands

    def set_routes(
            self, routes: Iterable[Iterable[int]] | None = None,
            n_machines_per_product: IntDistribution | None = None,
            machine_order_offset: IntDistribution | None = None) -> None:
        """
        Set the routes on which the products pass through machines.

        The routes can either be set directly as a constance, by providing
        `routes`. Otherwise, they can be sampled randomly.

        If you specify `routes`, then this parameter provides a sequence of
        sequences: For each product, it specifies the machines through which
        the product must pass. No product can pass through the same machine
        twice. No machine must be left unused. Each product passes through
        at least one machine.

        If you want to randomly sample the machines, you can provide a
        distribution from which the number of machines through which each
        product needs to go is drawn via `n_machines_per_product`.

        Now, in the real world, even if we have different products with
        different production steps, the order in which the products pass
        through machines is not completely "random". If we randomly sample
        machines, then our idea to realize a reasonable order is as follows:
        If a product goes through a set of machines `{a, b, c}`, then it tends
        to go through the machine with the lower indices first. In some cases,
        the order may be different, but generally lower-index machines tend to
        come earlier in production cycles. To realize this, once we have the
        machines for a product selected, each machine gets base priority
        `p = 10*i`, where `i` is the machine index. To this base priority, we
        add a random number distributed as `machine_order_offset`. This offset
        can thus change the order priority of the machines. The machines are
        then used by the product based on the order priority.

        :param routes: the routes
        :param n_machines_per_product: the distribution for machines per
            product
        :param machine_order_offset: the machine order offset
        """
        if self.__routes is not None:
            raise ValueError("Routes already set.")

        if routes is not None:
            self.__infos[INFO_ROUTES] = INFO_FIXED
            self.__do_set_routes(routes)
            return

        n_products: Final[int] = self.n_products()
        n_machines: Final[int] = self.n_machines()
        if (n_products == 1) or (n_machines == 1):
            self.__infos[INFO_ROUTES] = INFO_DETERMINISTIC
            self.__do_set_routes([list(range(n_machines))] * n_products)
            return

        self.__infos[INFO_ROUTES] = INFO_SAMPLED

        if n_machines_per_product is None:
            n_machines_per_product = Normal(max(1, min(
                3, n_machines // 2, (3 * n_machines) // n_products)),
                n_machines // 4)
        elif not isinstance(n_machines_per_product, IntDistribution):
            raise type_error(n_machines_per_product, "n_machines_per_product",
                             IntDistribution)
        n_machines_per_product = In(1, n_machines, n_machines_per_product)
        self.__infos[INFO_MACHINES_PER_PRODUCT] = repr(n_machines_per_product)
        rnd: Final[Generator] = self.__rnd()
        trials: int = _MAX_TRIALS
        while True:
            npm: list[int] = [
                n_machines_per_product.sample(rnd) for _ in range(n_products)]
            todo: int = sum(npm)
            if todo > n_machines:
                break
            trials -= 1
            if trials <= 0:
                raise ValueError("Failed to sample number of machines.")

        new_routes: Final[list[list[int]]] = [[] for _ in range(n_products)]
        choose: list[int] = []
        n_choose: int = 0
        trials = _MAX_TRIALS
        while todo > 0:
            trials -= 1
            if trials <= 0:
                raise ValueError("Failed to sample machines.")
            for prod, machs in enumerate(new_routes):
                if list.__len__(machs) >= npm[prod]:
                    continue
                if n_choose <= 0:
                    choose.extend(range(n_machines))
                    n_choose = n_machines
                chosen = choose.pop(rnd.integers(n_choose))
                n_choose -= 1
                if chosen not in machs:
                    machs.append(chosen)
                    todo -= 1

        # bring some order into the machines
        if machine_order_offset is None:
            machine_order_offset = Normal(0, max(3, (10 * n_machines) // 4))
        elif not isinstance(machine_order_offset, IntDistribution):
            raise type_error(machine_order_offset, "machine_order_offset",
                             IntDistribution)
        self.__infos[INFO_MACHINE_ORDER_OFFSET] = repr(machine_order_offset)
        for route in new_routes:
            srx = sorted((10 * ma + machine_order_offset.sample(rnd), ma)
                         for ma in route)
            route.clear()
            route.extend(x[1] for x in srx)

        self.__do_set_routes(new_routes)

    def __do_set_routes(self, routes: Iterable[Iterable[int]]) -> None:
        """
        Set the fully realized product routs.

        :param routes: the routes
        """
        n_products: int | None = self.__n_products
        n_machines: int | None = self.__n_machines

        uroutes: tuple[tuple[int, ...], ...] = tuple(map(tuple, routes))

        n_routes: int = tuple.__len__(uroutes)
        if (n_routes <= 0) or ((n_products is not None) and (
                n_routes != n_products)):
            raise ValueError(f"Invalid number of uroutes {n_routes} "
                             f"for {n_products} products.")
        max_machine: int = -1
        used_machines: set[int] = set()
        all_machines: set[int] = set()
        for p, route in enumerate(uroutes):
            rlen: int = tuple.__len__(route)
            if rlen <= 0:
                raise ValueError(f"Empty route for product {p}.")
            used_machines.clear()
            for idx, mach in enumerate(route):
                if not isinstance(mach, int):
                    raise TypeError(f"invalid machine {mach} at route step "
                                    f"{idx} for product {p}")
                if (mach < 0) or ((n_machines is not None) and (
                        mach >= n_machines)):
                    raise ValueError(f"invalid machine {mach} at route step "
                                     f"{idx} for product {p} and n_machines="
                                     f"{n_machines}")
                used_machines.add(mach)
                max_machine = max(mach, max_machine)
            if set.__len__(used_machines) != rlen:
                raise ValueError(f"machine used twice in route {p}")
            all_machines.update(used_machines)

        max_machine += 1
        if set.__len__(all_machines) != max_machine:
            raise ValueError(
                f"Did not use some of the {max_machine} machines")

        if n_machines is None:
            self.set_n_machines(max_machine)
        if n_products is None:
            self.set_n_products(n_routes)
        self.__routes = uroutes

    def routes(self) -> tuple[tuple[int, ...], ...]:
        """
        Get the routes for the products through the machines.

        :return: the routes for the products through the machines
        """
        if self.__routes is None:
            self.set_routes()
        return self.__routes

    def set_day_time_units(self, tu: int | IntDistribution) -> None:
        """
        Set the distribution of the time units per day.

        :param tu: the time units per day
        """
        if self.__day_time_unis is not None:
            raise ValueError("Day time units already set.")

        tu = AtLeast(8, distribution(tu)).simplify()
        self.__day_time_unis = tu
        self.__infos[INFO_DAY_TIME_UNIS] = repr(tu)

    def day_time_units(self) -> IntDistribution:
        """
        Get the day time units distribution.

        :return: the day time units distribution
        """
        if self.__day_time_unis is None:
            gen: IntDistribution = AtLeast(100, Choice((
                Normal(60 * 60 * 8, 60 * 4), Normal(60 * 60 * 24, 60 * 4))))
            self.__infos[INFO_DAY_TIME_UNIS_GEN] = repr(gen)
            nm: int = gen.sample(self.__rnd())
            self.set_day_time_units(Normal(nm, ceil(nm / 12)))
        return self.__day_time_unis

    def set_production_time_units(
            self, tu: int | IntDistribution) -> None:
        """
        Set the distribution of the base time units per production step.

        :param tu: the time units per day
        """
        if self.__production_time_units is not None:
            raise ValueError("Production time units already set.")

        tu = AtLeast(1, distribution(tu)).simplify()
        self.__production_time_units = tu
        self.__infos[INFO_PRODUCTION_TIME_UNITS] = repr(tu)

    def production_time_units(self) -> IntDistribution:
        """
        Get the production time units base distribution.

        :return: the production time units base distribution
        """
        if self.__production_time_units is None:
            gen: IntDistribution = AtLeast(4, Normal(60, 10))
            self.__infos[INFO_PRODUCTION_TIME_UNITS_GEN] = repr(gen)
            nm: int = gen.sample(self.__rnd())
            self.set_production_time_units(Normal(nm, ceil(nm / 12)))
        return self.__production_time_units

    def set_product_base_demands(
            self, base_demands: Iterable[IntDistribution] | None = None,
            base_demand: int | IntDistribution | None = None) -> None:
        """
        Set the base unit distributions for the products.

        :param base_demands: the base demand units per product
        :param base_demand: the distribution from which we draw the base
            demands
        """
        if self.__product_base_demands is not None:
            raise ValueError("Cannot set product base demand")

        if base_demands is None:
            self.__infos[INFO_PRODUCT_BASE_DEMANDS] = INFO_SAMPLED
            if base_demand is None:
                base_demand = Choice((Const(1), Const(10), Const(30)))
            else:
                base_demand = distribution(base_demand)
            base_demand = AtLeast(1, base_demand).simplify()
            self.__infos[INFO_PRODUCT_BASE_DEMAND] = repr(base_demand)
            rnd: Final[Generator] = self.__rnd()
            base_demands = (AtLeast(1, (Normal(base_demand.sample(rnd), 1)))
                            for _ in range(self.n_products()))
        else:
            self.__infos[INFO_PRODUCT_BASE_DEMANDS] = INFO_FIXED

        base_demands = tuple(d.simplify() for d in base_demands)
        n_base_demands: int = tuple.__len__(base_demands)
        if self.__n_products is None:
            self.set_n_products(n_base_demands)
        elif n_base_demands != self.__n_products:
            raise ValueError("Inconsistent number of product demands")
        self.__product_base_demands = base_demands

    def product_base_demands(self) -> tuple[IntDistribution, ...]:
        """
        Get the product base demand unit.

        :return: the base demand distributions
        """
        if self.__product_base_demands is None:
            self.set_product_base_demands()
        return self.__product_base_demands

    def set_customers(
            self, customers: Iterable[Customer] | None = None,
            time_offset: int | IntDistribution | None = None,
            demands_per_customer: int | IntDistribution | None = None,
            products_per_customer: int | IntDistribution | None = None,
            customer_size: int | IntDistribution | None = None,
            between_order_days: int | IntDistribution | None = None,
            between_order_jitter: int | IntDistribution | None = None,
            to_deadline_days: int | IntDistribution | None = None,
            to_deadline_jitter: int | IntDistribution | None = None) \
            -> None:
        """
        Set the customers.

        You can either set the customers directly, by providing `customers`.
        Or they can be randomly sampled.

        :param customers: the customers
        :param time_offset: the distribution for the time offset when
            customers begin ordering
        :param demands_per_customer: an optional distribution of the number of
            demands that each customer may issue. In total, there will always
            be exactly the number of expected demands. The distributions are
            sampled again and again, until that number is reached.
        :param products_per_customer: the products to be sampled per customer
        :param customer_size: the multiplier of the order amount
        :param between_order_days: a distribution for the number of
            days between customer orders
        :param between_order_jitter: the jitter for the time between orders
        :param to_deadline_days: the normal amount of days that a
            customer gives as deadline
        :param to_deadline_jitter: the jitter for the above
        """
        if customers is not None:
            self.__infos[INFO_CUSTOMERS] = INFO_FIXED
            self.__do_set_customers(customers)
            return
        if (self.__customers is not None) or (self.__demands is not None):
            raise ValueError("Cannot set customers anymore.")

        self.__infos[INFO_CUSTOMERS] = INFO_SAMPLED
        n_demands: Final[int] = self.n_demands()
        n_customers: Final[int] = self.n_customers()
        n_products: Final[int] = self.n_products()

        # Assign the demands to customers
        if demands_per_customer is None:
            frac: float = max(1.0, n_demands / n_customers)
            demands_per_customer = Normal(frac, ceil(0.3 * frac))
        else:
            demands_per_customer = distribution(demands_per_customer)
        demands_per_customer = In(
            1, n_demands + 1, demands_per_customer).simplify()

        self.__infos[INFO_DEMANDS_PER_CUSTOMER] =\
            f"{INFO_AGGREGATED}{demands_per_customer!r}"

        prod_base_dem: Final[tuple[
            IntDistribution, ...]] = self.product_base_demands()

        # Assign the products to customers
        if products_per_customer is None:
            frac = max(1.0, n_products / n_customers)
            products_per_customer = Normal(frac, ceil(0.3 * frac))
        else:
            products_per_customer = distribution(products_per_customer)
        products_per_customer = In(
            1, n_products + 1, products_per_customer).simplify()
        self.__infos[INFO_PRODUCTS_PER_CUSTOMER] =\
            f"{INFO_AGGREGATED}{products_per_customer!r}"

        tod: int = 100
        if time_offset is None:
            time_offset = Choice((Const(0), Normal(
                tod * 10, tod * 10), Normal(tod * 100, tod * 100)))
        else:
            tod = 1
            time_offset = distribution(time_offset)
        time_offset = AtLeast(0, time_offset).simplify()
        self.__infos[INFO_CUSTOMER_TIME_OFFSET] = f"{time_offset!r}/{tod}"

        if customer_size is None:
            customer_size = Normal(1, 5)
        else:
            customer_size = distribution(customer_size)
        customer_size = AtLeast(1, customer_size)
        self.__infos[INFO_CUSTOMER_SIZE] = repr(customer_size)

        #: Set up the required distributions
        if between_order_days is None:
            between_order_days = Choice((
                Normal(3, 1), Normal(10, 2), Normal(30, 2)))
        else:
            between_order_days = distribution(
                between_order_days)
        between_order_days = AtLeast(
            0, between_order_days).simplify()
        self.__infos[INFO_CUSTOMER_BETWEEN_ORDER_DAYS] = repr(
            between_order_days)

        bojd: int = 100
        if between_order_jitter is None:
            between_order_jitter = Normal(bojd, bojd / 3)
        else:
            bojd = 1
            between_order_jitter = distribution(between_order_jitter)
        between_order_jitter = AtLeast(0, between_order_jitter).simplify()
        self.__infos[INFO_BETWEEN_ORDER_JITTER] = \
            f"{between_order_jitter!r}/{bojd}"

        #: Set up the required distributions
        if to_deadline_days is None:
            to_deadline_days = Choice((
                Normal(3, 1), Normal(10, 2), Normal(30, 2)))
        else:
            to_deadline_days = distribution(to_deadline_days)
        to_deadline_days = AtLeast(0, to_deadline_days).simplify()
        self.__infos[INFO_CUSTOMER_TO_DEADLINE_DAYS] = repr(
            to_deadline_days)

        tdjd: int = 100
        if to_deadline_jitter is None:
            to_deadline_jitter = Normal(tdjd, tdjd / 3)
        else:
            to_deadline_jitter = distribution(to_deadline_jitter)
            tdjd = 1
        to_deadline_jitter = AtLeast(0, to_deadline_jitter).simplify()
        self.__infos[INFO_TO_DEADLINE_JITTER] = \
            f"{to_deadline_jitter!r}/{tdjd}"

        rnd: Final[Generator] = self.__rnd()

        cust_dem_prod: Final[list[tuple[list[int], list[int]]]] = [
            ([0], []) for _ in range(n_customers)]

        outer_trials: int = (10 + isqrt(_MAX_TRIALS))
        products: list[int] = []

        success: bool = False
        while not success:
            outer_trials -= 1
            if outer_trials <= 0:
                raise ValueError("Could not distribute demands to customers.")

            inner_trials: int = (10 + isqrt(_MAX_TRIALS))
            while not success:
                inner_trials -= 1
                if inner_trials <= 0:
                    raise ValueError(
                        "Could not distribute demands to customers.")
                for dems, _ in cust_dem_prod:
                    dems[0] = 0
                total: int = n_demands
                current: int = 0
                while total > 0:
                    n = In(1, total + 1, demands_per_customer).sample(rnd)
                    cust_dem_prod[current][0][0] += n
                    total -= n
                    current = (current + 1) % n_customers
                success = all(k[0][0] > 0 for k in cust_dem_prod)
            if not success:
                continue
            rnd.shuffle(cust_dem_prod)

            inner_trials = (10 + isqrt(_MAX_TRIALS))
            success = False
            while not success:
                inner_trials -= 1
                if inner_trials <= 0:
                    raise ValueError(
                        "Could not distribute products to customers.")

                for _, prods in cust_dem_prod:
                    prods.clear()

                products.extend(range(n_products))
                for dems, prods in cust_dem_prod:
                    for _ in range(In(  # no more products than orders
                            1, dems[0] + 1, products_per_customer).sample(
                            rnd)):
                        while True:
                            if list.__len__(products) <= 0:
                                success = True
                                products.extend(range(n_products))
                            prod = products.pop(rnd.integers(
                                list.__len__(products)))
                            if prod not in prods:
                                prods.append(prod)
                                break

        # At this stage, we have created the customers.
        # Each customer will order from a set of products.
        # Each customer has a number of demands.
        rnd.shuffle(cust_dem_prod)
        output: list[Customer] = []
        dtu_dist: IntDistribution = self.day_time_units()
        for cust_id, cust_data in enumerate(cust_dem_prod):
            cust_demands = cust_data[0][0]
            cust_products = cust_data[1]
            rnd.shuffle(cust_products)
            cust_prd_amount: list[tuple[int, IntDistribution]] = []
            for product in cust_products:
                base_amount = AtLeast(1, Mul((
                    customer_size, prod_base_dem[product]))).sample(rnd)
                cust_prd_amount.append((product, AtLeast(1, Normal(
                    base_amount, base_amount / 4))))

            use_dtu: int = dtu_dist.sample(rnd)
            bod_mean = use_dtu * between_order_days.sample(rnd)
            bod_sd = max(1, (
                use_dtu * between_order_jitter.sample(rnd) // bojd))
            cust_beto: IntDistribution = AtLeast(
                0, Normal(bod_mean, bod_sd)).simplify()
            tld_mean = use_dtu * to_deadline_days.sample(rnd)
            tld_sd = max(
                1, (use_dtu * to_deadline_jitter.sample(rnd)) // tdjd)
            cust_tdl: IntDistribution = AtLeast(0, Normal(
                tld_mean, tld_sd)).simplify()
            output.append(Customer(cust_id, cust_demands, (
                use_dtu * time_offset.sample(rnd)) // tod, tuple(
                cust_prd_amount), cust_beto, cust_tdl))

        self.__do_set_customers(output)

    def __do_set_customers(self, customers: Iterable[Customer]) -> None:
        """
        Set the customers.

        :param customers: the customers
        """
        if self.__customers is not None:
            raise ValueError("Customers already set.")
        if self.__demands is not None:
            raise ValueError("Cannot set customers after demands are set.")

        customers = tuple(customers)
        n_customers: Final[int] = tuple.__len__(customers)
        if n_customers < 0:
            raise ValueError("Number of customers invalid.")
        if self.__n_customers is None:
            self.set_n_customers(n_customers)
        elif self.__n_customers != n_customers:
            raise ValueError(f"Cannot set {n_customers} customers, "
                             f"must use {self.__n_customers}.")
        for cust in customers:
            if not isinstance(cust, Customer):
                raise type_error(cust, "customer", Customer)
        all_cust = set(customers)
        if set.__len__(all_cust) != n_customers:
            raise ValueError("Customer appears twice?")

        prods: set[int] = set()
        custs: set[int] = set()
        n_demands: int = 0
        for cust in customers:
            prods.update(map(itemgetter(0), cust.demands_products))
            custs.add(cust.id)
            n_demands += cust.n_demands

        if n_customers != (max(custs) - min(custs) + 1):
            raise ValueError("Missing customers?")
        if self.__n_demands is None:
            self.set_n_demands(n_demands)
        elif n_demands != self.__n_demands:
            raise ValueError("Inconsistent number of demands.")

        n_products = set.__len__(prods)
        if n_products != (max(prods) - min(prods) + 1):
            raise ValueError("Missing products?")
        if self.__n_products is None:
            self.set_n_products(n_products)
        elif self.__n_products != n_products:
            raise ValueError("Inconsistent number of products.")
        self.__customers = customers

    def customers(self) -> tuple[Customer, ...]:
        """
        Get the customer settings.

        :return: the customer settings
        """
        if self.__customers is None:
            self.set_customers()
        return self.__customers

    def set_demands(self, demands: Iterable[Demand] | None = None) -> None:
        """
        Set the demands.

        You can either set the demands directly, by providing `demands`.
        Or they can be randomly sampled.

        :param demands: the demands
        """
        if demands is not None:
            self.__infos[INFO_DEMANDS] = INFO_FIXED
            self.__do_set_demands(demands)
            return

        rnd: Final[Generator] = self.__rnd()
        customers: tuple[Customer, ...] = self.customers()

        #: Now we can sample the customer orders and their associated
        #: time windows.
        orders: list[list[int]] = []
        for customer in customers:
            n_demands: int = customer.n_demands
            prod_dist: dict[int, IntDistribution] = {
                xxx[0]: xxx[1] for xxx in customer.demands_products}
            products: tuple[int, ...] = tuple(prod_dist.keys())
            cust_orders: list[list[int]] = [[0] * 6 for _ in range(n_demands)]
            prod_src: list[int] = list(products)

            for order in cust_orders:
                psl: int = list.__len__(prod_src)
                if psl <= 0:
                    prod_src.extend(products)
                    psl = list.__len__(prod_src)
                order[DEMAND_PRODUCT] = prod_src.pop(rnd.integers(psl))
                order[DEMAND_CUSTOMER] = customer.id
            rnd.shuffle(cust_orders)

            time: int = customer.time_offset
            for order in cust_orders:
                use_prod = order[DEMAND_PRODUCT]
                order[DEMAND_AMOUNT] = prod_dist[use_prod].sample(rnd)
                time += customer.between_order_times.sample(rnd)
                order[DEMAND_TIME] = time
                order[DEMAND_DEADLINE] = (
                    time + customer.until_deadline.sample(rnd))
            orders.extend(cust_orders)

        # Now we got all the orders
        rnd.shuffle(orders)
        orders.sort(key=itemgetter(DEMAND_TIME))

        self.__do_set_demands(Demand(
            release_time=order[DEMAND_TIME],
            deadline=order[DEMAND_DEADLINE],
            demand_id=did,
            customer_id=order[DEMAND_CUSTOMER],
            product_id=order[DEMAND_PRODUCT],
            amount=order[DEMAND_AMOUNT]) for did, order in enumerate(orders))

    def __do_set_demands(self, demands: Iterable[Demand]) -> None:
        """
        Set the demands.

        :param demands: the demands
        """
        if self.__demands is not None:
            raise ValueError("Demands already set.")

        demands = tuple(demands)
        n_demands: Final[int] = tuple.__len__(demands)

        if n_demands < 0:
            raise ValueError("Number of demands invalid.")
        if self.__n_demands is None:
            self.set_n_demands(n_demands)
        elif self.__n_demands != n_demands:
            raise ValueError(f"Cannot set {n_demands} demands, "
                             f"must use {self.__n_demands}.")
        for d in demands:
            if not isinstance(d, Demand):
                raise type_error(d, "demand", Demand)
        alld = set(demands)
        if set.__len__(alld) != n_demands:
            raise ValueError("Demand appears twice?")

        prods: set[int] = set()
        custs: set[int] = set()
        for demand in demands:
            prods.add(demand.product_id)
            custs.add(demand.customer_id)

        n_customers: Final[int] = set.__len__(custs)
        if (max(custs) - min(custs) + 1) != n_customers:
            raise ValueError("Missing customers?")
        if self.__n_customers is None:
            self.set_n_customers(n_customers)
        elif n_customers != self.__n_customers:
            raise ValueError("Invalid number of customers!")

        n_products: Final[int] = set.__len__(prods)
        if (max(prods) - min(prods) + 1) != n_products:
            raise ValueError("Missing products?")
        if self.__n_products is None:
            self.set_n_products(n_products)
        elif n_products != self.__n_products:
            raise ValueError("Invalid number of products!")

        self.__demands = demands

    def demands(self) -> tuple[Demand, ...]:
        """
        Get the demands of the scenario.

        :return: the demands
        """
        if self.__demands is None:
            self.set_demands()
        return self.__demands
