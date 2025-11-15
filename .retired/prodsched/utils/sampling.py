"""
Some utilities for random sampling.

The goal that we follow with class :class:`IntDistribution` is to have
clearly defined integer-producing random distributions. We want to be
able to say exactly how to generate some random numbers.

>>> from moptipy.utils.nputils import rand_generator
>>> rnd = rand_generator(0)

>>> nd = Normal(1, 2)
>>> nd
Normal(mu=1, sd=2)
>>> [nd.sample(rnd) for _ in range(20)]
[1, 1, 2, 1, 0, 2, 4, 3, 0, -2, 0, 1, -4, 1, -1, 0, 0, 0, 2, 3]

>>> lb = AtLeast(2, nd)
>>> lb
AtLeast(lb=2, d=Normal(mu=1, sd=2))

>>> [lb.sample(rnd) for _ in range(20)]
[4, 2, 3, 2, 2, 3, 4, 4, 4, 3, 2, 4, 5, 5, 4, 2, 2, 2, 2, 2]

>>> u = Uniform(10, 17)
>>> u
Uniform(low=10, high=17)
>>> [u.sample(rnd) for _ in range(10)]
[15, 11, 16, 10, 14, 13, 17, 11, 17, 10]

>>> s = Sum((u, nd, ))
>>> s
Sum(ds=(Uniform(low=10, high=17), Normal(mu=1, sd=2)))

>>> [s.sample(rnd) for _ in range(10)]
[16, 14, 21, 13, 16, 12, 13, 18, 12, 20]

>>> g = Gamma(2, 8)
>>> g
Gamma(k=2, scale=8)
>>> [g.sample(rnd) for _ in range(10)]
[14, 15, 10, 4, 20, 7, 11, 9, 16, 15]

>>> b = In(5, 10, g)
>>> b
In(lb=5, ub=10, d=Gamma(k=2, scale=8))
>>> [b.sample(rnd) for _ in range(10)]
[7, 8, 8, 5, 7, 6, 9, 9, 9, 9]

>>> s2 = Sum((Const(10), nd, ))
>>> s2
Sum(ds=(Const(v=10), Normal(mu=1, sd=2)))

>>> [s2.sample(rnd) for _ in range(10)]
[10, 10, 8, 12, 13, 10, 11, 8, 10, 14]

>>> ch = Choice((2, 20, 3000, g))
>>> ch
Choice(ch=(2, 20, 3000, Gamma(k=2, scale=8)))

>>> [ch.sample(rnd) for _ in range(10)]
[20, 20, 7, 11, 20, 2, 3000, 2, 2, 2]
"""

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Final, cast

from numpy.random import Generator
from pycommons.math.int_math import try_int
from pycommons.types import check_int_range, type_error

#: the maximum number of trials during a sampling process
_MAX_TRIALS: int = 1_000_000


class IntDistribution:
    """A base class for integer distributions."""

    def sample(self, random: Generator) -> int:
        """
        Sample an integer from a random number generator.

        :param random: the random number generator
        :return: the integer
        """
        raise NotImplementedError

    def simplify(self) -> "IntDistribution":
        """
        Try to simplify this distribution.

        :returns: a simplified version of this distribution
        """
        if not isinstance(self, IntDistribution):
            raise type_error(self, "self", IntDistribution)
        return self


def distribution(d: int | IntDistribution) -> IntDistribution:
    """
    Get the distribution from the parameter.

    :param d: the integer value or distribution
    :return: the canonicalized distribution
    """
    if isinstance(d, int):
        return Const(d)
    if isinstance(d, IntDistribution):
        old_d: IntDistribution | None = None
        while old_d is not d:
            old_d = d
            d = d.simplify()
        return d
    raise type_error(d, "d", (IntDistribution, int))


@dataclass(order=True, frozen=True)
class Const(IntDistribution):
    """A constant value."""

    v: int

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        check_int_range(self.v, "v", -1_000_000_000_000, 1_000_000_000_000)

    def sample(self, random: Generator) -> int:
        """
        Sample the constant integer.

        :param random: the random number generator
        :return: the integer
        """
        return self.v


@dataclass(order=True, frozen=True)
class Normal(IntDistribution):
    """A class representing a normal distribution."""

    #: the expected value and center of the distribution
    mu: int | float
    #: the standard deviation of the distribution
    sd: int | float

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        if not (isfinite(self.mu) and isfinite(self.sd) and (self.sd > 0)):
            raise ValueError(f"Invalid parameters {self}.")
        object.__setattr__(self, "mu", try_int(self.mu))
        object.__setattr__(self, "sd", try_int(self.sd))

    def sample(self, random: Generator) -> int:
        """
        Sample from the normal distribution.

        :param random: the random number generator
        :return: the result
        """
        return round(random.normal(self.mu, self.sd))


@dataclass(order=True, frozen=True)
class Gamma(IntDistribution):
    """A class representing a Gamma distribution."""

    #: the shape parameter
    k: int | float
    #: the scale parameter
    scale: int | float

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        if not (isfinite(self.k) and isfinite(self.k) and (
                self.k > 0) and (self.scale > 0)):
            raise ValueError(f"Invalid parameters {self}.")
        object.__setattr__(self, "k", try_int(self.k))
        object.__setattr__(self, "scale", try_int(self.scale))

    def sample(self, random: Generator) -> int:
        """
        Sample from the Gamma distribution.

        :param random: the random number generator
        :return: the result
        """
        return round(random.gamma(self.k, self.scale))


@dataclass(order=True, frozen=True)
class Uniform(IntDistribution):
    """A class representing a uniform distribution."""

    #: the lowest permitted value
    low: int
    #: the highest permitted value
    high: int

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        if not (isinstance(self.low, int) and isinstance(self.high, int)):
            raise TypeError(f"Invalid types {self}.")
        if self.high <= self.low:
            raise ValueError(f"Invalid parameters {self}.")

    def sample(self, random: Generator) -> int:
        """
        Sample from the uniform distribution.

        :param random: the random number generator
        :return: the result
        """
        return int(random.integers(self.low, self.high + 1))

    def simplify(self) -> "IntDistribution":
        """
        Try to simplify this distribution.

        :returns: a simplified version of this distribution
        """
        if self.high == (self.low + 1):
            return Const(self.low)
        return self


@dataclass(order=True, frozen=True)
class Choice(IntDistribution):
    """A class representing a uniform choice."""

    #: the choices
    ch: tuple[int | IntDistribution, ...]

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        if not isinstance(self.ch, tuple):
            raise TypeError(f"Invalid types {self}.")
        for v in self.ch:
            if isinstance(v, IntDistribution):
                continue
            check_int_range(v, "v", -1_000_000_000_000, 1_000_000_000_000)

    def sample(self, random: Generator) -> int:
        """
        Sample from the uniform distribution.

        :param random: the random number generator
        :return: the result
        """
        res: Final = self.ch[random.integers(tuple.__len__(self.ch))]
        return res.sample(random) if isinstance(res, IntDistribution) else res


@dataclass(order=True, frozen=True)
class AtLeast(IntDistribution):
    """
    A distribution that is lower-bounded.

    >>> AtLeast(5, Const(7))
    AtLeast(lb=5, d=Const(v=7))

    >>> AtLeast(5, AtLeast(8, Const(17)))
    AtLeast(lb=8, d=Const(v=17))

    >>> AtLeast(8, AtLeast(5, Const(17)))
    AtLeast(lb=8, d=Const(v=17))
    """

    #: the inclusive lower bound
    lb: int
    #: the inner distribution to sample from
    d: IntDistribution

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        if not (isinstance(self.lb, int) and isinstance(
                self.d, IntDistribution)):
            raise TypeError(f"Invalid types {self}.")
        dd: Final[IntDistribution] = self.d
        if isinstance(dd, Const):
            if cast("Const", dd).v < self.lb:
                raise ValueError(f"Invalid distribution {self!r}.")
        elif isinstance(dd, AtLeast):
            dlb: AtLeast = cast("AtLeast", dd)
            object.__setattr__(self, "lb", max(self.lb, dlb.lb))
            object.__setattr__(self, "d", dlb.d)
        if isinstance(dd, In):
            idd: In = cast("In", dd)
            ulb: int = max(idd.lb, self.lb)
            if ulb >= idd.ub:
                raise ValueError(f"Invalid distribution {self!r}.")

    def simplify(self) -> "IntDistribution":
        """
        Try to simplify this distribution.

        :returns: a simplified version of this distribution
        """
        dd: Final[IntDistribution] = self.d
        if isinstance(dd, Const):
            return dd
        if isinstance(dd, In):
            idd: In = cast("In", dd)
            return In(max(idd.lb, self.lb), idd.ub, idd.d).simplify()
        return self

    def sample(self, random: Generator) -> int:
        """
        Sample from the lower-bounded distribution.

        :param random: the random number generator
        :return: the result
        """
        s: Final[Callable[[Generator], int]] = self.d.sample
        lb: Final[int] = self.lb
        for _ in range(_MAX_TRIALS):
            v = s(random)
            if lb <= v:
                return v
        raise ValueError(f"Failed to sample from {self!r}.")


@dataclass(order=True, frozen=True)
class In(IntDistribution):
    """
    A distribution that is lower and upper-bounded.

    >>> In(1, 10, Const(6))
    In(lb=1, ub=10, d=Const(v=6))

    >>> In(1, 10, In(5, 12, Const(6)))
    In(lb=5, ub=10, d=Const(v=6))

    >>> In(1, 10, AtLeast(6, Const(6)))
    In(lb=6, ub=10, d=Const(v=6))
    """

    #: the inclusive lower bound
    lb: int
    #: the exclusive upper bound
    ub: int
    #: the inner distribution to sample from
    d: IntDistribution

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        if not (isinstance(self.lb, int) and isinstance(self.ub, int)
                and isinstance(self.d, IntDistribution)):
            raise TypeError(f"Invalid types {self}.")
        if self.ub <= self.lb:
            raise ValueError(f"Invalid range {self}.")
        dd: IntDistribution = self.d
        lb: Final[int] = self.lb
        ub: Final[int] = self.ub
        if isinstance(dd, In):
            idd: In = cast("In", dd)
            ulb: int = max(idd.lb, lb)
            uub: int = min(idd.ub, ub)
            if ulb >= uub:
                raise ValueError(f"Invalid distribution {self!r}.")
            object.__setattr__(self, "lb", ulb)
            object.__setattr__(self, "ub", uub)
            dd = idd.d
            object.__setattr__(self, "d", dd)
        elif isinstance(dd, AtLeast):
            ldd: AtLeast = cast("AtLeast", dd)
            ulb = max(ldd.lb, lb)
            if ulb >= ub:
                raise ValueError(f"Invalid distribution {self!r}.")
            object.__setattr__(self, "lb", ulb)
            dd = ldd.d
            object.__setattr__(self, "d", dd)
        if isinstance(dd, Const) and not (lb <= cast("Const", dd).v <= ub):
            raise ValueError(f"Invalid distribution {self!r}.")

    def sample(self, random: Generator) -> int:
        """
        Sample from the lower-bounded distribution.

        :param random: the random number generator
        :return: the result
        """
        s: Final[Callable[[Generator], int]] = self.d.sample
        lb: Final[int] = self.lb
        ub: Final[int] = self.ub
        for _ in range(_MAX_TRIALS):
            v = s(random)
            if lb <= v <= ub:
                return v
        raise ValueError(f"Failed to sample from {self!r}.")

    def simplify(self) -> IntDistribution:
        """
        Simplify this distribution.

        :return: the simplified distribution
        """
        return Const(self.lb) if self.lb >= (self.ub + 1) else self


@dataclass(order=True, frozen=True)
class Sum(IntDistribution):
    """
    The sum of several distributions.

    >>> Sum((Const(1), Const(2))).simplify()
    Const(v=3)

    >>> Sum((Const(1), Normal(2, 3))).simplify()
    Sum(ds=(Const(v=1), Normal(mu=2, sd=3)))

    >>> Sum((Const(0), Normal(2, 3))).simplify()
    Normal(mu=2, sd=3)

    >>> Sum((Const(0), Normal(2, 3), Const(5))).simplify()
    Sum(ds=(Normal(mu=2, sd=3), Const(v=5)))
    """

    #: the distributions to sum over
    ds: tuple[IntDistribution, ...]

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        is_wrong: bool = False
        if tuple.__len__(self.ds) <= 0:
            is_wrong = True
        else:
            for d in self.ds:
                if not isinstance(d, IntDistribution):
                    is_wrong = True
                    break
        if is_wrong:
            raise ValueError(f"Invalid parameters {self}.")

    def simplify(self) -> IntDistribution:
        """
        Simplify this distribution.

        :return: the simplified distribution
        """
        count: Final[int] = tuple.__len__(self.ds)
        if count <= 1:
            return self.ds[0]
        const_sum: int = 0
        all_const: bool = True
        needed: list[IntDistribution] = []
        recreate: bool = False
        for d in self.ds:
            new_d = d.simplify()
            recreate = recreate or (new_d != d)
            if isinstance(new_d, Const):
                const_sum += cast("Const", new_d).v
            else:
                needed.append(new_d)
                all_const = False
        if all_const:
            return Const(const_sum)

        nl: int = list.__len__(needed)
        if recreate or (nl < (count - 1)) or (
                (nl < count) and (const_sum == 0)):
            if const_sum != 0:
                needed.append(Const(const_sum))
            if list.__len__(needed) <= 1:
                return needed[0]
            return Sum(tuple(needed))
        return self

    def sample(self, random: Generator) -> int:
        """
        Sample from the sum distribution.

        :param random: the random number generator
        :return: the result
        """
        total: int = 0
        for d in self.ds:
            total += d.sample(random)
        return total


@dataclass(order=True, frozen=True)
class Mul(IntDistribution):
    """
    The multiplication of several distributions.

    >>> Mul((Const(1), Const(2))).simplify()
    Const(v=2)

    >>> Mul((Const(2), Normal(2, 3))).simplify()
    Mul(ds=(Const(v=2), Normal(mu=2, sd=3)))

    >>> Mul((Const(1), Normal(2, 3))).simplify()
    Normal(mu=2, sd=3)

    >>> Mul((Const(1), Normal(2, 3), Const(5))).simplify()
    Mul(ds=(Normal(mu=2, sd=3), Const(v=5)))

    >>> Mul((Const(1), Normal(2, 3), Const(0))).simplify()
    Const(v=0)
    """

    #: the distributions to sum over
    ds: tuple[IntDistribution, ...]

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        is_wrong: bool = False
        if tuple.__len__(self.ds) <= 0:
            is_wrong = True
        else:
            for d in self.ds:
                if not isinstance(d, IntDistribution):
                    is_wrong = True
                    break
        if is_wrong:
            raise ValueError(f"Invalid parameters {self}.")

    def simplify(self) -> IntDistribution:
        """
        Simplify this distribution.

        :return: the simplified distribution
        """
        count: Final[int] = tuple.__len__(self.ds)
        if count <= 1:
            return self.ds[0]
        all_const: bool = True
        const_prod: int = 1
        needed: list[IntDistribution] = []
        recreate: bool = False
        for d in self.ds:
            new_d = d.simplify()
            recreate = recreate or (d != new_d)
            if isinstance(new_d, Const):
                cd = cast("Const", new_d)
                if cd.v == 0:
                    return cd
                const_prod *= cd.v
            else:
                all_const = False
                needed.append(new_d)

        if all_const:
            return Const(const_prod)
        nl: int = list.__len__(needed)
        if recreate or (nl < (count - 1)) or ((nl < count) and (
                const_prod == 1)):
            if const_prod != 1:
                needed.append(Const(const_prod))
            if list.__len__(needed) <= 1:
                return needed[0]
            return Mul(tuple(needed))
        return self

    def sample(self, random: Generator) -> int:
        """
        Sample from the sum distribution.

        :param random: the random number generator
        :return: the result
        """
        total: int = 1
        for d in self.ds:
            total *= d.sample(random)
        return total
