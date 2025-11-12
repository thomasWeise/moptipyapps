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
from typing import Callable, Final

from numpy.random import Generator
from pycommons.math.int_math import try_int
from pycommons.types import check_int_range

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
    """A distribution that is lower-bounded."""

    #: the inclusive lower bound
    lb: int
    #: the inner distribution to sample from
    d: IntDistribution

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        if not (isinstance(self.lb, int) and isinstance(
                self.d, IntDistribution)):
            raise TypeError(f"Invalid types {self}.")

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
    """A distribution that is lower and upper-bounded."""

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


@dataclass(order=True, frozen=True)
class Sum(IntDistribution):
    """The sum of several distributions."""

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
