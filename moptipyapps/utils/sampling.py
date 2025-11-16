"""
Some utilities for random sampling.

The goal that we follow with class :class:`Distribution` is to have
clearly defined integer-producing random distributions. We want to be
able to say exactly how to generate some random numbers.

A distribution can be sampled using the method :meth:`~Distribution.sample`.
Each distribution has a mean value, which either may be an exact value or
an approximate result, that can be obtained via :meth:`~Distribution.mean`.
Sometimes, distributions can be simplified, which is supported by
:meth:`~Distribution.simplify`.

>>> from moptipy.utils.nputils import rand_generator
>>> from statistics import mean
>>> rnd = rand_generator(0)

>>> const = Const(12.3)
>>> const
Const(v=12.3)
>>> const.mean()
12.3
>>> const.sample(rnd)
12.3

>>> normal = Normal(1, 2.0)
>>> normal
Normal(mu=1, sd=2)
>>> x = [normal.sample(rnd) for _ in range(200)]
>>> x[:20]
[1.2514604421867865, 0.7357902734173962, 2.280845300886564,\
 1.2098002343060794, -0.07133874632222192, 1.7231901098189695,\
 3.6080000902602745, 2.8941619262584846, -0.4074704716139852,\
 -1.530842942092105, -0.24654892507470438, 1.0826519586944872,\
 -3.6500615492776687, 0.5624166721349085, -1.4918218945061303,\
 -0.4645347094069032, -0.08851796571461978, 0.3673996872616909,\
 1.8232610727482657, 3.085026738885355]
>>> mean(x) / normal.mean()
1.0305262793198813

>>> exponential = Exponential(3)
>>> x = [exponential.sample(rnd) for _ in range(200)]
>>> x[:20]
[1.3203895772033505, 1.152983246827425, 4.527545171626064,\
 5.080441711712409, 0.6498200242245252, 3.408652958826374,\
 2.842677620245357, 2.1132194336022, 3.5838508479713393,\
 1.2223825336486978, 2.8454397976498504, 0.8653980789905962,\
 0.19572166348792452, 3.2506725793229854, 0.9426446058446336,\
 3.246902386754473, 10.294412603282666, 5.275683923067543,\
 0.49517363091492944, 0.2982336402218482]
>>> mean(x) / exponential.mean()
1.088857332669081

>>> gamma = Gamma(3.0, 0.26)
>>> gamma
Gamma(k=3, theta=0.26)
>>> x = [gamma.sample(rnd) for _ in range(200)]
>>> x[:20]
[1.1617763743292708, 1.305755109480284, 0.7627948403389954,\
 1.2735522637897285, 0.7742951665621697, 1.074233520618276,\
 0.6324661100546898, 1.4627037699922791, 0.5739033567160827,\
 0.5555065636904546, 0.629236234283296, 0.3666171387296996,\
 0.3780976936750937, 0.9511433672028858, 1.2607313263258062,\
 1.4442096466925938, 0.48758642085808085, 1.247724803721524,\
 1.9359140456080306, 1.3935246884396764]
>>> mean(x) / gamma.mean()
0.962254253258444

>>> Gamma.from_alpha_beta(2, 0.5)
Erlang(k=2, theta=2)

>>> Gamma.from_alpha_beta(2.5, 0.5)
Gamma(k=2.5, theta=2)

>>> Erlang.from_alpha_beta(2, 0.5)
Erlang(k=2, theta=2)

>>> erlang = Erlang(1.0, 0.26)
>>> erlang
Erlang(k=1, theta=0.26)
>>> x = [erlang.sample(rnd) for _ in range(200)]
>>> x[:20]
[0.29911329228981776, 0.3630768060267626, 0.14111385731543394,\
 0.0745673280234536, 0.029950507989979877, 0.04741877104350835,\
 0.38599089026561223, 0.047919114170390194, 0.06921557868837301,\
 0.4066084140331242, 0.07170887998378667, 0.022061870233843223,\
 0.04904717644388396, 0.32082097064821674, 0.001884448999141546,\
 0.6687964040577958, 0.060598863807579915, 0.21491377996577304,\
 0.23088301776258766, 0.23667780086315618]
>>> mean(x) / erlang.mean()
0.9212193955320179

>>> uniform = Uniform(10.5, 17)
>>> uniform
Uniform(low=10.5, high=17)
>>> x = [uniform.sample(rnd) for _ in range(200)]
>>> x[:20]
[11.235929257903324, 13.89459856391694, 14.196833332199233,\
 13.871456027849515, 14.485310406702393, 16.20468251006836,\
 13.7773314921396, 12.964459911549145, 12.167722566772781,\
 12.49450317595701, 14.145245846833722, 15.669920217581366,\
 13.367286694558258, 10.764955127505994, 11.723004274328007,\
 11.089239517262232, 12.666732191193558, 14.948448277461127,\
 14.339645757381653, 14.803829334728565]
>>> mean(x) / uniform.mean()
0.9968578545459643

>>> choice = Choice((Const(2), gamma, normal))
>>> choice
Choice(ch=(Const(v=2), Gamma(k=3, theta=0.26), Normal(mu=1, sd=2)))
>>> x = [choice.sample(rnd) for _ in range(200)]
>>> x[:20]
[0.5499364190576, -0.10428920251005325, 2, 2, 0.44263544084840273,\
 0.6088189450771303, 2, 2, 2, 2, -0.32003290715104904, 2,\
 0.6165299227577784, 1.0445083345086352, 2, 2, 2, -0.5970322857539738,\
 -1.6672705710198277, 2]
>>> mean(x) / choice.mean()
0.9849730453309948

>>> lower = AtLeast(2, normal)
>>> lower
AtLeast(lb=2, d=Normal(mu=1, sd=2))
>>> x = [lower.sample(rnd) for _ in range(200)]
>>> x[:20]
[2.9715155848474772, 3.407503493034132, 2.59634461519561,\
 2.0472458472897714, 3.9865670827840334, 2.015117730344058,\
 2.0316584714999935, 5.36625737470408, 3.042848158343226,\
 3.390407444638451, 4.5503184215686, 3.7682459882073007,\
 2.0539760253651305, 2.134886147372958, 2.500182239395479,\
 2.891402111997337, 2.7393826524907228, 2.3449577842364766,\
 2.9043074694017195, 4.7173482582723825]
>>> mean(x) / lower.mean()
1.0178179732075965

>>> interval = In(1, 10, gamma)
>>> interval
In(lb=1, ub=10, d=Gamma(k=3, theta=0.26))
>>> x = [interval.sample(rnd) for _ in range(200)]
>>> x[:20]
[1.3488423972739083, 1.0929631399361601, 1.681162621901135,\
 1.655614926246918, 1.041948891842002, 2.0773395958990175,\
 1.3338891374921853, 1.2478964188175743, 1.9417070894505217,\
 1.3990572178987266, 1.1216118870312337, 1.0641160239253207,\
 2.131253600219639, 1.0337453883221577, 1.396499618416345,\
 1.9865175145136038, 1.2555473269031396, 1.4412027583435465,\
 1.320740351247919, 1.0407411942999665]
>>> mean(x) / interval.mean()
1.0256329849020747

>>> erlang2 = Gamma.from_k_and_mean(3, 10)
>>> erlang2
Erlang(k=3, theta=3.3333333333333335)
>>> erlang2.mean()
10
>>> x = [erlang2.sample(rnd) for _ in range(200)]
>>> x[:20]
[10.087506399523226, 12.928131914870168, 12.330250639007767,\
 5.305123692562998, 21.085037136404374, 6.6603691824173135,\
 2.961302890492059, 9.810557147180853, 8.051620919921454,\
 8.750329405836668, 3.9511445189935763, 5.570300668751883,\
 16.70132947692463, 7.831425379483914, 11.154757962484842,\
 8.78943102381046, 8.395847820234795, 16.42251602814587,\
 17.1628992966332, 9.684008648356015]
>>> mean(x) / erlang2.mean()
1.022444624140316
"""

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Callable, Final, cast

from moptipy.utils.nputils import rand_generator, rand_seeds_from_str
from numpy.random import Generator
from pycommons.math.int_math import try_int
from pycommons.types import type_error

#: the maximum number of trials during a sampling process
_MAX_TRIALS: int = 1_000_000


class Distribution:
    """A base class for distributions."""

    def sample(self, random: Generator) -> int | float:
        """
        Sample a random number following this distribution generator.

        Each call to this function returns exactly one number.

        :param random: the random number generator
        :return: the number
        """
        raise NotImplementedError

    def simplify(self) -> "Distribution":
        """
        Try to simplify this distribution.

        Some distributions can trivially be simplified. For example, if you
        have applied a range limit (:class:`In`) to a constant distribution
        (class:`Const`), then this can be simplified to just the constant.
        If such simplification is possible, this method returns the simplified
        distribution. Otherwise, it just returns the distribution itself.

        :returns: a simplified version of this distribution
        """
        if not isinstance(self, Distribution):
            raise type_error(self, "self", Distribution)
        return self

    def mean(self) -> int | float:
        """
        Get the mean or approximate mean of the distribution.

        Some distribution overwrite this method to produce an exact computed
        expected value or mean. This default implementation just computes the
        arithmetic mean of 10'000 samples of the distribution. This serves as
        baseline approximation for any case where a closed form mathematical
        definition of the expected value is not available.

        :return: the mean or approximated mean
        """
        sample: Callable[[Generator], int | float] = self.sample
        gen: Final[Generator] = rand_generator(rand_seeds_from_str(
            repr(self), 1)[0])
        return try_int(fsum(sample(gen) for _ in range(10_000)) / 10_000)


@dataclass(order=True, frozen=True)
class Const(Distribution):
    """A constant value."""

    #: the constant value
    v: int | float

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        object.__setattr__(self, "v", try_int(self.v))

    def sample(self, random: Generator) -> int | float:
        """
        Sample the constant integer.

        :param random: the random number generator
        :return: the integer
        """
        return self.v

    def mean(self) -> int | float:
        """
        Get the mean of this distribution.

        :return: the mean
        """
        return self.v


@dataclass(order=True, frozen=True)
class Normal(Distribution):
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

    def sample(self, random: Generator) -> float:
        """
        Sample from the normal distribution.

        :param random: the random number generator
        :return: the result
        """
        return random.normal(self.mu, self.sd)

    def mean(self) -> int | float:
        """
        Get the mean of this distribution.

        :return: the mean
        """
        return self.mu


@dataclass(order=True, frozen=True)
class Exponential(Distribution):
    """A class representing an exponential distribution."""

    #: the exponential distribution parameter
    eta: int | float

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        object.__setattr__(self, "eta", try_int(self.eta))
        if self.eta <= 0:
            raise ValueError(f"Invalid setup {self!r}.")

    def sample(self, random: Generator) -> float:
        """
        Sample from the Exponential distribution.

        :param random: the random number generator
        :return: the result
        """
        return random.exponential(self.eta)

    def mean(self) -> int | float:
        """
        Get the mean of this distribution.

        :return: the mean
        """
        return try_int(self.eta)


@dataclass(order=True, frozen=True)
class Gamma(Distribution):
    """
    A class representing a Gamma distribution.

    Here, `k` is the shape and `theta` is the scale parameter.
    If you use a parameterization with `alpha` and `beta`, you need to create
    the distribution using :meth:`~Gamma.from_alpha_beta` instead. The reason
    is that `shape = 1/beta`, see
    https://www.statlect.com/probability-distributions/gamma-distribution.
    """

    #: the shape parameter
    k: int | float
    #: the scale parameter
    theta: int | float

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        object.__setattr__(self, "k", try_int(self.k))
        object.__setattr__(self, "theta", try_int(self.theta))
        if not (self.k > 0) and (self.theta > 0):
            raise ValueError(f"Invalid parameters {self}.")

    def sample(self, random: Generator) -> float:
        """
        Sample from the Gamma distribution.

        :param random: the random number generator
        :return: the result
        """
        return random.gamma(self.k, self.theta)

    def simplify(self) -> "Distribution":
        """
        Try to simplify this distribution.

        A Gamma distribution may simplify to either an :class:`Erlang` or an
        :class:`Exponential` distribution, depending on its parameters.
        If the :attr:`~Gamma.k` is `1`, then it is actually an
        :class:`Exponential` distribution. If :attr:`~Gamma.k` is an integer,
        then the distribution is an :class:`Erlang` distribution.

        1. https://www.statisticshowto.com/gamma-distribution
        2. https://www.statisticshowto.com/erlang-distribution

        :returns: a simplified version of this distribution
        """
        return Exponential(self.k) if self.k == 1 else (
            Erlang(self.k, self.theta) if isinstance(
                self.k, int) and not isinstance(self, Erlang) else self)

    def mean(self) -> int | float:
        """
        Get the mean of this distribution.

        :return: the mean
        """
        return try_int(self.k * self.theta)

    @classmethod
    def from_alpha_beta(cls, alpha: int | float, beta: int | float) \
            -> "Distribution":
        """
        Create a Gamma distribution from `alpha` and `beta`.

        :param alpha: the alpha parameter
        :param beta: the beta parameter
        :return: the distribution
        """
        beta = try_int(beta)
        if beta == 0:
            raise ValueError(f"beta cannot be {beta}.")
        return cls(alpha, 1 / beta).simplify()

    @classmethod
    def from_k_and_mean(cls, k: int | float, mean: int | float) \
            -> "Distribution":
        """
        Create the Gamma distribution from the value of `k` and a mean.

        :param k: the shape parameter
        :param mean: the mean
        :return: the distribution
        """
        k = try_int(k)
        mean = try_int(mean)
        if (mean <= 0) or (k <= 0):
            raise ValueError(f"Invalid values k={k}, mean={mean}.")
        return Gamma(k, mean / k).simplify()


class Erlang(Gamma):
    """The Erlang distribution."""

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        super().__post_init__()
        if not isinstance(self.k, int):
            raise type_error(self.k, "k", int)


@dataclass(order=True, frozen=True)
class Uniform(Distribution):
    """A class representing a uniform distribution."""

    #: the lowest permitted value
    low: int | float
    #: the highest permitted value
    high: int | float

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        object.__setattr__(self, "low", try_int(self.low))
        object.__setattr__(self, "high", try_int(self.high))
        if self.high <= self.low:
            raise ValueError(f"Invalid parameters {self}.")

    def sample(self, random: Generator) -> float:
        """
        Sample from the uniform distribution.

        :param random: the random number generator
        :return: the result
        """
        return random.uniform(self.low, self.high)

    def mean(self) -> int | float:
        """
        Get the mean of this distribution.

        :return: the mean
        """
        return try_int((self.high + self.low) / 2)


@dataclass(order=True, frozen=True)
class Choice(Distribution):
    """
    A class representing a uniform choice.

    >>> Choice((Uniform(1, 2), Uniform(3, 4))).simplify()
    Choice(ch=(Uniform(low=1, high=2), Uniform(low=3, high=4)))

    >>> Choice((Uniform(1, 2), Uniform(1.0, 2))).simplify()
    Uniform(low=1, high=2)

    >>> Choice((Uniform(1, 2), Choice(
    ...     (Const(1), Uniform(1, 2))))).simplify()
    Choice(ch=(Uniform(low=1, high=2), Const(v=1), Uniform(low=1, high=2)))
    """

    #: the choices
    ch: tuple[Distribution, ...]

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        if not isinstance(self.ch, tuple):
            raise TypeError(f"Invalid types {self}.")
        for v in self.ch:
            if not isinstance(v, Distribution):
                raise type_error(v, "choice", Distribution)

    def mean(self) -> int | float:
        """
        Get the mean of this distribution.

        :return: the mean
        """
        return try_int(fsum(d.mean() for d in self.ch) / tuple.__len__(
            self.ch))

    def sample(self, random: Generator) -> int | float:
        """
        Sample from the uniform distribution.

        :param random: the random number generator
        :return: the result
        """
        return self.ch[random.integers(tuple.__len__(self.ch))].sample(random)

    def simplify(self) -> Distribution:
        """
        Try to simplify this distribution.

        :returns: a simplified version of this distribution
        """
        ch: Final[tuple[Distribution, ...]] = self.ch
        if tuple.__len__(ch) <= 1:
            return ch[0].simplify()
        done: list[Distribution] = []
        needs: bool = False
        for dist in ch:
            use: Distribution = dist.simplify()
            if use != dist:
                needs = True
            if isinstance(use, Choice):
                needs = True
                done.extend(use.ch)
            else:
                done.append(use)

        total: int = list.__len__(done)
        dc: Distribution = done[0]
        if total <= 1:
            return dc

        all_same: bool = True
        for oth in done:
            if oth != dc:
                all_same = False
                break
        return dc if all_same else (Choice(tuple(done)) if needs else self)


@dataclass(order=True, frozen=True)
class AtLeast(Distribution):
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
    lb: int | float
    #: the inner distribution to sample from
    d: Distribution

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        object.__setattr__(self, "lb", try_int(self.lb))
        dd: Final[Distribution] = self.d
        if isinstance(dd, Const):
            if cast("Const", dd).v < self.lb:
                raise ValueError(f"Invalid distribution {self!r}.")
        elif isinstance(dd, AtLeast):
            dlb: AtLeast = cast("AtLeast", dd)
            object.__setattr__(self, "lb", max(self.lb, dlb.lb))
            object.__setattr__(self, "d", dlb.d)
        if isinstance(dd, In):
            idd: In = cast("In", dd)
            ulb: int | float = max(idd.lb, self.lb)
            if ulb >= idd.ub:
                raise ValueError(f"Invalid distribution {self!r}.")

    def simplify(self) -> "Distribution":
        """
        Try to simplify this distribution.

        :returns: a simplified version of this distribution
        """
        dd: Final[Distribution] = self.d
        if isinstance(dd, Const):
            return dd
        if isinstance(dd, In):
            idd: In = cast("In", dd)
            return In(max(idd.lb, self.lb), idd.ub, idd.d).simplify()
        if (self.lb <= 0) and (isinstance(dd, Exponential | Gamma | Erlang)):
            return dd
        return self

    def sample(self, random: Generator) -> int | float:
        """
        Sample from the lower-bounded distribution.

        :param random: the random number generator
        :return: the result
        """
        s: Final[Callable[[Generator], int | float]] = self.d.sample
        lb: Final[int | float] = self.lb
        for _ in range(_MAX_TRIALS):
            v = s(random)
            if lb <= v:
                return v
        raise ValueError(f"Failed to sample from {self!r}.")


@dataclass(order=True, frozen=True)
class In(Distribution):
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
    lb: int | float
    #: the exclusive upper bound
    ub: int | float
    #: the inner distribution to sample from
    d: Distribution

    def __post_init__(self) -> None:
        """Perform some basic sanity checks and cleanup."""
        object.__setattr__(self, "lb", try_int(self.lb))
        object.__setattr__(self, "ub", try_int(self.ub))
        if not isinstance(self.d, Distribution):
            raise TypeError(f"Invalid types {self}.")
        if self.ub <= self.lb:
            raise ValueError(f"Invalid range {self}.")
        dd: Distribution = self.d
        lb: Final[int | float] = self.lb
        ub: Final[int | float] = self.ub
        if isinstance(dd, In):
            idd: In = cast("In", dd)
            ulb: int | float = max(idd.lb, lb)
            uub: int | float = min(idd.ub, ub)
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
        elif isinstance(dd, Uniform):  # fix a uniform distribution
            udd: Uniform = cast("Uniform", dd)
            ulb = max(udd.low, lb)
            uub = min(udd.high, ub)
            if ulb >= uub:
                raise ValueError(f"Invalid distribution {self!r}.")
            if (ulb != udd.low) or (uub != udd.high):
                object.__setattr__(self, "d", Uniform(ulb, uub))
            object.__setattr__(self, "lb", ulb)
            object.__setattr__(self, "ub", uub)
        if isinstance(dd, Const) and not lb <= cast("Const", dd).v < ub:
            raise ValueError(f"Invalid distribution {self!r}.")

    def sample(self, random: Generator) -> int | float:
        """
        Sample from the lower-bounded distribution.

        :param random: the random number generator
        :return: the result
        """
        s: Final[Callable[[Generator], int | float]] = self.d.sample
        lb: Final[int | float] = self.lb
        ub: Final[int | float] = self.ub
        for _ in range(_MAX_TRIALS):
            v = s(random)
            if lb <= v < ub:
                return v
        raise ValueError(f"Failed to sample from {self!r}.")

    def simplify(self) -> Distribution:
        """
        Simplify this distribution.

        :return: the simplified distribution

        >>> In(-3, 4, Const(3)).simplify()
        Const(v=3)
        >>> ii = In(-10, 10, Uniform(3, 20))
        >>> ii
        In(lb=3, ub=10, d=Uniform(low=3, high=10))
        >>> ii.simplify()
        Uniform(low=3, high=10)
        """
        return self.d if isinstance(self.d, Const | Uniform) else self

    def mean(self) -> int | float:
        """
        Get the mean of this distribution.

        :return: the mean
        """
        return self.d.mean() if isinstance(self.d, Const | Uniform) \
            else super().mean()


def distribution(d: int | float | Distribution) -> Distribution:
    """
    Get the distribution from the parameter.

    :param d: the integer value or distribution
    :return: the canonicalized distribution

    >>> distribution(7)
    Const(v=7)

    >>> distribution(3.4)
    Const(v=3.4)

    >>> distribution(Choice((Const(4.0), )))
    Const(v=4)
    """
    if isinstance(d, int | float):
        return Const(d)
    if isinstance(d, Distribution):
        old_d: Distribution | None = None
        while old_d is not d:
            old_d = d
            d = d.simplify()
            if not isinstance(d, Distribution):
                break
        return d
    raise type_error(d, "d", (Distribution, int))
