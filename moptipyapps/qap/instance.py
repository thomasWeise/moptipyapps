"""
An instance of the Quadratic Assignment Problem.

In this module, we provide the class :class:`~Instance` that encapsulates all
information of a problem instance of the Quadratic Assignment Problem (QAP).
The QAP aims to locate facilities at locations such that the flow-distance
product sum combining a flows of goods between instances with the distances of
the locations becomes minimal. Each instance therefore presents a matrix with
:attr:`~moptipyapps.qap.instance.Instance.distances` and a matrix with flows
:attr:`~moptipyapps.qap.instance.Instance.flows`.

1. Eliane Maria Loiola, Nair Maria Maia de Abreu, Paulo Oswaldo
   Boaventura-Netto, Peter Hahn, and Tania Querido. A survey for the
   Quadratic Assignment Problem. European Journal of Operational Research.
   176(2):657-690. January 2007. https://doi.org/10.1016/j.ejor.2005.09.032.
2. Rainer E. Burkard, Eranda Çela, Panos M. Pardalos, and
   Leonidas S. Pitsoulis. The Quadratic Assignment Problem. In Ding-Zhu Du,
   Panos M. Pardalos, eds., Handbook of Combinatorial Optimization,
   pages 1713-1809, 1998, Springer New York, NY, USA.
   https://doi.org/10.1007/978-1-4613-0303-9_27.


We additionally provide access to several standard QAP benchmark instances
via the :meth:`~Instance.from_resource` and :meth:`~Instance.list_resources`
methods. The standard benchmark instances stem from QAPLIB, a library of QAP
instances, which can be found at <https://qaplib.mgi.polymtl.ca> and
<https://coral.ise.lehigh.edu/data-sets/qaplib>.

1. QAPLIB - A Quadratic Assignment Problem Library. The Websites
   <https://qaplib.mgi.polymtl.ca/> (updated 2018) and
   <https://coral.ise.lehigh.edu/data-sets/qaplib/> (updated 2011), including
   the benchmark instances, on visited 2023-10-21.
2. Rainer E. Burkard, Stefan E. Karisch, and Franz Rendl. QAPLIB - A Quadratic
   Assignment Problem Library. Journal of Global Optimization. 10:391-403,
   1997. https://doi.org/10.1023/A:1008293323270.
"""


from typing import Final, Iterable, cast

import numba  # type: ignore
import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import (
    DEFAULT_UNSIGNED_INT,
    int_range_to_dtype,
    is_np_int,
)
from moptipy.utils.strings import sanitize_name
from pycommons.types import check_int_range, check_to_int_range, type_error

from moptipyapps.qap.qaplib import open_resource_stream


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def trivial_bounds(distances: np.ndarray, flows: np.ndarray) \
        -> tuple[int, int]:
    """
    Compute the trivial bounds for the QAP objective.

    A trivial upper bound is to multiply the largest flow with the largest
    distance, the second-largest flow with the second-largest distance, the
    third-largest flow with the third-largest distance, and so on.

    A trivial lower bound is to multiply the largest flow with the shortest
    distance, the second-largest flow with the second-shortest distance, the
    third-largest flow with the third-shortest distance, and so on.

    :param distances: the distance matrix
    :param flows: the flow matrix
    :return: the lower and upper bounds

    >>> dst = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]])
    >>> flws = np.array([[0, 95, 86], [23, 0, 55], [24, 43, 0]])
    >>> 0*95 + 0*86 + 0*55 + 1*43 + 2*24 + 3*23 + 4*0 + 5*0 + 6*0
    160
    >>> 6*95 + 5*86 + 4*55 + 3*43 + 2*24 + 1*23 + 0*0 + 0*0 + 0*0
    1420
    >>> trivial_bounds(dst, flws)
    (160, 1420)
    """
    n: int = len(distances)
    n *= n
    df_ub: Final[np.ndarray] = np.empty(n, DEFAULT_UNSIGNED_INT)
    df_ub[:] = distances.flatten()
    df_ub.sort()
    df_lb: Final[np.ndarray] = np.empty(n, DEFAULT_UNSIGNED_INT)
    df_lb[:] = df_ub[::-1]
    ff: Final[np.ndarray] = np.empty(n, DEFAULT_UNSIGNED_INT)
    ff[:] = flows.flatten()
    ff.sort()
    return (int(np.multiply(df_lb, ff, df_lb).sum()),
            int(np.multiply(df_ub, ff, df_ub).sum()))


def _flow_or_dist_to_int(val: str) -> int:
    """
    Convert a flow or distance string to an integer.

    :param val: the value
    :return: the integer
    """
    return check_to_int_range(val, "value", 0, 1_000_000_000_000_000)


#: the instances of the QAPLib library
_INSTANCES: Final[tuple[str, ...]] = (
    "bur26a", "bur26b", "bur26c", "bur26d", "bur26e", "bur26f", "bur26g",
    "bur26h", "chr12a", "chr12b", "chr12c", "chr15a", "chr15b", "chr15c",
    "chr18a", "chr18b", "chr20a", "chr20b", "chr20c", "chr22a", "chr22b",
    "chr25a", "els19", "esc16a", "esc16b", "esc16c", "esc16d", "esc16e",
    "esc16f", "esc16g", "esc16h", "esc16i", "esc16j", "esc32a", "esc32b",
    "esc32c", "esc32d", "esc32e", "esc32g", "esc32h", "esc64a", "esc128",
    "had12", "had14", "had16", "had18", "had20", "kra30a", "kra30b", "kra32",
    "lipa20a", "lipa20b", "lipa30a", "lipa30b", "lipa40a", "lipa40b",
    "lipa50a", "lipa50b", "lipa60a", "lipa60b", "lipa70a", "lipa70b",
    "lipa80a", "lipa80b", "lipa90a", "lipa90b", "nug12", "nug14", "nug15",
    "nug16a", "nug16b", "nug17", "nug18", "nug20", "nug21", "nug22", "nug24",
    "nug25", "nug27", "nug28", "nug30", "rou12", "rou15", "rou20", "scr12",
    "scr15", "scr20", "sko42", "sko49", "sko56", "sko64", "sko72", "sko81",
    "sko90", "sko100a", "sko100b", "sko100c", "sko100d", "sko100e", "sko100f",
    "ste36a", "ste36b", "ste36c", "tai12a", "tai12b", "tai15a", "tai15b",
    "tai17a", "tai20a", "tai20b", "tai25a", "tai25b", "tai30a", "tai30b",
    "tai35a", "tai35b", "tai40a", "tai40b", "tai50a", "tai50b", "tai60a",
    "tai60b", "tai64c", "tai80a", "tai80b", "tai100a", "tai100b", "tai150b",
    "tai256c", "tho30", "tho40", "tho150", "wil50", "wil100")

#: the lower bounds provided at <https://qaplib.mgi.polymtl.ca/>
_BOUNDS: Final[dict[str, int]] = {
    "nug14": 1014, "nug15": 1150, "nug16a": 1610, "nug16b": 1240,
    "nug17": 1732, "nug18": 1930, "nug20": 2570, "nug21": 2438, "nug22": 3596,
    "nug24": 3488, "nug25": 3744, "nug27": 5234, "nug28": 5166, "nug30": 6124,
    "rou12": 235528, "rou15": 354210, "rou20": 725522, "scr12": 31410,
    "scr15": 51140, "scr20": 110030, "sko100a": 143846, "sko100b": 145522,
    "sko100c": 139881, "sko100d": 141289, "sko100e": 140893, "sko100f": 140691,
    "sko42": 15332, "sko49": 22650, "sko56": 33385, "sko64": 47017,
    "sko72": 64455, "sko81": 88359, "sko90": 112423, "ste36a": 9526,
    "ste36b": 15852, "ste36c": 8239110, "tai100a": 17853840,
    "tai100b": 1151591000, "tai12a": 224416, "tai12b": 39464925,
    "tai150b": 441786736, "tai15a": 388214, "tai15b": 51765268,
    "tai17a": 491812, "tai20a": 703482, "tai20b": 122455319,
    "tai256c": 44095032, "tai25a": 1167256, "tai25b": 344355646,
    "tai30a": 1706855, "tai30b": 637117113, "tai35a": 2216627,
    "tai35b": 269532400, "tai40a": 2843274, "tai40b": 608808400,
    "tai50a": 4390920, "tai50b": 431090700, "tai60a": 6325978,
    "tai60b": 592371800, "tai64c": 1855928, "tai80a": 11657010,
    "tai80b": 786298800, "tho150": 7620628, "tho30": 149936, "tho40": 226490,
    "wil100": 268955, "wil50": 48121, "bur26a": 5426670, "bur26b": 3817852,
    "bur26c": 5426795, "bur26d": 3821225, "bur26e": 5386879, "bur26f": 3782044,
    "bur26g": 10117172, "bur26h": 7098658, "chr12a": 9552, "chr12b": 9742,
    "chr12c": 11156, "chr15a": 9896, "chr15b": 7990, "chr15c": 9504,
    "chr18a": 11098, "chr18b": 1534, "chr20a": 2192, "chr20b": 2298,
    "chr20c": 14142, "chr22a": 6156, "chr22b": 6194, "chr25a": 3796,
    "els19": 17212548, "esc128": 64, "esc16a": 68, "esc16b": 292,
    "esc16c": 160, "esc16d": 16, "esc16e": 28, "esc16f": 0, "esc16g": 26,
    "esc16h": 996, "esc16i": 14, "esc16j": 8, "esc32a": 130, "esc32b": 168,
    "esc32c": 642, "esc32d": 200, "esc32e": 2, "esc32g": 6, "esc32h": 438,
    "esc64a": 116, "had12": 1652, "had14": 2724, "had16": 3720, "had18": 5358,
    "had20": 6922, "kra30a": 88900, "kra30b": 91420, "kra32": 88700,
    "lipa20a": 3683, "lipa20b": 27076, "lipa30a": 13178, "lipa30b": 151426,
    "lipa40a": 31538, "lipa40b": 476581, "lipa50a": 62093, "lipa50b": 1210244,
    "lipa60a": 107218, "lipa60b": 2520135, "lipa70a": 169755,
    "lipa70b": 4603200, "lipa80a": 253195, "lipa80b": 7763962,
    "lipa90a": 360630, "lipa90b": 12490441, "nug12": 578}

#: The best-known solutions of the QAPLIB instances that are not yet solved to
#: optimality, as of 2024-05-09 on https://qaplib.mgi.polymtl.ca/
_BKS: Final[dict[str, tuple[str, int]]] = {
    "sko90": ("T1991RTSFTQAP,T1995COISFTQAP", 115534),
    "sko100a": ("FF1993GHFTQAP", 152002),
    "sko100b": ("FF1993GHFTQAP", 153890),
    "sko100c": ("FF1993GHFTQAP", 147862),
    "sko100d": ("FF1993GHFTQAP", 149576),
    "sko100e": ("FF1993GHFTQAP", 149150),
    "sko100f": ("FF1993GHFTQAP", 149036),
    "tai30a": ("T1991RTSFTQAP,T1995COISFTQAP", 1818146),
    "tai35a": ("T1991RTSFTQAP,T1995COISFTQAP", 2422002),
    "tai35b": ("T1991RTSFTQAP,T1995COISFTQAP", 283315445),
    "tai40a": ("T1991RTSFTQAP,T1995COISFTQAP", 3139370),
    "tai40b": ("T1991RTSFTQAP,T1995COISFTQAP", 637250948),
    "tai50a": ("M2008AIOTITSAFTQAP", 4938796),
    "tai50b": ("T1991RTSFTQAP,T1995COISFTQAP", 458821517),
    "tai60a": ("M2005ATSAFTQAP", 7205962),
    "tai60b": ("T1991RTSFTQAP,T1995COISFTQAP", 608215054),
    "tai80a": ("M2008AIOTITSAFTQAP", 13499184),
    "tai80b": ("T1991RTSFTQAP,T1995COISFTQAP", 818415043),
    "tai100a": ("M2008AIOTITSAFTQAP", 21044752),
    "tai100b": ("T1991RTSFTQAP,T1995COISFTQAP", 1185996137),
    "tai150b": ("TG1997AMFTQAP", 498896643),
    "tai256c": ("S1997MMASFQAP", 44759294),
    "tho40": ("TB1994AISAAFTQAP", 240516),
    "tho150": ("M2003AMSAAFTQAP", 8133398),
    "wil100": ("FF1993GHFTQAP", 273038),
}


class Instance(Component):
    """An instance of the Quadratic Assignment Problem."""

    def __init__(self, distances: np.ndarray, flows: np.ndarray,
                 lower_bound: int | None = None,
                 upper_bound: int | None = None,
                 name: str | None = None) -> None:
        """
        Create an instance of the quadratic assignment problem.

        :param distances: the distance matrix
        :param flows: the flow matrix
        :param lower_bound: the optional lower bound
        :param upper_bound: the optional upper bound
        :param name: the name of this instance
        """
        super().__init__()
        if not isinstance(distances, np.ndarray):
            raise type_error(distances, "distances", np.ndarray)
        if not isinstance(flows, np.ndarray):
            raise type_error(flows, "flows", np.ndarray)
        shape: tuple[int, ...] = distances.shape
        if len(shape) != 2:
            raise ValueError("distance matrix must have two dimensions, but "
                             f"has {len(shape)}, namely {shape}.")
        if shape[0] != shape[1]:
            raise ValueError(
                f"distance matrix must be square, but has shape {shape}.")
        if not is_np_int(distances.dtype):
            raise ValueError("distance matrix must be integer, but has "
                             f"dtype {distances.dtype}.")
        if shape != flows.shape:
            raise ValueError(
                f"flow matrix has shape {flows.shape} and distance matrix has"
                f" shape {shape}, which are different.")
        if not is_np_int(flows.dtype):
            raise ValueError(
                f"flow matrix must be integer, but has dtype {flows.dtype}.")

        lb, ub = trivial_bounds(distances, flows)
        if lower_bound is not None:
            lb = max(lb, check_int_range(
                lower_bound, "lower_bound", 0, 1_000_000_000_000_000))
        if upper_bound is not None:
            ub = min(ub, check_int_range(
                upper_bound, "upper_bound", 0, 1_000_000_000_000_000))
        if lb > ub:
            raise ValueError(f"lower bound = {lb} > upper_bound = {ub}!")
        dtype: Final[np.dtype] = int_range_to_dtype(min_value=0, max_value=ub)
        #: the scale of the problem
        self.n: Final[int] = shape[0]
        if name is None:
            name = f"qap{self.n}_{lb}_{ub}"
        else:
            sn: Final[str] = sanitize_name(name)
            if name != sn:
                raise ValueError(f"name={name!r} sanitizes to {sn!r}.")
        #: the name of this instance
        self.name: Final[str] = name
        #: the distance matrix
        self.distances: Final[np.ndarray] = \
            distances.astype(dtype) if distances.dtype != dtype else distances
        #: the flows
        self.flows: Final[np.ndarray] = \
            flows.astype(dtype) if flows.dtype != dtype else flows
        #: the lower bound for the QAP objective
        self.lower_bound: Final[int] = lb
        #: the upper bound for the QAP objective
        self.upper_bound: Final[int] = ub

    def __str__(self):
        """
        Get the name of this instance.

        :return: :attr:`~name`
        """
        return self.name

    def bks(self) -> tuple[str, int]:
        """
        Get the best-known solution, if known, the optimum, or lower bound.

        A tuple with a string identifying the source of the value and a value
        corresponding to the best-known solution:

        - `("OPT", xxx)`: the problem instance has been solved to optimality
          and `xxx` is the objective value of the optimum
        - `("LB", xxx)`: neither the optimum nor a best-known solution are
          available for this instance, so we return the lower bound

        The data is based on https://qaplib.mgi.polymtl.ca/, visited on
        2024-05-09. The following sources are included:

        - "FF1993GHFTQAP": Charles Fleurent and Jacques A. Ferland. Genetic
          Hybrids for the Quadratic Assignment Problem. In Panos M. Pardalos
          and Henry Wolkowicz, eds, *Quadratic Assignment and Related
          Problems, Proceedings of a DIMACS Workshop,* May 20-21, 1993.
          pages 173-187. Providence, RI, USA: American Mathematical Society.
        - "M2003AMSAAFTQAP": Alfonsas Misevičius, A Modified Simulated
          Annealing Algorithm for the Quadratic Assignment Problem.
          *Informatica* 14(4):497-514. January 2003.
          https://doi.org/10.15388/Informatica.2003.037.
        - "M2005ATSAFTQAP": Alfonsas Misevičius. A Tabu Search Algorithm for
          the Quadratic Assignment Problem.
          *Computational Optimization and Applications* 30(1):95-111. 2005.
          https://doi.org/10.1007/s10589-005-4562-x.
        - "M2008AIOTITSAFTQAP": Alfonsas Misevičius. An Implementation of
          the Iterated Tabu Search Algorithm for the Quadratic Assignment
          Problem. Working Paper. 2008. Kaunas, Lithuania: Kaunas University
          of Technology.
        - "S1997MMASFQAP": Thomas Stützle. MAX-MIN Ant System for Quadratic
          Assignment Problems. Research Report AIDA-97-04. 1997. Darmstadt,
          Germany: Department of Computer Schience, Darmstadt University of
          Technology.
        - "T1991RTSFTQAP": Éric Taillard. Robust Taboo Search for the
          Quadratic Assignment Problem. *Parallel Computing.*
          17(4-5):443-455. July 1991.
        - "T1995COISFTQAP": Éric D. Taillard. Comparison of Iterative
          Searches for the Quadratic Assignment Problem. *Location Science*
          3(2):87-105. 1995.
        - "TB1994AISAAFTQAP": Ulrich Wilhelm Thonemann and Andreas Bölte.
          An Improved Simulated Annealing Algorithm for the Quadratic
          Assignment Problem. Working Paper 1994. Paderborn, Germany: School
          of Business, Department of Production and Operations Research,
          University of Paderborn.
        - "TG1997AMFTQAP": Éric D. Taillard and Luca-Maria Gambardella.
          Adaptive Memories for the Quadratic Assignment Problem. 1997.
          Technical Report IDSIA-87-97. Lugano, Switzerland: IDSIA.

        :return: a `tuple[str, int]` with the objective value of the
            best-known (if any is available), the optimum, or the lower bound.
        """
        name: Final[str] = self.name
        if name in _BKS:
            return _BKS[name]
        if name in _BOUNDS:
            return "OPT", _BOUNDS[name]
        return "LB", self.lower_bound

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this instance as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("n", self.n)
        logger.key_value("qapLowerBound", self.lower_bound)
        logger.key_value("qapUpperBound", self.upper_bound)

    @staticmethod
    def from_qaplib_stream(stream: Iterable[str],
                           lower_bound: int | None = None,
                           upper_bound: int | None = None,
                           name: str | None = None) -> "Instance":
        """
        Load an instance from a QAPLib-formatted stream.

        :param stream: the stream to load the data from
        :param lower_bound: the optional lower bound
        :param upper_bound: the optional upper bound
        :param name: the name of this instance
        :return: the instance

        >>> ins = Instance.from_qaplib_stream([
        ...         "4", "",
        ...         "1 2 3 4 5 6 7 8 9 10 11 12 13",
        ...         "   14    15   16  ", "",
        ...         "17 18 19 20 21 22 23 24 25 26     27",
        ...         " 28   29 30 31   32"])
        >>> ins.distances
        array([[17, 18, 19, 20],
               [21, 22, 23, 24],
               [25, 26, 27, 28],
               [29, 30, 31, 32]], dtype=int16)
        >>> ins.flows
        array([[ 1,  2,  3,  4],
               [ 5,  6,  7,  8],
               [ 9, 10, 11, 12],
               [13, 14, 15, 16]], dtype=int16)
        >>> ins.lower_bound
        2992
        >>> ins.upper_bound
        3672
        >>> ins.name
        'qap4_2992_3672'
        """
        state: int = 0
        n: int | None = None
        n2: int = -1
        flows: list[int] = []
        dists: list[int] = []
        for oline in stream:
            line = oline.strip()
            if len(line) <= 0:
                continue
            if state == 0:
                n = check_to_int_range(line, "n", 1, 1_000_000)
                n2 = n * n
                state = 1
            else:
                row: Iterable[int] = map(_flow_or_dist_to_int, line.split())
                if state == 1:
                    flows.extend(row)
                    if len(flows) >= n2:
                        state = 2
                        continue
                dists.extend(row)
                if len(dists) >= n2:
                    state = 3
                    break

        if (n is None) or (n <= 0):
            raise ValueError(f"Invalid or unspecified size n={n}.")
        lll: int = len(flows)
        if lll != n2:
            raise ValueError(
                f"Invalid number of flows {lll}, should be n²={n}²={n2}.")
        lll = len(dists)
        if lll != n2:
            raise ValueError(
                f"Invalid number of distances {lll}, should be n²={n}²={n2}.")
        if state != 3:
            raise ValueError(f"Stream is incomplete, state={state}.")

        return Instance(np.array(dists, DEFAULT_UNSIGNED_INT).reshape((n, n)),
                        np.array(flows, DEFAULT_UNSIGNED_INT).reshape((n, n)),
                        lower_bound, upper_bound, name)

    @staticmethod
    def list_resources() -> tuple[str, ...]:
        """
        Get a tuple of all the QAP-lib instances available as resource.

        The original data can be found at <https://qaplib.mgi.polymtl.ca> and
        <https://coral.ise.lehigh.edu/data-sets/qaplib>.

        :return: the tuple with the instance names

        >>> len(Instance.list_resources())
        134
        """
        return _INSTANCES

    @staticmethod
    def from_resource(name: str) -> "Instance":
        """
        Load a QAP-Lib instance from a resource.

        The original data  can be found at <https://qaplib.mgi.polymtl.ca> and
        <https://coral.ise.lehigh.edu/data-sets/qaplib>.

        :param name: the name string
        :return: the instance

        >>> insta = Instance.from_resource("chr25a")
        >>> print(insta.n)
        25
        >>> print(insta.name)
        chr25a
        >>> print(insta.lower_bound)
        3796
        >>> print(insta.upper_bound)
        50474
        """
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        container: Final = Instance.from_resource
        inst_attr: Final[str] = f"__inst_{name}"
        if hasattr(container, inst_attr):  # instance loaded?
            return cast("Instance", getattr(container, inst_attr))

        lb: int | None = _BOUNDS.get(name, None)
        with open_resource_stream(f"{name}.dat") as stream:
            inst: Final[Instance] = Instance.from_qaplib_stream(
                stream, name=name, lower_bound=lb)

        if inst.n <= 1000:
            setattr(container, inst_attr, inst)
        return inst
