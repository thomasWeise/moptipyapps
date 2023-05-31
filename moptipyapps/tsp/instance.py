"""
An instance of the Traveling Salesperson Problem (TSP) as distance matrix.

An instance of the Traveling Salesperson Problem (TSP) is defined as a
fully-connected graph with :attr:`Instance.n_cities` nodes. Each edge in the
graph has a weight, which identifies the distance between the nodes. The goal
is to find the *shortest* tour that visits every single node in the graph
exactly once and then returns back to its starting node. Then nodes are
usually called cities. In this file, we present methods for loading instances
of the TSP as distance matrices `A`. In other words, the value at `A[i, j]`
identifies the travel distance from `i` to `j`.

We can load files in a subset of the TSPLib format [1, 2] and also include the
instances of TSPLib with no more than 2000 cities. We load instances as full
distance matrices, which makes writing algorithms to solve them easier but
handling instances with more than 2000 cities problematic due to the high
memory consumption. Of course, you could still load them, but it would be
ill-advised to do so on a normal PC (at the time of this writing).

The TSPLib contains many instances of the symmetric TSP, where the distance
`A[i, j]` from city `i` to city `j` is the same as the distance `A[j, i]` from
`j` to `i`. Here, we provide 110 of them as resource. The cities of some of
these instances are points in the Euclidean plain and the distances are the
(approximate) Euclidean distances. Others use geographical coordinates
(longitude/latitude), and yet others are provides as distances matrices
directly. We also provide 19 of the asymmetric TSPLib instances, where the
distance `A[i, j]` from `i` to `j` may be different from the distance
`A[j, i]` from `j` to `i`.

You can obtain lists of either all, only the symmetric, or only the asymmetric
instance resources via

>>> print(Instance.list_resources()[0:10])  # get all instances (1..10)
('a280', 'ali535', 'att48', 'att532', 'bayg29', 'bays29', 'berlin52', \
'bier127', 'br17', 'brazil58')

>>> print(Instance.list_resources(asymmetric=False)[0:10])  # only symmetric
('a280', 'ali535', 'att48', 'att532', 'bayg29', 'bays29', 'berlin52', \
'bier127', 'brazil58', 'brg180')

>>> print(Instance.list_resources(symmetric=False)[0:10])  # only asymmetric
('br17', 'ft53', 'ft70', 'ftv170', 'ftv33', 'ftv35', 'ftv38', 'ftv44', \
'ftv47', 'ftv55')

You can load any of these instances from the resources via
:meth:`Instance.from_resource` as follows:

>>> inst = Instance.from_resource("a280")
>>> print(inst.n_cities)
280

If you want to read an instance from an external TSPLib file, you can use
:meth:`Instance.from_file`. Be aware that not the whole TSPLib standard is
supported right now, but only a reasonably large subset.

Every TSP instance automatically provides a lower bound
:attr:`Instance.tour_length_lower_bound` and an upper bound
:attr:`Instance.tour_length_upper_bound` of the lengths of valid tours.
For the TSPLib instances, the globally optimal solutions and their tour
lengths are known, so we can use them as lower bounds directly. Otherwise,
we currently use a very crude approximation: We assume that, for each city
`i`, the next city `j` to be visited would be the nearest neighbor of `i`.
Of course, in reality, such a tour may not be feasible, as it could contain
disjoint sets of loops. But no tour can be shorter than this.
As upper bound, we do the same but assume that `j` would be the cities
farthest away from `i`.

>>> print(inst.tour_length_lower_bound)
2579
>>> print(inst.tour_length_upper_bound)
65406

It should be noted that all TSPLib instances by definition have integer
distances. This means that there are never floating point distances and, for
example, Euclidean distances are rounded and are only approximate Euclidean.
Then again, since floating point numbers could also only represent things such
as `sqrt(2)` approximately, using integers instead of floating point numbers
does not really change anything - distances would be approximately Euclidean
or approximately geographical either way.

TSPLib also provides some known optimal solutions in path representation,
i.e., as permutations of the numbers `0` to `n_cities-1`. The optimal
solutions corresponding to the instances provides as resources can be obtained
via :mod:`~moptipyapps.tsp.known_optima`.

The original data of TSPLib can be found at
<http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>. Before doing
anything with these data directly, you should make sure to read the FAQ
<http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html> and the
documentation
<http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf>.

Important initial work on this code has been contributed by Mr. Tianyu LIANG
(梁天宇), <liangty@stu.hfuu.edu.cn> a Master's student at the Institute of
Applied Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School
of Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei
University (合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the
supervision of Prof. Dr. Thomas Weise (汤卫思教授).

1. Gerhard Reinelt. TSPLIB - A Traveling Salesman Problem Library.
   *ORSA Journal on Computing* 3(4):376-384. November 1991.
   https://doi.org/10.1287/ijoc.3.4.376.
   http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
2. Gerhard Reinelt. *TSPLIB95.* Heidelberg, Germany: Universität
   Heidelberg, Institut für Angewandte Mathematik. 1995.
   http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
3. David Lee Applegate, Robert E. Bixby, Vašek Chvátal, and William John Cook.
   *The Traveling Salesman Problem: A Computational Study.* Second Edition,
   2007. Princeton, NJ, USA: Princeton University Press. Volume 17 of
   Princeton Series in Applied Mathematics. ISBN: 0-691-12993-2.
4. Gregory Z. Gutin and Abraham P. Punnen. *The Traveling Salesman Problem and
   its Variations.* 2002. Kluwer Academic Publishers. Volume 12 of
   Combinatorial Optimization. https://doi.org/10.1007/b101971.
5. Pedro Larrañaga, Cindy M. H. Kuijpers, Roberto H. Murga, Iñaki Inza, and
   Sejla Dizdarevic. Genetic Algorithms for the Travelling Salesman Problem: A
   Review of Representations and Operators. *Artificial Intelligence Review*
   13(2):129-170. April 1999. https://doi.org/10.1023/A:1006529012972.
6. Eugene Leighton Lawler, Jan Karel Lenstra, Alexander Hendrik George Rinnooy
   Kan, and David B. Shmoys. *The Traveling Salesman Problem: A Guided Tour of
   Combinatorial Optimization.* September 1985. Chichester, West Sussex, UK:
   Wiley Interscience. In Estimation, Simulation, and
   Control - Wiley-Interscience Series in Discrete Mathematics and
   Optimization. ISBN: 0-471-90413-9
7. Tianyu Liang, Zhize Wu, Jörg Lässig, Daan van den Berg, and Thomas Weise.
   Solving the Traveling Salesperson Problem using Frequency Fitness
   Assignment. In *Proceedings of the IEEE Symposium on Foundations of
   Computational Intelligence (IEEE FOCI'22),* part of the *IEEE Symposium
   Series on Computational Intelligence (SSCI'22),* December 4-7, 2022,
   Singapore. Pages 360-367. IEEE.
   https://doi.org/10.1109/SSCI51031.2022.10022296.
8. Thomas Weise, Raymond Chiong, Ke Tang, Jörg Lässig, Shigeyoshi Tsutsui,
   Wenxiang Chen, Zbigniew Michalewicz, and Xin Yao. Benchmarking Optimization
   Algorithms: An Open Source Framework for the Traveling Salesman Problem.
   *IEEE Computational Intelligence Magazine.* 9(3):40-52. August 2014.
   https://doi.org/10.1109/MCI.2014.2326101.
"""

from math import acos, cos, isfinite, sqrt
from typing import Any, Callable, Final, TextIO, cast

import moptipy.utils.nputils as npu
import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.path import Path
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import check_int_range, check_to_int_range, type_error

from moptipyapps.tsp.tsplib import open_resource_stream

#: the known optimal tour lengths and lower bounds of TSP Lib
_LOWER_BOUNDS: dict[str, int] = {
    "a280": 2579, "ali535": 202339, "att48": 10628, "att532": 27686,
    "bayg29": 1610, "bays29": 2020, "berlin52": 7542, "bier127": 118282,
    "br17": 39, "brazil58": 25395, "brd14051": 469385, "brg180": 1950,
    "burma14": 3323, "ch130": 6110, "ch150": 6528, "d1291": 50801,
    "d15112": 1573084, "d1655": 62128, "d18512": 645238, "d198": 15780,
    "d2103": 80450, "d493": 35002, "d657": 48912, "dantzig42": 699,
    "dsj1000": 18660188, "eil101": 629, "eil51": 426, "eil76": 538,
    "fl1400": 20127, "fl1577": 22249, "fl3795": 28772, "fl417": 11861,
    "fnl4461": 182566, "fri26": 937, "ft53": 6905, "ft70": 38673,
    "ftv100": 1788, "ftv110": 1958, "ftv120": 2166, "ftv130": 2307,
    "ftv140": 2420, "ftv150": 2611, "ftv160": 2683, "ftv170": 2755,
    "ftv33": 1286, "ftv35": 1473, "ftv38": 1530, "ftv44": 1613, "ftv47": 1776,
    "ftv55": 1608, "ftv64": 1839, "ftv70": 1950, "ftv90": 1579,
    "gil262": 2378, "gr120": 6942, "gr137": 69853, "gr17": 2085,
    "gr202": 40160, "gr21": 2707, "gr229": 134602, "gr24": 1272,
    "gr431": 171414, "gr48": 5046, "gr666": 294358, "gr96": 55209,
    "hk48": 11461, "kro124p": 36230, "kroA100": 21282, "kroA150": 26524,
    "kroA200": 29368, "kroB100": 22141, "kroB150": 26130, "kroB200": 29437,
    "kroC100": 20749, "kroD100": 21294, "kroE100": 22068, "lin105": 14379,
    "lin318": 42029, "nrw1379": 56638, "p43": 5620, "p654": 34643,
    "pa561": 2763, "pcb1173": 56892, "pcb3038": 137694, "pcb442": 50778,
    "pla33810": 66048945, "pla7397": 23260728, "pla85900": 142382641,
    "pr1002": 259045, "pr107": 44303, "pr124": 59030, "pr136": 96772,
    "pr144": 58537, "pr152": 73682, "pr226": 80369, "pr2392": 378032,
    "pr264": 49135, "pr299": 48191, "pr439": 107217, "pr76": 108159,
    "rat195": 2323, "rat575": 6773, "rat783": 8806, "rat99": 1211,
    "rbg323": 1326, "rbg358": 1163, "rbg403": 2465, "rbg443": 2720,
    "rd100": 7910, "rd400": 15281, "rl11849": 923288, "rl1304": 252948,
    "rl1323": 270199, "rl1889": 316536, "rl5915": 565530, "rl5934": 556045,
    "ry48p": 14422, "si1032": 92650, "si175": 21407, "si535": 48450,
    "st70": 675, "swiss42": 1273, "ts225": 126643, "tsp225": 3916,
    "u1060": 224094, "u1432": 152970, "u159": 42080, "u1817": 57201,
    "u2152": 64253, "u2319": 234256, "u574": 36905, "u724": 41910,
    "ulysses16": 6859, "ulysses22": 7013, "usa13509": 19982859,
    "vm1084": 239297, "vm1748": 336556}

#: the TSPLib instances
_SYMMETRIC_INSTANCES: Final[tuple[str, ...]] = (
    "a280", "ali535", "att48", "att532", "bayg29", "bays29", "berlin52",
    "bier127", "brazil58", "brg180", "burma14", "ch130", "ch150", "d1291",
    "d1655", "d198", "d493", "d657", "dantzig42", "eil101", "eil51", "eil76",
    "fl1400", "fl1577", "fl417", "fri26", "gil262", "gr120", "gr137", "gr17",
    "gr202", "gr21", "gr229", "gr24", "gr431", "gr48", "gr666", "gr96",
    "hk48", "kroA100", "kroA150", "kroA200", "kroB100", "kroB150", "kroB200",
    "kroC100", "kroD100", "kroE100", "lin105", "lin318", "nrw1379", "p654",
    "pcb1173", "pcb442", "pr1002", "pr107", "pr124", "pr136", "pr144",
    "pr152", "pr226", "pr264", "pr299", "pr439", "pr76", "rat195", "rat575",
    "rat783", "rat99", "rd100", "rd400", "rl1304", "rl1323", "rl1889",
    "si1032", "si175", "si535", "st70", "swiss42", "ts225", "tsp225", "u1060",
    "u1432", "u159", "u1817", "u574", "u724", "ulysses16", "ulysses22",
    "vm1084", "vm1748")

#: The set of asymmetric instances
_ASYMMETRIC_INSTANCES: Final[tuple[str, ...]] = (
    "br17", "ft53", "ft70", "ftv170", "ftv33", "ftv35", "ftv38", "ftv44",
    "ftv47", "ftv55", "ftv64", "ftv70", "kro124p", "p43", "rbg323", "rbg358",
    "rbg403", "rbg443", "ry48p")

#: The set of all TSP instances
_INSTANCES: Final[tuple[str, ...]] = tuple(sorted(
    _SYMMETRIC_INSTANCES + _ASYMMETRIC_INSTANCES))

#: the set of asymmetric resources
_ASYMMETRIC_RESOURCES: Final[set[str]] = set(_ASYMMETRIC_INSTANCES)

#: the problem is a symmetric tsp
__TYPE_SYMMETRIC_TSP: Final[str] = "TSP"
#: the problem is an asymmetric tsp
__TYPE_ASYMMETRIC_TSP: Final[str] = "ATSP"
#: the permitted types
__TYPES: Final[set[str]] = {__TYPE_SYMMETRIC_TSP, __TYPE_ASYMMETRIC_TSP}
#: the name start
__KEY_NAME: Final[str] = "NAME"
#: the type start
__KEY_TYPE: Final[str] = "TYPE"
#: the dimension start
__KEY_DIMENSION: Final[str] = "DIMENSION"
#: the comment start
__KEY_COMMENT: Final[str] = "COMMENT"
#: EUC_2D coordinates
__EWT_EUC_2D: Final[str] = "EUC_2D"
#: geographical coordinates
__EWT_GEO: Final[str] = "GEO"
#: ATT coordinates
__EWT_ATT: Final[str] = "ATT"
#: ceiling 2D coordinates
__EWT_CEIL2D: Final[str] = "CEIL_2D"
#: the explicit edge weight type
__EWT_EXPLICIT: Final[str] = "EXPLICIT"
#: the edge weight type start
__KEY_EDGE_WEIGHT_TYPE: Final[str] = "EDGE_WEIGHT_TYPE"
#: the permitted edge weight types
__EDGE_WEIGHT_TYPES: Final[set[str]] = {
    __EWT_EUC_2D, __EWT_GEO, __EWT_ATT, __EWT_CEIL2D, __EWT_EXPLICIT}
#: the edge weight format "function"
__EWF_FUNCTION: Final[str] = "FUNCTION"
#: the full matrix edge weight format
__EWF_FULL_MATRIX: Final[str] = "FULL_MATRIX"
#: the upper row edge weight format
__EWF_UPPER_ROW: Final[str] = "UPPER_ROW"
#: the lower diagonal row
__EWF_LOWER_DIAG_ROW: Final[str] = "LOWER_DIAG_ROW"
#: the upper diagonal row
__EWF_UPPER_DIAG_ROW: Final[str] = "UPPER_DIAG_ROW"
#: the edge weight format start
__KEY_EDGE_WEIGHT_FORMAT: Final[str] = "EDGE_WEIGHT_FORMAT"
#: the permitted edge weight formats
__EDGE_WEIGHT_FORMATS: Final[set[str]] = {
    __EWF_FUNCTION, __EWF_FULL_MATRIX, __EWF_UPPER_ROW,
    __EWF_LOWER_DIAG_ROW, __EWF_UPPER_DIAG_ROW}
#: the start of the node coord type
__KEY_NODE_COORD_TYPE: Final[str] = "NODE_COORD_TYPE"
#: 2d coordinates
__NODE_COORD_TYPE_2D: Final[str] = "TWOD_COORDS"
#: no coordinates
__NODE_COORD_TYPE_NONE: Final[str] = "NO_COORDS"
#: the permitted node coordinate types
__NODE_COORD_TYPES: Final[set[str]] = {
    __NODE_COORD_TYPE_2D, __NODE_COORD_TYPE_NONE}
#: the node coordinate section starts
__START_NODE_COORD_SECTION: Final[str] = "NODE_COORD_SECTION"
#: start the edge weight section
__START_EDGE_WEIGHT_SECTION: Final[str] = "EDGE_WEIGHT_SECTION"
#: the end of the file
__EOF: Final[str] = "EOF"
#: the fixed edges section
__FIXED_EDGES: Final[str] = "FIXED_EDGES_SECTION"


def __line_to_nums(line: str,
                   collector: Callable[[int | float], Any]) -> None:
    """
    Split a line to a list of integers.

    :param line: the line string
    :param collector: the collector
    """
    if not isinstance(line, str):
        raise type_error(line, "line", str)
    idx: int = 0
    line = line.strip()
    str_len: Final[int] = len(line)

    while idx < str_len:
        while (idx < str_len) and (line[idx] == " "):
            idx += 1
        if idx >= str_len:
            return

        next_space = line.find(" ", idx)
        if next_space < 0:
            next_space = str_len

        part: str = line[idx:next_space]
        if ("." in part) or ("E" in part) or ("e" in part):
            f: float = float(part)
            if not isfinite(f):
                raise ValueError(
                    f"{part!r} translates to non-finite float {f}.")
            collector(f)
        else:
            collector(check_to_int_range(
                part, "line fragment", -1_000_000_000_000, 1_000_000_000_000))
        idx = next_space


def __nint(v: int | float) -> int:
    """
    Get the nearest integer in the TSPLIB format.

    :param v: the value
    :return: the integer
    """
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(0.5 + v)
    raise type_error(v, "value", (int, float))


def __dist_2deuc(a: list[int | float], b: list[int | float]) -> int:
    """
    Compute the two-dimensional Euclidean distance function.

    :param a: the first point
    :param b: the second point
    :return: the distance
    """
    return __nint(sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2)))


def __matrix_from_points(
        n_cities: int, coord_dim: int, stream: TextIO,
        dist_func: Callable[[list[int | float], list[int | float]], int]) \
        -> np.ndarray:
    """
    Load a distance matrix from 2D coordinates.

    :param n_cities: the dimension
    :param coord_dim: the coordinate dimension
    :param stream: the stream
    :param dist_func: the distance function
    :return: the matrix
    """
    index: int = 0
    coordinates: Final[list[list[int | float]]] = []
    row: Final[list[int | float]] = []
    for the_line in stream:
        if not isinstance(the_line, str):
            raise type_error(the_line, "line", str)
        line = the_line.strip()
        if len(line) <= 0:
            continue
        if line == __EOF:
            break
        __line_to_nums(line, row.append)
        index += 1
        if (len(row) != (coord_dim + 1)) or (not isinstance(row[0], int)) \
                or (row[0] != index):
            raise ValueError(f"invalid row {line!r} at index {index}, "
                             f"gives values {row}.")
        coordinates.append(row[1:])
        row.clear()
    if index != n_cities:
        raise ValueError(f"only found {index} rows, but expected {n_cities}")

    # now construct the matrix
    matrix: Final[np.ndarray] = np.zeros((n_cities, n_cities),
                                         npu.DEFAULT_INT)
    for i in range(n_cities):
        a = coordinates[i]
        for j in range(i):
            b = coordinates[j]
            dist = dist_func(a, b)
            if not isinstance(dist, int):
                raise type_error(dist, f"distance[{i},{j}]", int)
            if dist < 0:
                raise ValueError(
                    f"Cannot have node distance {dist}: ({a}) at index="
                    f"{i + 1} and ({b} at index={j + 1}.")
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix


def __coord_to_rad(x: int | float) -> float:
    """
    Convert coordinates to longitude/latitude.

    :param x: the coordinate
    """
    degrees: int = int(x)
    return (3.141592 * (degrees + (5.0 * (x - degrees)) / 3.0)) / 180.0


def __dist_loglat(a: list[int | float], b: list[int | float]) -> int:
    """
    Convert a longitude-latitude pair to a distance.

    :param a: the first point
    :param b: the second point
    :return: the distance
    """
    lat1: float = __coord_to_rad(a[0])
    long1: float = __coord_to_rad(a[1])
    lat2: float = __coord_to_rad(b[0])
    long2: float = __coord_to_rad(b[1])
    q1: Final[float] = cos(long1 - long2)
    q2: Final[float] = cos(lat1 - lat2)
    q3: Final[float] = cos(lat1 + lat2)
    return int(6378.388 * acos(0.5 * (
        (1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)


def __dist_att(a: list[int | float], b: list[int | float]) -> int:
    """
    Compute the ATT Pseudo-Euclidean distance function.

    :param a: the first point
    :param b: the second point
    :return: the distance
    """
    xd: Final[int | float] = a[0] - b[0]
    yd: Final[int | float] = a[1] - b[1]
    rij: Final[float] = sqrt((xd * xd + yd * yd) / 10.0)
    tij: Final[int] = __nint(rij)
    return (tij + 1) if tij < rij else tij


def __dist_2dceil(a: list[int | float], b: list[int | float]) -> int:
    """
    Compute the ceiling of the two-dimensional Euclidean distance function.

    :param a: the first point
    :param b: the second point
    :return: the distance
    """
    dist: Final[float] = sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
    disti: Final[int] = int(dist)
    return disti if dist == disti else (disti + 1)


def _matrix_from_node_coord_section(
        n_cities: int | None, edge_weight_type: str | None,
        node_coord_type: str | None, stream: TextIO) -> np.ndarray:
    """
    Get a data matrix from a node coordinate section.

    :param n_cities: the dimension
    :param edge_weight_type: the edge weight type
    :param node_coord_type: the node coordinate type
    :param stream: the node coordinate stream
    :return: the data matrix
    """
    check_int_range(n_cities, "n_cities", 2, 1_000_000_000_000)
    if (node_coord_type is not None) and \
            (not isinstance(node_coord_type, str)):
        raise type_error(node_coord_type, "node_coord_type", str)
    if not isinstance(edge_weight_type, str):
        raise type_error(edge_weight_type, "edge_weight_type", str)

    dist_fun = None
    coord_dim: int | None = None
    if (edge_weight_type == __EWT_EUC_2D) \
            and (node_coord_type in (None, __NODE_COORD_TYPE_2D)):
        coord_dim = 2
        dist_fun = __dist_2deuc
    elif (edge_weight_type == __EWT_GEO) \
            and (node_coord_type in (None, __NODE_COORD_TYPE_2D)):
        coord_dim = 2
        dist_fun = __dist_loglat
    elif (edge_weight_type == __EWT_ATT) \
            and (node_coord_type in (None, __NODE_COORD_TYPE_2D)):
        coord_dim = 2
        dist_fun = __dist_att
    elif (edge_weight_type == __EWT_CEIL2D) \
            and (node_coord_type in (None, __NODE_COORD_TYPE_2D)):
        coord_dim = 2
        dist_fun = __dist_2dceil

    if (coord_dim is not None) and (dist_fun is not None):
        return __matrix_from_points(n_cities, coord_dim, stream, dist_fun)

    raise ValueError(f"invalid combination of {__KEY_EDGE_WEIGHT_TYPE} "
                     f"and {__KEY_NODE_COORD_TYPE}")


def __read_n_ints(n: int, stream: TextIO) -> list[int]:
    """
    Read exactly `n` integers from a stream.

    :param n: the number of integers to read
    :param stream: the stream to read from
    :return: the list of integers
    """
    res: list[int] = []

    def __append(i: int | float, fwd=res.append) -> None:
        if isinstance(i, int):
            fwd(i)
        else:
            i2 = int(i)
            if i2 != i:
                raise ValueError(f"{i} is not integer")
            fwd(i2)

    for line in stream:
        __line_to_nums(line, __append)
        if len(res) == n:
            break

    if len(res) != n:
        raise ValueError(f"expected {n} integers, got {len(res)}.")
    return res


def _matrix_from_edge_weights(
        n_cities: int | None, edge_weight_type: str | None,
        edge_weight_format: str | None, stream: TextIO) -> np.ndarray:
    """
    Get a data matrix from a edge weights section.

    :param n_cities: the dimension
    :param edge_weight_type: the edge weight type
    :param edge_weight_format: the edge weight format
    :param stream: the node coordinate stream
    :return: the data matrix
    """
    check_int_range(n_cities, "n_cities", 2, 1_000_000_000_000)
    if not isinstance(edge_weight_type, str):
        raise type_error(edge_weight_type, "node_coord_type", str)
    if not isinstance(edge_weight_type, str):
        raise type_error(edge_weight_type, "edge_weight_type", str)
    if edge_weight_type == __EWT_EXPLICIT:
        if edge_weight_format == __EWF_FULL_MATRIX:
            res = np.array(__read_n_ints(n_cities * n_cities, stream),
                           dtype=npu.DEFAULT_INT).reshape(
                (n_cities, n_cities))
            np.fill_diagonal(res, 0)
            return res
        if edge_weight_format == __EWF_UPPER_ROW:
            ints = __read_n_ints((n_cities * (n_cities - 1)) // 2, stream)
            res = np.zeros((n_cities, n_cities), dtype=npu.DEFAULT_INT)
            i: int = 1
            j: int = 0
            for v in ints:
                res[j, i] = res[i, j] = v
                i = i + 1
                if i >= n_cities:
                    j = j + 1
                    i = j + 1
            return res
        if edge_weight_format == __EWF_LOWER_DIAG_ROW:
            ints = __read_n_ints(
                n_cities + ((n_cities * (n_cities - 1)) // 2), stream)
            res = np.zeros((n_cities, n_cities), dtype=npu.DEFAULT_INT)
            i = 0
            j = 0
            for v in ints:
                if i != j:
                    res[j, i] = res[i, j] = v
                i = i + 1
                if i > j:
                    j = j + 1
                    i = 0
            return res
        if edge_weight_format == __EWF_UPPER_DIAG_ROW:
            ints = __read_n_ints(
                n_cities + ((n_cities * (n_cities - 1)) // 2), stream)
            res = np.zeros((n_cities, n_cities), dtype=npu.DEFAULT_INT)
            i = 0
            j = 0
            for v in ints:
                if i != j:
                    res[j, i] = res[i, j] = v
                i = i + 1
                if i >= n_cities:
                    j = j + 1
                    i = 0
            return res
    raise ValueError(
        f"unsupported combination of {__KEY_EDGE_WEIGHT_TYPE}="
        f"{edge_weight_type!r} and {__KEY_EDGE_WEIGHT_FORMAT}="
        f"{edge_weight_format!r}")


def _from_stream(
        stream: TextIO,
        lower_bound_getter: Callable[[str], int] | None =
        _LOWER_BOUNDS.__getitem__) -> "Instance":
    """
    Read a TSP Lib instance from a TSP-lib formatted stream.

    :param stream: the text stream
    :param lower_bound_getter: a function returning a lower bound for an
        instance name, or `None` to use a simple lower bound approximation
    :return: the instance
    """
    if (lower_bound_getter is not None) \
            and (not callable(lower_bound_getter)):
        raise type_error(
            lower_bound_getter, "lower_bound_getter", None, call=True)

    the_name: str | None = None
    the_type: str | None = None
    the_n_cities: int | None = None
    the_ewt: str | None = None
    the_ewf: str | None = None
    the_nct: str | None = None
    the_matrix: np.ndarray | None = None

    for the_line in stream:
        if not isinstance(the_line, str):
            raise type_error(the_line, "line", str)
        line = the_line.strip()
        if len(line) <= 0:
            continue

        sep_idx: int = line.find(":")
        if sep_idx > 0:
            key: str = line[:sep_idx].strip()
            value: str = line[sep_idx + 1:].strip()
            if len(value) <= 0:
                raise ValueError(f"{line!r} has empty value "
                                 f"{value!r} for key {key!r}.")

            if key == __KEY_NAME:
                if the_name is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position,"
                        f" already got {__KEY_NAME}={the_name!r}.")
                if value.endswith(".tsp"):
                    value = value[0:-4]
                the_name = value
                continue

            if key == __KEY_TYPE:
                if the_type is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position, already "
                        f"got {__KEY_TYPE}={the_type!r}.")
                the_type = __TYPE_SYMMETRIC_TSP \
                    if value == "TSP (M.~Hofmeister)" else value
                if the_type not in __TYPES:
                    raise ValueError(
                        f"only {__TYPES!r} are permitted as {__KEY_TYPE}, "
                        f"but got {the_type!r} from {line!r}.")
                continue

            if key == __KEY_DIMENSION:
                if the_n_cities is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position,"
                        f" already got {__KEY_DIMENSION}={the_n_cities}.")
                the_n_cities = check_to_int_range(value, "dimension",
                                                  2, 1_000_000_000)

            if key == __KEY_EDGE_WEIGHT_TYPE:
                if the_ewt is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position, already "
                        f"got {__KEY_EDGE_WEIGHT_TYPE}={the_ewt!r}.")
                the_ewt = value
                if the_ewt not in __EDGE_WEIGHT_TYPES:
                    raise ValueError(
                        f"only {__EDGE_WEIGHT_TYPES!r} are permitted as "
                        f"{__KEY_EDGE_WEIGHT_TYPE}, but got {the_ewt!r} "
                        f"in {line!r}")
                continue

            if key == __KEY_EDGE_WEIGHT_FORMAT:
                if the_ewf is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position, already "
                        f"got {__KEY_EDGE_WEIGHT_FORMAT}={the_ewf!r}.")
                the_ewf = value
                if the_ewf not in __EDGE_WEIGHT_FORMATS:
                    raise ValueError(
                        f"only {__EDGE_WEIGHT_FORMATS!r} are permitted as "
                        f"{__KEY_EDGE_WEIGHT_FORMAT}, but got {the_ewf} "
                        f"in {line!r}")
                continue
            if key == __KEY_NODE_COORD_TYPE:
                if the_nct is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position, already "
                        f"got {__KEY_NODE_COORD_TYPE}={the_nct!r}.")
                the_nct = value
                if the_nct not in __NODE_COORD_TYPES:
                    raise ValueError(
                        f"only {__NODE_COORD_TYPES!r} are permitted as node "
                        f"{__KEY_NODE_COORD_TYPE}, but got {the_nct!r} "
                        f"in {line!r}")
                continue
        elif line == __START_NODE_COORD_SECTION:
            if the_matrix is not None:
                raise ValueError(
                    f"already got matrix, cannot have {line!r} here!")
            the_matrix = _matrix_from_node_coord_section(
                the_n_cities, the_ewt, the_nct, stream)
            continue
        elif line == __START_EDGE_WEIGHT_SECTION:
            if the_matrix is not None:
                raise ValueError(
                    f"already got matrix, cannot have {line!r} here!")
            the_matrix = _matrix_from_edge_weights(
                the_n_cities, the_ewt, the_ewf, stream)
            continue
        elif line == __EOF:
            break
        elif line == __FIXED_EDGES:
            raise ValueError(f"{__FIXED_EDGES!r} not supported")

    if the_name is None:
        raise ValueError("did not find any name.")
    if the_matrix is None:
        raise ValueError(f"did not find any matrix for {the_name!r}.")

    inst: Final[Instance] = Instance(
        the_name, 0 if (lower_bound_getter is None) else
        lower_bound_getter(the_name), the_matrix)
    if (the_type == __TYPE_SYMMETRIC_TSP) and (not inst.is_symmetric):
        raise ValueError("found asymmetric TSP instance but expected "
                         f"{the_name!r} to be a symmetric one?")

    return inst


class Instance(Component, np.ndarray):
    """An instance of the Traveling Salesperson Problem."""

    #: the name of the instance
    name: str
    #: the number of cities
    n_cities: int
    #: the lower bound of the tour length
    tour_length_lower_bound: int
    #: the upper bound of the tour length
    tour_length_upper_bound: int
    #: is the TSP instance symmetric?
    is_symmetric: bool

    def __new__(cls, name: str, tour_length_lower_bound: int,
                matrix: np.ndarray) -> "Instance":
        """
        Create an instance of the Traveling Salesperson Problem.

        :param cls: the class
        :param name: the name of the instance
        :param tour_length_lower_bound: the lower bound of the tour length
        :param matrix: the matrix with the data (will be copied)
        """
        use_name: Final[str] = sanitize_name(name)
        if name != use_name:
            raise ValueError(f"Name {name!r} is not a valid name.")

        check_int_range(tour_length_lower_bound, "tour_length_lower_bound",
                        0, 1_000_000_000_000_000)

        n_cities: int = len(matrix)
        if n_cities <= 1:
            raise ValueError(f"There must be at least two cities in a TSP "
                             f"instance, but we got {n_cities}.")

        use_shape: Final[tuple[int, int]] = (n_cities, n_cities)
        if isinstance(matrix, np.ndarray):
            if matrix.shape != use_shape:
                raise ValueError(
                    f"Unexpected shape {matrix.shape} for {n_cities} cities, "
                    f"expected {use_shape}.")
        else:
            raise type_error(matrix, "matrix", np.ndarray)

        # validate the matrix and compute the upper bound
        upper_bound: int = 0
        lower_bound_2: int = 0
        is_symmetric: bool = True
        for i in range(n_cities):
            farthest_neighbor: int = -1
            nearest_neighbor: int = 9_223_372_036_854_775_807

            for j in range(n_cities):
                dist = int(matrix[i, j])
                if i == j:
                    if dist != 0:
                        raise ValueError(
                            f"if i=j={i}, then dist must be zero "
                            f"but is {dist}.")
                else:
                    if dist > farthest_neighbor:
                        farthest_neighbor = dist
                    if dist < nearest_neighbor:
                        nearest_neighbor = dist
                if dist != matrix[j, i]:
                    is_symmetric = False
            if farthest_neighbor <= 0:
                raise ValueError(f"farthest neighbor distance of node {i} is"
                                 f" {farthest_neighbor}?")
            upper_bound = upper_bound + farthest_neighbor
            lower_bound_2 = lower_bound_2 + nearest_neighbor

        tour_length_lower_bound = max(
            tour_length_lower_bound, check_int_range(
                lower_bound_2, "lower_bound_2", 0, 1_000_000_000_000_000))
        check_int_range(upper_bound, "upper_bound",
                        tour_length_lower_bound, 1_000_000_000_000_001)

        # create the object
        obj: Final[Instance] = super().__new__(
            cls, use_shape, int_range_to_dtype(
                min_value=0, max_value=max(upper_bound, n_cities)))
        np.copyto(obj, matrix, "unsafe")

        #: the name of the instance
        obj.name = use_name
        #: the number of cities
        obj.n_cities = n_cities
        #: the lower bound of the tour length
        obj.tour_length_lower_bound = tour_length_lower_bound
        #: the upper bound of the tour length
        obj.tour_length_upper_bound = upper_bound
        #: is this instance symmetric?
        obj.is_symmetric = is_symmetric
        return obj

    def __str__(self):
        """
        Get the name of this instance.

        :return: the name of this instance

        >>> str(Instance.from_resource("gr17"))
        'gr17'
        """
        return self.name

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the instance to the given logger.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("I") as kv:
        ...         Instance.from_resource("kroE100").log_parameters_to(kv)
        ...     print(repr('@'.join(l.get_log())))
        'BEGIN_I@name: kroE100@class: moptipyapps.tsp.instance.Instance\
@nCities: 100@tourLengthLowerBound: 22068@tourLengthUpperBound: 330799@\
symmetric: T@dtype: i@END_I'
        """
        super().log_parameters_to(logger)
        logger.key_value("nCities", self.n_cities)
        logger.key_value("tourLengthLowerBound", self.tour_length_lower_bound)
        logger.key_value("tourLengthUpperBound", self.tour_length_upper_bound)
        logger.key_value("symmetric", self.is_symmetric)
        logger.key_value(npu.KEY_NUMPY_TYPE, self.dtype.char)

    @staticmethod
    def from_file(
            path: str,
            lower_bound_getter: Callable[[str], int] | None =
            _LOWER_BOUNDS.__getitem__) -> "Instance":
        """
        Read a TSP Lib instance from a TSP-lib formatted file.

        :param path: the path to the file
        :param lower_bound_getter: a function returning a lower bound for an
            instance name, or `None` to use a simple lower bound approximation
        :return: the instance
        """
        file: Final[Path] = Path.file(path)
        with file.open_for_read() as stream:
            try:
                return _from_stream(
                    cast(TextIO, stream), lower_bound_getter)
            except (TypeError, ValueError) as err:
                raise ValueError(f"error when parsing file {file!r}") from err

    @staticmethod
    def from_resource(name: str) -> "Instance":
        """
        Load an instance from a resource.

        :param name: the name string
        :return: the instance

        >>> Instance.from_resource("br17").n_cities
        17
        """
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        container: Final = Instance.from_resource
        inst_attr: Final[str] = f"__inst_{name}"
        if hasattr(container, inst_attr):  # instance loaded?
            return cast(Instance, getattr(container, inst_attr))

        is_symmetric: Final[bool] = name not in _ASYMMETRIC_INSTANCES
        suffix: Final[str] = ".tsp" if is_symmetric else ".atsp"
        with open_resource_stream(f"{name}{suffix}") as stream:
            inst: Final[Instance] = _from_stream(stream)
            if inst.name != name:
                raise ValueError(f"got {inst.name!r} for instance {name!r}?")
            if is_symmetric and (not inst.is_symmetric):
                raise ValueError(f"{name!r} should be symmetric but is not?")
            if inst.n_cities <= 1000:
                setattr(container, inst_attr, inst)
            return inst

    @staticmethod
    def list_resources(symmetric: bool = True,
                       asymmetric: bool = True) -> tuple[str, ...]:
        """
        Get a tuple of all the instances available as resource.

        :param symmetric: include the symmetric instances
        :param asymmetric: include the asymmetric instances
        :return: the tuple with the instance names

        >>> a = len(Instance.list_resources(True, True))
        >>> print(a)
        110
        >>> b = len(Instance.list_resources(True, False))
        >>> print(b)
        91
        >>> c = len(Instance.list_resources(False, True))
        >>> print(c)
        19
        >>> print(a == (b + c))
        True
        >>> print(len(Instance.list_resources(False, False)))
        0
        """
        return _INSTANCES if (symmetric and asymmetric) else (
            _SYMMETRIC_INSTANCES if symmetric else (
                _ASYMMETRIC_INSTANCES if asymmetric else ()))
