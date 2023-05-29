"""An instance of the Traveling Salesperson Problem."""

from math import acos, cos, sqrt
from typing import Any, Callable, Final, TextIO, cast

import moptipy.utils.nputils as npu
import numpy as np
from moptipy.api.component import Component
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.path import Path
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import check_int_range, check_to_int_range, type_error

from moptipyapps.tsp.tsplib import open_resource_stream

#: the known optimal tour lengths and lower bounds of TSP Lib
_LOWER_BOUNDS: dict[str, int] = {
    "a280": 2579, "ali535": 202339, "att48": 10628, "att532": 27686,
    "bayg29": 1610, "bays29": 2020, "berlin52": 7542, "bier127": 118282,
    "brazil58": 25395, "brd14051": 469385, "brg180": 1950, "burma14": 3323,
    "ch130": 6110, "ch150": 6528, "d198": 15780, "d493": 35002, "d657": 48912,
    "d1291": 50801, "d1655": 62128, "d2103": 80450, "d15112": 1573084,
    "d18512": 645238, "dantzig42": 699, "eil51": 426, "eil76": 538,
    "eil101": 629, "fl417": 11861, "fl1400": 20127, "fl1577": 22249,
    "fl3795": 28772, "fnl4461": 182566, "fri26": 937, "gil262": 2378,
    "gr17": 2085, "gr21": 2707, "gr24": 1272, "gr48": 5046, "gr96": 55209,
    "gr120": 6942, "gr137": 69853, "gr202": 40160, "gr229": 134602,
    "gr431": 171414, "gr666": 294358, "hk48": 11461, "kroA100": 21282,
    "kroB100": 22141, "kroC100": 20749, "kroD100": 21294, "kroE100": 22068,
    "kroA150": 26524, "kroB150": 26130, "kroA200": 29368, "kroB200": 29437,
    "lin105": 14379, "lin318": 42029, "nrw1379": 56638,
    "p654": 34643, "pa561": 2763, "pcb442": 50778, "pcb1173": 56892,
    "pcb3038": 137694, "pla7397": 23260728, "pla33810": 66048945,
    "pla85900": 142382641, "pr76": 108159, "pr107": 44303, "pr124": 59030,
    "pr136": 96772, "pr144": 58537, "pr152": 73682, "pr226": 80369,
    "pr264": 49135, "pr299": 48191, "pr439": 107217, "pr1002": 259045,
    "pr2392": 378032, "rat99": 1211, "rat195": 2323, "rat575": 6773,
    "rat783": 8806, "rd100": 7910, "rd400": 15281, "rl1304": 252948,
    "rl1323": 270199, "rl1889": 316536, "rl5915": 565530,
    "rl5934": 556045, "rl11849": 923288, "si175": 21407, "si535": 48450,
    "si1032": 92650, "st70": 675, "swiss42": 1273, "ts225": 126643,
    "tsp225": 3916, "u159": 42080, "u574": 36905, "u724": 41910,
    "u1060": 224094, "u1432": 152970, "u1817": 57201, "u2152": 64253,
    "u2319": 234256, "ulysses16": 6859, "ulysses22": 7013,
    "usa13509": 19982859, "vm1084": 239297, "vm1748": 336556, "br17": 39,
    "ft53": 6905, "ft70": 38673, "ftv33": 1286, "ftv35": 1473, "ftv38": 1530,
    "ftv44": 1613, "ftv47": 1776, "ftv55": 1608, "ftv64": 1839, "ftv70": 1950,
    "ftv90": 1579, "ftv100": 1788, "ftv110": 1958, "ftv120": 2166,
    "ftv130": 2307, "ftv140": 2420, "ftv150": 2611, "ftv160": 2683,
    "ftv170": 2755, "kro124": 36230, "p43": 5620, "rbg323": 1326,
    "rbg358": 1163, "rbg403": 2465, "rbg443": 2720, "ry48p": 14422,
    "dsj1000": 18660188}

#: the TSPLib instances
_INSTANCES: Final[tuple[str, ...]] = (
    "a280", "ali535", "att48", "att532", "berlin52", "bier127", "burma14",
    "ch130", "ch150", "d198", "d493", "d657", "d1291", "d1655",
    "eil51", "eil76", "eil101", "fl417", "fl1400", "fl1577", "gil262",
    "kroA100", "kroA150", "kroA200", "kroB100", "kroB150", "kroB200",
    "kroC100", "kroD100", "kroE100", "lin105", "lin318",
    "nrw1379", "p654", "pcb442", "pcb1173", "pr76", "pr107", "pr124", "pr136",
    "pr144", "pr152", "pr226", "pr264", "pr299", "pr439", "pr1002",
    "rat99", "rat195", "rat575", "rat783", "rd100", "rd400", "rl1304",
    "rl1323", "rl1889", "st70", "ts225", "tsp225", "u159", "u574", "u724",
    "u1060", "u1432", "u1817", "ulysses16", "ulysses22",
    "vm1084", "vm1748")

#: the problem is a symmetric tsp
_TYPE_SYMMETRIC_TSP: Final[str] = "TSP"
#: the problem is an asymmetric tsp
_TYPE_ASYMMETRIC_TSP: Final[str] = "ATSP"
#: the permitted types
_TYPES: Final[set[str]] = {_TYPE_SYMMETRIC_TSP, _TYPE_ASYMMETRIC_TSP}
#: the name start
_KEY_NAME: Final[str] = "NAME"
#: the type start
_KEY_TYPE: Final[str] = "TYPE"
#: the dimension start
_KEY_DIMENSION: Final[str] = "DIMENSION"
#: the comment start
_KEY_COMMENT: Final[str] = "COMMENT"
#: EUC_2D coordinates
_EWT_EUC_2D: Final[str] = "EUC_2D"
#: geographical coordinates
_EWT_GEO: Final[str] = "GEO"
#: ATT coordinates
_EWT_ATT: Final[str] = "ATT"
#: ceiling 2D coordinates
_EWT_CEIL2D: Final[str] = "CEIL_2D"
#: the edge weight type start
_KEY_EDGE_WEIGHT_TYPE: Final[str] = "EDGE_WEIGHT_TYPE"
#: the permitted edge weight types
_EDGE_WEIGHT_TYPES: Final[set[str]] = {
    _EWT_EUC_2D, _EWT_GEO, _EWT_ATT, _EWT_CEIL2D}
#: the edge weight format "function"
_EWF_FUNCTION: Final[str] = "FUNCTION"
#: the edge weight format start
_KEY_EDGE_WEIGHT_FORMAT: Final[str] = "EDGE_WEIGHT_FORMAT"
#: the permitted edge weight formats
_EDGE_WEIGHT_FORMATS: Final[set[str]] = {_EWF_FUNCTION}
#: the start of the node coord type
_KEY_NODE_COORD_TYPE: Final[str] = "NODE_COORD_TYPE"
#: 2d coordinates
_NODE_COORD_TYPE_2D: Final[str] = "TWOD_COORDS"
#: 3d coordinates
_NODE_COORD_TYPE_3D: Final[str] = "THREED_COORDS"
#: no coordinates
_NODE_COORD_TYPE_NONE: Final[str] = "NO_COORDS"
#: the permitted node coordinate types
_NODE_COORD_TYPES: Final[set[str]] = {
    _NODE_COORD_TYPE_2D, _NODE_COORD_TYPE_3D, _NODE_COORD_TYPE_NONE}
#: the node coordinate section starts
_START_NODE_COORD_SECTION: Final[str] = "NODE_COORD_SECTION"
#: the end of the file
_EOF: Final[str] = "EOF"


def __line_to_ints(line: str,
                   collector: Callable[[int | float], Any]) -> None:
    """
    Split a line to a list of integers.

    :param line: the line string
    :param collector: the collector
    """
    if not isinstance(line, str):
        raise type_error(line, "line", str)
    idx: int = 0
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
            collector(float(part))
        else:
            collector(check_to_int_range(
                part, "line_fragment", -1_000_000_000_000, 1_000_000_000_000))
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
        if line == _EOF:
            break
        __line_to_ints(line, row.append)
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
    if (edge_weight_type == _EWT_EUC_2D) \
            and (node_coord_type in (None, _NODE_COORD_TYPE_2D)):
        coord_dim = 2
        dist_fun = __dist_2deuc
    elif (edge_weight_type == _EWT_GEO) \
            and (node_coord_type in (None, _NODE_COORD_TYPE_2D)):
        coord_dim = 2
        dist_fun = __dist_loglat
    elif (edge_weight_type == _EWT_ATT) \
            and (node_coord_type in (None, _NODE_COORD_TYPE_2D)):
        coord_dim = 2
        dist_fun = __dist_att
    elif (edge_weight_type == _EWT_CEIL2D) \
            and (node_coord_type in (None, _NODE_COORD_TYPE_2D)):
        coord_dim = 2
        dist_fun = __dist_2dceil

    if (coord_dim is not None) and (dist_fun is not None):
        return __matrix_from_points(n_cities, coord_dim, stream, dist_fun)

    raise ValueError(f"invalid combination of {_KEY_EDGE_WEIGHT_TYPE} "
                     f"and {_KEY_NODE_COORD_TYPE}")


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

            if key == _KEY_NAME:
                if the_name is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position,"
                        f" already got {_KEY_NAME}={the_name!r}.")
                if value.endswith(".tsp"):
                    value = value[0:-4]
                the_name = value
                continue

            if key == _KEY_TYPE:
                if the_type is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position, already "
                        f"got {_KEY_TYPE}={the_type!r}.")
                the_type = value
                if the_type not in _TYPES:
                    raise ValueError(
                        f"only {_TYPES!r} are permitted as {_KEY_TYPE}, "
                        f"but got {the_type!r} from {line!r}.")
                continue

            if key == _KEY_DIMENSION:
                if the_n_cities is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position,"
                        f" already got {_KEY_DIMENSION}={the_n_cities}.")
                the_n_cities = check_to_int_range(value, "dimension",
                                                  2, 1_000_000_000)

            if key == _KEY_EDGE_WEIGHT_TYPE:
                if the_ewt is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position, already "
                        f"got {_KEY_EDGE_WEIGHT_TYPE}={the_ewt!r}.")
                the_ewt = value
                if the_ewt not in _EDGE_WEIGHT_TYPES:
                    raise ValueError(
                        f"only {_EDGE_WEIGHT_TYPES!r} are permitted as "
                        f"{_KEY_EDGE_WEIGHT_TYPE}, but got {the_ewt!r} "
                        f"in {line!r}")
                continue

            if key == _KEY_EDGE_WEIGHT_FORMAT:
                if the_ewf is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position, already "
                        f"got {_KEY_EDGE_WEIGHT_FORMAT}={the_ewf!r}.")
                the_ewf = value
                if the_ewf not in _EDGE_WEIGHT_FORMATS:
                    raise ValueError(
                        f"only {_EDGE_WEIGHT_FORMATS!r} are permitted as "
                        f"{_KEY_EDGE_WEIGHT_FORMAT}, but got {the_ewf} "
                        f"in {line!r}")
                continue
            if key == _KEY_NODE_COORD_TYPE:
                if the_nct is not None:
                    raise ValueError(
                        f"{line!r} cannot come at this position, already "
                        f"got {_KEY_NODE_COORD_TYPE}={the_nct!r}.")
                the_nct = value
                if the_nct not in _NODE_COORD_TYPES:
                    raise ValueError(
                        f"only {_NODE_COORD_TYPES!r} are permitted as node "
                        f"{_KEY_NODE_COORD_TYPE}, but got {the_nct!r} "
                        f"in {line!r}")
                continue
        elif line == _START_NODE_COORD_SECTION:
            the_matrix = _matrix_from_node_coord_section(
                the_n_cities, the_ewt, the_nct, stream)
            continue
        elif line == _EOF:
            break

    if the_name is None:
        raise ValueError("did not find any name.")
    if the_matrix is None:
        raise ValueError(f"did not find any matrix for {the_name!r}.")

    inst: Final[Instance] = Instance(
        the_name, 1 if (lower_bound_getter is None) else
        lower_bound_getter(the_name), the_matrix)
    if (the_type == _TYPE_SYMMETRIC_TSP) and (not inst.is_symmetric):
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
                        1, 1_000_000_000_000_000)

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
                lower_bound_2, "lower_bound_2", 1, 1_000_000_000_000_000))
        check_int_range(upper_bound, "upper_bound",
                        tour_length_lower_bound + 1, 1_000_000_000_000_001)

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
        """
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        container: Final = Instance.from_resource
        inst_attr: Final[str] = f"__inst_{name}"
        if hasattr(container, inst_attr):  # instance loaded?
            return cast(Instance, getattr(container, inst_attr))

        with open_resource_stream(f"{name}.tsp") as stream:
            inst: Final[Instance] = _from_stream(stream)
            if inst.name != name:
                raise ValueError(f"got {inst.name!r} for instance {name!r}?")
            if inst.n_cities <= 1000:
                setattr(container, inst_attr, inst)
            return inst

    def __str__(self):
        """
        Get the name of this instance.

        :return: the name of this instance
        """
        return self.name

    @staticmethod
    def list_resources() -> tuple[str, ...]:
        """
        Get a tuple of all the instances available as resource.

        :return: the tuple with the instance names
        """
        return _INSTANCES
