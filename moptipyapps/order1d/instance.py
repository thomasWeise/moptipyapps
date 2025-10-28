"""An instance of the ordering problem."""


from math import isfinite
from typing import Callable, Final, Iterable, TypeVar, cast

import numpy as np
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_INT
from moptipy.utils.strings import (
    num_to_str_for_name,
    sanitize_name,
)
from pycommons.types import check_int_range, type_error
from scipy.stats import rankdata  # type: ignore

from moptipyapps.qap.instance import Instance as QAPInstance

#: the type variable for the source object
T = TypeVar("T")

#: the zero-based index column name
_ZERO_BASED_INDEX: Final[str] = "indexZeroBased"
#: the suggested x-coordinate
_SUGGESTED_X_IN_0_1: Final[str] = "suggestedXin01"


class Instance(QAPInstance):
    """
    An instance of the One-Dimensional Ordering Problem.

    Such an instance represents the ranking of objects by their distance to
    each other as a :mod:`~moptipyapps.qap` problem.

    >>> def _dist(a, b):
    ...     return abs(a - b)
    >>> def _tags(a):
    ...     return f"t{a}"
    >>> the_instance = Instance.from_sequence_and_distance(
    ...     [1, 2, 3], _dist, 1, 100, ("bla", ), _tags)
    >>> print(the_instance)
    order1d_3_1
    >>> print(the_instance.distances)
    [[0 1 2]
     [1 0 1]
     [2 1 0]]
    >>> print(the_instance.flows)
    [[0 4 2]
     [3 0 3]
     [2 4 0]]
    """

    def __init__(self, distances: np.ndarray,
                 flow_power: int | float,
                 horizon: int,
                 tag_titles: Iterable[str],
                 tags: Iterable[tuple[Iterable[str] | str, int]],
                 name: str | None = None) -> None:
        """
        Create an instance of the quadratic assignment problem.

        :param distances: the original distance matrix
        :param flow_power: the power to be used for constructing the flows
            based on the original distances
        :param horizon: the horizon for distance ranks, after which all
            elements are ignored
        :param tag_titles: the tag titles
        :param tags: the assignment of rows to names
        :param name: the instance name
        """
        if not isinstance(flow_power, int | float):
            raise type_error(flow_power, "flow_power", (int, float))
        if not (isfinite(flow_power) and (0 < flow_power < 100)):
            raise ValueError(
                f"flow_power must be in (0, 100), but is {flow_power}.")

        n: Final[int] = len(distances)  # the number of elements
        horizon = check_int_range(horizon, "horizon", 1, 1_000_000_000_000)

        # construct the distance matrix
        dist_matrix = np.zeros((n, n), DEFAULT_INT)
        for i in range(n):
            for j in range(i + 1, n):
                dist_matrix[i, j] = dist_matrix[j, i] = j - i

        # construct the flow matrix
        flows: np.ndarray = rankdata(
            distances, axis=1, method="average") - 1.0

        # get the flow rank multiplier
        multiplier: float = 1.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                f = flows[i, j]
                if int(f) < f <= horizon:  # dow we need the multiplier?
                    multiplier = 0.5 ** (-flow_power)
                    break

        flow_matrix = np.zeros((n, n), DEFAULT_INT)
        max_val: Final[int] = min(n - 1, horizon)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                f = flows[i, j]
                if f > horizon:
                    continue
                flow_matrix[i, j] = round(
                    multiplier * ((max_val - f + 1) ** flow_power))

        # construct a name if necessary
        if name is None:
            name = f"order1d_{n}_{num_to_str_for_name(flow_power)}"
            if max_val < (n - 1):
                name = f"{name}_{max_val}"

        super().__init__(dist_matrix, flow_matrix, name=name)

        if self.n != n:
            raise ValueError("error when assigning number of elements!")

        #: the flow power
        self.flow_power: Final[int | float] = flow_power
        #: the horizon
        self.horizon: Final[int] = max_val

        if not isinstance(tags, Iterable):
            raise type_error(tags, "tags", Iterable)
        if not isinstance(tag_titles, Iterable):
            raise type_error(tag_titles, "tag_titles", Iterable)

        #: the tag titles
        self.tag_titles: Final[tuple[str, ...]] = tuple(map(
            sanitize_name, tag_titles))
        req_len: Final[int] = len(self.tag_titles)
        if req_len <= 0:
            raise ValueError("No tags specified.")

        #: the tags, i.e., additional data for each original element
        self.tags: Final[tuple[tuple[tuple[str, ...], int], ...]] = (
            tuple(((t, ) if isinstance(t, str)
                   else tuple(t), k) for t, k in tags))
        if _ZERO_BASED_INDEX in self.tags:
            raise ValueError(f"{_ZERO_BASED_INDEX} not permitted in tags.")
        if _SUGGESTED_X_IN_0_1 in self.tags:
            raise ValueError(f"{_SUGGESTED_X_IN_0_1} not permitted in tags.")

        if len(self.tags) < n:
            raise ValueError(f"there must be at least {self.n} tags, but got "
                             f"{len(self.tags)}, i.e., {self.tags}")
        for tag in self.tags:
            check_int_range(tag[1], "id", 0, n)
            if len(tag[0]) != req_len:
                raise ValueError(
                    f"all tags must have the same length. "
                    f"while the first tag ({self.tags[0]}) has "
                    f"length {req_len}, {tag} has length {len(tag[0])}.")

    @staticmethod
    def from_sequence_and_distance(
            data: Iterable[T | None],
            get_distance: Callable[[T, T], int | float],
            flow_power: int | float,
            horizon: int,
            tag_titles: Iterable[str],
            get_tags: Callable[[T], str | Iterable[str]],
            name: str | None = None) -> "Instance":
        """
        Turn a sequence of objects into a One-Dimensional Ordering instance.

        :param data: the data source, i.e., an iterable of data elements
        :param get_tags: the function for extracting tags from objects
        :param get_distance: the function for getting the distance between
            objects
        :param flow_power: the flow power
        :param horizon: the maximal considered rank
        :param tag_titles: the tag names
        :param name: the optional name
        :return: the ordering instance

        >>> def _dist(a, b):
        ...     return abs(a - b)
        >>> def _tags(a):
        ...     return f"x{a}", f"b{a}"
        >>> res = Instance.from_sequence_and_distance(
        ...     [1, 2, 5], _dist, 2, 100, ("a", "b"), _tags)
        >>> print(res)
        order1d_3_2
        >>> print(res.flows)
        [[0 4 1]
         [4 0 1]
         [1 4 0]]
        >>> print(res.distances)
        [[0 1 2]
         [1 0 1]
         [2 1 0]]
        >>> res = Instance.from_sequence_and_distance(
        ...     [1, 2, 35, 4], _dist, 2, 100, ("a", "b"), _tags)
        >>> print(res)
        order1d_4_2
        >>> print(res.flows)
        [[0 9 1 4]
         [9 0 1 4]
         [1 4 0 9]
         [4 9 1 0]]
        >>> print(res.distances)
        [[0 1 2 3]
         [1 0 1 2]
         [2 1 0 1]
         [3 2 1 0]]
        >>> res = Instance.from_sequence_and_distance(
        ...     [1, 2, 4, 4], _dist, 2, 100, ("a", "b"), _tags)
        >>> print(res)
        order1d_3_2
        >>> print(res.flows)
        [[0 4 1]
         [4 0 1]
         [1 4 0]]
        >>> print(res.distances)
        [[0 1 2]
         [1 0 1]
         [2 1 0]]
        >>> print(res.tags)
        ((('x1', 'b1'), 0), (('x2', 'b2'), 1), (('x4', 'b4'), 2), \
(('x4', 'b4'), 2))
        >>> def _dist2(a, b):
        ...     return abs(abs(a) - abs(b)) + 1
        >>> res = Instance.from_sequence_and_distance(
        ...     [1, 2, -4, 4], _dist2, 2, 100, ("a", "b"), _tags)
        >>> print(res)
        order1d_4_2
        >>> print(res.flows)
        [[ 0 36  9  9]
         [36  0  9  9]
         [ 4 16  0 36]
         [ 4 16 36  0]]
        >>> print(res.distances)
        [[0 1 2 3]
         [1 0 1 2]
         [2 1 0 1]
         [3 2 1 0]]
        >>> res = Instance.from_sequence_and_distance(
        ...     [1, 2, -4, 4], _dist2, 3, 100, ("a", "b"), _tags)
        >>> print(res)
        order1d_4_3
        >>> print(res.flows)
        [[  0 216  27  27]
         [216   0  27  27]
         [  8  64   0 216]
         [  8  64 216   0]]
        >>> print(res.distances)
        [[0 1 2 3]
         [1 0 1 2]
         [2 1 0 1]
         [3 2 1 0]]
        >>> print(res.tags)
        ((('x1', 'b1'), 0), (('x2', 'b2'), 1), (('x-4', 'b-4'), 2), \
(('x4', 'b4'), 3))
        >>> res = Instance.from_sequence_and_distance(
        ...     [1, 2, -4, 4], _dist2, 3, 2, ("a", "b"), _tags)
        >>> print(res)
        order1d_4_3_2
        >>> print(res.flows)
        [[0 8 0 0]
         [8 0 0 0]
         [0 1 0 8]
         [0 1 8 0]]
        >>> print(res.distances)
        [[0 1 2 3]
         [1 0 1 2]
         [2 1 0 1]
         [3 2 1 0]]
        >>> print(res.tags)
        ((('x1', 'b1'), 0), (('x2', 'b2'), 1), (('x-4', 'b-4'), 2), \
(('x4', 'b4'), 3))
        >>> res = Instance.from_sequence_and_distance(
        ...     [1, 2, -4, 4], _dist2, 2, 2, ("a", "b"), _tags)
        >>> print(res)
        order1d_4_2_2
        >>> print(res.flows)
        [[0 4 0 0]
         [4 0 0 0]
         [0 1 0 4]
         [0 1 4 0]]
        >>> print(res.distances)
        [[0 1 2 3]
         [1 0 1 2]
         [2 1 0 1]
         [3 2 1 0]]
        """
        if not isinstance(data, Iterable):
            raise type_error(data, "data", Iterable)
        if not callable(get_tags):
            raise type_error(get_tags, "get_tags", call=True)
        if not callable(get_distance):
            raise type_error(get_distance, "get_distance", call=True)
        if not isinstance(tag_titles, Iterable):
            raise type_error(tag_titles, "tag_titles", Iterable)

        # build a distance matrix and purge all zero-distance elements
        datal: list[T] = cast("list[T]", data) \
            if isinstance(data, list) else list(data)
        mappings: list[tuple[T, int]] = []
        distances: list[list[int | float]] = []
        i: int = 0
        while i < len(datal):
            o1: T = datal[i]
            j: int = i + 1
            current_dists: list[int | float] = [d[i] for d in distances]
            current_dists.append(0)
            while j < len(datal):
                o2: T = datal[j]
                dist: int | float = get_distance(o1, o2)
                if not (isfinite(dist) and (0 <= dist <= 1e100)):
                    raise ValueError(
                        f"invalid distance {dist} for objects {i}, {j}")
                if dist <= 0:  # distance == 0, must purge
                    mappings.append((o2, i))
                    del datal[j]
                    for ds in distances:
                        del ds[j]
                    continue
                current_dists.append(dist)
                j += 1
            mappings.append((o1, i))
            distances.append(current_dists)
            i += 1

        # we now got a full distance matrix, let's turn it into a rank matrix
        return Instance(np.array(distances),
                        flow_power, horizon, tag_titles,
                        ((get_tags(obj), idx) for obj, idx in mappings),
                        name)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("flowPower", self.flow_power)
        logger.key_value("horizon", self.horizon)
        logger.key_value("nOrig", len(self.tags))
