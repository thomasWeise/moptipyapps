"""An instance of the ordering problem."""


from typing import Callable, Final, Iterable, TypeVar, cast

import numpy as np
from moptipy.api.component import Component
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.types import check_int_range, check_to_int_range, type_error
from scipy.stats import rankdata  # type: ignore

#: the type variable for the source object
S = TypeVar("S")
#: the type variable for the object
OBJ = TypeVar("OBJ")


class Instance(Component, np.ndarray):
    """
    An instance of the One-Dimensional Ordering Problem.

    Such an instance represents the ranking of objects by their distance to
    each other. If we have `n` objects, then we compute the distance of each
    object to each of its `n-1` neighbors. The distance of an object to itself
    is `0`. Let's say that we have the three numbers `1, 2, 3` and the
    distance be their absolute difference. Then `1` has distance `1` to `2`
    and distance `2` to `3`. For `2`, the distance to `1` is `1` and to `3` is
    also `1`. We compute the average rank of the objects to each other on a
    per-row basis (including the object itself). For object `1`, this gives us
    `(1, 2, 3)` and for object `2`, it gives `(2.5, 1, 2.5)`, the latter
    because `1` and `3` are equally far from `2`. We multiply this with 2 to
    get integer ranks and subtract 1, i.e., we would get
    `[[1, 3, 5], [4, 1, 4], [5, 3, 1]]` for our three numbers.

    >>> def _dist(a, b):
    ...     return abs(a - b)
    >>> def _tags(a):
    ...     return f"t{a}"
    >>> the_instance = Instance.from_sequence_and_distance(
    ...     [1, 2, 3], _tags, _dist)
    >>> print(np.array(the_instance))
    [[1 3 5]
     [4 1 4]
     [5 3 1]]

    However, it is possible that some objects appear twice or have zero
    distance. In this case, they will be purged from the matrix:

    >>> the_instance = Instance.from_sequence_and_distance(
    ...     [1, 2, 3, 3, 2, 3], _tags, _dist)
    >>> print(np.array(the_instance))
    [[1 3 5]
     [4 1 4]
     [5 3 1]]

    But they will still be visible in the tags:

    >>> print(the_instance.tags)
    ((('t1',), 0), (('t2',), 1), (('t2',), 1), (('t3',), 2), \
(('t3',), 2), (('t3',), 2))
    """

    #: the assignment of string tags to object IDs
    tags: tuple[tuple[tuple[str, ...], int], ...]

    def __new__(cls, matrix: np.ndarray,
                tags: Iterable[tuple[Iterable[str] | str, int]]) -> "Instance":
        """
        Create an instance of the one-dimensional ordering problem.

        :param cls: the class
        :param matrix: the matrix with the rank data (will be copied)
        :param tags: the assignment of rows to names
        """
        n: Final[int] = len(matrix)
        use_shape: Final[tuple[int, int]] = (n, n)
        if isinstance(matrix, np.ndarray):
            if matrix.shape != use_shape:
                raise ValueError(
                    f"Unexpected shape {matrix.shape} for {n} objects, "
                    f"expected {use_shape}.")
        else:
            raise type_error(matrix, "matrix", np.ndarray)

        obj: Final[Instance] = super().__new__(
            cls, use_shape, int_range_to_dtype(min_value=-n, max_value=n))
        np.copyto(obj, matrix, "unsafe")
        for i in range(n):
            for j in range(n):
                if check_to_int_range(
                        obj[i, j], "element", 0, (2 * n) - 1) != matrix[i, j]:
                    raise ValueError(
                        f"error when copying: {obj[i, j]} != {matrix[i, j]}")

        #: the tags
        obj.tags = tuple(((t, ) if isinstance(t, str) else tuple(t),
                          k) for t, k in tags)
        if len(obj.tags) < n:
            raise ValueError(f"there must be at least {n} tags, but got "
                             f"{len(obj.tags)}, i.e., {obj.tags}")

        req_len: int = len(obj.tags[0][0])
        if req_len <= 0:
            raise ValueError(f"tag length must be >= 1, but is {req_len} "
                             f"for tag {obj.tags[0]}.")
        for tag in obj.tags:
            check_int_range(tag[1], "id", 0, n)
            if len(tag[0]) != req_len:
                raise ValueError(
                    f"all tags must have the same length. "
                    f"while the first tag ({obj.tags[0]}) has "
                    f"length {req_len}, {tag} has length {len(tag[0])}.")
        return obj

    @staticmethod
    def from_sequence_and_distance(
            data: Iterable[S | None],
            get_tags: Callable[[OBJ], str | Iterable[str]],
            get_distance: Callable[[OBJ, OBJ], int],
            unpack: Callable[[S], OBJ | None] = cast(
                Callable[[S], OBJ], lambda x: x)) -> "Instance":
        """
        Turn a sequence of objects into a One-Dimensional Ordering instance.

        :param data: the data source, i.e., an iterable of data elements
        :param get_tags: the function for extracting tags from objects
        :param get_distance: the function for getting the distance between
            objects
        :param unpack: a function unpacking a source element, e.g., a string,
            to an object
        :return: the ordering instance

        >>> def _dist(a, b):
        ...     return abs(a - b)
        >>> def _tags(a):
        ...     return f"x{a}", f"b{a}"
        >>> res = Instance.from_sequence_and_distance(
        ...     [4, 5, 1, 2, 3], _tags, _dist)
        >>> print(res)
        Instance
        >>> print(np.array(res))
        [[1 4 9 7 4]
         [3 1 9 7 5]
         [7 9 1 3 5]
         [7 9 4 1 4]
         [4 8 8 4 1]]
        >>> print(res.tags)
        ((('x4', 'b4'), 0), (('x5', 'b5'), 1), (('x1', 'b1'), 2), \
(('x2', 'b2'), 3), (('x3', 'b3'), 4))
        >>> res = Instance.from_sequence_and_distance(
        ...     [4, 5, 4, 2, 5], _tags, _dist)
        >>> print(np.array(res))
        [[1 3 5]
         [3 1 5]
         [3 5 1]]
        >>> print(res.tags)
        ((('x4', 'b4'), 0), (('x4', 'b4'), 0), (('x5', 'b5'), 1), \
(('x5', 'b5'), 1), (('x2', 'b2'), 2))
        >>> def _dist2(a, b):
        ...     return abs(abs(a) - abs(b))
        >>> res = Instance.from_sequence_and_distance(
        ...     [-4, 5, 4, 2, -5], _tags, _dist2)
        >>> print(np.array(res))
        [[1 3 5]
         [3 1 5]
         [3 5 1]]
        >>> print(res.tags)
        ((('x4', 'b4'), 0), (('x-4', 'b-4'), 0), (('x-5', 'b-5'), 1), \
(('x5', 'b5'), 1), (('x2', 'b2'), 2))
        """
        if not isinstance(data, Iterable):
            raise type_error(data, "data", Iterable)
        if not callable(get_tags):
            raise type_error(get_tags, "get_tags", call=True)
        if not callable(get_distance):
            raise type_error(get_distance, "get_distance", call=True)
        if not callable(unpack):
            raise type_error(unpack, "unpack", call=True)

        # first extract all the objects
        raw: list[OBJ] = []
        for d in data:
            if d is None:
                continue
            objx: OBJ | None = unpack(d)
            if objx is None:
                continue
            raw.append(objx)

        # build a distance matrix and purge all zero-distance elements
        mappings: list[tuple[OBJ, int]] = []
        distances: list[list[int]] = []
        i: int = 0
        while i < len(raw):
            o1: OBJ = raw[i]
            j: int = i + 1
            current_dists: list[int] = [d[i] for d in distances]
            current_dists.append(0)
            while j < len(raw):
                o2: OBJ = raw[j]
                dist: int = check_int_range(
                    get_distance(o1, o2), "get_distance", 0,
                    1_000_000_000_000)
                if dist <= 0:  # distance == 0, must purge
                    mappings.append((o2, i))
                    del raw[j]
                    for ds in distances:
                        del ds[j]
                    continue
                current_dists.append(dist)
                j += 1
            mappings.append((o1, i))
            distances.append(current_dists)
            i += 1

        # we now got a full distance matrix, let's turn it into a rank matrix
        return Instance(
            (2.0 * rankdata(distances, axis=1, method="average") - 1.0),
            ((get_tags(obj), idx) for obj, idx in mappings))
