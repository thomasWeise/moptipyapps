"""
Synthesize some interesting starting points.

Here we have some basic functionality to synthesize starting points, i.e.,
training cases, for the dynamic systems control task.
The points synthesized here by function
:func:`make_interesting_starting_points` try to fulfill two goals:

1. the points should be as far away from each other as possible in the state
   space,
2. there should be points of many different distances from the state space
   origin.

These two goals are slightly contradicting and are achieved by forcing the
points to be located on rings of increasing distance from the origin via
:func:`interesting_point_transform` while maximizing their mean distance
to each other via :func:`interesting_point_objective`.
Since :func:`make_interesting_starting_points` is a bit slow, it makes sense
to pre-compute the points and then store them as array constants.
"""

from typing import Callable, Final, Iterable, cast

import numba  # type: ignore
import numpy as np
from moptipy.algorithms.so.vector.cmaes_lib import BiPopCMAES
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.console import logger


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def interesting_point_transform(
        x: np.ndarray, max_radius: float, dim: int) -> np.ndarray:
    """
    Transform interesting points.

    >>> xx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.0])
    >>> ppp = interesting_point_transform(xx, 10.0, 3)
    >>> print(ppp)
    [[0.6681531  1.33630621 2.00445931]
     [2.27921153 2.84901441 3.41881729]
     [3.76928033 4.30774895 4.84621757]
     [5.23423923 5.75766315 6.28108707]]
    >>> print(xx)
    [0.6681531  1.33630621 2.00445931 2.27921153 2.84901441 3.41881729
     3.76928033 4.30774895 4.84621757 5.23423923 5.75766315 6.28108707]
    >>> print([np.sqrt(np.square(pppp).sum()) for pppp in ppp])
    [2.5, 5.0, 7.5, 10.0]
    >>> ppp = interesting_point_transform(xx, 10.0, 3)
    >>> print(ppp)
    [[0.6681531  1.33630621 2.00445931]
     [2.27921153 2.84901441 3.41881729]
     [3.76928033 4.30774895 4.84621757]
     [5.23423923 5.75766315 6.28108707]]
    >>> print([np.sqrt(np.square(pppp).sum()) for pppp in ppp])
    [2.5, 5.0, 7.5, 10.0]
    >>> ppp = interesting_point_transform(xx, 10.0, 2)
    >>> print(ppp)
    [[0.74535599 1.49071198]
     [2.20132119 2.50305736]
     [3.200922   3.8411064 ]
     [4.39003072 5.01717796]
     [5.66154475 6.11484713]
     [6.75724629 7.3715414 ]]
    """
    n: Final[int] = len(x) // dim
    p: Final[np.ndarray] = np.reshape(x, (n, dim))
    for i in range(n):
        pp: np.ndarray = p[i, :]

        cur_radius: float = np.sqrt(np.square(pp).sum())
        if cur_radius <= 0.0:
            continue
        goal_radius: float = max_radius * ((i + 1) / n)
        if goal_radius != cur_radius:
            first: bool = True
            while True:
                old_radius = cur_radius
                pp2 = pp * goal_radius / cur_radius
                if np.array_equal(pp2, pp):
                    break
                cur_radius = np.sqrt(np.square(pp2).sum())
                if cur_radius == goal_radius:
                    pp = pp2
                    break
                if first:
                    first = False
                elif cur_radius > old_radius:
                    break
                pp = pp2
        p[i, :] = pp
    return p


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def interesting_point_objective(x: np.ndarray, other: np.ndarray,
                                max_radius: float, dim: int) -> float:
    """Compute the point diversity."""
    pts: np.ndarray = interesting_point_transform(x, max_radius, dim)
    n: Final[int] = len(pts)
    f: float = 0.0
    for i in range(n):
        pt: np.ndarray = pts[i]
        scale: float = (((i + 1) / n) ** 2)  # we know this
        for j in range(i + 1, n):
            dst: float = np.sqrt(np.square(pt - pts[j]).sum())
            f += scale / dst if dst > 0.0 else 1e10
        for oth in other:
            dst = np.sqrt(np.square(pt - oth).sum())
            f += scale / dst if dst > 0.0 else 1e10
    return f


def make_interesting_starting_points(
        n: int, other: Iterable[Iterable[float]],
        log: bool = True) -> np.ndarray:
    """
    Create some reasonably diverse starting points.

    :param n: the number of starting points
    :param other: the other points
    :param log: write log output
    :return: the starting points

    >>> p = make_interesting_starting_points(
    ...     3, np.array([[1.0, 2.0], [3.0, 2.9]]), False)
    >>> print(",".join(";".join(f"{x:.5f}" for x in row) for row in p))
    0.92722;-1.55225,-3.61077;0.19795,-0.91540;-5.34649

    >>> p = make_interesting_starting_points(
    ...     3, np.array([[1.0, 2.0, 7.0], [3.0, 2.9, 1.1]]), False)
    >>> print(",".join(";".join(f"{x:.5f}" for x in row) for row in p))
    -0.35688;-3.08116;0.72047,-5.49372;1.93835;-2.57330,0.22616;-7.21443;-6.25786
    """
    other_points: Final[np.ndarray] = np.array(other)
    dim: Final[int] = other_points.shape[1]
    max_fes: Final[int] = 2048 + int(40 * (n ** (dim / 1.7)))

    max_radius: float = max(np.sqrt(np.square(pt).sum())
                            for pt in other_points)
    max_radius = max(max_radius ** 1.1, max_radius ** 0.6,
                     max_radius * 1.3)

    max_dim: float = np.max(np.abs(
        np.array(other_points).flatten()))
    max_dim = max(max_dim ** 1.2, max_dim ** (1.0 / 1.2),
                  max_dim ** 2.5, max_radius * 1.2)

    if log:
        logger(
            f"now determining {n} hopefully diverse samples of dimension "
            f"{dim} using {len(other_points)} other points, "
            f"max_radius={max_radius}, and max_dim={max_dim} for "
            f"{max_fes} FEs.")

    space: Final[VectorSpace] = VectorSpace(dim * n, -max_dim, max_dim)
    best: np.ndarray = space.create()

    class __Obj(Objective):
        def __init__(self):
            nonlocal other_points
            nonlocal max_radius
            nonlocal dim
            self.evaluate = cast(  # type: ignore
                Callable[[np.ndarray], float],
                lambda x, o=other_points, mr=max_radius, dd=dim:
                interesting_point_objective(x, o, mr, dd))

    with Execution().set_solution_space(space).set_rand_seed(1)\
            .set_algorithm(BiPopCMAES(space))\
            .set_max_fes(max_fes)\
            .set_objective(__Obj()).execute() as process:
        f = process.get_best_f()
        process.get_copy_of_best_x(best)
        best = interesting_point_transform(best, max_radius, dim)
        if log:
            logger(f"generated {n} points with objective {f}:\n{best!r}")
    return best
