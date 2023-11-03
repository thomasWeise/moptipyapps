"""Some examples for distance metrics."""

from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.utils.nputils import DEFAULT_BOOL


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def swap_distance(p1: np.ndarray, p2: np.ndarray) -> int:
    """
    Compute the swap distance between two permutations `p1` and `p1`.

    This is the minimum number of swaps required to translate `p1` to `p2` and
    vice versa. This function is symmatric.

    An upper bound for the number of maximum number of swaps that could be
    required is the length of the permutation. This upper bound can be derived
    from Selection Sort. Imagine that I want to translate the array `p1` to
    `p2`. I go through `p1` from beginning to end. If, at index `i`, I find
    the right element (`p1[i] == p2[i]`), then I do nothing. If not, then the
    right element must come at some index `j>i` (because all elements before I
    already have fixed). So I swap `p1[i]` with `p1[j]`. Now `p1[i] == p2[i]`
    and I increment `i`. Once I arrive at the end of `p1`, it must hold that
    `all(p1[i] == p2[i])`. At the same time, I have performed at most one swap
    at each index during the iteration. Hence, I can never need more swaps
    than the arrays are long.

    :param p1: the first permutation
    :param p2: the second permutation
    :return: the swap distance, always between `0` and `len(p1)`

    >>> swap_distance(np.array([0, 1, 2, 3]), np.array([3, 1, 2, 0]))
    1
    >>> swap_distance(np.array([0, 1, 2]), np.array([0, 1, 2]))
    0
    >>> swap_distance(np.array([1, 0, 2]), np.array([0, 1, 2]))
    1
    >>> swap_distance(np.array([0, 1, 2]), np.array([1, 0, 2]))
    1
    >>> swap_distance(np.array([0, 1, 2]), np.array([2, 0, 1]))
    2
    >>> swap_distance(np.array([2, 0, 1]), np.array([0, 1, 2]))
    2
    >>> swap_distance(np.arange(10), np.array([4, 8, 1, 5, 9, 3, 6, 0, 7, 2]))
    7
    >>> swap_distance(np.array([4, 8, 1, 5, 9, 3, 6, 0, 7, 2]), np.arange(10))
    7
    """
    n: Final[int] = len(p1)
    x: np.ndarray = p2[np.argsort(p1)]
    unchecked: np.ndarray = np.ones(n, DEFAULT_BOOL)
    result: int = 0

    for i in range(n):
        if unchecked[i]:
            result += 1
            unchecked[i] = False
            j = x[i]
            while j != i:
                unchecked[j] = False
                j = x[j]

    return n - result
