"""The permutation-based encoding."""


import numba  # type: ignore
import numpy as np
from moptipy.api.encoding import Encoding

from moptipyapps.spoc.spoc_4.challenge_1.beginner.base_obj import (
    BaseObjectWithArrays,
)


@numba.njit(cache=True, inline="always", fastmath=False, boundscheck=False)
def _decode(x: np.ndarray, y: np.ndarray, data: np.ndarray,
            earth: np.ndarray, lunar: np.ndarray, dest: np.ndarray) -> None:
    """
    Decode the permutation to a selection.

    :param x: the candidate solution
    :param y: the candidate solution
    :param data: the orbit data
    :param earth: the earth orbits
    :param lunar: the lunar orbits
    :param dest: the destination orbits
    """
    earth.fill(0)
    lunar.fill(0)
    dest.fill(0)
    y.fill(0)
    for index in x:
        ue = data[index, 0]
        if earth[ue]:
            continue
        earth[ue] = 1
        ul = data[index, 1]
        if lunar[ul]:
            continue
        lunar[ul] = 1
        ud = data[index, 2]
        if dest[ud]:
            continue
        dest[ud] = 1
        y[index] = True


class PermutationEncoding(BaseObjectWithArrays, Encoding):
    """The permutation-based decoding function of the beginner problem."""

    def decode(self, x, y) -> None:
        """
        Decode a permutation to a selection.

        :param x: the permutation
        :param y: the selection
        """
        _decode(x, y, self.instance, self.earth, self.lunar, self.dest)
