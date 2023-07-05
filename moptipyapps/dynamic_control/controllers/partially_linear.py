"""
Partially linear controllers.

Partially linear controllers are encoded as sets of linear controllers and
anchor points. For each state, the linear controller with closest anchor
point is used.
"""

from typing import Final, Iterable

import numba  # type: ignore
import numpy as np

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __linear_2d_1o_2(state: np.ndarray, _: float,
                     params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a double linear function for 2d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]

    d: float = ((s0 - params[0]) ** 2.0) + ((s1 - params[1]) ** 2.0)
    o: float = (s0 * params[2]) + (s1 * params[3])

    d2: float = ((s0 - params[4]) ** 2.0) + ((s1 - params[5]) ** 2.0)
    if d2 < d:
        o = (s0 * params[6]) + (s1 * params[7])
    out[0] = o


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __linear_3d_1o_2(state: np.ndarray, _: float,
                     params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a double linear function for 3d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]

    d: float = (((s0 - params[0]) ** 2.0) + ((s1 - params[1]) ** 2.0)
                + ((s2 - params[2]) ** 2.0))
    o: float = (s0 * params[3]) + (s1 * params[4]) + (s2 * params[5])

    d2: float = (((s0 - params[6]) ** 2.0) + ((s1 - params[7]) ** 2.0)
                 + ((s2 - params[8]) ** 2.0))
    if d2 < d:
        o = (s0 * params[9]) + (s1 * params[10]) + (s2 * params[11])

    out[0] = o


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __linear_2d_1o_3(state: np.ndarray, _: float,
                     params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a triple linear function for 2d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]

    d: float = ((s0 - params[0]) ** 2.0) + ((s1 - params[1]) ** 2.0)
    o: float = (s0 * params[2]) + (s1 * params[3])

    d2: float = ((s0 - params[4]) ** 2.0) + ((s1 - params[5]) ** 2.0)
    if d2 < d:
        o = (s0 * params[6]) + (s1 * params[7])

    d2 = ((s0 - params[8]) ** 2.0) + ((s1 - params[9]) ** 2.0)
    if d2 < d:
        o = (s0 * params[10]) + (s1 * params[11])

    out[0] = o


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __linear_3d_1o_3(state: np.ndarray, _: float,
                     params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a double linear function for 3d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]

    d: float = (((s0 - params[0]) ** 2.0) + ((s1 - params[1]) ** 2.0)
                + ((s2 - params[2]) ** 2.0))
    o: float = (s0 * params[3]) + (s1 * params[4]) + (s2 * params[5])

    d2: float = (((s0 - params[6]) ** 2.0) + ((s1 - params[7]) ** 2.0)
                 + ((s2 - params[8]) ** 2.0))
    if d2 < d:
        o = (s0 * params[9]) + (s1 * params[10]) + (s2 * params[11])

    d2 = (((s0 - params[12]) ** 2.0) + ((s1 - params[13]) ** 2.0)
          + ((s2 - params[14]) ** 2.0))
    if d2 < d:
        o = (s0 * params[15]) + (s1 * params[16]) + (s2 * params[17])

    out[0] = o


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __linear_2d_1o_4(state: np.ndarray, _: float,
                     params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a quadruple linear function for 2d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]

    d: float = ((s0 - params[0]) ** 2.0) + ((s1 - params[1]) ** 2.0)
    o: float = (s0 * params[2]) + (s1 * params[3])

    d2: float = ((s0 - params[4]) ** 2.0) + ((s1 - params[5]) ** 2.0)
    if d2 < d:
        o = (s0 * params[6]) + (s1 * params[7])

    d2 = ((s0 - params[8]) ** 2.0) + ((s1 - params[9]) ** 2.0)
    if d2 < d:
        o = (s0 * params[10]) + (s1 * params[11])

    d2 = ((s0 - params[12]) ** 2.0) + ((s1 - params[13]) ** 2.0)
    if d2 < d:
        o = (s0 * params[14]) + (s1 * params[15])

    out[0] = o


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __linear_3d_1o_4(state: np.ndarray, _: float,
                     params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a quadruple linear function for 3d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]

    d: float = (((s0 - params[0]) ** 2.0) + ((s1 - params[1]) ** 2.0)
                + ((s2 - params[2]) ** 2.0))
    o: float = (s0 * params[3]) + (s1 * params[4]) + (s2 * params[5])

    d2: float = (((s0 - params[6]) ** 2.0) + ((s1 - params[7]) ** 2.0)
                 + ((s2 - params[8]) ** 2.0))
    if d2 < d:
        o = (s0 * params[9]) + (s1 * params[10]) + (s2 * params[11])

    d2 = (((s0 - params[12]) ** 2.0) + ((s1 - params[13]) ** 2.0)
          + ((s2 - params[14]) ** 2.0))
    if d2 < d:
        o = (s0 * params[15]) + (s1 * params[16]) + (s2 * params[17])

    d2 = (((s0 - params[18]) ** 2.0) + ((s1 - params[19]) ** 2.0)
          + ((s2 - params[20]) ** 2.0))
    if d2 < d:
        o = (s0 * params[21]) + (s1 * params[22]) + (s2 * params[23])

    out[0] = o


def partially_linear(system: System) -> Iterable[Controller]:
    """
    Create a several linear controllers for the given equations object.

    :param system: the equations object
    :return: the partially linear controllers
    """
    if system.control_dims != 1:
        raise ValueError("invalid controller dimensions "
                         f"{system.control_dims} for {system!r}.")
    if system.state_dims == 2:
        return (Controller("linear_2", 2, 1, 8, __linear_2d_1o_2),
                Controller("linear_3", 2, 1, 12, __linear_2d_1o_3),
                Controller("linear_4", 2, 1, 16, __linear_2d_1o_4))
    if system.state_dims == 3:
        return (Controller("linear_2", 3, 1, 12, __linear_3d_1o_2),
                Controller("linear_3", 3, 1, 18, __linear_3d_1o_3),
                Controller("linear_4", 3, 1, 24, __linear_3d_1o_4))
    raise ValueError("invalid state dimensions "
                     f"{system.state_dims} for {system!r}.")
