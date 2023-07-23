"""
A cubic controller.

A cubic controller is a function where all value of the state vector enter
the computation plainly, squared, and raised to the third power. The powers
of their combinations do not exceed the third power, e.g., the controller is
a linear combination of A, B, A², AB, B², A³, A²B, B²A, and B³ if the state
has values A and B. The controller represents the multipliers for these
coefficients.
"""

from typing import Final

import numba  # type: ignore
import numpy as np

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __cubic_2d_1o(state: np.ndarray, _: float,
                  params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a cubic polynomial for 2d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s02: Final[float] = s0 * s0
    s12: Final[float] = s1 * s1
    out[0] = (s0 * params[0]) + (s1 * params[1]) \
        + (s02 * params[2]) + (s0 * s1 * params[3]) + (s12 * params[4]) \
        + (s02 * s0 * params[5]) + (s02 * s1 * params[6]) \
        + (s0 * s12 * params[7]) + (s12 * s1 * params[8])


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __cubic_3d_1o(state: np.ndarray, _: float,
                  params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a cubic polynomial for 3d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]
    s02: Final[float] = s0 * s0
    s12: Final[float] = s1 * s1
    s22: Final[float] = s2 * s2
    out[0] = (s0 * params[0]) + (s1 * params[1]) + (s2 * params[2]) \
        + (s02 * params[3]) + (s0 * s1 * params[4]) \
        + (s0 * s2 * params[5]) + (s12 * params[6]) \
        + (s1 * s2 * params[7]) + (s22 * params[8]) \
        + (s0 * s02 * params[9]) + (s02 * s1 * params[10]) \
        + (s02 * s2 * params[11]) + (s12 * s1 * params[12]) \
        + (s12 * s0 * params[13]) + (s12 * s2 * params[14]) \
        + (s22 * s2 * params[15]) + (s22 * s0 * params[16]) \
        + (s22 * s1 * params[17])


def cubic(system: System) -> Controller:
    """
    Create a cubic controller for the given equations object.

    :param system: the equations object
    :return: the cubic controller
    """
    if system.control_dims != 1:
        raise ValueError("invalid controller dimensions "
                         f"{system.control_dims} for {system!r}.")
    name: Final[str] = "cubic"
    if system.state_dims == 2:
        return Controller(name, 2, 1, 9, __cubic_2d_1o)
    if system.state_dims == 3:
        return Controller(name, 3, 1, 19, __cubic_3d_1o)
    raise ValueError("invalid state dimensions "
                     f"{system.state_dims} for {system!r}.")
