"""
A quadratic controller.

A quadratic controller is a function where all value of the state vector enter
the computation plainly and squared. The powers of their combinations do not
exceed two, e.g., the controller is a linear combination of A, B, A², AB, and
B² if the state has values A and B. The controller represents the multipliers
for these coefficients.
"""

from typing import Final

import numba  # type: ignore
import numpy as np

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __quadratic_2d_1o(state: np.ndarray, _: float,
                      params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a quadratic function for 2d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    out[0] = (s0 * params[0]) + (s1 * params[1]) + (s0 * s0 * params[2]) \
        + (s0 * s1 * params[3]) + (s1 * s1 * params[4])


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __quadratic_3d_1o(state: np.ndarray, _: float,
                      params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a quadratic function for 3d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]
    out[0] = (s0 * params[0]) + (s1 * params[1]) + (s2 * params[2]) \
        + (s0 * s0 * params[3]) + (s0 * s1 * params[4]) \
        + (s0 * s2 * params[5]) + (s1 * s1 * params[6]) \
        + (s1 * s2 * params[7]) + (s2 * s2 * params[8])


def quadratic(system: System) -> Controller:
    """
    Create a quadratic controller for the given equations object.

    :param system: the equations object
    :return: the quadratic controller
    """
    if system.control_dims != 1:
        raise ValueError("invalid controller dimensions "
                         f"{system.control_dims} for {system!r}.")
    name: Final[str] = "quadratic"
    if system.state_dims == 2:
        return Controller(name, 2, 1, 5, __quadratic_2d_1o)
    if system.state_dims == 3:
        return Controller(name, 3, 1, 9, __quadratic_3d_1o)
    raise ValueError("invalid state dimensions "
                     f"{system.state_dims} for {system!r}.")
