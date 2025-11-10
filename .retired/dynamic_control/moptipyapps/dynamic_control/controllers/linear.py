"""
A linear controller.

In a linear controller, all values of the state vector enter only as-is,
i.e., it is a linear combination of A and B if the state is composed of
the two values A and B. We then optimize the weights of these coefficients.
"""

from typing import Final

import numba  # type: ignore
import numpy as np

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __linear_2d_1o(state: np.ndarray, _: float,
                   params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a linear function for 2d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    out[0] = (state[0] * params[0]) + (state[1] * params[1])


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __linear_3d_1o(state: np.ndarray, _: float,
                   params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a linear function for 3d state spaces.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    out[0] = (state[0] * params[0]) + (state[1] * params[1]) \
        + (state[2] * params[2])


def linear(system: System) -> Controller:
    """
    Create a linear controller for the given equations object.

    :param system: the equations object
    :return: the linear controller
    """
    if system.control_dims != 1:
        raise ValueError("invalid controller dimensions "
                         f"{system.control_dims} for {system!r}.")
    name: Final[str] = "linear"
    if system.state_dims == 2:
        return Controller(name, 2, 1, 2, __linear_2d_1o)
    if system.state_dims == 3:
        return Controller(name, 3, 1, 3, __linear_3d_1o)
    raise ValueError("invalid state dimensions "
                     f"{system.state_dims} for {system!r}.")
