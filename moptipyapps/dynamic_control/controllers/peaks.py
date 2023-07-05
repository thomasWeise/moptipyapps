"""Peak functions."""

from typing import Final, Iterable

import numba  # type: ignore
import numpy as np

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __peak(a: float) -> float:
    """
    Compute the peak function.

    :param a: the input
    :return: the output
    """
    return np.exp(-(a * a))


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __peaks_2d_1o_1(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a single peak for 2d-spaces.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    out[0] = params[0] * __peak(params[1] + (params[2] * state[0])
                                + (params[3] * state[1]))


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __peaks_3d_1o_1(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a single peak for 3d-spaces.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    out[0] = params[0] * __peak(
        params[1] + (params[2] * state[0]) + (params[3] * state[1])
        + (params[4] * state[2]))


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __peaks_2d_1o_2(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute two peaks for 2d-spaces.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    out[0] = params[0] * __peak(params[1] + (params[2] * s0)
                                + (params[3] * s1)) \
        + params[4] * __peak(params[5] + (params[6] * s0)
                             + (params[7] * s1))


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __peaks_3d_1o_2(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a two peaks for 3d-spaces.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]
    out[0] = params[0] * __peak(
        params[1] + (params[2] * s0) + (params[3] * s1) + (params[4] * s2))\
        + params[5] * __peak(
        params[6] + (params[7] * s0) + (params[8] * s1) + (params[9] * s2))


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __peaks_2d_1o_3(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute three peaks for 2d-spaces.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    out[0] = params[0] * __peak(params[1] + (params[2] * s0)
                                + (params[3] * s1)) \
        + params[4] * __peak(params[5] + (params[6] * s0)
                             + (params[7] * s1)) \
        + params[8] * __peak(params[9] + (params[10] * s0)
                             + (params[11] * s1))


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __peaks_3d_1o_3(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a single peak for 3d-spaces.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]
    out[0] = params[0] * __peak(
        params[1] + (params[2] * s0) + (params[3] * s1) + (params[4] * s2))\
        + params[5] * __peak(
        params[6] + (params[7] * s0) + (params[8] * s1) + (params[9] * s2))\
        + params[10] * __peak(
        params[11] + (params[12] * s0) + (params[13] * s1)
        + (params[14] * s2))


def peaks(system: System) -> Iterable[Controller]:
    """
    Create poor man's PNNs fitting to a given system.

    Based on the dimensionality of the state space, we generate a set of PNNs
    with different numbers of layers and neurons. The weights of the neurons
    can then directly be optimized by a numerical optimization algorithm.
    This is, of course, probably much less efficient that doing some proper
    learning like back-propagation. However, it allows us to easily plug the
    PNNs into the same optimization routines as other controllers.

    :param system: the equations object
    :return: the PNNs
    """
    if system.control_dims != 1:
        raise ValueError("invalid controller dimensions "
                         f"{system.control_dims} for {system!r}.")
    if system.state_dims == 2:
        return (Controller("peaks_1", 2, 1, 4, __peaks_2d_1o_1),
                Controller("peaks_2", 2, 1, 8, __peaks_2d_1o_2),
                Controller("peaks_3", 2, 1, 12, __peaks_2d_1o_3))
    if system.state_dims == 3:
        return (Controller("peaks_1", 3, 1, 5, __peaks_3d_1o_1),
                Controller("peaks_2", 3, 1, 10, __peaks_3d_1o_2),
                Controller("peaks_3", 3, 1, 15, __peaks_3d_1o_3))
    raise ValueError("invalid state dimensions "
                     f"{system.state_dims} for {system!r}.")
