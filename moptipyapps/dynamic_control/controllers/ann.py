"""
Poor man's Artificial Neural Networks.

Here, artificial neural networks (ANNs) are defined as plain mathematical
functions which are parameterized by their weights. The weights are subject
to black-box optimization and all together put into a single vector.
In other words, we do not use proper back-propagation learning or any other
sophisticated neural network specific training strategy. Instead, we treat the
neural networks as black boxes that can be parameterized using the weight
vector. Different ANN architectures have different weight vectors.
As activation functions, we use `arctan`.
"""

from typing import Final, Iterable

import numba  # type: ignore
import numpy as np

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_2d_1o_1(state: np.ndarray, _: float,
                  params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a man's ANN for 2d-spaces with one layer containing 1 node.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    out[0] = params[0] * np.arctan(params[1] + (params[2] * state[0])
                                   + (params[3] * state[1]))


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_3d_1o_1(state: np.ndarray, _: float,
                  params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 3d-spaces with one layer containing 1 node.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    out[0] = params[0] * np.arctan(
        params[1] + (params[2] * state[0]) + (params[3] * state[1])
        + (params[4] * state[2]))


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_2d_1o_2(state: np.ndarray, _: float,
                  params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 2d-spaces with one layer containing 2 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    hl_1_1 = np.arctan(params[0] + (params[1] * s0) + (params[2] * s1))
    hl_1_2 = np.arctan(params[3] + (params[4] * s0) + (params[5] * s1))
    out[0] = (params[6] * hl_1_1) + (params[7] * hl_1_2)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_3d_1o_2(state: np.ndarray, _: float,
                  params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 3d-spaces with one layer containing 2 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]
    hl_1_1 = np.arctan(params[0] + (params[1] * s0) + (params[2] * s1)
                       + (params[3] * s2))
    hl_1_2 = np.arctan(params[4] + (params[5] * s0) + (params[6] * s1)
                       + (params[7] * s2))
    out[0] = (params[8] * hl_1_1) + (params[9] * hl_1_2)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_2d_1o_3(state: np.ndarray, _: float,
                  params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 2d-spaces with one layer containing 3 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    hl_1_1 = np.arctan(params[0] + (params[1] * s0) + (params[2] * s1))
    hl_1_2 = np.arctan(params[3] + (params[4] * s0) + (params[5] * s1))
    hl_1_3 = np.arctan(params[6] + (params[7] * s0) + (params[8] * s1))
    out[0] = (params[9] * hl_1_1) + (params[10] * hl_1_2) \
        + (params[11] * hl_1_3)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_3d_1o_3(state: np.ndarray, _: float,
                  params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 3d-spaces with one layer containing 3 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]
    hl_1_1 = np.arctan(params[0] + (params[1] * s0) + (params[2] * s1)
                       + (params[3] * s2))
    hl_1_2 = np.arctan(params[4] + (params[5] * s0) + (params[6] * s1)
                       + (params[7] * s2))
    hl_1_3 = np.arctan(params[8] + (params[9] * s0) + (params[10] * s1)
                       + (params[11] * s2))
    out[0] = (params[12] * hl_1_1) + (params[13] * hl_1_2) \
        + (params[14] * hl_1_3)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_2d_1o_2_2(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 2d-spaces with 2 layers with 2 nodes each.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    hl_1_1 = np.arctan(params[0] + (params[1] * s0) + (params[2] * s1))
    hl_1_2 = np.arctan(params[3] + (params[4] * s0) + (params[5] * s1))
    hl_2_1 = np.arctan(params[6] + (params[7] * hl_1_1)
                       + (params[8] * hl_1_2))
    hl_2_2 = np.arctan(params[9] + (params[10] * hl_1_1)
                       + (params[11] * hl_1_2))
    out[0] = (params[12] * hl_2_1) + (params[13] * hl_2_2)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_3d_1o_2_2(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 3d-spaces with 2 layers with 2 nodes each.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]
    hl_1_1 = np.arctan(params[0] + (params[1] * s0) + (params[2] * s1)
                       + (params[3] * s2))
    hl_1_2 = np.arctan(params[4] + (params[5] * s0) + (params[6] * s1)
                       + (params[7] * s2))
    hl_2_1 = np.arctan(params[8] + (params[9] * hl_1_1)
                       + (params[10] * hl_1_2))
    hl_2_2 = np.arctan(params[11] + (params[12] * hl_1_1)
                       + (params[13] * hl_1_2))
    out[0] = (params[14] * hl_2_1) + (params[15] * hl_2_2)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_2d_1o_3_2(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 2d-spaces with 2 layers with 3 and 2 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    hl_1_1 = np.arctan(params[0] + (params[1] * s0) + (params[2] * s1))
    hl_1_2 = np.arctan(params[3] + (params[4] * s0) + (params[5] * s1))
    hl_1_3 = np.arctan(params[6] + (params[7] * s0) + (params[8] * s1))
    hl_2_1 = np.arctan(params[9] + (params[10] * hl_1_1)
                       + (params[11] * hl_1_2) + (params[12] * hl_1_3))
    hl_2_2 = np.arctan(params[13] + (params[14] * hl_1_1)
                       + (params[15] * hl_1_2) + (params[16] * hl_1_2))
    out[0] = (params[17] * hl_2_1) + (params[18] * hl_2_2)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __ann_3d_1o_3_2(state: np.ndarray, _: float,
                    params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a poor man's ANN for 3d-spaces with 2 layers with 3 and 2 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    s0: Final[float] = state[0]
    s1: Final[float] = state[1]
    s2: Final[float] = state[2]
    hl_1_1 = np.arctan(params[0] + (params[1] * s0) + (params[2] * s1)
                       + (params[3] * s2))
    hl_1_2 = np.arctan(params[4] + (params[5] * s0) + (params[6] * s1)
                       + (params[7] * s2))
    hl_1_3 = np.arctan(params[8] + (params[9] * s0) + (params[10] * s1)
                       + (params[11] * s2))
    hl_2_1 = np.arctan(params[12] + (params[13] * hl_1_1)
                       + (params[14] * hl_1_2) + (params[15] * hl_1_3))
    hl_2_2 = np.arctan(params[16] + (params[17] * hl_1_1)
                       + (params[18] * hl_1_2) + (params[19] * hl_1_2))
    out[0] = (params[20] * hl_2_1) + (params[21] * hl_2_2) \
        + (params[22] * hl_2_2)


def anns(system: System) -> Iterable[Controller]:
    """
    Create poor man's ANNs fitting to a given system.

    Based on the dimensionality of the state space, we generate a set of ANNs
    with different numbers of layers and neurons. The weights of the neurons
    can then directly be optimized by a numerical optimization algorithm.
    This is, of course, probably much less efficient that doing some proper
    learning like back-propagation. However, it allows us to easily plug the
    ANNs into the same optimization routines as other controllers.

    :param system: the equations object
    :return: the ANNs
    """
    if system.control_dims != 1:
        raise ValueError("invalid controller dimensions "
                         f"{system.control_dims} for {system!r}.")
    if system.state_dims == 2:
        return (Controller("ann_1", 2, 1, 4, __ann_2d_1o_1),
                Controller("ann_2", 2, 1, 8, __ann_2d_1o_2),
                Controller("ann_3", 2, 1, 12, __ann_2d_1o_3),
                Controller("ann_2_2", 2, 1, 14, __ann_2d_1o_2_2),
                Controller("ann_3_2", 2, 1, 19, __ann_2d_1o_3_2))
    if system.state_dims == 3:
        return (Controller("ann_1", 3, 1, 5, __ann_3d_1o_1),
                Controller("ann_2", 3, 1, 10, __ann_3d_1o_2),
                Controller("ann_3", 3, 1, 15, __ann_3d_1o_3),
                Controller("ann_2_2", 3, 1, 16, __ann_3d_1o_2_2),
                Controller("ann_3_2", 3, 1, 23, __ann_3d_1o_3_2))
    raise ValueError("invalid state dimensions "
                     f"{system.state_dims} for {system!r}.")
