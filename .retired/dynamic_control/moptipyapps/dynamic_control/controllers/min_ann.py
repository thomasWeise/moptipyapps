"""
Poor man's Artificial Neural Networks with minimized input.

ANNs that include the state as input variable together with an additional
variable, say `z`. The controller output is then the value `z*` for which
the ANN takes on the smallest value (under the current state). In other
words, the ANN is supposed to model the system's objective function. The
idea is similar to :mod:`~moptipyapps.dynamic_control.controllers.ann`, but
instead of using the output of the ANNs as controller values, we use the value
`z*` for which the output of the ANN becomes minimal as controller value.
"""

from typing import Final, Iterable

import numba  # type: ignore
import numpy as np
from numpy import inf

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System

#: the golden ratio
PHI: Final[float] = 0.5 * (np.sqrt(5.0) + 1.0)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __min_ann_3d_1o_1(state: np.ndarray, _: float,
                      params: np.ndarray, out: np.ndarray) -> None:
    """
    Minimize a poor man's ANN for 3d-spaces with one layer containing 1 node.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed. The ANN has one additional parameter besides the `state`.
    We then find the value of this parameter for which the ANN takes on the
    smallest value via an iterated bracket - golden ratio search.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    hl_1_1_in: Final[float] = (state * params[0:3]).sum()
    x_3: Final[float] = params[3]

# first estimate of the best value
    x_best: float = 0.0
    f_best: float = np.arctan(hl_1_1_in + x_3 * x_best)

# now do bracketing
    x_low = x_a = -1000.0
    f_a: float = np.arctan(hl_1_1_in + x_3 * x_a)
    if f_a < f_best:
        x_best = x_a
        f_best = f_a

    x_c = x_b = -990.0
    f_c = f_b = np.arctan(hl_1_1_in + x_3 * x_b)
    if f_b < f_best:
        x_best = x_b
        f_best = f_b

# we try optimization in every single bracket we find
    found_bracket: bool = True
    ever_found_backet: bool = False
    while found_bracket:

        found_bracket = False
        while x_b < 1000.0:
            x_c = x_b + 10.0
            f_c = np.arctan(hl_1_1_in + x_3 * x_c)
            if f_c < f_best:
                x_best = x_c
                f_best = f_c
            if (f_c > f_b) and (f_b < f_a):
                ever_found_backet = found_bracket = True
                break
            x_a = x_b
            f_a = f_b
            x_b = x_c
            f_b = f_c

    # use huge range -1e3 ... 1e3
        if found_bracket:
            x_low = np.nextafter(x_a, inf)
            x_high = np.nextafter(x_c, -inf)
        elif ever_found_backet:
            break
        else:
            x_high = x_c

    # golden ratio algorithm
        delta = x_high - x_low
        while delta > 1e-12:
            delta /= PHI
            x_cc = x_high - delta

            f_cc: float = np.arctan(hl_1_1_in + x_3 * x_cc)
            if f_cc < f_best:
                x_best = x_cc
                f_best = f_cc

            x_dd = x_low + delta
            f_dd: float = np.arctan(hl_1_1_in + x_3 * x_dd)
            if f_dd < f_best:
                x_best = x_dd
                f_best = f_dd

            if f_cc < f_dd:
                x_high = np.nextafter(x_dd, -inf)
            else:
                x_low = np.nextafter(x_cc, inf)
            delta = x_high - x_low

        # move forward to next bracket
        x_a = x_b
        f_a = f_b
        x_b = x_c
        f_b = f_c

    out[0] = x_best


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __min_ann_2d_1o_1(state: np.ndarray, _: float,
                      params: np.ndarray, out: np.ndarray) -> None:
    """
    Minimize a poor man's ANN for 2d-spaces with one layer containing 1 node.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed. The ANN has one additional parameter besides the `state`.
    We then find the value of this parameter for which the ANN takes on the
    smallest value via an iterated bracket - golden ratio search.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    hl_1_1_in: Final[float] = (state * params[0:2]).sum()
    x_2: Final[float] = params[2]

# first estimate of the best value
    x_best: float = 0.0
    f_best: float = np.arctan(hl_1_1_in + x_2 * x_best)

# now do bracketing
    x_low = x_a = -1000.0
    f_a: float = np.arctan(hl_1_1_in + x_2 * x_a)
    if f_a < f_best:
        x_best = x_a
        f_best = f_a

    x_c = x_b = -990.0
    f_c = f_b = np.arctan(hl_1_1_in + x_2 * x_b)
    if f_b < f_best:
        x_best = x_b
        f_best = f_b

# we try optimization in every single bracket we find
    found_bracket: bool = True
    ever_found_backet: bool = False
    while found_bracket:

        found_bracket = False
        while x_b < 1000.0:
            x_c = x_b + 10.0
            f_c = np.arctan(hl_1_1_in + x_2 * x_c)
            if f_c < f_best:
                x_best = x_c
                f_best = f_c
            if (f_c > f_b) and (f_b < f_a):
                ever_found_backet = found_bracket = True
                break
            x_a = x_b
            f_a = f_b
            x_b = x_c
            f_b = f_c

    # use huge range -1e3 ... 1e3
        if found_bracket:
            x_low = np.nextafter(x_a, inf)
            x_high = np.nextafter(x_c, -inf)
        elif ever_found_backet:
            break
        else:
            x_high = x_c

    # golden ratio algorithm
        delta = x_high - x_low
        while delta > 1e-12:
            delta /= PHI
            x_cc = x_high - delta

            f_cc: float = np.arctan(hl_1_1_in + x_2 * x_cc)
            if f_cc < f_best:
                x_best = x_cc
                f_best = f_cc

            x_dd = x_low + delta
            f_dd: float = np.arctan(hl_1_1_in + x_2 * x_dd)
            if f_dd < f_best:
                x_best = x_dd
                f_best = f_dd

            if f_cc < f_dd:
                x_high = np.nextafter(x_dd, -inf)
            else:
                x_low = np.nextafter(x_cc, inf)
            delta = x_high - x_low

        # move forward to next bracket
        x_a = x_b
        f_a = f_b
        x_b = x_c
        f_b = f_c

    out[0] = x_best


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __min_ann_3d_1o_2(state: np.ndarray, _: float,
                      params: np.ndarray, out: np.ndarray) -> None:
    """
    Minimize a poor man's ANN for 3d-spaces with one layer containing 2 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed. The ANN has one additional parameter besides the `state`.
    We then find the value of this parameter for which the ANN takes on the
    smallest value via an iterated bracket - golden ratio search.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    hl_1_1_in: Final[float] = (state * params[0:3]).sum()
    x_3: Final[float] = params[3]
    x_4: Final[float] = params[4]
    hl_1_2_in: Final[float] = (state * params[5:8]).sum()
    x_8: Final[float] = params[8]
    x_9: Final[float] = params[9]
    x_10: Final[float] = params[10]
    x_11: Final[float] = params[11]

# first estimate of the best value
    x_best: float = 0.0
    hl_1_1: float = np.arctan(hl_1_1_in + x_3 * x_best + x_4)
    hl_1_2: float = np.arctan(hl_1_2_in + x_8 * x_best + x_9)
    f_best: float = hl_1_1 * x_10 + hl_1_2 * x_11

# now do bracketing
    x_low = x_a = -1000.0
    hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_a + x_4)
    hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_a + x_9)
    f_a: float = hl_1_1 * x_10 + hl_1_2 * x_11
    if f_a < f_best:
        x_best = x_a
        f_best = f_a

    x_c = x_b = -990.0
    hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_b + x_4)
    hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_b + x_9)
    f_c = f_b = hl_1_1 * x_10 + hl_1_2 * x_11
    if f_b < f_best:
        x_best = x_b
        f_best = f_b

# we try optimization in every single bracket we find
    found_bracket: bool = True
    ever_found_backet: bool = False
    while found_bracket:

        found_bracket = False
        while x_b < 1000.0:
            x_c = x_b + 10.0
            hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_c + x_4)
            hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_c + x_9)
            f_c = hl_1_1 * x_10 + hl_1_2 * x_11
            if f_c < f_best:
                x_best = x_c
                f_best = f_c
            if (f_c > f_b) and (f_b < f_a):
                ever_found_backet = found_bracket = True
                break
            x_a = x_b
            f_a = f_b
            x_b = x_c
            f_b = f_c

    # use huge range -1e3 ... 1e3
        if found_bracket:
            x_low = np.nextafter(x_a, inf)
            x_high = np.nextafter(x_c, -inf)
        elif ever_found_backet:
            break
        else:
            x_high = x_c

    # golden ratio algorithm
        delta = x_high - x_low
        while delta > 1e-12:
            delta /= PHI
            x_cc = x_high - delta

            hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_cc + x_4)
            hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_cc + x_9)
            f_cc: float = hl_1_1 * x_10 + hl_1_2 * x_11
            if f_cc < f_best:
                x_best = x_cc
                f_best = f_cc

            x_dd = x_low + delta
            hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_dd + x_4)
            hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_dd + x_9)
            f_dd: float = hl_1_1 * x_10 + hl_1_2 * x_11
            if f_dd < f_best:
                x_best = x_dd
                f_best = f_dd

            if f_cc < f_dd:
                x_high = np.nextafter(x_dd, -inf)
            else:
                x_low = np.nextafter(x_cc, inf)
            delta = x_high - x_low

        # move forward to next bracket
        x_a = x_b
        f_a = f_b
        x_b = x_c
        f_b = f_c

    out[0] = x_best


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __min_ann_2d_1o_2(state: np.ndarray, _: float,
                      params: np.ndarray, out: np.ndarray) -> None:
    """
    Minimize a poor man's ANN for 2d-spaces with one layer containing 2 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed. The ANN has one additional parameter besides the `state`.
    We then find the value of this parameter for which the ANN takes on the
    smallest value via an iterated bracket - golden ratio search.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    hl_1_1_in: Final[float] = (state * params[0:2]).sum()
    x_2: Final[float] = params[2]
    x_3: Final[float] = params[3]
    hl_1_2_in: Final[float] = (state * params[4:6]).sum()
    x_6: Final[float] = params[6]
    x_7: Final[float] = params[7]
    x_8: Final[float] = params[8]
    x_9: Final[float] = params[9]

# first estimate of the best value
    x_best: float = 0.0
    hl_1_1: float = np.arctan(hl_1_1_in + x_2 * x_best + x_3)
    hl_1_2: float = np.arctan(hl_1_2_in + x_6 * x_best + x_7)
    f_best: float = hl_1_1 * x_8 + hl_1_2 * x_9

# now do bracketing
    x_low = x_a = -1000.0
    hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_a + x_3)
    hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_a + x_7)
    f_a: float = hl_1_1 * x_8 + hl_1_2 * x_9
    if f_a < f_best:
        x_best = x_a
        f_best = f_a

    x_c = x_b = -990.0
    hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_b + x_3)
    hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_b + x_7)
    f_c = f_b = hl_1_1 * x_8 + hl_1_2 * x_9
    if f_b < f_best:
        x_best = x_b
        f_best = f_b

# we try optimization in every single bracket we find
    found_bracket: bool = True
    ever_found_backet: bool = False
    while found_bracket:

        found_bracket = False
        while x_b < 1000.0:
            x_c = x_b + 10.0
            hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_c + x_3)
            hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_c + x_7)
            f_c = hl_1_1 * x_8 + hl_1_2 * x_9
            if f_c < f_best:
                x_best = x_c
                f_best = f_c
            if (f_c > f_b) and (f_b < f_a):
                ever_found_backet = found_bracket = True
                break
            x_a = x_b
            f_a = f_b
            x_b = x_c
            f_b = f_c

    # use huge range -1e3 ... 1e3
        if found_bracket:
            x_low = np.nextafter(x_a, inf)
            x_high = np.nextafter(x_c, -inf)
        elif ever_found_backet:
            break
        else:
            x_high = x_c

    # golden ratio algorithm
        delta = x_high - x_low
        while delta > 1e-12:
            delta /= PHI
            x_cc = x_high - delta

            hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_cc + x_3)
            hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_cc + x_7)
            f_cc: float = hl_1_1 * x_8 + hl_1_2 * x_9
            if f_cc < f_best:
                x_best = x_cc
                f_best = f_cc

            x_dd = x_low + delta
            hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_dd + x_3)
            hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_dd + x_7)
            f_dd: float = hl_1_1 * x_8 + hl_1_2 * x_9
            if f_dd < f_best:
                x_best = x_dd
                f_best = f_dd

            if f_cc < f_dd:
                x_high = np.nextafter(x_dd, -inf)
            else:
                x_low = np.nextafter(x_cc, inf)
            delta = x_high - x_low

        # move forward to next bracket
        x_a = x_b
        f_a = f_b
        x_b = x_c
        f_b = f_c

    out[0] = x_best


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __min_ann_3d_1o_3(state: np.ndarray, _: float,
                      params: np.ndarray, out: np.ndarray) -> None:
    """
    Minimize a poor man's ANN for 3d-spaces with one layer containing 3 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed. The ANN has one additional parameter besides the `state`.
    We then find the value of this parameter for which the ANN takes on the
    smallest value via an iterated bracket - golden ratio search.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    hl_1_1_in: Final[float] = (state * params[0:3]).sum()
    x_3: Final[float] = params[3]
    x_4: Final[float] = params[4]
    hl_1_2_in: Final[float] = (state * params[5:8]).sum()
    x_8: Final[float] = params[8]
    x_9: Final[float] = params[9]
    hl_1_3_in: Final[float] = (state * params[10:13]).sum()
    x_13: Final[float] = params[13]
    x_14: Final[float] = params[14]
    x_15: Final[float] = params[15]
    x_16: Final[float] = params[16]
    x_17: Final[float] = params[17]

# first estimate of the best value
    x_best: float = 0.0
    hl_1_1: float = np.arctan(hl_1_1_in + x_3 * x_best + x_4)
    hl_1_2: float = np.arctan(hl_1_2_in + x_8 * x_best + x_9)
    hl_1_3: float = np.arctan(hl_1_3_in + x_13 * x_best + x_14)
    f_best: float = hl_1_1 * x_15 + hl_1_2 * x_16 + hl_1_3 * x_17

# now do bracketing
    x_low = x_a = -1000.0
    hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_a + x_4)
    hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_a + x_9)
    hl_1_3 = np.arctan(hl_1_3_in + x_13 * x_a + x_14)
    f_a = hl_1_1 * x_15 + hl_1_2 * x_16 + hl_1_3 * x_17
    if f_a < f_best:
        x_best = x_a
        f_best = f_a

    x_c = x_b = -990.0
    hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_b + x_4)
    hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_b + x_9)
    hl_1_3 = np.arctan(hl_1_3_in + x_13 * x_b + x_14)
    f_c = f_b = hl_1_1 * x_15 + hl_1_2 * x_16 + hl_1_3 * x_17
    if f_b < f_best:
        x_best = x_b
        f_best = f_b

# we try optimization in every single bracket we find
    found_bracket: bool = True
    ever_found_backet: bool = False
    while found_bracket:

        found_bracket = False
        while x_b < 1000.0:
            x_c = x_b + 10.0
            hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_c + x_4)
            hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_c + x_9)
            hl_1_3 = np.arctan(hl_1_3_in + x_13 * x_c + x_14)
            f_c = hl_1_1 * x_15 + hl_1_2 * x_16 + hl_1_3 * x_17
            if f_c < f_best:
                x_best = x_c
                f_best = f_c
            if (f_c > f_b) and (f_b < f_a):
                ever_found_backet = found_bracket = True
                break
            x_a = x_b
            f_a = f_b
            x_b = x_c
            f_b = f_c

    # use huge range -1e3 ... 1e3
        if found_bracket:
            x_low = np.nextafter(x_a, inf)
            x_high = np.nextafter(x_c, -inf)
        elif ever_found_backet:
            break
        else:
            x_high = x_c

    # golden ratio algorithm
        delta = x_high - x_low
        while delta > 1e-12:
            delta /= PHI
            x_cc = x_high - delta
            hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_cc + x_4)
            hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_cc + x_9)
            hl_1_3 = np.arctan(hl_1_3_in + x_13 * x_cc + x_14)
            f_cc = hl_1_1 * x_15 + hl_1_2 * x_16 + hl_1_3 * x_17
            if f_cc < f_best:
                x_best = x_cc
                f_best = f_cc

            x_dd = x_low + delta
            hl_1_1 = np.arctan(hl_1_1_in + x_3 * x_dd + x_4)
            hl_1_2 = np.arctan(hl_1_2_in + x_8 * x_dd + x_9)
            hl_1_3 = np.arctan(hl_1_3_in + x_13 * x_dd + x_14)
            f_dd = hl_1_1 * x_15 + hl_1_2 * x_16 + hl_1_3 * x_17
            if f_dd < f_best:
                x_best = x_dd
                f_best = f_dd

            if f_cc < f_dd:
                x_high = np.nextafter(x_dd, -inf)
            else:
                x_low = np.nextafter(x_cc, inf)
            delta = x_high - x_low

        # move forward to next bracket
        x_a = x_b
        f_a = f_b
        x_b = x_c
        f_b = f_c

    out[0] = x_best


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __min_ann_2d_1o_3(state: np.ndarray, _: float,
                      params: np.ndarray, out: np.ndarray) -> None:
    """
    Minimize a poor man's ANN for 2d-spaces with one layer containing 3 nodes.

    We use :func:`numpy.arctan` as activation function. The output layer
    is just a weighted sum of the hidden layer and no other transformation
    is performed. The ANN has one additional parameter besides the `state`.
    We then find the value of this parameter for which the ANN takes on the
    smallest value via an iterated bracket - golden ratio search.

    :param state: the current state of the system
    :param params: the weight vector for the neurons.
    :param out: the control vector, receiving one single element
    """
    hl_1_1_in: Final[float] = (state * params[0:2]).sum()
    x_2: Final[float] = params[2]
    x_3: Final[float] = params[3]
    hl_1_2_in: Final[float] = (state * params[4:6]).sum()
    x_6: Final[float] = params[6]
    x_7: Final[float] = params[7]
    hl_1_3_in: Final[float] = (state * params[8:10]).sum()
    x_10: Final[float] = params[10]
    x_11: Final[float] = params[11]
    x_12: Final[float] = params[12]
    x_13: Final[float] = params[13]
    x_14: Final[float] = params[14]

# first estimate of the best value
    x_best: float = 0.0
    hl_1_1: float = np.arctan(hl_1_1_in + x_2 * x_best + x_3)
    hl_1_2: float = np.arctan(hl_1_2_in + x_6 * x_best + x_7)
    hl_1_3: float = np.arctan(hl_1_3_in + x_10 * x_best + x_11)
    f_best: float = hl_1_1 * x_12 + hl_1_2 * x_13 + hl_1_3 * x_14

# now do bracketing
    x_low = x_a = -1000.0
    hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_a + x_3)
    hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_a + x_7)
    hl_1_3 = np.arctan(hl_1_3_in + x_10 * x_a + x_11)
    f_a: float = hl_1_1 * x_12 + hl_1_2 * x_13 + hl_1_3 * x_14
    if f_a < f_best:
        x_best = x_a
        f_best = f_a

    x_c = x_b = -990.0

    hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_b + x_3)
    hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_b + x_7)
    hl_1_3 = np.arctan(hl_1_3_in + x_10 * x_b + x_11)
    f_c = f_b = hl_1_1 * x_12 + hl_1_2 * x_13 + hl_1_3 * x_14
    if f_b < f_best:
        x_best = x_b
        f_best = f_b

# we try optimization in every single bracket we find
    found_bracket: bool = True
    ever_found_backet: bool = False
    while found_bracket:

        found_bracket = False
        while x_b < 1000.0:
            x_c = x_b + 10.0
            hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_c + x_3)
            hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_c + x_7)
            hl_1_3 = np.arctan(hl_1_3_in + x_10 * x_c + x_11)
            f_c = hl_1_1 * x_12 + hl_1_2 * x_13 + hl_1_3 * x_14
            if f_c < f_best:
                x_best = x_c
                f_best = f_c
            if (f_c > f_b) and (f_b < f_a):
                ever_found_backet = found_bracket = True
                break
            x_a = x_b
            f_a = f_b
            x_b = x_c
            f_b = f_c

    # use huge range -1e3 ... 1e3
        if found_bracket:
            x_low = np.nextafter(x_a, inf)
            x_high = np.nextafter(x_c, -inf)
        elif ever_found_backet:
            break
        else:
            x_high = x_c

    # golden ratio algorithm
        delta = x_high - x_low
        while delta > 1e-12:
            delta /= PHI
            x_cc = x_high - delta
            hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_cc + x_3)
            hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_cc + x_7)
            hl_1_3 = np.arctan(hl_1_3_in + x_10 * x_cc + x_11)
            f_cc: float = hl_1_1 * x_12 + hl_1_2 * x_13 + hl_1_3 * x_14
            if f_cc < f_best:
                x_best = x_cc
                f_best = f_cc

            x_dd = x_low + delta
            hl_1_1 = np.arctan(hl_1_1_in + x_2 * x_dd + x_3)
            hl_1_2 = np.arctan(hl_1_2_in + x_6 * x_dd + x_7)
            hl_1_3 = np.arctan(hl_1_3_in + x_10 * x_dd + x_11)
            f_dd: float = hl_1_1 * x_12 + hl_1_2 * x_13 + hl_1_3 * x_14
            if f_dd < f_best:
                x_best = x_dd
                f_best = f_dd

            if f_cc < f_dd:
                x_high = np.nextafter(x_dd, -inf)
            else:
                x_low = np.nextafter(x_cc, inf)
            delta = x_high - x_low

        # move forward to next bracket
        x_a = x_b
        f_a = f_b
        x_b = x_c
        f_b = f_c

    out[0] = x_best


def min_anns(system: System) -> Iterable[Controller]:
    """
    Create poor man's ANNs for modeling the objective function.

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
        return (Controller("min_ann_1", 2, 1, 3, __min_ann_2d_1o_1),
                Controller("min_ann_2", 2, 1, 10, __min_ann_2d_1o_2),
                Controller("min_ann_3", 2, 1, 15, __min_ann_2d_1o_3))
    if system.state_dims == 3:
        return (Controller("min_ann_1", 3, 1, 4, __min_ann_3d_1o_1),
                Controller("min_ann_2", 3, 1, 12, __min_ann_3d_1o_2),
                Controller("min_ann_3", 3, 1, 18, __min_ann_3d_1o_3))
    raise ValueError("invalid state dimensions "
                     f"{system.state_dims} for {system!r}.")
