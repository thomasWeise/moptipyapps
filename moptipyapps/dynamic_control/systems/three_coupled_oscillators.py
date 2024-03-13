"""
A system of three coupled oscillators.

There are three oscillators located in a two-dimensional plane. Thus, there
are a total of six coordinates.

The initial starting point of the work here were conversations with
Prof. Dr. Bernd NOACK and Guy Yoslan CORNEJO MACEDA of the Harbin Institute
of Technology in Shenzhen, China (哈尔滨工业大学(深圳)) as well as the
following paper:

1. Ruiying Li, Bernd R. Noack, Laurent Cordier, Jacques Borée, Eurika Kaiser,
   and Fabien Harambat. Linear genetic programming control for strongly
   nonlinear dynamics with frequency crosstalk. *Archives of Mechanics.*
   70(6):505-534. Warszawa 2018. Seventy Years of the Archives of Mechanics.
   https://doi.org/10.24423/aom.3000. Also: arXiv:1705.00367v1
   [physics.flu-dyn] 30 Apr 2017. https://arxiv.org/abs/1705.00367.
"""

from math import pi
from typing import Final

import numba  # type: ignore
import numpy as np
from pycommons.io.path import Path
from pycommons.types import check_int_range

from moptipyapps.dynamic_control.starting_points import (
    make_interesting_starting_points,
)
from moptipyapps.dynamic_control.system import System

#: pi to the square
__PI2: Final[float] = pi * pi


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __3_coupled_oscillators(state: np.ndarray, _: float,
                            control: np.ndarray, out: np.ndarray) -> None:
    """
    Compute the differential equations of a controlled 3-oscillators system.

    The system is six-dimensional. There are three oscillators, and each one
    is updated. There is one control value.
    This function implements equation (3.1) of the paper.

    :param state: the state of the system
    :param _: the time index, which is ignored
    :param control: the output of the controller
    :param out: the differential, i.e., the output of this function
    """
    a1: Final[float] = state[0]
    a2: Final[float] = state[1]
    a3: Final[float] = state[2]
    a4: Final[float] = state[3]
    a5: Final[float] = state[4]
    a6: Final[float] = state[5]

    r1sqr: Final[float] = (a1 * a1) + (a2 * a2)
    r2sqr: Final[float] = (a3 * a3) + (a4 * a4)
    r3sqr: Final[float] = (a5 * a5) + (a6 * a6)
    sigma1: Final[float] = -r1sqr + r2sqr - r3sqr
    sigma2: Final[float] = 0.1 - r2sqr
    sigma3: Final[float] = -0.1

    b: Final[float] = control[0]

    # compute the differential system
    out[0] = (sigma1 * a1) - a2
    out[1] = (sigma1 * a2) + a1
    out[2] = (sigma2 * a3) - (pi * a4)
    out[3] = (sigma2 * a4) + (pi * a3) + b
    out[4] = (sigma3 * a5) - (__PI2 * a6)
    out[5] = (sigma3 * a6) + (__PI2 * a5) + b


def _sinus_control(
        _: np.ndarray, time: float, __: float, out: np.ndarray) -> None:
    """
    Present the sinus-based control law from the paper.

    :param _: the state, ignored
    :param time: the time
    :param __: ignored
    :param out: the output destination
    """
    out[0] = 0.07 * np.sin(__PI2 * time)


def _lgpc3_36(
        state: np.ndarray, time: float, _: float, out: np.ndarray) -> None:
    """
    Present the LGPC-3 equation 3.6.

    :param state: the state
    :param time: the time
    :param _: ignored
    :param out: the output destination
    """
    out[0] = np.tanh(np.sin(np.tanh(
        3.0 * state[1] * np.sin(time) * np.sin(__PI2 * time) - state[3])))


class __TO(System):
    """The internal three oscillators class."""

    def describe_system_without_control(
            self, dest_dir: str, skip_if_exists: bool = True) \
            -> tuple[Path, ...]:
        result = list(super().describe_system_without_control(
            dest_dir, skip_if_exists))
        result.extend(self.describe_system(
            "open loop 0.07sin(\u03C0²t)", _sinus_control, 0.0,
            f"{self.name}_open_loop", dest_dir, skip_if_exists))
        result.extend(self.describe_system(
            "LGPC-3", _lgpc3_36, 0.0,
            f"{self.name}_lgpc3", dest_dir, skip_if_exists))
        return tuple(result)


def make_3_couple_oscillators(n_points: int) -> System:
    """
    Create the oscillator system.

    :param n_points: the number of training points
    :return: the Lorenz system
    """
    check_int_range(n_points, "n_points", 1, 1_000)
    tests: Final[np.ndarray] = \
        np.array([[0.1, 0, 0.1, 0, 0.1, 0]], float)
    # similar to training = make_interesting_starting_points(111, tests)
    training: Final[np.ndarray] = np.array([[
        -0.00721145, -0.00399194, -0.01078522, 0.0795534, 0.02929588,
        0.01588136], [
        0.13202113, 0.05538969, -0.01011213, -0.06584996, 0.02154825,
        -0.07136868], [
        0.18476107, 0.10267522, -0.05342748, 0.12922333, -0.06481063,
        -0.0133719], [
        -0.20572899, 0.08377647, 0.21218986, 0.00109397, 0.16296447,
        0.03239565]]) if n_points == 4 else (
        make_interesting_starting_points(n_points, tests))

    three: Final[System] = __TO(
        "3oscillators", 6, 1, 2, 2, 1.0,
        tests, training, 5000, 50.0, 5000, 50.0,
        (0, 1, 2, len(training) + len(tests) - 1))
    three.equations = __3_coupled_oscillators  # type: ignore
    return three


#: The 3 oscillators system with 4 training points.
THREE_COUPLED_OSCILLATORS: Final[System] = make_3_couple_oscillators(4)
