"""A test of the controller equations."""

from typing import Callable, Final

import numpy as np
from moptipy.utils.nputils import is_all_finite
from numpy.random import Generator, default_rng

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.controllers.ann import anns, make_ann
from moptipyapps.dynamic_control.controllers.cubic import cubic
from moptipyapps.dynamic_control.controllers.linear import linear
from moptipyapps.dynamic_control.controllers.min_ann import min_anns
from moptipyapps.dynamic_control.controllers.partially_linear import (
    partially_linear,
)
from moptipyapps.dynamic_control.controllers.peaks import peaks
from moptipyapps.dynamic_control.controllers.predefined import predefined
from moptipyapps.dynamic_control.controllers.quadratic import quadratic
from moptipyapps.dynamic_control.systems.lorenz import LORENZ_4
from moptipyapps.dynamic_control.systems.stuart_landau import STUART_LANDAU_4


def __controller_test(controller: Controller,
                      random: Generator = default_rng()) -> None:
    """
    Test a controller.

    :param controller: the controller equations
    :param random: the random number generator
    """
    state_dim: Final[int] = controller.state_dims
    param_dim: Final[int] = controller.param_dims
    ctrl_dim: Final[int] = controller.control_dims

    n_tests: int = int(random.integers(111, 1111))
    n_reps: Final[int] = int(random.integers(3, 15))

    states: Final[np.ndarray] = random.uniform(
        -100.0, 100.0, (n_tests, state_dim))
    s2: Final[np.ndarray] = np.copy(states)
    params: Final[np.ndarray] = random.uniform(
        -100.0, 100.0, (n_tests, param_dim))
    p2: Final[np.ndarray] = np.copy(params)
    ts: Final[np.ndarray] = random.uniform(1e-10, 5000.0, n_tests)
    t2: Final[np.ndarray] = np.copy(ts)
    out: Final[np.ndarray] = np.empty((n_tests, ctrl_dim))

    buf: Final[np.ndarray] = np.empty(ctrl_dim)
    call: Final[Callable[[np.ndarray, float, np.ndarray, np.ndarray], None]] \
        = controller.controller

    for i in range(n_tests):
        call(states[i], float(ts[i]), params[i], buf)
        if not is_all_finite(buf):
            raise ValueError(
                f"error on controller {controller}: encountered invalid "
                f"output {buf!r} for input {states[i]!r},"
                f" {float(ts[i])!r}, and {params[i]!r}.")
        out[i, :] = buf

    if not np.all(s2 == states):
        raise ValueError(
            f"error on controller {controller}: corrupted states")
    if not np.all(p2 == params):
        raise ValueError(
            f"error on controller {controller}: corrupted controls")
    if not np.all(t2 == ts):
        raise ValueError(f"error on controller {controller}: corrupted ts")

    for _j in range(n_reps):
        for i in range(n_tests):
            call(states[i], float(ts[i]), params[i], buf)
            if not np.all(buf == out[i]):
                raise ValueError(
                    f"error on controller {controller}: encountered different"
                    f" outputs {buf!r} and {out[i]!r} for input {states[i]!r}"
                    f", {float(ts[i])!r}, and {params[i]!r}.")
        if not np.all(s2 == states):
            raise ValueError(
                f"error on controller {controller}: corrupted states")
        if not np.all(p2 == params):
            raise ValueError(
                f"error on controller {controller}: corrupted controls")
        if not np.all(t2 == ts):
            raise ValueError(f"error on controller {controller}: corrupted ts")


def test_linear() -> None:
    """Test the linear controllers."""
    __controller_test(linear(STUART_LANDAU_4))
    __controller_test(linear(LORENZ_4))


def test_quadratic() -> None:
    """Test the quadratic controllers."""
    __controller_test(quadratic(STUART_LANDAU_4))
    __controller_test(quadratic(LORENZ_4))


def test_cubic() -> None:
    """Test the cubic controllers."""
    __controller_test(cubic(STUART_LANDAU_4))
    __controller_test(cubic(LORENZ_4))


def test_anns() -> None:
    """Test the ANN controllers."""
    for ann in anns(STUART_LANDAU_4):
        __controller_test(ann)
    for ann in anns(LORENZ_4):
        __controller_test(ann)

    for sd in range(2, 5):
        for cd in range(1, 5):
            for layers in [[1], [1, 1], [2], [2, 2], [4, 4]]:
                __controller_test(make_ann(sd, cd, layers))


def test_min_anns() -> None:
    """Test the Min-ANN controllers."""
    for ann in min_anns(STUART_LANDAU_4):
        __controller_test(ann)
    for ann in min_anns(LORENZ_4):
        __controller_test(ann)


def test_partially_linear() -> None:
    """Test the partially linear controllers."""
    for c in partially_linear(STUART_LANDAU_4):
        __controller_test(c)
    for c in partially_linear(LORENZ_4):
        __controller_test(c)


def test_predefined() -> None:
    """Test the predefined controllers."""
    for c in predefined(STUART_LANDAU_4):
        __controller_test(c)
    for c in predefined(LORENZ_4):
        __controller_test(c)


def test_peaks() -> None:
    """Test the peaks controllers."""
    for c in peaks(STUART_LANDAU_4):
        __controller_test(c)
    for c in peaks(LORENZ_4):
        __controller_test(c)
