"""A test of the systems equations."""

from time import monotonic_ns
from typing import Callable, Final

import numpy as np
from moptipy.utils.nputils import is_all_finite
from numpy.random import Generator, default_rng

from moptipyapps.dynamic_control.system import System
from moptipyapps.dynamic_control.systems.lorenz import LORENZ_4, LORENZ_111
from moptipyapps.dynamic_control.systems.stuart_landau import (
    STUART_LANDAU_4,
    STUART_LANDAU_111,
)


def __system_test(system: System,
                  random: Generator = default_rng()) -> None:
    """
    Test a system.

    :param system: the system of equations
    :param random: the random number generator
    """
    state_dim: Final[int] = system.state_dims
    ctrl_dim: Final[int] = system.control_dims

    min_tests = max(111, len(system.training_starting_states),
                    len(system.test_starting_states))

    n_tests: int = int(random.integers(min_tests, min_tests + 10000))
    n_reps: Final[int] = int(random.integers(3, 15))

    states: Final[np.ndarray] = random.uniform(
        -100.0, 100.0, (n_tests, state_dim))
    s2: Final[np.ndarray] = np.copy(states)
    ctrl: Final[np.ndarray] = random.uniform(
        -100.0, 100.0, (n_tests, ctrl_dim))
    c2: Final[np.ndarray] = np.copy(ctrl)
    ts: Final[np.ndarray] = random.uniform(1e-10, 5000.0, n_tests)
    t2: Final[np.ndarray] = np.copy(ts)
    out: Final[np.ndarray] = np.empty((n_tests, state_dim))
    buf: Final[np.ndarray] = np.empty(state_dim)
    call: Final[Callable[[np.ndarray, float, np.ndarray, np.ndarray], None]] \
        = system.equations
    end_time: Final[int] = monotonic_ns() + 20_000_000_000

    for mode in range(3):

        for i in range(n_tests):
            if monotonic_ns() >= end_time:
                return
            call(states[i], float(ts[i]), ctrl[i], buf)
            if not is_all_finite(buf):
                raise ValueError(
                    f"error on system {system}: encountered invalid output "
                    f"{buf!r} for input {states[i]!r},"
                    f" {float(ts[i])!r}, and {ctrl[i]!r}.")
            out[i, :] = buf

        if (mode == 0) and not np.all(s2 == states):
            raise ValueError(f"error on system {system}: corrupted states")
        if (mode == 0) and not np.all(c2 == ctrl):
            raise ValueError(f"error on system {system}: corrupted controls")
        if (mode == 0) and not np.all(t2 == ts):
            raise ValueError(f"error on system {system}: corrupted ts")

        for _j in range(n_reps):
            for i in range(n_tests):
                if monotonic_ns() >= end_time:
                    return
                call(states[i], float(ts[i]), ctrl[i], buf)
                if not np.all(buf == out[i]):
                    raise ValueError(
                        f"error on system {system}: encountered different "
                        f"outputs {buf!r} and {out[i]!r} for input "
                        f"{states[i]!r}, {float(ts[i])!r}, and "
                        f"{ctrl[i]!r}.")
            if (mode == 0) and not np.all(s2 == states):
                raise ValueError(f"error on system {system}: corrupted states")
            if (mode == 0) and not np.all(c2 == ctrl):
                raise ValueError(
                    f"error on system {system}: corrupted controls")
            if (mode == 0) and not np.all(t2 == ts):
                raise ValueError(f"error on system {system}: corrupted ts")

        if mode == 0:
            n_tests = len(system.training_starting_states)
            states[0:n_tests, :] = system.training_starting_states
        elif mode == 1:
            n_tests = len(system.test_starting_states)
            states[0:n_tests, :] = system.test_starting_states


def test_stuart_landau_4() -> None:
    """Test the Stuart-Landau model with 4 training cases."""
    __system_test(STUART_LANDAU_4)


def test_stuart_landau_111() -> None:
    """Test the Stuart-Landau model with 111 training cases."""
    __system_test(STUART_LANDAU_111)


def test_lorenz_4() -> None:
    """Test the Lorenz model with 4 training cases."""
    __system_test(LORENZ_4)


def test_lorenz_111() -> None:
    """Test the Lorenz model with 111 training cases."""
    __system_test(LORENZ_111)
