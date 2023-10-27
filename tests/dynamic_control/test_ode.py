"""A test of the ODE integration."""

from math import isfinite
from typing import Callable, Final

import numpy as np
from moptipy.utils.nputils import is_all_finite
from numpy.random import Generator, default_rng

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.experiment_raw import make_instances
from moptipyapps.dynamic_control.experiment_surrogate import (
    make_instances as make_instances2,
)
from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.ode import (
    diff_from_ode,
    j_from_ode,
    run_ode,
    t_from_ode,
)
from moptipyapps.dynamic_control.system import System


def __run_ode_test(instance: Instance,
                   random: Generator = default_rng()) -> None:
    """
    Test the ode integration.

    :param instance: the instance
    :param random: the random number generator
    """
    system: Final[System] = instance.system
    controller: Final[Controller] = instance.controller
    state_dim: Final[int] = controller.state_dims
    if state_dim != system.state_dims:
        raise ValueError("state dimensions incompatible: "
                         f"{state_dim}!={system.state_dims}.")
    param_dim: Final[int] = controller.param_dims
    ctrl_dim: Final[int] = controller.control_dims
    if ctrl_dim != system.control_dims:
        raise ValueError("control dimensions incompatible: "
                         f"{ctrl_dim}!={system.control_dims}.")

    n_tests: int = 2
    n_reps: Final[int] = 2

    starting_states: Final[np.ndarray] = random.uniform(
        -10.0, 10.0, (n_tests, state_dim))
    s2: Final[np.ndarray] = np.copy(starting_states)
    params: Final[np.ndarray] = random.uniform(
        -10.0, 10.0, (n_tests, param_dim))
    p2: Final[np.ndarray] = np.copy(params)
    steps: Final[list[int]] = list(map(int, random.integers(10, 15, n_tests)))
    t2: Final[np.ndarray] = np.copy(steps)
    ctrl: Final[Callable[[np.ndarray, float, np.ndarray, np.ndarray], None]] \
        = controller.controller
    sys: Final[Callable[[np.ndarray, float, np.ndarray, np.ndarray], None]] \
        = system.equations
    js: Final[list[float]] = []
    ts: Final[list[float]] = []
    results: Final[list[np.ndarray]] = []
    diffs1: Final[list[np.ndarray]] = []
    diffs2: Final[list[np.ndarray]] = []

    for mode in range(3):
        results.clear()
        js.clear()
        ts.clear()
        diffs1.clear()
        diffs2.clear()

        for i in range(n_tests):
            res = run_ode(starting_states[i], sys, ctrl, params[i],
                          ctrl_dim, steps[i])
            if not all(is_all_finite(r) for r in res):
                raise ValueError(
                    f"error on instance {instance}: encountered invalid "
                    f"output {res!r} for input {starting_states[i]!r} and "
                    f"{params[i]!r}.")
            results.append(res)
            rcp = np.copy(res)
            jj = j_from_ode(res, state_dim)
            if (not isfinite(jj)) or (jj < 0.0):
                raise ValueError(f"invalid j={jj} on instance {instance}.")
            js.append(jj)
            if not np.all(rcp == res):
                raise ValueError(
                    f"j_from_ode corrupts result on instance {instance}.")
            t = t_from_ode(res)
            if (not isfinite(t)) or (t < 0.0):
                raise ValueError(f"invalid t={t} on instance {instance}.")
            ts.append(t)
            df, cf = diff_from_ode(res, state_dim)
            if not all(is_all_finite(dff) for dff in df):
                raise ValueError(
                    f"invalid state on instance {instance}: {df}")
            if not all(is_all_finite(cff) for cff in cf):
                raise ValueError(
                    f"invalid differential on instance {instance}: {cf}")
            diffs1.append(df)
            diffs2.append(cf)
            if not np.all(rcp == res):
                raise ValueError(
                    f"t_from_ode corrupts result on instance {instance}.")

        if (mode == 0) and not np.all(s2 == starting_states):
            raise ValueError(
                f"error on instance {instance}: corrupted states")
        if (mode == 0) and not np.all(p2 == params):
            raise ValueError(
                f"error on instance {instance}: corrupted params")
        if (mode == 0) and not np.all(t2 == steps):
            raise ValueError(f"error on instance {instance}: corrupted steps")

        for _j in range(n_reps):
            for i in range(n_tests):
                res = run_ode(starting_states[i], sys, ctrl, params[i],
                              ctrl_dim, steps[i])
                if not np.all(res == results[i]):
                    raise ValueError(
                        f"error on instance {instance}: encountered different "
                        f"outputs {res!r} and {results[i]!r} for input "
                        f"{starting_states[i]!r} and {params[i]!r}.")
                rcp = np.copy(res)
                jj = j_from_ode(res, state_dim)
                if (not isfinite(jj)) or (jj < 0.0):
                    raise ValueError(f"invalid j={jj} on instance {instance}.")
                if jj != js[i]:
                    raise ValueError(f"inconsistent j={jj}, should be {js[i]} "
                                     f"on instance {instance}")
                if not np.all(rcp == res):
                    raise ValueError(
                        f"j_from_ode corrupts result on instance {instance}.")
                t = t_from_ode(res)
                if (not isfinite(t)) or (t < 0.0):
                    raise ValueError(f"invalid t={t} on instance {instance}.")
                if t != ts[i]:
                    raise ValueError(f"inconsistent t={t}, should be {ts[i]} "
                                     f"on instance {instance}")
                df, cf = diff_from_ode(res, state_dim)
                if not all(is_all_finite(dff) for dff in df):
                    raise ValueError(
                        f"invalid state on instance {instance}: {df}")
                if not np.all(df == diffs1[i]):
                    raise ValueError(
                        f"invalid state on {instance}: {df!r} should be {df}")
                if not all(is_all_finite(cff) for cff in cf):
                    raise ValueError("invalid differential on instance "
                                     f"{instance}: {diffs1[i]!r}")
                if not np.all(cf == diffs2[i]):
                    raise ValueError(f"invalid differential on {instance}: "
                                     f"{cf!r} should be {diffs2[i]!r}")
                diffs1.append(df)
                diffs2.append(cf)
                if not np.all(rcp == res):
                    raise ValueError(
                        f"t_from_ode corrupts result on instance {instance}.")

            if (mode == 0) and not np.all(s2 == starting_states):
                raise ValueError(
                    f"error on instance {instance}: corrupted states")
            if (mode == 0) and not np.all(p2 == params):
                raise ValueError(
                    f"error on instance {instance}: corrupted params")
            if (mode == 0) and not np.all(t2 == steps):
                raise ValueError(
                    f"error on instance {instance}: corrupted steps")

        if mode == 0:
            n_tests = min(2, len(system.training_starting_states))
            starting_states[0:n_tests, :] = system.training_starting_states[
                0:n_tests, :]
        elif mode == 1:
            n_tests = min(2, len(system.test_starting_states))
            starting_states[0:n_tests, :] = system.test_starting_states[
                0:n_tests, :]


def test_run_ode_on_experiment_raw() -> None:
    """Test the raw experiment."""
    insts = list(make_instances())
    __run_ode_test(insts[default_rng().integers(len(insts))]())


def test_run_ode_on_experiment_surrogate() -> None:
    """Test the surrogate experiment."""
    insts = list(make_instances2())
    __run_ode_test(insts[default_rng().integers(len(insts))]())
