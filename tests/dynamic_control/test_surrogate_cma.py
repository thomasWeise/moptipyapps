"""Test the model equations."""


from typing import Final

import numpy as np
from moptipy.api.execution import Execution
from numpy.random import Generator, default_rng

import moptipyapps.dynamic_control.model_objective as mo
from moptipyapps.dynamic_control.experiment_surrogate import make_instances
from moptipyapps.dynamic_control.objective import (
    FigureOfMerit,
    FigureOfMeritLE,
)
from moptipyapps.dynamic_control.surrogate_cma import SurrogateCmaEs
from moptipyapps.dynamic_control.system_model import SystemModel


def test_surrogate_cmaes() -> None:
    """Test whether surrogate cmaes really invokes the model."""
    old: Final = mo._evaluate
    mo._evaluate = old.py_func
    random: Final[Generator] = default_rng()

    model_counter: Final[np.ndarray] = np.array([0], int)
    real_count: Final[np.ndarray] = np.array([0], int)

    instance: SystemModel = next(iter(make_instances()))()

    n_ode_steps_on_raw_model_per_training_case: Final[int] = (
        int(random.integers(10, 20)))
    setattr(instance.system, "training_steps",
            n_ode_steps_on_raw_model_per_training_case)

    n_training_cases: Final[int] = int(random.integers(1, 3))
    setattr(instance.system, "training_starting_states",
            np.array(instance.system.training_starting_states[
                     0:n_training_cases]))

    def __model(_: np.ndarray, __: float, ___: np.ndarray, out: np.ndarray,
                cnt=model_counter) -> None:
        cnt[0] += 1
        out.fill(-4.0)

    def __equations(_: np.ndarray, __: float, ___: np.ndarray, out: np.ndarray,
                    cnt=real_count) -> None:
        cnt[0] += 1
        out.fill(5.0)

    setattr(instance.system, "equations", __equations)
    setattr(instance.model, "controller", __model)

    objective: Final[FigureOfMerit] = FigureOfMeritLE(instance, True)
    rsm: Final = objective.set_model

    def __sm(eq, _rsm=rsm) -> None:
        if hasattr(eq, "py_func"):
            eq = eq.py_func
        _rsm(eq)

    objective.set_model = __sm

    space = instance.controller.parameter_space()

    n_total_fes: Final[int] = int(random.integers(5, 10))
    n_warmup_fes: Final[int] = int(random.integers(2, n_total_fes - 2))

    n_fes_for_model_training: Final[int] = int(random.integers(4, 10))
    n_fes_for_controller_synthesis_on_model: Final[int] = int(random.integers(
        4, 10))

    with Execution().set_max_fes(n_total_fes)\
        .set_objective(objective).set_solution_space(space)\
        .set_solution_space(space)\
        .set_algorithm(SurrogateCmaEs(
            instance, space, objective, n_warmup_fes,
            n_fes_for_model_training,
            n_fes_for_controller_synthesis_on_model)).execute():
        pass

    n_real_ode_steps: Final[int] = (
        n_total_fes * n_training_cases
        * n_ode_steps_on_raw_model_per_training_case)
    assert real_count >= n_real_ode_steps

    n_training_invocations: int = 0
    for i in range(n_warmup_fes, n_total_fes):
        data_set_size: int = i * n_training_cases * (
            n_ode_steps_on_raw_model_per_training_case - 1)
        n_training_invocations += n_fes_for_model_training * data_set_size

    n_model_ode_steps: Final[int] = (
        (n_total_fes - n_warmup_fes)
        * n_training_cases * n_ode_steps_on_raw_model_per_training_case)
    assert model_counter >= n_training_invocations + n_model_ode_steps

    mo._evaluate = old
