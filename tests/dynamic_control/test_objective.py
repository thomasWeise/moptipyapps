"""A test of the objective functions."""

from typing import Any, Callable, Final

from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.tests.objective import validate_objective
from numpy.random import Generator, default_rng

from moptipyapps.dynamic_control.controllers.ann import anns
from moptipyapps.dynamic_control.controllers.cubic import cubic
from moptipyapps.dynamic_control.controllers.linear import linear
from moptipyapps.dynamic_control.controllers.min_ann import min_anns
from moptipyapps.dynamic_control.controllers.partially_linear import (
    partially_linear,
)
from moptipyapps.dynamic_control.controllers.peaks import peaks
from moptipyapps.dynamic_control.controllers.predefined import predefined
from moptipyapps.dynamic_control.controllers.quadratic import quadratic
from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.objective import (
    FigureOfMerit,
    FigureOfMeritLE,
)
from moptipyapps.dynamic_control.system import System
from moptipyapps.dynamic_control.systems.lorenz import LORENZ_4, LORENZ_111
from moptipyapps.dynamic_control.systems.stuart_landau import (
    STUART_LANDAU_4,
    STUART_LANDAU_111,
)


def __objective_test(fc: Callable[[Instance, bool], FigureOfMerit],
                     instance: Instance) -> None:
    """
    Test an objective function.

    :param fc: the creator for the figure of merit
    :param instance: the instance
    """
    f: Final[FigureOfMerit] = fc(instance, False)
    solution_space: Final[VectorSpace] = (
        instance.controller.parameter_space())
    op0: Final[Op0Uniform] = Op0Uniform(solution_space)

    def __mf(r: Generator, x: Any, __o=op0.op0) -> Any:
        __o(r, x)
        return x

    try:
        validate_objective(f, solution_space, __mf, True)
    except ValueError as ve:
        raise ValueError(f"error on instance {str(instance)!r}") from ve
    except TypeError as te:
        raise TypeError(f"error on instance {str(instance)!r}") from te


def __objective_tests(fc: Callable[[Instance, bool], FigureOfMerit],
                      random: Generator = default_rng()) -> None:
    """
    Test the raw figure of merit.

    :param fc: the figure of merit
    :param random: the generator
    """
    for orig_system in (STUART_LANDAU_111, LORENZ_111, STUART_LANDAU_4,
                        LORENZ_4):
        system = System(orig_system.name,
                        orig_system.state_dims, orig_system.control_dims,
                        orig_system.state_dim_mod,
                        orig_system.state_dims_in_j,
                        orig_system.gamma,
                        orig_system.test_starting_states,
                        orig_system.training_starting_states,
                        int(random.integers(10, 32)), 10.0,
                        int(random.integers(10, 32)), 10.0,
                        orig_system.plot_examples)

        controllers = [linear(system), quadratic(system), cubic(system)]
        controllers.extend(anns(system))
        controllers.extend(min_anns(system))
        controllers.extend(partially_linear(system))
        controllers.extend(predefined(system))
        controllers.extend(peaks(system))
        for c in controllers:
            __objective_test(fc, Instance(system, c))


def test_figure_of_merit() -> None:
    """Test the original figure of merit."""
    __objective_tests(FigureOfMerit)


def test_figure_of_merit_le() -> None:
    """Test the figure of merit with LE extension."""
    __objective_tests(FigureOfMeritLE)
