"""Perform tests on the Traveling Salesperson Problem."""

from time import monotonic_ns
from typing import Callable, Final, Iterable

import numpy as np
from moptipy.api.algorithm import Algorithm
from moptipy.api.objective import Objective
from moptipy.spaces.permutations import Permutations
from moptipy.tests.algorithm import validate_algorithm
from moptipy.tests.objective import validate_objective
from numpy.random import Generator, default_rng
from pycommons.types import type_error

from moptipyapps.tsp.instance import Instance
from moptipyapps.tsp.tour_length import TourLength

#: the internal random number generator
__RANDOM: Final[Generator] = default_rng()


def tsp_instances_for_tests(
        random: Generator = __RANDOM,
        symmetric: bool = True,
        asymmetric: bool = True) -> Iterable[str]:
    """
    Get a sequence of TSP instances to test on.

    :param random: the random number generator to use
    :param symmetric: include symmetric instances
    :param asymmetric: include asymmetric instances
    :returns: an iterable of TSP instance names
    """
    if not isinstance(symmetric, bool):
        raise type_error(symmetric, "symmetric", bool)
    if not isinstance(asymmetric, bool):
        raise type_error(asymmetric, "asymmetric", bool)

    instances: tuple[str, ...]
    if symmetric and asymmetric:
        instances = Instance.list_resources(True, True)
    elif symmetric:
        instances = Instance.list_resources(True, False)
    elif asymmetric:
        instances = Instance.list_resources(False, True)
    else:
        raise ValueError(
            "at least of one symmetric or asymmetric must be TRUE")
    use_insts = list(instances)
    while len(use_insts) > 20:
        del use_insts[random.integers(len(use_insts))]
    random.shuffle(use_insts)
    return use_insts


def make_tour_valid(random: Generator, y: np.ndarray) -> np.ndarray:
    """
    Create valid TSP tours.

    :param random: the random number generator to use
    :param y: the input tour
    :returns: the valid version of `y`
    """
    y[0:len(y)] = range(len(y))
    random.shuffle(y)
    return y


def make_tour_invalid(random: Generator, y: np.ndarray) -> np.ndarray:
    """
    Create invalid tours.

    :param random: the random number generator to use
    :param y: the input tour
    :returns: the invalid version of `y`
    """
    ly: Final[int] = len(y)
    y[0:ly] = range(ly)
    random.shuffle(y)
    yorig = np.copy(y)

    ri = random.integers
    end_time: Final[int] = monotonic_ns() + 20_000_000_000
    while np.all(y == yorig):
        if monotonic_ns() >= end_time:
            y[0] = y[1]
            return y
        if ri(2) <= 0:
            z1 = z2 = ri(ly)
            while z1 == z2:
                if monotonic_ns() >= end_time:
                    y[0] = y[1]
                    return y
                z2 = ri(ly)
            y[z1] = y[z2]
        if ri(2) <= 0:
            y[ri(ly)] = ri(ly, 10 * ly)
        if ri(2) <= 0:
            y[ri(ly)] = ri(-2 * ly, -1)
    return y


def validate_algorithm_on_1_tsp(
        algorithm: Algorithm | Callable[
            [Instance, Permutations], Algorithm],
        instance: str | None = None, max_fes: int = 256,
        random: Generator = __RANDOM) -> None:
    """
    Check the validity of a black-box algorithm on one TSP instance.

    :param algorithm: the algorithm or algorithm factory
    :param instance: the instance name, or `None` to randomly pick one
    :param max_fes: the maximum number of FEs
    :param random: the default random generator to use
    """
    if not (isinstance(algorithm, Algorithm) or callable(algorithm)):
        raise type_error(algorithm, "algorithm", Algorithm, True)
    if instance is None:
        instance = str(random.choice(Instance.list_resources()))
    if not isinstance(instance, str):
        raise type_error(instance, "TSP instance name", (str, None))
    inst = Instance.from_resource(instance)
    if not isinstance(inst, Instance):
        raise type_error(inst, f"loaded bin TSP instance {instance!r}",
                         Instance)

    search_space = Permutations.standard(inst.n_cities)
    objective = TourLength(inst)
    if callable(algorithm):
        algorithm = algorithm(inst, search_space)
    if not isinstance(algorithm, Algorithm):
        raise type_error(algorithm, "algorithm", Algorithm, call=True)

    validate_algorithm(algorithm=algorithm,
                       solution_space=search_space,
                       objective=objective,
                       max_fes=max_fes)


def validate_algorithm_on_tsp(
        algorithm: Callable[[Instance, Permutations], Algorithm],
        symmetric: bool = True, asymmetric: bool = True,
        max_fes: int = 256, random: Generator = __RANDOM) -> None:
    """
    Validate an algorithm on a set of TSP instances.

    :param algorithm: the algorithm factory
    :param symmetric: include symmetric instances
    :param asymmetric: include asymmetric instances
    :param max_fes: the maximum FEs
    :param random: the random number generator
    """
    end_time: Final[int] = monotonic_ns() + 20_000_000_000
    for i in tsp_instances_for_tests(random, symmetric, asymmetric):
        if monotonic_ns() >= end_time:
            return
        validate_algorithm_on_1_tsp(algorithm, i, max_fes, random)


def validate_objective_on_1_tsp(
        objective: Objective | Callable[[Instance], Objective],
        instance: str | None = None,
        random: Generator = __RANDOM) -> None:
    """
    Validate an objective function on 1 TSP instance.

    :param objective: the objective function or a factory creating it
    :param instance: the instance name
    :param random: the random number generator
    """
    if instance is None:
        instance = str(random.choice(Instance.list_resources()))
    if not isinstance(instance, str):
        raise type_error(instance, "TSP instance name", (str, None))
    inst = Instance.from_resource(instance)
    if not isinstance(inst, Instance):
        raise type_error(inst, f"loaded TSP instance {instance!r}",
                         Instance)

    if callable(objective):
        objective = objective(inst)

    validate_objective(
        objective=objective,
        solution_space=Permutations.standard(inst.n_cities),
        make_solution_space_element_valid=make_tour_valid,
        is_deterministic=True)


def validate_objective_on_tsp(
        objective: Objective | Callable[[Instance], Objective],
        symmetric: bool = True, asymmetric: bool = True,
        random: Generator = __RANDOM) -> None:
    """
    Validate an objective function on TSP instances.

    :param objective: the objective function or a factory creating it
    :param symmetric: include symmetric instances
    :param asymmetric: include asymmetric instances
    :param random: the random number generator
    """
    end_time: Final[int] = monotonic_ns() + 20_000_000_000
    for i in tsp_instances_for_tests(random, symmetric, asymmetric):
        if monotonic_ns() >= end_time:
            return
        validate_objective_on_1_tsp(objective, i, random)
