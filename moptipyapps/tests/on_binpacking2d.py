"""Perform tests on the Two-Dimensional Bin Packing Problem."""

from typing import Callable, Final, Iterable, cast

import numpy as np
import numpy.random as rnd
from moptipy.api.algorithm import Algorithm
from moptipy.api.encoding import Encoding
from moptipy.api.objective import Objective
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.tests.algorithm import validate_algorithm
from moptipy.tests.encoding import validate_encoding
from moptipy.tests.objective import validate_objective
from moptipy.tests.space import validate_space
from moptipy.utils.types import type_error
from numpy.random import Generator, default_rng

from moptipyapps.binpacking2d.bin_count_and_last_empty import (
    BinCountAndLastEmpty,
)
from moptipyapps.binpacking2d.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
from moptipyapps.binpacking2d.ibl_encoding_2 import (
    ImprovedBottomLeftEncoding2,
)
from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.packing import Packing
from moptipyapps.binpacking2d.packing_space import PackingSpace

#: the internal random number generator
__RANDOM: Final[Generator] = default_rng()


def binpacking_instances_for_tests(
        random: Generator = __RANDOM) -> Iterable[str]:
    """
    Get a sequence of 2D Bin Packing instances to test on.

    :param random: the random number generator to use
    :returns: an iterable of 2D Bin Packing instance names
    """
    ri = random.integers
    insts: set[str] = {
        "a04", "a08", "beng10", f"a0{ri(1, 10)}", f"a1{ri(1, 10)}",
        f"a2{ri(1, 10)}", f"a3{ri(1, 10)}", f"a4{ri(1, 4)}",
        f"beng0{ri(1, 10)}", f"cl01_080_0{ri(1, 10)}",
        f"cl06_020_0{ri(1, 10)}", f"cl09_040_0{ri(1, 10)}",
        f"cl10_100_0{ri(1, 10)}"}
    insn: list[str] = list(Instance.list_resources())
    while len(insts) < 16:
        insts.add(insn.pop(ri(len(insn))))
    use_insts: list[str] = list(insts)
    random.shuffle(cast(list, use_insts))
    return use_insts


def make_packing_valid(inst: Instance,
                       random: Generator = __RANDOM) \
        -> Callable[[Generator, Packing], Packing]:
    """
    Make a function that creates valid packings.

    :param inst: the two-dimensional bin packing instance
    :param random: the random number generator to use
    :returns: a function that can make packings valid
    """
    search_space = SignedPermutations(inst.get_standard_item_sequence())
    encoding = (ImprovedBottomLeftEncoding1 if random.integers(2) == 0
                else ImprovedBottomLeftEncoding2)(inst)
    op0 = Op0ShuffleAndFlip(search_space)

    def __make_valid(ra: rnd.Generator,
                     y: Packing, ss=search_space,
                     en=encoding, o0=op0) -> Packing:
        x = ss.create()
        o0.op0(ra, x)
        en.decode(x, y)
        return y

    return __make_valid


def make_packing_invalid(random: Generator = __RANDOM) \
        -> Callable[[Packing], Packing]:
    """
    Make a function that creates invalid packings.

    :param random: the random number generator to use
    :returns: a function that can make packings invalid
    """

    def __make_invalid(x: Packing, ri=random.integers) -> Packing:
        not_finished: bool = True
        while not_finished:
            while ri(2) == 0:
                x[ri(len(x)), ri(6)] = -1
                not_finished = False
            while ri(2) == 0:
                second = first = ri(len(x))
                while second == first:
                    second = ri(len(x))
                x[first, 1] = x[second, 1]
                x[first, 2] = x[second, 2] - 1
                x[first, 3] = x[second, 3] - 1
                x[first, 4] = x[second, 4] + 1
                x[first, 5] = x[second, 5] + 1
        return x

    return __make_invalid


def validate_algorithm_on_1_2dbinpacking(
        algorithm: Algorithm | Callable[
            [Instance, SignedPermutations, Objective], Algorithm],
        instance: str | None = None, max_fes: int = 100,
        random: Generator = __RANDOM) -> None:
    """
    Check the validity of a black-box algorithm on the 2d bin packing.

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
        raise type_error(instance, "bin packing instance name", (str, None))
    inst = Instance.from_resource(instance)
    if not isinstance(inst, Instance):
        raise type_error(inst, f"loaded bin packing instance {instance!r}",
                         Instance)

    search_space = SignedPermutations(inst.get_standard_item_sequence())
    solution_space = PackingSpace(inst)
    encoding = (ImprovedBottomLeftEncoding1 if random.integers(2) == 0
                else ImprovedBottomLeftEncoding2)(inst)
    objective = BinCountAndLastEmpty(inst)
    if callable(algorithm):
        algorithm = algorithm(inst, search_space, objective)
    if not isinstance(algorithm, Algorithm):
        raise type_error(algorithm, "algorithm", Algorithm, call=True)

    validate_algorithm(algorithm=algorithm,
                       solution_space=solution_space,
                       objective=objective,
                       search_space=search_space,
                       encoding=encoding,
                       max_fes=max_fes)


def validate_algorithm_on_2dbinpacking(
        algorithm: Callable[[Instance, SignedPermutations,
                             Objective], Algorithm],
        max_fes: int = 100, random: Generator = __RANDOM) -> None:
    """
    Validate an algorithm on a set of JSSP instances.

    :param algorithm: the algorithm factory
    :param max_fes: the maximum FEs
    :param random: the random number generator
    """
    for i in binpacking_instances_for_tests(random):
        validate_algorithm_on_1_2dbinpacking(algorithm, i, max_fes, random)


def validate_objective_on_1_2dbinpacking(
        objective: Objective | Callable[[Instance], Objective],
        instance: str | None = None,
        random: Generator = __RANDOM) -> None:
    """
    Validate an objective function on 1 2D bin packing instance.

    :param objective: the objective function or a factory creating it
    :param instance: the instance name
    :param random: the random number generator
    """
    if instance is None:
        instance = str(random.choice(Instance.list_resources()))
    if not isinstance(instance, str):
        raise type_error(instance, "bin packing instance name", (str, None))
    inst = Instance.from_resource(instance)
    if not isinstance(inst, Instance):
        raise type_error(inst, f"loaded bin packing instance {instance!r}",
                         Instance)

    if callable(objective):
        objective = objective(inst)

    validate_objective(
        objective=objective,
        solution_space=PackingSpace(inst),
        make_solution_space_element_valid=make_packing_valid(inst),
        is_deterministic=True)


def validate_objective_on_2dbinpacking(
        objective: Objective | Callable[[Instance], Objective],
        random: Generator = __RANDOM) -> None:
    """
    Validate an objective function on bin packing instances.

    :param objective: the objective function or a factory creating it
    :param random: the random number generator
    """
    for i in binpacking_instances_for_tests(random):
        validate_objective_on_1_2dbinpacking(objective, i, random)


def validate_signed_permutation_encoding_on_1_2dbinpacking(
        encoding: Encoding | Callable[[Instance], Encoding],
        instance: str | None = None,
        random: Generator = __RANDOM) -> None:
    """
    Validate a signed permutation encoding on one 2D bin packing instance.

    :param encoding: the encoding or a factory creating it
    :param instance: the instance name
    :param random: the random number generator
    """
    if instance is None:
        instance = str(random.choice(Instance.list_resources()))
    if not isinstance(instance, str):
        raise type_error(instance, "bin packing instance name", (str, None))
    inst = Instance.from_resource(instance)
    if not isinstance(inst, Instance):
        raise type_error(inst, f"loaded bin packing instance {instance!r}",
                         Instance)

    if callable(encoding):
        encoding = encoding(inst)
    inst = Instance.from_resource(instance)

    x_space = SignedPermutations(inst.get_standard_item_sequence())
    validate_space(x_space)

    y_space = PackingSpace(inst)
    validate_space(y_space, make_element_valid=None)

    validate_encoding(encoding, x_space, y_space)

    x = x_space.create()
    x_space.validate(x)

    y = y_space.create()
    encoding.decode(x, y)
    y_space.validate(y)

    random.shuffle(x)
    ri = random.integers
    for i, xx in enumerate(x):
        if ri(2) == 0:
            x[i] = -xx
    encoding.decode(x, y)
    y_space.validate(y)

    x_str = x_space.to_str(x)
    x_2 = x_space.from_str(x_str)
    if not x_space.is_equal(x, x_2):
        raise ValueError("error in space to/from_str")
    if not np.array_equal(x, x_2):
        raise ValueError("error in space to/from_str and is_equal")

    y_2 = y_space.create()
    encoding.decode(x_2, y_2)
    if not y_space.is_equal(y, y_2):
        raise ValueError("encoding is not deterministic")
    if not np.array_equal(y, y_2):
        raise ValueError(
            "encoding is not deterministic and error in space.is_equal")


def validate_signed_permutation_encoding_on_2dbinpacking(
        encoding: Encoding | Callable[[Instance], Encoding],
        random: Generator = __RANDOM) -> None:
    """
    Validate a signed permutation encoding function on bin packing instances.

    :param encoding: the encoding or a factory creating it
    :param random: the random number generator
    """
    for i in binpacking_instances_for_tests(random):
        validate_signed_permutation_encoding_on_1_2dbinpacking(
            encoding, i, random)
