"""Test the errors and hardness objective."""
import numpy.random as rnd
from moptipy.operators.vectors.op0_uniform import Op0Uniform
from moptipy.tests.objective import validate_objective

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.instgen.errors_and_hardness import (
    ErrorsAndHardness,
)
from moptipyapps.binpacking2d.instgen.problem import Problem


def __check_for_instance(inst: Problem) -> None:
    """
    Check the objective for one problem instance.

    :param inst: the instance
    """
    op0 = Op0Uniform(inst.search_space)

    def __make_valid(ra: rnd.Generator,
                     y: list[Instance], ins=inst, o0=op0) -> list[Instance]:
        x = ins.search_space.create()
        o0.op0(ra, x)
        ins.encoding.decode(x, y)
        return y

    validate_objective(ErrorsAndHardness(inst.solution_space, 20, 2),
                       inst.solution_space, __make_valid)


def test_errors_hardness() -> None:
    """Test the errors and hardness objective function."""
    random: rnd.Generator = rnd.default_rng()

    for s in Instance.list_resources():
        __check_for_instance(Problem(s, int(random.integers(0, 10)) / 9))
