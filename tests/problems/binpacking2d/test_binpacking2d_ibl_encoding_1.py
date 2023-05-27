"""Test the first realization of the improved bottom left encoding-."""
import numpy as np
import numpy.random as rnd
from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.tests.encoding import validate_encoding
from moptipy.tests.space import validate_space

from moptipyapps.binpacking2d.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.packing_space import PackingSpace
from moptipyapps.tests.on_binpacking2d import (
    validate_signed_permutation_encoding_on_2dbinpacking,
)


def __check_for_instance(instance: str,
                         random: rnd.Generator = rnd.default_rng()) -> None:
    inst = Instance.from_resource(instance)

    x_space = SignedPermutations(inst.get_standard_item_sequence())
    validate_space(x_space)

    y_space = PackingSpace(inst)
    validate_space(y_space, make_element_valid=None)

    g = ImprovedBottomLeftEncoding1(inst)
    validate_encoding(g, x_space, y_space)

    x = x_space.create()
    x_space.validate(x)

    y = y_space.create()
    g.decode(x, y)
    y_space.validate(y)

    random.shuffle(x)
    g.decode(x, y)
    y_space.validate(y)

    x_str = x_space.to_str(x)
    x_2 = x_space.from_str(x_str)
    assert x_space.is_equal(x, x_2)
    assert np.array_equal(x, x_2)

    y_str = y_space.to_str(y)
    y_2 = y_space.from_str(y_str)
    assert y_space.is_equal(y, y_2)
    assert np.array_equal(y, y_2)


def test_for_selected() -> None:
    """Test the ibf for a selected number of instances."""
    random: rnd.Generator = rnd.default_rng()

    checks: set[str] = {"a01", "a10", "a20", "beng03", "beng10",
                        "cl01_040_08", "cl04_100_10", "cl10_060_03"}
    choices = list(Instance.list_resources())
    while len(checks) < 10:
        checks.add(choices.pop(random.integers(len(choices))))

    for s in checks:
        __check_for_instance(s)

    validate_signed_permutation_encoding_on_2dbinpacking(
        ImprovedBottomLeftEncoding1, random)
