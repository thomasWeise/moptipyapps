"""Test the bin-count-and--small objective."""
import numpy.random as rnd
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.tests.objective import validate_objective

from moptipyapps.binpacking2d.encodings.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
from moptipyapps.binpacking2d.encodings.ibl_encoding_2 import (
    ImprovedBottomLeftEncoding2,
)
from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.objectives.bin_count import BinCount
from moptipyapps.binpacking2d.objectives.bin_count_and_small import (
    BinCountAndSmall,
)
from moptipyapps.binpacking2d.packing import Packing
from moptipyapps.binpacking2d.packing_space import PackingSpace
from moptipyapps.tests.on_binpacking2d import (
    validate_objective_on_2dbinpacking,
)


def __check_for_instance(inst: Instance, random: rnd.Generator) -> None:
    """
    Check the objective for one problem instance.

    :param inst: the instance
    """
    search_space = SignedPermutations(inst.get_standard_item_sequence())
    solution_space = PackingSpace(inst)
    encoding = (ImprovedBottomLeftEncoding1 if random.integers(2) == 0
                else ImprovedBottomLeftEncoding2)(inst)
    objective = BinCountAndSmall(inst)
    op0 = Op0ShuffleAndFlip(search_space)

    def __make_valid(ra: rnd.Generator,
                     y: Packing, ss=search_space,
                     en=encoding, o0=op0) -> Packing:
        x = ss.create()
        o0.op0(ra, x)
        en.decode(x, y)
        return y

    validate_objective(objective, solution_space, __make_valid)

    f1: BinCount = BinCount(inst)
    for _ in range(10):
        pa = __make_valid(random, Packing(inst))
        assert f1.evaluate(pa) == objective.to_bin_count(
            objective.evaluate(pa))


def test_bin_count_and_small_objective() -> None:
    """Test the bin-count-and--small-area objective function."""
    random: rnd.Generator = rnd.default_rng()

    choices = list(Instance.list_resources())
    checks: set[str] = {c for c in choices if c.startswith(("a", "b"))}
    min_len: int = len(checks) + 10
    while len(checks) < min_len:
        checks.add(choices.pop(random.integers(len(choices))))

    for s in checks:
        __check_for_instance(Instance.from_resource(s), random)

    validate_objective_on_2dbinpacking(BinCountAndSmall, random)


def test_bin_count_and_small_objective_2() -> None:
    """Test the bin-count-and--small-area function."""
    random: rnd.Generator = rnd.default_rng()
    for inst in Instance.list_resources():
        if not inst.startswith(("a", "b")):
            continue
        instance = Instance.from_resource(inst)
        search_space = SignedPermutations(
            instance.get_standard_item_sequence())
        solution_space = PackingSpace(instance)
        encoding = (ImprovedBottomLeftEncoding1 if random.integers(2) == 0
                    else ImprovedBottomLeftEncoding2)(instance)
        objective = BinCountAndSmall(instance)
        op0 = Op0ShuffleAndFlip(search_space)
        x = search_space.create()
        op0.op0(random, x)
        y = solution_space.create()
        encoding.decode(x, y)
        assert 0 <= objective.lower_bound() <= objective.evaluate(y) \
               <= objective.upper_bound() <= 1_000_000_000_000_000
