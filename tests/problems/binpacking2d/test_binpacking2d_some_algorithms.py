"""Test applying some algorithms to the 2D bin packing problem."""


from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.rls import RLS
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.operators.signed_permutations.op1_swap_2_or_flip import (
    Op1Swap2OrFlip,
)

from moptipyapps.tests.on_binpacking2d import (
    validate_algorithm_on_2dbinpacking,
)


def test_rls() -> None:
    """Test randomized local search."""
    validate_algorithm_on_2dbinpacking(
        lambda _, perm, __: RLS(Op0ShuffleAndFlip(perm), Op1Swap2OrFlip()))


def test_rs() -> None:
    """Test random sampling."""
    validate_algorithm_on_2dbinpacking(
        lambda _, perm, __: RandomSampling(Op0ShuffleAndFlip(perm)))
