"""Test the bin packing 2D packing space."""
import numpy as np
from moptipy.tests.space import validate_space
from numpy.random import default_rng

from moptipyapps.binpacking2d.encodings.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
from moptipyapps.binpacking2d.encodings.ibl_encoding_2 import (
    ImprovedBottomLeftEncoding2,
)
from moptipyapps.binpacking2d.instance import IDX_REPETITION, Instance
from moptipyapps.binpacking2d.packing import Packing
from moptipyapps.binpacking2d.packing_space import PackingSpace


def test_packing_space() -> None:
    """Test the packing space."""
    use_random = default_rng()

    def __make_invalid(x: Packing, random=use_random) -> Packing:
        x[random.integers(len(x)), random.integers(6)] = -1
        return x

    res = list(Instance.list_resources())

    for i in range(30):
        if len(res) <= 0:
            break
        name = res.pop(use_random.integers(len(res)))
        inst = Instance.from_resource(name)

        def __make_valid(x: Packing, ins=inst,
                         random=use_random) -> Packing:
            encoding = (ImprovedBottomLeftEncoding1 if random.integers(2) == 0
                        else ImprovedBottomLeftEncoding2)(ins)
            # generate the data for a random packing
            x_data = [i if random.integers(2) == 0 else -i
                      for i in range(1, ins.n_different_items + 1)
                      for _ in range(ins[i - 1, IDX_REPETITION])]
            xx = np.array(x_data, ins.dtype)  # convert data to numpy array
            random.shuffle(xx)  # shuffle the data
            encoding.decode(xx, x)
            return x

        validate_space(PackingSpace(inst), __make_valid, __make_invalid)
