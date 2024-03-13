"""Compare the runtime of encodings on several problem instances."""
import os
from statistics import mean, median
from timeit import timeit
from typing import Any, Callable, cast

import numpy as np
import psutil
from moptipy.utils.nputils import rand_generator
from pycommons.types import type_name

from moptipyapps.binpacking2d.encodings.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
from moptipyapps.binpacking2d.encodings.ibl_encoding_2 import (
    ImprovedBottomLeftEncoding2,
)
from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.packing_space import PackingSpace

# Check if process is a sub-process of make?
ns = lambda prc: False if prc is None else (  # noqa: E731
    "make" in prc.name() or ns(prc.parent()))
# Is this a make build?
IS_MAKE_BUILD = ns(psutil.Process(os.getppid()))

# Create the random number generator.
random = rand_generator(1)

# If it is a make build, use only 1 repetition, otherwise 20.
REPETITIONS = 1 if IS_MAKE_BUILD else 20

# The instances to iterate over: All if not make build, 20 otherwise.
INSTANCES = random.choice(Instance.list_resources(), 20, False) \
    if IS_MAKE_BUILD else Instance.list_resources()


# We test two versions of the Improved Bottom Left Encoding
encodings = [ImprovedBottomLeftEncoding1,  # the 1st encoding
             ImprovedBottomLeftEncoding2]  # the 2nd encoding

# The array to receive the timing measurements
timings = [[] for _ in encodings]

# Iterate over all instances.
for inst_name in INSTANCES:
    instance = Instance.from_resource(inst_name)  # load the instance

# We create 10 points in the search space to be mapped by the encodings.
    all_x_data_lst: list[np.ndarray] = []
    for _ in range(10):
        x_data = instance.get_standard_item_sequence()
        for i, e in enumerate(x_data):
            if random.integers(2) != 0:
                x_data[i] = -e
        x = np.array(x_data, instance.dtype)
        random.shuffle(x)
        all_x_data_lst.append(x)
    all_x_data: tuple[np.ndarray, ...] = tuple(all_x_data_lst)

    y_space = PackingSpace(instance)
    y = y_space.create()
    benchmarks = []
    for encoding in encodings:
        def __f(_x=all_x_data, _y=y,
                _e=cast(Callable[[Any, Any], Any],
                        encoding(instance).decode)) -> None:
            for __x in _x:
                _e(__x, _y)
        benchmarks.append(__f)

    for f in benchmarks:
        timeit(f, number=2)

    for i, f in enumerate(benchmarks):
        t = timeit(f, number=10)
        timings[i].append(t)


def get_short_name(the_type) -> str:
    """Get the short name of the type."""
    s = type_name(the_type)
    last_dot = s.rfind(".")
    if last_dot > 0:
        return s[(last_dot + 1):]
    return s


# Print the times measured for the different encodings.
for i, e in enumerate(encodings):
    print(f"{get_short_name(e)}: "
          f"{min(timings[i]):.4f}/{mean(timings[i]):.4f}/"
          f"{median(timings[i]):.4f}/{max(timings[i]):.4f}")
