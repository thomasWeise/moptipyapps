"""
A combination of the errors and the hardness objective.

>>> orig = Instance.from_resource("a04")
>>> space = InstanceSpace(orig)
>>> print(f"{space.inst_name!r} with {space.n_different_items}/"
...       f"{space.n_items} items with area {space.total_item_area} "
...       f"in {space.min_bins} bins of "
...       f"size {space.bin_width}*{space.bin_height}.")
'a04n' with 2/16 items with area 7305688 in 3 bins of size 2750*1220.

>>> from moptipyapps.binpacking2d.instgen.inst_decoding import InstanceDecoder
>>> decoder = InstanceDecoder(space)
>>> import numpy as np
>>> x = np.array([ 0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, ])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 15/16 items with area 10065000 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;15;2750;1220;1101,1098;2750,244;2750,98;1101,171;1649,171;2750,976;\
441,122;1649,122;2750,10;2750,1,2;2750,3;1649,1098;2750,878;2750,58;660,122

>>> obj = ErrorsAndHardness(space, max_fes=100)
>>> obj.lower_bound()
0.0
>>> obj.upper_bound()
1.0
>>> obj.evaluate(y)
0.5439113595945743

>>> obj.evaluate(orig)
0.9275907603155895
"""
from typing import Callable, Final, Iterable

from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection

from moptipyapps.binpacking2d.instance import (
    Instance,
)
from moptipyapps.binpacking2d.instgen.errors import Errors
from moptipyapps.binpacking2d.instgen.hardness import (
    DEFAULT_EXECUTORS,
    Hardness,
)
from moptipyapps.binpacking2d.instgen.instance_space import InstanceSpace


class ErrorsAndHardness(Objective):
    """Compute the errors and hardness."""

    def __init__(self, space: InstanceSpace,
                 max_fes: int = 1_000_000, n_runs: int = 3,
                 executors: Iterable[
                     Callable[[Instance], tuple[
                         Execution, Objective]]] = DEFAULT_EXECUTORS) -> None:
        """
        Initialize the errors objective function.

        :param space: the instance space
        :param max_fes: the maximum FEs
        :param n_runs: the maximum runs
        :param executors: the functions creating the executions
        """
        super().__init__()
        #: the errors objective
        self.errors: Final[Errors] = Errors(space)
        #: the hardness objective function
        self.hardness: Final[Hardness] = Hardness(
            max_fes, n_runs, executors)

    def evaluate(self, x: list[Instance] | Instance) -> float:
        """
        Compute the combination of hardness and errors.

        :param x: the instance
        :return: the hardness and errors
        """
        return max(0.0, min(1.0, ((self.hardness.evaluate(
            x) * 1000.0) + self.errors.evaluate(x)) / 1001.0))

    def lower_bound(self) -> float:
        """
        Get the lower bound of the instance error and hardness.

        :returns 0.0: always
        """
        return 0.0

    def upper_bound(self) -> float:
        """
        Get the upper bound of the instance error and hardness.

        :returns 1.0: always
        """
        return 1.0

    def is_always_integer(self) -> bool:
        """
        Return `False` because the hardness function returns `float`.

        :retval False: always
        """
        return False

    def __str__(self) -> str:
        """
        Get the name of the errors objective function.

        :return: `errorsAndHardness`
        :retval "errorsAndHardness": always
        """
        return "errorsAndHardness"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this instance.

        :param logger: the logger
        """
        super().log_parameters_to(logger)
        with logger.scope("err") as err:
            self.errors.log_parameters_to(err)
        with logger.scope("hard") as hard:
            self.hardness.log_parameters_to(hard)
