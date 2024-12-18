"""
An objective function assessing the hardness of an instance.

>>> from moptipyapps.binpacking2d.instgen.instance_space import InstanceSpace
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

>>> hardness = Hardness(max_fes=1000)
>>> hardness.lower_bound()
0.0
>>> hardness.upper_bound()
1.0
>>> hardness.evaluate(y)
0.6688894353015722

>>> y[0] = orig
>>> hardness.evaluate(y)
0.9298025539793962

>>> z = Instance.from_compact_str(
...     "cl04_020_01n;19;100;100;1,10;2,38;2,62;1,4,2;3,38;1,7;27,93;1,62;1,"
...     "3;13,38;1,38;1,17;1,45;36,62;39,3;1,2;20,10;3,24;12,4")
>>> hardness.evaluate(z)
0.998823040627722
"""
from math import fsum, isfinite
from typing import Callable, Final, Iterable

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.objective import Objective
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.operators.signed_permutations.op1_swap_2_or_flip import (
    Op1Swap2OrFlip,
)
from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import rand_seeds_from_str
from pycommons.types import check_int_range

from moptipyapps.binpacking2d.encodings.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
from moptipyapps.binpacking2d.instance import (
    Instance,
)
from moptipyapps.binpacking2d.objectives.bin_count import BinCount
from moptipyapps.binpacking2d.objectives.bin_count_and_last_skyline import (
    BinCountAndLastSkyline,
)
from moptipyapps.binpacking2d.packing_space import PackingSpace


def setup_rls_f7(instance: Instance) -> tuple[Execution, Objective]:
    """
    Set up the randomized local search for an instance.

    :param instance: the instance
    :return: the execution and upper bound of the objective
    """
    search_space = SignedPermutations(
        instance.get_standard_item_sequence())  # Create the search space.
    solution_space = PackingSpace(instance)  # Create the space of packings.
    objective: Final[BinCountAndLastSkyline] = BinCountAndLastSkyline(instance)

    # Build a single execution of a single run of a single algorithm and
    # return the upper bound of the objective value
    return (Execution()
            .set_search_space(search_space)
            .set_solution_space(solution_space)
            .set_encoding(ImprovedBottomLeftEncoding1(instance))
            .set_algorithm(  # This is the algorithm: Randomized Local Search.
                RLS(Op0ShuffleAndFlip(search_space), Op1Swap2OrFlip()))
            .set_objective(objective), objective)


def setup_rls_f1(instance: Instance) -> tuple[Execution, Objective]:
    """
    Set up the randomized local search for an instance.

    :param instance: the instance
    :return: the execution
    """
    search_space = SignedPermutations(
        instance.get_standard_item_sequence())  # Create the search space.
    solution_space = PackingSpace(instance)  # Create the space of packings.
    objective: Final[BinCount] = BinCount(instance)

    # Build a single execution of a single run of a single algorithm and
    # return the lower bound of the objective value
    return (Execution()
            .set_search_space(search_space)
            .set_solution_space(solution_space)
            .set_encoding(ImprovedBottomLeftEncoding1(instance))
            .set_algorithm(  # This is the algorithm: Randomized Local Search.
                RLS(Op0ShuffleAndFlip(search_space), Op1Swap2OrFlip()))
            .set_objective(objective), objective)


#: the default executors
DEFAULT_EXECUTORS: Final[tuple[Callable[[Instance], tuple[
    Execution, Objective]], ...]] = (setup_rls_f1, setup_rls_f7)


class Hardness(Objective):
    """Compute the hardness of an instance."""

    def __init__(
            self, max_fes: int = 1_000_000, n_runs: int = 3,
            executors: Iterable[Callable[[Instance], tuple[
                Execution, Objective]]] = DEFAULT_EXECUTORS) -> None:
        """
        Initialize the hardness objective function.

        :param max_fes: the maximum FEs
        :param n_runs: the maximum runs
        :param executors: the functions creating the executions
        """
        super().__init__()
        #: the maximum FEs per setup.
        self.max_fes: Final[int] = check_int_range(
            max_fes, "max_fes", 2, 1_000_000_000_000)
        #: the maximum FEs per setup.
        self.n_runs: Final[int] = check_int_range(
            n_runs, "n_runs", 1, 1_000_000)
        #: the executors
        self.executors: Final[tuple[Callable[[Instance], tuple[
            Execution, Objective]], ...]] = tuple(executors)

        #: the last instance name
        self.__last_inst: str | None = None
        #: the last seeds name
        self.__last_seeds: tuple[int, ...] | None = None
        #: the internal results list
        self.__results: Final[list[float]] = []

    def evaluate(self, x: list[Instance] | Instance) -> float:
        """
        Compute the hardness of an instance.

        :param x: the instance
        :return: the hardness
        """
        instance: Final[Instance] = x[0] if isinstance(x, list) else x
        seeds: tuple[int, ...]

        name: str = instance.name
        if (self.__last_seeds is None) or (self.__last_inst is None) or (
                self.__last_inst != name):
            self.__last_seeds = seeds = tuple(rand_seeds_from_str(
                f"seed for {instance.name}", self.n_runs))
            self.__last_inst = name
        else:
            seeds = self.__last_seeds

        max_fes: Final[int] = self.max_fes
        runs: int = 0
        results: Final[list[float]] = self.__results
        results.clear()
        for executor in self.executors:
            execs, f = executor(instance)
            lb: int | float = f.lower_bound()
            ub: int | float = f.upper_bound()
            if not (isfinite(lb) and isfinite(ub) and (lb < ub)):
                raise ValueError(f"Invalid lower and upper bound {lb}, {ub}.")
            execs.set_max_fes(max_fes)
            for seed in seeds:
                execs.set_rand_seed(seed)
                with execs.execute() as proc:
                    runs += 1
                    quality: int | float = proc.get_best_f()
                    if not (isfinite(quality) and (lb <= quality <= ub)):
                        raise ValueError(
                            f"quality={quality} invalid, must be in "
                            f"[{lb}, {ub}] for objective {f}.")
                    runtime: int | float = proc.get_last_improvement_fe()
                    if not (0 < runtime <= max_fes):
                        raise ValueError(f"invalid FEs {runtime}, must "
                                         f"be in 1..{max_fes}.")
                    runtime = (max_fes - runtime) / max_fes
                    if not (0.0 <= runtime < 1.0):
                        raise ValueError(
                            f"invalid normalized runtime {runtime}.")
                    quality = ((ub - quality) + runtime) / (ub - lb + 1)
                    if not (0.0 <= quality <= 1.0):
                        raise ValueError(
                            f"invalid normalized quality {quality} "
                            f"for objective {f}.")
                    results.append(quality ** 4)
        ret: Final[float] = max(0.0, min(1.0, fsum(results) / runs))
        results.clear()
        return ret

    def lower_bound(self) -> float:
        """
        Get the lower bound of the instance hardness.

        :return: the lower bound for the instance hardness
        :returns 0.0: always
        """
        return 0.0

    def upper_bound(self) -> float:
        """
        Get the upper bound of the instance hardness.

        :return: the upper bound for the instance hardness
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
        Get the name of the hardness objective function.

        :return: `hardness`
        :retval "hardness": always
        """
        return "hardness"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this instance.

        :param logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value("nRuns", self.n_runs)
        logger.key_value("maxFEs", self.max_fes)
        logger.key_value("nExecutors", len(self.executors))
