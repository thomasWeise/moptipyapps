"""Test the packing result."""


from typing import Final

from moptipy.algorithms.random_sampling import RandomSampling
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.operators.signed_permutations.op1_swap_2_or_flip import (
    Op1Swap2OrFlip,
)
from moptipy.spaces.signed_permutations import SignedPermutations
from numpy.random import Generator, default_rng
from pycommons.io.temp import temp_dir, temp_file

from moptipyapps.binpacking2d.encodings.ibl_encoding_2 import (
    ImprovedBottomLeftEncoding2,
)
from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.objectives.bin_count_and_last_empty import (
    BinCountAndLastEmpty,
)
from moptipyapps.binpacking2d.objectives.bin_count_and_last_small import (
    BinCountAndLastSmall,
)
from moptipyapps.binpacking2d.packing_result import PackingResult
from moptipyapps.binpacking2d.packing_result import from_csv as pr_from_csv
from moptipyapps.binpacking2d.packing_result import from_logs as pr_from_logs
from moptipyapps.binpacking2d.packing_result import to_csv as pr_to_csv
from moptipyapps.binpacking2d.packing_space import PackingSpace
from moptipyapps.binpacking2d.packing_statistics import PackingStatistics
from moptipyapps.binpacking2d.packing_statistics import from_csv as ps_from_csv
from moptipyapps.binpacking2d.packing_statistics import (
    from_packing_results as ps_from_packing_results,
)
from moptipyapps.binpacking2d.packing_statistics import to_csv as ps_to_csv

#: the maximum permitted FEs
__MAX_FES: Final[int] = 64


def __make_setup_1(instance: Instance) -> Execution:
    """
    Create the setup 1.

    :param instance: the instance
    :returns: the execution
    """
    search_space = SignedPermutations(
        instance.get_standard_item_sequence())  # Create the search space.
    solution_space = PackingSpace(instance)  # Create the space of packings.
    return Execution().set_search_space(search_space)\
        .set_solution_space(solution_space)\
        .set_encoding(ImprovedBottomLeftEncoding2(instance))\
        .set_algorithm(  # This is the algorithm: Randomized Local Search.
            RLS(Op0ShuffleAndFlip(search_space), Op1Swap2OrFlip()))\
        .set_objective(BinCountAndLastEmpty(instance))\
        .set_max_fes(__MAX_FES)


def __make_setup_2(instance: Instance) -> Execution:
    """
    Create the setup 2.

    :param instance: the instance
    :returns: the execution
    """
    search_space = SignedPermutations(
        instance.get_standard_item_sequence())  # Create the search space.
    solution_space = PackingSpace(instance)  # Create the space of packings.
    return Execution().set_search_space(search_space)\
        .set_solution_space(solution_space)\
        .set_encoding(ImprovedBottomLeftEncoding2(instance))\
        .set_algorithm(  # This is the algorithm: Randomized Local Search.
            RandomSampling(Op0ShuffleAndFlip(search_space)))\
        .set_objective(BinCountAndLastSmall(instance))\
        .set_max_fes(__MAX_FES)


def test_packing_results_experiment() -> None:
    """Test a small experiment with packing results."""
    random: Generator = default_rng()
    all_instances: tuple[str, ...] = Instance.list_resources()
    instances: set[str] = set()
    while True:
        instances.add(all_instances[random.integers(len(all_instances))])
        if (len(instances) > 3) and (random.integers(5) <= 0):
            break

    instance_factories = [
        lambda _i=i: Instance.from_resource(_i) for i in instances]
    algorithms = [__make_setup_1, __make_setup_2]

    n_runs: Final[int] = int(random.integers(2, 7))

    with temp_dir() as td:
        run_experiment(base_dir=td,
                       instances=instance_factories,
                       setups=algorithms,
                       n_runs=n_runs)
        results_1: list[PackingResult] = list(pr_from_logs(td))
        results_2: list[PackingResult] = []
        assert len(results_1) == \
               len(algorithms) * n_runs * len(instance_factories)
        results_1.sort()
        all_objectives: set[str] = set()
        all_algorithms: set[str] = set()
        for res in results_1:
            assert res.end_result.instance in all_instances
            assert res.end_result.max_fes == __MAX_FES
            all_objectives.add(res.end_result.objective)
            all_algorithms.add(res.end_result.algorithm)
        assert len(all_algorithms) == len(algorithms)
        assert len(all_objectives) == 2
        with temp_file() as tf:
            pr_to_csv(results_1, tf)
            results_2.extend(pr_from_csv(tf))
            assert len(results_2) == len(results_1)
            results_2.sort()
            assert results_1 == results_2

        end_stats_1: Final[list[PackingStatistics]] = []
        ps_from_packing_results(results_1, end_stats_1.append)
        assert len(end_stats_1) == len(algorithms) * len(instance_factories)
        end_stats_2: Final[list[PackingStatistics]] = []
        ps_from_packing_results(results_2, end_stats_2.append)
        assert len(end_stats_1) == len(end_stats_2)
        end_stats_3: Final[list[PackingStatistics]] = []
        with temp_file() as tf2:
            ps_to_csv(end_stats_1, tf2)
            end_stats_3.extend(ps_from_csv(tf2))
        assert end_stats_3 == end_stats_2
