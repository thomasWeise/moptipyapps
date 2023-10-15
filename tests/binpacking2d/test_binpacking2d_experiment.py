"""Test applying some algorithms to the 2D bin packing problem."""
from typing import Any, Callable, Final, cast

from moptipy.algorithms.so.fea1plus1 import FEA1plus1
from moptipy.algorithms.so.rls import RLS
from moptipy.api.encoding import Encoding
from moptipy.api.execution import Execution
from moptipy.api.experiment import run_experiment
from moptipy.api.objective import Objective
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import EndStatistics
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.operators.signed_permutations.op1_swap_2_or_flip import (
    Op1Swap2OrFlip,
)
from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.utils.path import Path
from moptipy.utils.temp import TempDir

from moptipyapps.binpacking2d.encodings.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
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
from moptipyapps.binpacking2d.packing_space import PackingSpace
from moptipyapps.binpacking2d.packing_statistics import PackingStatistics

INSTANCES: Final[list[str]] = [s for s in Instance.list_resources()
                               if s.startswith(("a", "b"))]

#: load the problem instance
PROBLEMS: Final = [lambda ss=s: Instance.from_resource(ss) for s in INSTANCES]

#: the maximum FEs
MAX_FES: Final[int] = 100

#: the number of runs
N_RUNS: Final[int] = 3


def __basic(problem: Instance,
            objective: Callable[[Instance], Objective],
            encoding: Callable[[Instance], Encoding])\
        -> tuple[Execution, SignedPermutations]:
    """Create the basic execution and search space."""
    search_space: Final[SignedPermutations] = SignedPermutations(
        problem.get_standard_item_sequence())
    return Execution().set_search_space(search_space)\
        .set_solution_space(PackingSpace(problem))\
        .set_encoding(encoding(problem))\
        .set_objective(objective(problem))\
        .set_max_fes(MAX_FES)\
        .set_log_improvements(True), search_space


def __ea(problem: Instance,
         objective: Callable[[Instance], Objective],
         encoding: Callable[[Instance], Encoding]) -> Execution:
    """Create the (1+1) EA execution."""
    ex, search_space = __basic(problem, objective, encoding)
    return ex.set_algorithm(RLS(
        Op0ShuffleAndFlip(search_space), Op1Swap2OrFlip()))


def __fea(problem: Instance,
          objective: Callable[[Instance], Objective],
          encoding: Callable[[Instance], Encoding]) -> Execution:
    """Create the (1+1) FEA execution."""
    ex, search_space = __basic(problem, objective, encoding)
    return ex.set_algorithm(FEA1plus1(
        Op0ShuffleAndFlip(search_space), Op1Swap2OrFlip()))


def __make_ea(objective: Callable[[Instance], Objective],
              encoding: Callable[[Instance], Encoding])\
        -> Callable[[Any], Execution]:
    """Make the EA execution creator."""
    return cast(Callable[[Any], Execution],
                lambda inst, __ob=objective, __en=encoding:
                __ea(cast(Instance, inst), __ob, __en))


def __make_fea(objective: Callable[[Instance], Objective],
               encoding: Callable[[Instance], Encoding])\
        -> Callable[[Any], Execution]:
    """Make the FEA execution creator."""
    return cast(Callable[[Any], Execution],
                lambda inst, __ob=objective, __en=encoding:
                __fea(cast(Instance, inst), __ob, __en))


def __evaluate(results: Path, evaluation: Path) -> None:
    """
    Apply the evaluation procedure.

    :param results: the results directory
    :param evaluation: the evaluation directory
    """
    end_results: Final[list[EndResult]] = []
    EndResult.from_logs(results, end_results.append)
    for err in end_results:
        assert err.max_fes == MAX_FES
        assert err.instance in INSTANCES
    assert len(end_results) == N_RUNS * len(INSTANCES) * 2
    instances_1 = {er.instance for er in end_results}
    assert instances_1 == set(INSTANCES)
    algorithms_1 = {er.algorithm for er in end_results}
    assert len(algorithms_1) == 2
    seeds_1 = {er.rand_seed for er in end_results}
    assert N_RUNS <= len(seeds_1) <= N_RUNS * len(INSTANCES)
    p = Path.resolve_inside(evaluation, "results.txt")
    EndResult.to_csv(end_results, p)
    end_results_2: Final[list[EndResult]] = []
    EndResult.from_csv(p, end_results_2.append)

    assert sorted(end_results, key=lambda er: (
        er.algorithm, er.instance, er.rand_seed)) == sorted(
        end_results_2, key=lambda er: (
            er.algorithm, er.instance, er.rand_seed))

    end_statistics: Final[list[EndStatistics]] = []
    EndStatistics.from_end_results(end_results, end_statistics.append)
    assert len(end_statistics) == len(INSTANCES) * 2
    algorithms_2 = {es.algorithm for es in end_statistics}
    assert len(algorithms_2) == 2
    assert algorithms_2 == algorithms_1
    instances_2 = {es.instance for es in end_statistics}
    assert set(INSTANCES) == instances_2
    p = Path.resolve_inside(evaluation, "statistics.txt")
    EndStatistics.to_csv(end_statistics, p)
    end_statistics_2: Final[list[EndStatistics]] = []
    EndStatistics.from_csv(p, end_statistics_2.append)
    assert sorted(
        end_statistics, key=lambda er: (
            er.algorithm, er.instance)) == sorted(
        end_statistics_2, key=lambda er: (
            er.algorithm, er.instance))

    packing_results: Final[list[PackingResult]] = []
    PackingResult.from_logs(results, packing_results.append)
    for exr in packing_results:
        assert exr.end_result.rand_seed in seeds_1
        assert exr.end_result.algorithm in algorithms_1
        assert exr.end_result.max_fes == MAX_FES
        assert exr.end_result.instance in instances_1
    algorithms_3 = {er.end_result.algorithm for er in packing_results}
    assert algorithms_3 == algorithms_1
    instances_3 = {er.end_result.instance for er in packing_results}
    assert instances_3 == instances_1
    assert len(packing_results) == N_RUNS * len(INSTANCES) * 2
    seeds_2 = {er.end_result.rand_seed for er in packing_results}
    assert seeds_2 == seeds_1
    p = Path.resolve_inside(evaluation, "pack_results.txt")
    PackingResult.to_csv(packing_results, p)
    packing_results_2: Final[list[PackingResult]] = []
    PackingResult.from_csv(p, packing_results_2.append)
    assert sorted(packing_results, key=lambda er: (
        er.end_result.algorithm, er.end_result.instance,
        er.end_result.rand_seed)) == sorted(
        packing_results_2, key=lambda er: (
            er.end_result.algorithm, er.end_result.instance,
            er.end_result.rand_seed))

    packing_statistics: Final[list[PackingStatistics]] = []
    PackingStatistics.from_packing_results(
        packing_results, packing_statistics.append)
    assert len(packing_statistics) == len(INSTANCES) * 2
    assert len({es.end_statistics.algorithm
                for es in packing_statistics}) == 2
    instances_4 = {es.end_statistics.instance for es in packing_statistics}
    assert instances_4 == instances_1
    algorithms_4 = {er.end_statistics.algorithm for er in packing_statistics}
    assert algorithms_4 == algorithms_1
    p = Path.resolve_inside(evaluation, "pack_statistics.txt")
    PackingStatistics.to_csv(packing_statistics, p)


def test_experiment_empty_1() -> None:
    """Test the bin packing with BinCountAndLastEmpty and encoding 1."""
    with TempDir.create() as td:
        results = td.resolve_inside("results")
        results.ensure_dir_exists()
        evaluation = td.resolve_inside("evaluation")
        evaluation.ensure_dir_exists()
        run_experiment(base_dir=results,
                       instances=PROBLEMS,
                       setups=[__make_ea(BinCountAndLastEmpty,
                                         ImprovedBottomLeftEncoding1),
                               __make_fea(BinCountAndLastEmpty,
                                          ImprovedBottomLeftEncoding1)],
                       n_runs=N_RUNS)
        __evaluate(results, evaluation)


def test_experiment_empty_2() -> None:
    """Test the bin packing with BinCountAndLastEmpty and encoding 2."""
    with TempDir.create() as td:
        results = td.resolve_inside("results")
        results.ensure_dir_exists()
        evaluation = td.resolve_inside("evaluation")
        evaluation.ensure_dir_exists()
        run_experiment(base_dir=results,
                       instances=PROBLEMS,
                       setups=[__make_ea(BinCountAndLastEmpty,
                                         ImprovedBottomLeftEncoding2),
                               __make_fea(BinCountAndLastEmpty,
                                          ImprovedBottomLeftEncoding2)],
                       n_runs=N_RUNS)
        __evaluate(results, evaluation)


def test_experiment_small_1() -> None:
    """Test the bin packing with BinCountAndLastSmall and encoding 1."""
    with TempDir.create() as td:
        results = td.resolve_inside("results")
        results.ensure_dir_exists()
        evaluation = td.resolve_inside("evaluation")
        evaluation.ensure_dir_exists()
        run_experiment(base_dir=results,
                       instances=PROBLEMS,
                       setups=[__make_ea(BinCountAndLastSmall,
                                         ImprovedBottomLeftEncoding1),
                               __make_fea(BinCountAndLastSmall,
                                          ImprovedBottomLeftEncoding1)],
                       n_runs=N_RUNS)
        __evaluate(results, evaluation)


def test_experiment_small_2() -> None:
    """Test the bin packing with BinCountAndLastSmall and encoding 2."""
    with TempDir.create() as td:
        results = td.resolve_inside("results")
        results.ensure_dir_exists()
        evaluation = td.resolve_inside("evaluation")
        evaluation.ensure_dir_exists()
        run_experiment(base_dir=results,
                       instances=PROBLEMS,
                       setups=[__make_ea(BinCountAndLastSmall,
                                         ImprovedBottomLeftEncoding2),
                               __make_fea(BinCountAndLastSmall,
                                          ImprovedBottomLeftEncoding2)],
                       n_runs=N_RUNS)
        __evaluate(results, evaluation)
