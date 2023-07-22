"""An extended end result statistics record to represent packings."""
import os.path
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Final, Iterable, Mapping, cast

from moptipy.api.logging import (
    KEY_ALGORITHM,
    KEY_BEST_F,
    KEY_GOAL_F,
    KEY_INSTANCE,
    KEY_LAST_IMPROVEMENT_FE,
    KEY_LAST_IMPROVEMENT_TIME_MILLIS,
    KEY_MAX_FES,
    KEY_MAX_TIME_MILLIS,
    KEY_TOTAL_FES,
    KEY_TOTAL_TIME_MILLIS,
)
from moptipy.evaluation.base import (
    KEY_N,
)
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import (
    KEY_BEST_F_SCALED,
    KEY_ERT_FES,
    KEY_ERT_TIME_MILLIS,
    KEY_N_SUCCESS,
    KEY_SUCCESS_FES,
    KEY_SUCCESS_TIME_MILLIS,
    EndStatistics,
)
from moptipy.evaluation.statistics import (
    EMPTY_CSV_ROW,
    Statistics,
)
from moptipy.utils.console import logger
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.path import Path
from moptipy.utils.strings import (
    num_to_str,
)
from moptipy.utils.types import check_int_range, immutable_mapping, type_error

from moptipyapps.binpacking2d.bin_count import BIN_COUNT_NAME
from moptipyapps.binpacking2d.packing_result import (
    _OBJECTIVE_LOWER,
    _OBJECTIVE_UPPER,
    KEY_BIN_HEIGHT,
    KEY_BIN_WIDTH,
    KEY_N_DIFFERENT_ITEMS,
    KEY_N_ITEMS,
    KEY_USED_OBJECTIVE,
    PackingResult,
)


@dataclass(frozen=True, init=False, order=True)
class PackingStatistics:
    """
    An end statistics record of one run of one algorithm on one problem.

    This record provides the information of the outcome of one application of
    one algorithm to one problem instance in an immutable way.
    """

    #: the original end statistics record
    end_statistics: EndStatistics
    #: the number of items in the instance
    n_items: int
    #: the number of different items in the instance
    n_different_items: int
    #: the bin width
    bin_width: int
    #: the bin heigth
    bin_height: int
    #: the objective values evaluated after the optimization
    objectives: Mapping[str, Statistics]
    #: the objective function used for optimization
    used_objective: str
    #: the bounds for the objective values (append ".lowerBound" and
    #: ".upperBound" to all objective function names)
    objective_bounds: Mapping[str, int]
    #: the bounds for the minimum number of bins of the instance
    bin_bounds: Mapping[str, int]

    def __init__(self,
                 n_items: int,
                 n_different_items: int,
                 bin_width: int,
                 bin_height: int,
                 end_statistics: EndStatistics,
                 objectives: Mapping[str, Statistics],
                 used_objective: str,
                 objective_bounds: Mapping[str, int | float],
                 bin_bounds: Mapping[str, int]):
        """
        Create a consistent instance of :class:`PackingStatistics`.

        :param end_statistics: the end statistics
        :param n_items: the number of items
        :param n_different_items: the number of different items
        :param bin_width: the bin width
        :param bin_height: the bin height
        :param objectives: the objective values computed after the
            optimization
        :param used_objective: the objective function used
        :param bin_bounds: the different bounds for the number of bins
        :param objective_bounds: the bounds for the objective functions
        :raises TypeError: if any parameter has a wrong type
        :raises ValueError: if the parameter values are inconsistent
        """
        super().__init__()
        if not isinstance(end_statistics, EndStatistics):
            raise type_error(end_statistics, "end_statistics", EndResult)
        if not isinstance(used_objective, str):
            raise type_error(used_objective, "used_objective", str)
        if end_statistics.best_f != objectives[used_objective]:
            raise ValueError(
                f"end_statistics.best_f={end_statistics.best_f}, but objecti"
                f"ves[{used_objective!r}]={objectives[used_objective]}.")
        if not isinstance(objectives, Mapping):
            raise type_error(objectives, "objectives", Mapping)
        if not isinstance(objective_bounds, Mapping):
            raise type_error(objective_bounds, "objective_bounds", Mapping)
        if not isinstance(bin_bounds, Mapping):
            raise type_error(bin_bounds, "bin_bounds", Mapping)
        if len(objective_bounds) != (2 * len(objectives)):
            raise ValueError(f"it is required that there is a lower and an "
                             f"upper bound for each of the {len(objectives)} "
                             f"functions, but we got {len(objective_bounds)} "
                             f"bounds, objectives={objectives}, "
                             f"objective_bounds={objective_bounds}.")

        for name, value in objectives.items():
            if not isinstance(name, str):
                raise type_error(
                    name, f"name of evaluation[{name!r}]={value!r}", str)
            if not isinstance(value, Statistics):
                raise type_error(
                    value, f"value of evaluation[{name!r}]={value!r}",
                    Statistics)
            lower = objective_bounds[f"{name}{_OBJECTIVE_LOWER}"]
            if not isfinite(lower):
                raise ValueError(f"{name}{_OBJECTIVE_LOWER}=={lower}.")
            upper = objective_bounds[f"{name}{_OBJECTIVE_UPPER}"]
            if not (lower <= value.minimum <= value.maximum <= upper):
                raise ValueError(
                    f"it is required that {name}{_OBJECTIVE_LOWER}<={name}."
                    f"min<={name}.max<={name}{_OBJECTIVE_UPPER}, but got "
                    f"{lower}, {value.minimum}, {value.maximum} and {upper}.")

        bins: Final[Statistics | None] = cast(
            Statistics, objectives[BIN_COUNT_NAME]) \
            if BIN_COUNT_NAME in objectives else None
        for name, value2 in bin_bounds.items():
            if not isinstance(name, str):
                raise type_error(
                    name, f"name of bounds[{name!r}]={value2!r}", str)
            check_int_range(value2, f"bounds[{name!r}]", 1, 1_000_000_000)
            if (bins is not None) and (bins.minimum < value2):
                raise ValueError(
                    f"number of bins={bins} is inconsistent with "
                    f"bound {name!r}={value2}.")

        object.__setattr__(self, "end_statistics", end_statistics)
        object.__setattr__(self, "objectives", immutable_mapping(objectives))
        object.__setattr__(self, "used_objective", used_objective)
        object.__setattr__(self, "objective_bounds",
                           immutable_mapping(objective_bounds))
        object.__setattr__(self, "bin_bounds", immutable_mapping(bin_bounds))
        object.__setattr__(self, "n_different_items", check_int_range(
            n_different_items, "n_different_items", 1, 1_000_000_000_000))
        object.__setattr__(self, "n_items", check_int_range(
            n_items, "n_items", n_different_items, 1_000_000_000_000))
        object.__setattr__(self, "bin_width", check_int_range(
            bin_width, "bin_width", 1, 1_000_000_000_000))
        object.__setattr__(self, "bin_height", check_int_range(
            bin_height, "bin_height", 1, 1_000_000_000_000))

    @staticmethod
    def from_packing_results(
            results: Iterable[PackingResult],
            collector: Callable[["PackingStatistics"], None]) -> None:
        """
        Create packing statistics from a sequence of packing results.

        :param results: the packing results
        :param collector: the collector receiving the created packing
            statistics
        """
        if not isinstance(results, Iterable):
            raise type_error(results, "results", Iterable)
        if not callable(collector):
            raise type_error(collector, "collector", call=True)
        groups: Final[dict[tuple[str, str, str], list[PackingResult]]] = {}
        objectives_set: set[str] = set()
        for i, pr in enumerate(results):
            if not isinstance(pr, PackingResult):
                raise type_error(pr, f"end_results[{i}]", PackingResult)
            setting: tuple[str, str, str] = \
                pr.end_result.algorithm, pr.end_result.instance, \
                pr.used_objective
            if setting in groups:
                groups[setting].append(pr)
            else:
                groups[setting] = [pr]
            objectives_set.update(pr.objectives.keys())

        if len(groups) <= 0:
            raise ValueError("results is empty!")
        if len(objectives_set) <= 0:
            raise ValueError("results has not objectives!")
        end_stats: Final[list[EndStatistics]] = []
        objectives: Final[list[str]] = sorted(objectives_set)

        for key in sorted(groups.keys()):
            data = groups[key]

            pr0 = data[0]
            n_items: int = pr0.n_items
            n_different_items: int = pr0.n_different_items
            bin_width: int = pr0.bin_width
            bin_height: int = pr0.bin_height
            used_objective: str = pr0.used_objective
            if used_objective not in objectives_set:
                raise ValueError(
                    f"{used_objective!r} not in {objectives_set!r}.")
            if used_objective != key[2]:
                raise ValueError(
                    f"used_objective={used_objective!r} different "
                    f"from key={key[2]}!?")
            objective_bounds: Mapping[str, int | float] = pr0.objective_bounds
            bin_bounds: Mapping[str, int] = pr0.bin_bounds
            for i, pr in enumerate(data):
                if n_items != pr.n_items:
                    raise ValueError(f"n_items={n_items} for data[0] but "
                                     f"{pr.n_items} for data[{i}]?")
                if n_different_items != pr.n_different_items:
                    raise ValueError(
                        f"n_different_items={n_different_items} for data[0] "
                        f"but {pr.n_different_items} for data[{i}]?")
                if bin_width != pr.bin_width:
                    raise ValueError(
                        f"bin_width={bin_width} for data[0] "
                        f"but {pr.bin_width} for data[{i}]?")
                if bin_height != pr.bin_height:
                    raise ValueError(
                        f"bin_height={bin_height} for data[0] "
                        f"but {pr.bin_height} for data[{i}]?")
                if used_objective != pr.used_objective:
                    raise ValueError(
                        f"used_objective={used_objective!r} for data[0] "
                        f"but {pr.used_objective!r} for data[{i}]?")
                if objective_bounds != pr.objective_bounds:
                    raise ValueError(
                        f"objective_bounds={objective_bounds!r} for data[0] "
                        f"but {pr.objective_bounds!r} for data[{i}]?")
                if bin_bounds != pr.bin_bounds:
                    raise ValueError(
                        f"bin_bounds={bin_bounds!r} for data[0] "
                        f"but {pr.bin_bounds!r} for data[{i}]?")

            EndStatistics.from_end_results((pr.end_result for pr in data),
                                           end_stats.append)
            if len(end_stats) != 1:
                raise ValueError(f"got {end_stats} from {data}?")

            collector(PackingStatistics(
                n_items=n_items,
                n_different_items=n_different_items,
                bin_width=bin_width,
                bin_height=bin_height,
                end_statistics=end_stats[0],
                objectives={
                    o: Statistics.create([pr.objectives[o] for pr in data])
                    for o in objectives
                },
                used_objective=used_objective,
                objective_bounds=objective_bounds,
                bin_bounds=bin_bounds,
            ))
            end_stats.clear()

    @staticmethod
    def to_csv(results: Iterable["PackingStatistics"], file: str) -> Path:
        """
        Write a sequence of packing statistics to a file in CSV format.

        :param results: the end statistics
        :param file: the path
        :return: the path of the file that was written
        """
        path: Final[Path] = Path.path(file)
        logger(f"Writing packing results to CSV file {path!r}.")
        Path.path(os.path.dirname(path)).ensure_dir_exists()

        # get a nicely sorted view on the statistics
        use_stats = sorted(
            results, key=lambda ppr: (
                ppr.end_statistics.algorithm,
                ppr.end_statistics.instance,
                ppr.used_objective,
                ppr.end_statistics))

        has_goal_f: int = 0  # 1
        has_best_f_scaled: bool = False  # 2
        has_n_success: bool = False  # 4
        has_success_fes: bool = False  # 8
        has_success_time_millis: bool = False  # 16
        has_ert_fes: bool = False  # 32
        has_ert_time_millis: bool = False  # 64
        has_max_fes: int = 0  # 128
        has_max_time_millis: int = 0  # 256
        checker: int = 511

        for ess in use_stats:
            es = ess.end_statistics
            if es.goal_f is not None:
                if isinstance(es.goal_f, Statistics):
                    has_goal_f = 2
                    checker &= ~1
                elif has_goal_f == 0:
                    has_goal_f = 1
            if es.best_f_scaled is not None:
                has_best_f_scaled = True
                checker &= ~2
            if es.n_success is not None:
                has_n_success = True
                checker &= ~4
            if es.success_fes is not None:
                has_success_fes = True
                checker &= ~8
            if es.success_time_millis is not None:
                has_success_time_millis = True
                checker &= ~16
            if es.ert_fes is not None:
                has_ert_fes = True
                checker &= ~32
            if es.ert_time_millis is not None:
                has_ert_time_millis = True
                checker &= ~64
            if es.max_fes is not None:
                if isinstance(es.max_fes, Statistics):
                    has_max_fes = 2
                    checker &= ~128
                elif has_max_fes == 0:
                    has_max_fes = 1
            if es.max_time_millis is not None:
                if isinstance(es.max_time_millis, Statistics):
                    has_max_time_millis = 2
                    checker &= ~256
                elif has_max_time_millis == 0:
                    has_max_time_millis = 1
            if checker == 0:
                break

        # get the names of the bounds and objectives
        bin_bounds_set: set[str] = set()
        objectives_set: set[str] = set()
        for pr in use_stats:
            bin_bounds_set.update(pr.bin_bounds.keys())
            objectives_set.update(pr.objectives.keys())
        bin_bounds: Final[list[str]] = sorted(bin_bounds_set)
        objectives: Final[list[str]] = sorted(objectives_set)

        with path.open_for_write() as out:
            wrt: Final[Callable] = out.write
            sep: Final[str] = CSV_SEPARATOR
            wrt(KEY_ALGORITHM)
            wrt(sep)
            wrt(KEY_INSTANCE)
            wrt(sep)
            wrt(KEY_USED_OBJECTIVE)
            wrt(sep)
            wrt(KEY_N_ITEMS)
            wrt(sep)
            wrt(KEY_N_DIFFERENT_ITEMS)
            wrt(sep)
            wrt(KEY_BIN_WIDTH)
            wrt(sep)
            wrt(KEY_BIN_HEIGHT)
            wrt(sep)
            for bb in bin_bounds:
                wrt(bb)
                wrt(sep)
            for oo in objectives:
                wrt(oo)
                wrt(_OBJECTIVE_LOWER)
                wrt(sep)
                wrt(oo)
                wrt(_OBJECTIVE_UPPER)
                wrt(sep)

            def h(p) -> None:
                wrt(sep.join(Statistics.csv_col_names(p)))

            wrt(KEY_N)
            wrt(sep)
            h(KEY_BEST_F)
            wrt(sep)
            h(KEY_LAST_IMPROVEMENT_FE)
            wrt(sep)
            h(KEY_LAST_IMPROVEMENT_TIME_MILLIS)
            wrt(sep)
            h(KEY_TOTAL_FES)
            wrt(sep)
            h(KEY_TOTAL_TIME_MILLIS)
            if has_goal_f == 1:
                wrt(sep)
                wrt(KEY_GOAL_F)
            elif has_goal_f == 2:
                wrt(sep)
                h(KEY_GOAL_F)
            if has_best_f_scaled:
                wrt(sep)
                h(KEY_BEST_F_SCALED)
            if has_n_success:
                wrt(sep)
                wrt(KEY_N_SUCCESS)
            if has_success_fes:
                wrt(sep)
                h(KEY_SUCCESS_FES)
            if has_success_time_millis:
                wrt(sep)
                h(KEY_SUCCESS_TIME_MILLIS)
            if has_ert_fes:
                wrt(sep)
                wrt(KEY_ERT_FES)
            if has_ert_time_millis:
                wrt(sep)
                wrt(KEY_ERT_TIME_MILLIS)
            if has_max_fes == 1:
                wrt(sep)
                wrt(KEY_MAX_FES)
            elif has_max_fes == 2:
                wrt(sep)
                h(KEY_MAX_FES)
            if has_max_time_millis == 1:
                wrt(sep)
                wrt(KEY_MAX_TIME_MILLIS)
            elif has_max_time_millis == 2:
                wrt(sep)
                h(KEY_MAX_TIME_MILLIS)
            for oo in objectives:
                wrt(sep)
                h(oo)

            out.write("\n")

            csv: Final[Callable] = Statistics.value_to_csv
            num: Final[Callable] = num_to_str

            for rec in use_stats:
                er = rec.end_statistics
                wrt(er.algorithm)
                wrt(sep)
                wrt(er.instance)
                wrt(sep)
                wrt(rec.used_objective)
                wrt(sep)
                wrt(str(rec.n_items))
                wrt(sep)
                wrt(str(rec.n_different_items))
                wrt(sep)
                wrt(str(rec.bin_width))
                wrt(sep)
                wrt(str(rec.bin_height))
                wrt(sep)

                for bb in bin_bounds:
                    wrt(str(rec.bin_bounds[bb]))
                    wrt(sep)
                for oo in objectives:
                    wrt(num_to_str(rec.objective_bounds[
                        f"{oo}{_OBJECTIVE_LOWER}"]))
                    wrt(sep)
                    wrt(num_to_str(rec.objective_bounds[
                        f"{oo}{_OBJECTIVE_UPPER}"]))
                    wrt(sep)

                wrt(str(er.n))
                wrt(sep)
                wrt(er.best_f.to_csv())
                wrt(sep)
                wrt(er.last_improvement_fe.to_csv())
                wrt(sep)
                wrt(er.last_improvement_time_millis.to_csv())
                wrt(sep)
                wrt(er.total_fes.to_csv())
                wrt(sep)
                wrt(er.total_time_millis.to_csv())
                if has_goal_f == 1:
                    wrt(sep)
                    if er.goal_f is not None:
                        wrt(num(er.goal_f))
                elif has_goal_f == 2:
                    wrt(sep)
                    if isinstance(er.goal_f, Statistics):
                        wrt(er.goal_f.to_csv())
                    elif isinstance(er.goal_f, int | float):
                        wrt(csv(er.goal_f))
                    else:
                        wrt(EMPTY_CSV_ROW)
                if has_best_f_scaled:
                    wrt(sep)
                    if er.best_f_scaled is None:
                        wrt(EMPTY_CSV_ROW)
                    else:
                        wrt(er.best_f_scaled.to_csv())
                if has_n_success:
                    wrt(sep)
                    if er.n_success is not None:
                        wrt(str(er.n_success))
                if has_success_fes:
                    wrt(sep)
                    if er.success_fes is None:
                        wrt(EMPTY_CSV_ROW)
                    else:
                        wrt(er.success_fes.to_csv())
                if has_success_time_millis:
                    wrt(sep)
                    if er.success_time_millis is None:
                        wrt(EMPTY_CSV_ROW)
                    else:
                        wrt(er.success_time_millis.to_csv())
                if has_ert_fes:
                    wrt(sep)
                    if er.ert_fes is not None:
                        wrt(num(er.ert_fes))
                if has_ert_time_millis:
                    wrt(sep)
                    if er.ert_time_millis is not None:
                        wrt(num(er.ert_time_millis))
                if has_max_fes == 1:
                    wrt(sep)
                    if er.max_fes is not None:
                        wrt(str(er.max_fes))
                elif has_max_fes == 2:
                    wrt(sep)
                    if isinstance(er.max_fes, Statistics):
                        wrt(er.max_fes.to_csv())
                    elif isinstance(er.max_fes, int | float):
                        wrt(csv(er.max_fes))
                    else:
                        wrt(EMPTY_CSV_ROW)
                if has_max_time_millis == 1:
                    wrt(sep)
                    if er.max_time_millis is not None:
                        wrt(str(er.max_time_millis))
                elif has_max_time_millis == 2:
                    wrt(sep)
                    if isinstance(er.max_time_millis, Statistics):
                        wrt(er.max_time_millis.to_csv())
                    elif isinstance(er.max_time_millis, int | float):
                        wrt(csv(er.max_time_millis))
                    else:
                        wrt(EMPTY_CSV_ROW)
                for oo in objectives:
                    wrt(sep)
                    wrt(rec.objectives[oo].to_csv())
                out.write("\n")

        # finally, return the path to the generated file
        return path
