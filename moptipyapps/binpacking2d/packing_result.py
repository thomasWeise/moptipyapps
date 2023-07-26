"""
An extended end result record to represent packings.

This class extends the information provided by
:mod:`~moptipy.evaluation.end_results`. It allows us to compare the results
of experiments over different objective functions. It also represents the
bounds for the number of bins and for the objective functions. It also
includes the problem-specific information of two-dimensional bin packing
instances.
"""
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
    KEY_RAND_SEED,
    KEY_TOTAL_FES,
    KEY_TOTAL_TIME_MILLIS,
)
from moptipy.api.objective import Objective
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.log_parser import LogParser
from moptipy.utils.console import logger
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.path import Path
from moptipy.utils.strings import (
    intfloatnone_to_str,
    intnone_to_str,
    num_to_str,
    str_to_intfloat,
    str_to_intfloatnone,
    str_to_intnone,
)
from moptipy.utils.types import check_int_range, immutable_mapping, type_error

from moptipyapps.binpacking2d.bin_count import BIN_COUNT_NAME, BinCount
from moptipyapps.binpacking2d.bin_count_and_last_empty import (
    BinCountAndLastEmpty,
)
from moptipyapps.binpacking2d.bin_count_and_last_small import (
    BinCountAndLastSmall,
)
from moptipyapps.binpacking2d.instance import Instance, _lower_bound_damv
from moptipyapps.binpacking2d.packing import Packing, _PackingParser

#: the name of the objective function
KEY_USED_OBJECTIVE: Final[str] = "objective"
#: the number of items
KEY_N_ITEMS: Final[str] = "nItems"
#: the number of different items
KEY_N_DIFFERENT_ITEMS: Final[str] = "nDifferentItems"
#: the bin width
KEY_BIN_WIDTH: Final[str] = "binWidth"
#: the bin height
KEY_BIN_HEIGHT: Final[str] = "binHeight"

#: The internal CSV header part 1, after which the bounds are inserted
_HEADER_1: Final[str] = (f"{KEY_ALGORITHM}{CSV_SEPARATOR}"
                         f"{KEY_INSTANCE}{CSV_SEPARATOR}"
                         f"{KEY_USED_OBJECTIVE}{CSV_SEPARATOR}"
                         f"{KEY_N_ITEMS}{CSV_SEPARATOR}"
                         f"{KEY_N_DIFFERENT_ITEMS}{CSV_SEPARATOR}"
                         f"{KEY_BIN_WIDTH}{CSV_SEPARATOR}"
                         f"{KEY_BIN_HEIGHT}")


#: The internal CSV header part 2, after which the objective values are listed
_HEADER_2: Final[str] = (f"{KEY_RAND_SEED}{CSV_SEPARATOR}"
                         f"{KEY_BEST_F}{CSV_SEPARATOR}"
                         f"{KEY_LAST_IMPROVEMENT_FE}{CSV_SEPARATOR}"
                         f"{KEY_LAST_IMPROVEMENT_TIME_MILLIS}"
                         f"{CSV_SEPARATOR}"
                         f"{KEY_TOTAL_FES}{CSV_SEPARATOR}"
                         f"{KEY_TOTAL_TIME_MILLIS}{CSV_SEPARATOR}"
                         f"{KEY_GOAL_F}{CSV_SEPARATOR}"
                         f"{KEY_MAX_FES}{CSV_SEPARATOR}"
                         f"{KEY_MAX_TIME_MILLIS}")


#: the default objective functions
_DEFAULT_OBJECTIVES: Final[tuple[Callable[[Instance], Objective], ...]] = (
    BinCount, BinCountAndLastEmpty, BinCountAndLastSmall,
)


def __lb_geometric(inst: Instance) -> int:
    """
    Compute the geometric lower bound.

    :param inst: the instance
    :return: the lower bound
    """
    area: Final[int] = sum(int(row[0]) * int(row[1]) * int(row[2])
                           for row in inst)
    bin_size: Final[int] = inst.bin_width * inst.bin_height
    res: int = area // bin_size
    return (res + 1) if ((res * bin_size) != area) else res


#: the lower bound of an objective
_OBJECTIVE_LOWER: Final[str] = ".lowerBound"
#: the upper bound of an objective
_OBJECTIVE_UPPER: Final[str] = ".upperBound"
#: the start string for bin bounds
_BINS_START: Final[str] = f"bins{_OBJECTIVE_LOWER}"
#: the default bounds
_DEFAULT_BOUNDS: Final[Mapping[str, Callable[[Instance], int]]] = \
    immutable_mapping({
        _BINS_START: lambda i: i.lower_bound_bins,
        f"{_BINS_START}.geometric": __lb_geometric,
        f"{_BINS_START}.damv": lambda i: _lower_bound_damv(
            i.bin_width, i.bin_height, i),
    })


@dataclass(frozen=True, init=False, order=True)
class PackingResult:
    """
    An end result record of one run of one packing algorithm on one problem.

    This record provides the information of the outcome of one application of
    one algorithm to one problem instance in an immutable way.
    """

    #: the original end result record
    end_result: EndResult
    #: the number of items in the instance
    n_items: int
    #: the number of different items in the instance
    n_different_items: int
    #: the bin width
    bin_width: int
    #: the bin heigth
    bin_height: int
    #: the objective values evaluated after the optimization
    objectives: Mapping[str, int]
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
                 end_result: EndResult,
                 objectives: Mapping[str, int | float],
                 used_objective: str,
                 objective_bounds: Mapping[str, int | float],
                 bin_bounds: Mapping[str, int]):
        """
        Create a consistent instance of :class:`PackingResult`.

        :param end_result: the end result
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
        if not isinstance(end_result, EndResult):
            raise type_error(end_result, "end_result", EndResult)
        if not isinstance(used_objective, str):
            raise type_error(used_objective, "used_objective", str)
        if end_result.best_f != objectives[used_objective]:
            raise ValueError(
                f"end_result.best_f={end_result.best_f}, but objectives["
                f"{used_objective!r}]={objectives[used_objective]}.")
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
            if not isinstance(value, int | float):
                raise type_error(
                    value, f"value of evaluation[{name!r}]={value!r}",
                    (int, float))
            if not isfinite(value):
                raise ValueError(
                    f"non-finite value of evaluation[{name!r}]={value!r}")
            lower = objective_bounds[f"{name}{_OBJECTIVE_LOWER}"]
            if not isfinite(lower):
                raise ValueError(f"{name}{_OBJECTIVE_LOWER}=={lower}.")
            upper = objective_bounds[f"{name}{_OBJECTIVE_UPPER}"]
            if not (lower <= value <= upper):
                raise ValueError(
                    f"it is required that {name}{_OBJECTIVE_LOWER}<=f<={name}"
                    f"{_OBJECTIVE_UPPER}, but got {lower}, {value}, and "
                    f"{upper}.")

        bins: Final[int | None] = cast(int, objectives[BIN_COUNT_NAME]) \
            if BIN_COUNT_NAME in objectives else None
        for name, value in bin_bounds.items():
            if not isinstance(name, str):
                raise type_error(
                    name, f"name of bounds[{name!r}]={value!r}", str)
            check_int_range(value, f"bounds[{name!r}]", 1, 1_000_000_000)
            if (bins is not None) and (bins < value):
                raise ValueError(
                    f"number of bins={bins} is inconsistent with "
                    f"bound {name!r}={value}.")

        object.__setattr__(self, "end_result", end_result)
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
    def from_packing_and_end_result(  # pylint: disable=W0102
            end_result: EndResult, packing: Packing, used_objective: str,
            objectives: Iterable[Callable[[Instance], Objective]] =
            _DEFAULT_OBJECTIVES,
            bin_bounds: Mapping[str, Callable[[Instance], int]] =
            _DEFAULT_BOUNDS,
            cache: Mapping[str, tuple[Mapping[str, int], tuple[
                Objective, ...], Mapping[str, int | float]]] | None =
            None) -> "PackingResult":
        """
        Create a `PackingResult` from an `EndResult` and a `Packing`.

        :param end_result: the end results record
        :param packing: the packing
        :param used_objective: the used objective function
        :param bin_bounds: the bounds computing functions
        :param objectives: the objective function factories
        :param cache: a cache that can store stuff if this function is to be
            called repeatedly
        :return: the packing result
        """
        if not isinstance(end_result, EndResult):
            raise type_error(end_result, "end_result", EndResult)
        if not isinstance(packing, Packing):
            raise type_error(packing, "packing", Packing)
        if not isinstance(used_objective, str):
            raise type_error(used_objective, "used_objective", str)
        if not isinstance(objectives, Iterable):
            raise type_error(objectives, "objectives", Iterable)
        if not isinstance(bin_bounds, Mapping):
            raise type_error(bin_bounds, "bin_bounds", Mapping)
        if (cache is not None) and (not isinstance(cache, dict)):
            raise type_error(cache, "cache", (None, dict))

        instance: Final[Instance] = packing.instance
        if instance.name != end_result.instance:
            raise ValueError(
                f"packing.instance.name={instance.name!r}, but "
                f"end_result.instance={end_result.instance!r}.")

        row: tuple[Mapping[str, int], tuple[Objective, ...],
                   Mapping[str, int | float]] | None = None \
            if (cache is None) else cache.get(instance.name, None)
        if row is None:
            objfs = tuple(sorted((obj(instance) for obj in objectives),
                                 key=str))
            obounds = {}
            for objf in objfs:
                obounds[f"{objf}{_OBJECTIVE_LOWER}"] = objf.lower_bound()
                obounds[f"{objf}{_OBJECTIVE_UPPER}"] = objf.upper_bound()
            row = ({key: bin_bounds[key](instance)
                    for key in sorted(bin_bounds.keys())}, objfs,
                   immutable_mapping(obounds))
        if cache is not None:
            cache[instance.name] = row
        return PackingResult(
            n_items=instance.n_items,
            n_different_items=instance.n_different_items,
            bin_width=instance.bin_width, bin_height=instance.bin_height,
            end_result=end_result,
            objectives={str(objf): cast(Objective, objf).evaluate(packing)
                        for objf in row[1]},
            used_objective=used_objective,
            objective_bounds=row[2],
            bin_bounds=row[0])

    @staticmethod
    def from_single_log(  # pylint: disable=W0102
            file: str,
            objectives: Iterable[Callable[[Instance], Objective]] =
            _DEFAULT_OBJECTIVES,
            bin_bounds: Mapping[str, Callable[[Instance], int]] =
            _DEFAULT_BOUNDS,
            cache: Mapping[str, tuple[Mapping[str, int], tuple[
                Objective, ...], Mapping[str, int | float]]] | None =
            None) -> "PackingResult":
        """
        Create a `PackingResult` from a file.

        :param file: the file path
        :param objectives: the objective function factories
        :param bin_bounds: the bounds computing functions
        :param cache: a cache that can store stuff if this function is to be
            called repeatedly
        :return: the packing result
        """
        file_path = Path.file(file)
        end_results: Final[list[EndResult]] = []
        EndResult.from_logs(file_path, end_results.append)
        if len(end_results) != 1:
            raise ValueError(
                f"needs one end result record in file {file_path!r}, "
                f"but got {end_results}.")

        parser: Final[_PackingParser] = _PackingParser()
        parser.parse_file(file)
        # noinspection PyProtectedMember
        packing = parser._result
        if not isinstance(packing, Packing):
            raise type_error(packing, f"packing from {file!r}", Packing)
        # noinspection PyProtectedMember
        used_objective = parser._used_objective
        if not isinstance(used_objective, str):
            raise type_error(
                used_objective, f"used_objective from {file!r}", str)
        return PackingResult.from_packing_and_end_result(
            end_result=end_results[0], packing=packing,
            used_objective=used_objective, objectives=objectives,
            bin_bounds=bin_bounds, cache=cache)

    @staticmethod
    def from_logs(  # pylint: disable=W0102
            directory: str,
            collector: Callable[["PackingResult"], None],
            objectives: Iterable[Callable[[Instance], Objective]] =
            _DEFAULT_OBJECTIVES,
            bin_bounds: Mapping[str, Callable[[Instance], int]]
            = _DEFAULT_BOUNDS) -> None:
        """
        Parse a directory recursively to get all packing results.

        :param directory: the directory to parse
        :param collector: the collector for receiving the results
        :param objectives: the objective function factories
        :param bin_bounds: the bin bounds calculators
        """
        _LogParser(collector, objectives, bin_bounds).parse_dir(directory)

    @staticmethod
    def to_csv(results: Iterable["PackingResult"], file: str) -> Path:
        """
        Write a sequence of packing results to a file in CSV format.

        :param results: the end results
        :param file: the path
        :return: the path of the file that was written
        """
        path: Final[Path] = Path.path(file)
        logger(f"Writing packing results to CSV file {path!r}.")
        Path.path(os.path.dirname(path)).ensure_dir_exists()

        # get a nicely sorted view on the results
        use_results = sorted(
            results, key=lambda ppr: (
                ppr.end_result.algorithm,
                ppr.end_result.instance,
                ppr.used_objective,
                ppr.end_result.rand_seed,
                ppr.end_result))

        # get the names of the bounds and objectives
        bin_bounds_set: set[str] = set()
        objectives_set: set[str] = set()
        for pr in use_results:
            bin_bounds_set.update(pr.bin_bounds.keys())
            objectives_set.update(pr.objectives.keys())
        bin_bounds: Final[list[str]] = sorted(bin_bounds_set)
        objectives: Final[list[str]] = sorted(objectives_set)

        line: list[str] = [_HEADER_1]
        line.extend(bin_bounds)
        for ob in objectives:
            line.append(f"{ob}{_OBJECTIVE_LOWER}{CSV_SEPARATOR}"
                        f"{ob}{_OBJECTIVE_UPPER}")
        line.append(_HEADER_2)
        line.extend(objectives)

        with path.open_for_write() as out:
            out.write(CSV_SEPARATOR.join(line))
            line.clear()
            out.write("\n")
            for pr in use_results:
                e: EndResult = pr.end_result
                line.append(
                    f"{e.algorithm}{CSV_SEPARATOR}"
                    f"{e.instance}{CSV_SEPARATOR}"
                    f"{pr.used_objective}{CSV_SEPARATOR}"
                    f"{pr.n_items}{CSV_SEPARATOR}"
                    f"{pr.n_different_items}{CSV_SEPARATOR}"
                    f"{pr.bin_width}{CSV_SEPARATOR}"
                    f"{pr.bin_height}",
                )
                for bb in bin_bounds:
                    line.append(
                        str(pr.bin_bounds[bb]) if bb in pr.bin_bounds else "")
                for ob in objectives:
                    ox = f"{ob}{_OBJECTIVE_LOWER}"
                    line.append(num_to_str(pr.objective_bounds[ox])
                                if ox in pr.objective_bounds else "")
                    ox = f"{ob}{_OBJECTIVE_UPPER}"
                    line.append(num_to_str(pr.objective_bounds[ox])
                                if ox in pr.objective_bounds else "")
                line.append(
                    f"{hex(e.rand_seed)}{CSV_SEPARATOR}"
                    f"{num_to_str(e.best_f)}{CSV_SEPARATOR}"
                    f"{e.last_improvement_fe}{CSV_SEPARATOR}"
                    f"{e.last_improvement_time_millis}{CSV_SEPARATOR}"
                    f"{e.total_fes}{CSV_SEPARATOR}"
                    f"{e.total_time_millis}{CSV_SEPARATOR}"
                    f"{intfloatnone_to_str(e.goal_f)}{CSV_SEPARATOR}"
                    f"{intnone_to_str(e.max_fes)}{CSV_SEPARATOR}"
                    f"{intnone_to_str(e.max_time_millis)}")
                for ob in objectives:
                    line.append(num_to_str(pr.objectives[ob])
                                if ob in pr.objectives else "")
                out.write(CSV_SEPARATOR.join(line))
                line.clear()
                out.write("\n")

        logger(f"Done writing end results to CSV file {path!r}.")
        return path

    @staticmethod
    def from_csv(file: str,
                 collector: Callable[["PackingResult"], None]) -> None:
        """
        Load the packing results from a CSV file.

        :param file: the file to read from
        :param collector: the collector for the results
        """
        path = Path.file(file)
        if not callable(collector):
            raise type_error(collector, "collector", call=True)

        with path.open_for_read() as stream:
            header = stream.readline().strip()
            if not header.startswith(_HEADER_1):
                raise ValueError(f"expect header to start with {_HEADER_1!r}"
                                 f", but got {header!r}")
            idx = header.find(_HEADER_2)
            if idx <= 0:
                raise ValueError(f"expect header to contain {_HEADER_2!r}"
                                 f", but got {header!r}")
            bounds_str = header[len(_HEADER_1) + 1:idx - 1]
            if len(bounds_str) <= 0:
                raise ValueError(f"no bounds in header {header!r}.")
            bounds_strs = bounds_str.split(CSV_SEPARATOR)
            bounds: list[str] = []
            objectives: list[str] = []
            in_bin_bounds: bool = True
            is_lower: bool = True
            for ss in bounds_strs:
                if in_bin_bounds and ss.startswith(_BINS_START):
                    bounds.append(ss)
                else:
                    in_bin_bounds = False
                    end = _OBJECTIVE_LOWER if is_lower else _OBJECTIVE_UPPER
                    if not ss.endswith(end):
                        raise ValueError(
                            f"expected {ss!r} to end with {end!r}.")
                    if is_lower:
                        objectives.append(ss[:-len(end)])
                    is_lower = not is_lower
            end = CSV_SEPARATOR.join(objectives)
            if not header.endswith(end):
                raise ValueError(f"expected header to end with {end!r} but"
                                 f" got {header!r}.")

            for line in stream:
                data = line.strip().split(CSV_SEPARATOR)
                index: int = 0
                algorithm = data[index]
                index += 1
                instance = data[index]
                index += 1
                used_objective = data[index]
                index += 1
                n_items = int(data[index])
                index += 1
                n_different_items = int(data[index])
                index += 1
                bin_width = int(data[index])
                index += 1
                bin_height = int(data[index])
                index += 1
                the_bin_bounds: dict[str, int] = {}
                for bb in bounds:
                    the_bin_bounds[bb] = int(data[index])
                    index += 1
                the_objective_bounds: dict[str, int | float] = {}
                for oo in objectives:
                    the_objective_bounds[f"{oo}{_OBJECTIVE_LOWER}"] \
                        = str_to_intfloat(data[index])
                    index += 1
                    the_objective_bounds[f"{oo}{_OBJECTIVE_UPPER}"] \
                        = str_to_intfloat(data[index])
                    index += 1
                rand_seed = int((data[index])[2:], 16)  # rand seed
                index += 1
                best_f = str_to_intfloat(data[index])  # best_f
                index += 1
                last_improvement_fe = int(data[index])  # last_improvement_fe
                index += 1
                litm = int(data[index])
                index += 1
                total_fes = int(data[index])
                index += 1
                total_time_millis = int(data[index])  # total_time_millis
                index += 1
                goal_f = str_to_intfloatnone(data[index])  # goal_f
                index += 1
                max_fes = str_to_intnone(data[index])  # max_fes
                index += 1
                max_time_millis = str_to_intnone(data[index])
                index += 1
                the_objectives: dict[str, int | float] = {}
                for oo in objectives:
                    the_objectives[oo] = str_to_intfloat(data[index])
                    index += 1
                collector(PackingResult(
                    n_items=n_items,
                    n_different_items=n_different_items,
                    bin_width=bin_width,
                    bin_height=bin_height,
                    end_result=EndResult(
                        algorithm=algorithm,
                        instance=instance,
                        rand_seed=rand_seed,
                        best_f=best_f,
                        last_improvement_fe=last_improvement_fe,
                        last_improvement_time_millis=litm,
                        total_fes=total_fes,
                        total_time_millis=total_time_millis,
                        goal_f=goal_f,
                        max_fes=max_fes,
                        max_time_millis=max_time_millis),
                    objectives=the_objectives,
                    used_objective=used_objective,
                    objective_bounds=the_objective_bounds,
                    bin_bounds=the_bin_bounds))


class _LogParser(LogParser):
    """The internal log parser class."""

    def __init__(self, collector: Callable[["PackingResult"], None],
                 objectives: Iterable[Callable[[Instance], Objective]],
                 bin_bounds: Mapping[str, Callable[[Instance], int]]) -> None:
        """
        Parse a directory recursively to get all packing results.

        :param collector: the collector for receiving the results
        :param objectives: the objective function factories
        :param bin_bounds: the bin bounds calculators
        """
        super().__init__()
        if not callable(collector):
            raise type_error(collector, "collector", call=True)
        if not isinstance(objectives, Iterable):
            raise type_error(objectives, "objectives", Iterable)
        if not isinstance(bin_bounds, Mapping):
            raise type_error(bin_bounds, "bin_bounds", Mapping)
        #: the internal collector
        self.__collector: Final[Callable[[PackingResult], None]] = collector
        #: the objectives holder
        self.__objectives: Final[
            Iterable[Callable[[Instance], Objective]]] = objectives
        #: the bin bounds
        self.__bin_bounds: Final[
            Mapping[str, Callable[[Instance], int]]] = bin_bounds
        #: the internal cache
        self.__cache: Final[Mapping[
            str, tuple[Mapping[str, int], tuple[
                Objective, ...], Mapping[str, int | float]]]] = {}

    def parse_file(self, path: str) -> bool:
        """
        Parse a log file.

        :param path: the path to the log file
        :return: `True`
        """
        self.__collector(PackingResult.from_single_log(
            path, self.__objectives, self.__bin_bounds, self.__cache))
        return True
