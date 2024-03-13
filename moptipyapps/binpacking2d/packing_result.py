"""
An extended end result record to represent packings.

This class extends the information provided by
:mod:`~moptipy.evaluation.end_results`. It allows us to compare the results
of experiments over different objective functions. It also represents the
bounds for the number of bins and for the objective functions. It also
includes the problem-specific information of two-dimensional bin packing
instances.
"""
import argparse
import os.path
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Final, Iterable, Mapping, cast

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
from moptipy.evaluation.base import (
    KEY_ENCODING,
    KEY_OBJECTIVE_FUNCTION,
    EvaluationDataElement,
)
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.log_parser import LogParser
from moptipy.utils.logger import CSV_SEPARATOR
from pycommons.ds.immutable_map import immutable_mapping
from pycommons.io.console import logger
from pycommons.io.path import Path, file_path
from pycommons.strings.string_conv import (
    int_or_none_to_str,
    num_or_none_to_str,
    num_to_str,
    str_to_int_or_none,
    str_to_num,
    str_to_num_or_none,
)
from pycommons.types import check_int_range, type_error

from moptipyapps.binpacking2d.instance import Instance, _lower_bound_damv
from moptipyapps.binpacking2d.objectives.bin_count import (
    BIN_COUNT_NAME,
    BinCount,
)
from moptipyapps.binpacking2d.objectives.bin_count_and_empty import (
    BinCountAndEmpty,
)
from moptipyapps.binpacking2d.objectives.bin_count_and_last_empty import (
    BinCountAndLastEmpty,
)
from moptipyapps.binpacking2d.objectives.bin_count_and_last_skyline import (
    BinCountAndLastSkyline,
)
from moptipyapps.binpacking2d.objectives.bin_count_and_last_small import (
    BinCountAndLastSmall,
)
from moptipyapps.binpacking2d.objectives.bin_count_and_lowest_skyline import (
    BinCountAndLowestSkyline,
)
from moptipyapps.binpacking2d.objectives.bin_count_and_small import (
    BinCountAndSmall,
)
from moptipyapps.binpacking2d.packing import Packing
from moptipyapps.shared import moptipyapps_argparser

#: the number of items
KEY_N_ITEMS: Final[str] = "nItems"
#: the number of different items
KEY_N_DIFFERENT_ITEMS: Final[str] = "nDifferentItems"
#: the bin width
KEY_BIN_WIDTH: Final[str] = "binWidth"
#: the bin height
KEY_BIN_HEIGHT: Final[str] = "binHeight"

#: The internal CSV header part 1, after which the encoding may come
_HEADER_1: Final[str] = (f"{KEY_ALGORITHM}{CSV_SEPARATOR}"
                         f"{KEY_INSTANCE}{CSV_SEPARATOR}"
                         f"{KEY_OBJECTIVE_FUNCTION}")
#: The internal CSV header part 2, after which the bounds are inserted
_HEADER_2: Final[str] = (f"{KEY_N_ITEMS}{CSV_SEPARATOR}"
                         f"{KEY_N_DIFFERENT_ITEMS}{CSV_SEPARATOR}"
                         f"{KEY_BIN_WIDTH}{CSV_SEPARATOR}"
                         f"{KEY_BIN_HEIGHT}")
#: The internal CSV header part 3, after which the objective values are listed
_HEADER_3: Final[str] = (f"{KEY_RAND_SEED}{CSV_SEPARATOR}"
                         f"{KEY_BEST_F}{CSV_SEPARATOR}"
                         f"{KEY_LAST_IMPROVEMENT_FE}{CSV_SEPARATOR}"
                         f"{KEY_LAST_IMPROVEMENT_TIME_MILLIS}"
                         f"{CSV_SEPARATOR}"
                         f"{KEY_TOTAL_FES}{CSV_SEPARATOR}"
                         f"{KEY_TOTAL_TIME_MILLIS}")


#: the default objective functions
DEFAULT_OBJECTIVES: Final[tuple[Callable[[Instance], Objective], ...]] = (
    BinCount, BinCountAndLastEmpty, BinCountAndLastSmall,
    BinCountAndLastSkyline, BinCountAndEmpty, BinCountAndSmall,
    BinCountAndLowestSkyline,
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
LOWER_BOUNDS_BIN_COUNT: Final[str] = f"bins{_OBJECTIVE_LOWER}"
#: the default bounds
_DEFAULT_BOUNDS: Final[Mapping[str, Callable[[Instance], int]]] = \
    immutable_mapping({
        LOWER_BOUNDS_BIN_COUNT: lambda i: i.lower_bound_bins,
        f"{LOWER_BOUNDS_BIN_COUNT}.geometric": __lb_geometric,
        f"{LOWER_BOUNDS_BIN_COUNT}.damv": lambda i: _lower_bound_damv(
            i.bin_width, i.bin_height, i),
    })


@dataclass(frozen=True, init=False, order=False, eq=False)
class PackingResult(EvaluationDataElement):
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
    #: the bounds for the objective values (append ".lowerBound" and
    #: ".upperBound" to all objective function names)
    objective_bounds: Mapping[str, int]
    #: the bounds for the minimum number of bins of the instance
    bin_bounds: Mapping[str, int]

    def __init__(self,
                 end_result: EndResult,
                 n_items: int,
                 n_different_items: int,
                 bin_width: int,
                 bin_height: int,
                 objectives: Mapping[str, int | float],
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
        :param bin_bounds: the different bounds for the number of bins
        :param objective_bounds: the bounds for the objective functions
        :raises TypeError: if any parameter has a wrong type
        :raises ValueError: if the parameter values are inconsistent
        """
        super().__init__()
        if not isinstance(end_result, EndResult):
            raise type_error(end_result, "end_result", EndResult)
        if end_result.best_f != objectives[end_result.objective]:
            raise ValueError(
                f"end_result.best_f={end_result.best_f}, but objectives["
                f"{end_result.objective!r}]="
                f"{objectives[end_result.objective]}.")
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

    def _tuple(self) -> tuple[Any, ...]:
        """
        Create a tuple with all the data of this data class for comparison.

        :returns: a tuple with all the data of this class, where `None` values
            are masked out
        """
        # noinspection PyProtectedMember
        return self.end_result._tuple()

    @staticmethod
    def from_packing_and_end_result(  # pylint: disable=W0102
            end_result: EndResult, packing: Packing,
            objectives: Iterable[Callable[[Instance], Objective]] =
            DEFAULT_OBJECTIVES,
            bin_bounds: Mapping[str, Callable[[Instance], int]] =
            _DEFAULT_BOUNDS,
            cache: Mapping[str, tuple[Mapping[str, int], tuple[
                Objective, ...], Mapping[str, int | float]]] | None =
            None) -> "PackingResult":
        """
        Create a `PackingResult` from an `EndResult` and a `Packing`.

        :param end_result: the end results record
        :param packing: the packing
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

        objective_values: dict[str, int | float] = {}
        bin_count: int = -1
        bin_count_obj: str = ""
        for objf in row[1]:
            z: int | float = objf.evaluate(packing)
            objfn: str = str(objf)
            objective_values[objfn] = z
            if not isinstance(z, int):
                continue
            if not isinstance(objf, BinCount):
                continue
            bc: int = objf.to_bin_count(z)
            if bin_count == -1:
                bin_count = bc
                bin_count_obj = objfn
            elif bin_count != bc:
                raise ValueError(
                    f"found bin count disagreement: {bin_count} of "
                    f"{bin_count_obj!r} != {bc} of {objf!r}")

        return PackingResult(
            end_result=end_result,
            n_items=instance.n_items,
            n_different_items=instance.n_different_items,
            bin_width=instance.bin_width, bin_height=instance.bin_height,
            objectives=objective_values,
            objective_bounds=row[2],
            bin_bounds=row[0])

    @staticmethod
    def from_single_log(  # pylint: disable=W0102
            file: str,
            objectives: Iterable[Callable[[Instance], Objective]] =
            DEFAULT_OBJECTIVES,
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
        the_file_path = file_path(file)
        end_results: Final[list[EndResult]] = []
        EndResult.from_logs(the_file_path, end_results.append)
        if len(end_results) != 1:
            raise ValueError(
                f"needs one end result record in file {the_file_path!r}, "
                f"but got {end_results}.")

        packing = Packing.from_log(the_file_path)
        if not isinstance(packing, Packing):
            raise type_error(packing, f"packing from {file!r}", Packing)
        return PackingResult.from_packing_and_end_result(
            end_result=end_results[0], packing=packing,
            objectives=objectives, bin_bounds=bin_bounds, cache=cache)

    @staticmethod
    def from_logs(  # pylint: disable=W0102
            directory: str,
            collector: Callable[["PackingResult"], None],
            objectives: Iterable[Callable[[Instance], Objective]] =
            DEFAULT_OBJECTIVES,
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
        path: Final[Path] = Path(file)
        logger(f"Writing packing results to CSV file {path!r}.")
        Path(os.path.dirname(path)).ensure_dir_exists()

        # get a nicely sorted view on the results
        use_results = sorted(results)

        # get the names of the bounds and objectives
        bin_bounds_set: set[str] = set()
        objectives_set: set[str] = set()
        needs_encoding: bool = False
        needs_max_fes: bool = False
        needs_max_ms: bool = False
        needs_goal_f: bool = False
        for pr in use_results:
            bin_bounds_set.update(pr.bin_bounds.keys())
            objectives_set.update(pr.objectives.keys())
            er = pr.end_result
            if er.encoding is not None:
                needs_encoding = True
            if er.max_fes is not None:
                needs_max_fes = True
            if er.max_time_millis is not None:
                needs_max_ms = True
            if (er.goal_f is not None) and (isfinite(er.goal_f)):
                needs_goal_f = True
        bin_bounds: Final[list[str]] = sorted(bin_bounds_set)
        objectives: Final[list[str]] = sorted(objectives_set)

        line: list[str] = [_HEADER_1]
        if needs_encoding:
            line.append(KEY_ENCODING)
        line.append(_HEADER_2)
        line.extend(bin_bounds)
        for ob in objectives:
            line.append(f"{ob}{_OBJECTIVE_LOWER}{CSV_SEPARATOR}"
                        f"{ob}{_OBJECTIVE_UPPER}")
        line.append(_HEADER_3)
        if needs_goal_f:
            line.append(KEY_GOAL_F)
        if needs_max_fes:
            line.append(KEY_MAX_FES)
        if needs_max_ms:
            line.append(KEY_MAX_TIME_MILLIS)
        line.extend(objectives)

        with path.open_for_write() as out:
            out.write(CSV_SEPARATOR.join(line))
            line.clear()
            out.write("\n")
            for pr in use_results:
                e: EndResult = pr.end_result
                line.append(f"{e.algorithm}{CSV_SEPARATOR}"
                            f"{e.instance}{CSV_SEPARATOR}{e.objective}")
                if needs_encoding:
                    line.append("" if e.encoding is None else e.encoding)
                line.append(f"{pr.n_items}{CSV_SEPARATOR}"
                            f"{pr.n_different_items}{CSV_SEPARATOR}"
                            f"{pr.bin_width}{CSV_SEPARATOR}"
                            f"{pr.bin_height}")
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
                    f"{e.total_time_millis}")
                if needs_goal_f:
                    line.append(num_or_none_to_str(e.goal_f))
                if needs_max_fes:
                    line.append(int_or_none_to_str(e.max_fes))
                if needs_max_ms:
                    line.append(int_or_none_to_str(e.max_time_millis))
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
        path = file_path(file)
        if not callable(collector):
            raise type_error(collector, "collector", call=True)

        with (path.open_for_read() as stream):
            header = stream.readline()
            if not isinstance(header, str):
                raise type_error(header, f"{file!r}[0]", str)

            idx_algorithm: int = -1
            idx_instance: int = -1
            idx_objective: int = -1
            idx_encoding: int = -1
            idx_seed: int = -1
            idx_li_fe: int = -1
            idx_li_ms: int = -1
            idx_best_f: int = -1
            idx_tt_fe: int = -1
            idx_tt_ms: int = -1
            idx_goal_f: int = -1
            idx_max_fes: int = -1
            idx_max_ms: int = -1
            idx_n_items: int = -1
            idx_n_different: int = -1
            idx_bin_width: int = -1
            idx_bin_height: int = -1
            idx_bounds: dict[str, int] = {}
            idx_objective_bounds: dict[str, int] = {}
            idx_objectives: dict[str, int] = {}

            for i, cellstr in enumerate(header.strip().split(CSV_SEPARATOR)):
                cell = cellstr.strip()
                if cell == KEY_N_ITEMS:
                    idx_n_items = i
                elif cell == KEY_N_DIFFERENT_ITEMS:
                    idx_n_different = i
                elif cell == KEY_BIN_WIDTH:
                    idx_bin_width = i
                elif cell == KEY_BIN_HEIGHT:
                    idx_bin_height = i
                elif cell == KEY_ALGORITHM:
                    idx_algorithm = i
                elif cell == KEY_INSTANCE:
                    idx_instance = i
                elif cell == KEY_OBJECTIVE_FUNCTION:
                    idx_objective = i
                elif cell == KEY_ENCODING:
                    idx_encoding = i
                elif cell == KEY_RAND_SEED:
                    idx_seed = i
                elif cell == KEY_LAST_IMPROVEMENT_FE:
                    idx_li_fe = i
                elif cell == KEY_LAST_IMPROVEMENT_TIME_MILLIS:
                    idx_li_ms = i
                elif cell == KEY_BEST_F:
                    idx_best_f = i
                elif cell == KEY_TOTAL_FES:
                    idx_tt_fe = i
                elif cell == KEY_TOTAL_TIME_MILLIS:
                    idx_tt_ms = i
                elif cell == KEY_GOAL_F:
                    idx_goal_f = i
                elif cell == KEY_MAX_FES:
                    idx_max_fes = i
                elif cell == KEY_MAX_TIME_MILLIS:
                    idx_max_ms = i
                elif cell.startswith(LOWER_BOUNDS_BIN_COUNT):
                    idx_bounds[cell] = i
                elif cell.endswith((_OBJECTIVE_LOWER, _OBJECTIVE_UPPER)):
                    idx_objective_bounds[cell] = i
                else:
                    idx_objectives[cell] = i

            if idx_n_items < 0:
                raise ValueError(
                    f"Missing key {KEY_N_ITEMS!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_n_different < 0:
                raise ValueError(
                    f"Missing key {KEY_N_DIFFERENT_ITEMS!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_bin_width < 0:
                raise ValueError(
                    f"Missing key {KEY_BIN_WIDTH!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_bin_height < 0:
                raise ValueError(
                    f"Missing key {KEY_BIN_HEIGHT!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_algorithm < 0:
                raise ValueError(
                    f"Missing key {KEY_ALGORITHM!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_instance < 0:
                raise ValueError(
                    f"Missing key {KEY_INSTANCE!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_objective < 0:
                raise ValueError(
                    f"Missing key {KEY_OBJECTIVE_FUNCTION!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_seed < 0:
                raise ValueError(
                    f"Missing key {KEY_RAND_SEED!r} in "
                    f"header {header!r} of file {file!r}.")
            if idx_li_fe < 0:
                raise ValueError(
                    f"Missing key {KEY_LAST_IMPROVEMENT_FE!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_li_ms < 0:
                raise ValueError(
                    f"Missing key {KEY_LAST_IMPROVEMENT_TIME_MILLIS!r} in "
                    f"header {header!r} of file {file!r}.")
            if idx_best_f < 0:
                raise ValueError(
                    f"Missing key {KEY_BEST_F!r} in header "
                    f"{header!r} of file {file!r}.")
            if idx_tt_fe < 0:
                raise ValueError(
                    f"Missing key {KEY_TOTAL_FES!r} in "
                    f"header {header!r} of file {file!r}.")
            if idx_tt_ms < 0:
                raise ValueError(
                    f"Missing key {KEY_TOTAL_TIME_MILLIS!r} in "
                    f"header {header!r} of file {file!r}.")

            objectives: list[str] = sorted(idx_objectives.keys())
            if len(objectives) <= 0:
                raise ValueError(f"no objectives found in file {file!r}.")
            for kk in objectives:
                if f"{kk}{_OBJECTIVE_LOWER}" not in idx_objective_bounds:
                    raise ValueError(
                        f"objective {kk!r} has no lower bound in header "
                        f"{header} of file {file!r}.")
                if f"{kk}{_OBJECTIVE_UPPER}" not in idx_objective_bounds:
                    raise ValueError(
                        f"objective {kk!r} has no upper bound in header "
                        f"{header} of file {file!r}.")
            if len(idx_objective_bounds) != (2 * len(objectives)):
                raise ValueError(
                    f"inconsistent bounds {idx_objective_bounds.keys()} in "
                    f"header {header} of file {file!r}.")

            bounds: list[str] = sorted(idx_bounds.keys())
            if len(bounds) <= 0:
                raise ValueError(f"no bounds found in file {file!r}.")

            for line in stream:
                splt = line.strip().split(CSV_SEPARATOR)
                encoding: str | None
                if idx_encoding < 0:
                    encoding = None
                else:
                    encoding = splt[idx_encoding].strip()
                    if len(encoding) <= 0:
                        encoding = None

                er = EndResult(
                    splt[idx_algorithm].strip(),  # algorithm
                    splt[idx_instance].strip(),  # instance
                    splt[idx_objective].strip(),  # objective
                    encoding,  # encoding
                    int((splt[idx_seed])[2:], 16),  # rand seed
                    str_to_num(splt[idx_best_f]),  # best_f
                    int(splt[idx_li_fe]),  # last_improvement_fe
                    int(splt[idx_li_ms]),  # last_improvement_time_millis
                    int(splt[idx_tt_fe]),  # total_fes
                    int(splt[idx_tt_ms]),  # total_time_millis
                    None if idx_goal_f < 0 else
                    str_to_num_or_none(splt[idx_goal_f]),  # goal_f
                    None if idx_max_fes < 0 else
                    str_to_int_or_none(splt[idx_max_fes]),  # max_fes
                    None if idx_max_ms < 0 else
                    str_to_int_or_none(splt[idx_max_ms]))  # max_time_millis

                the_objectives: dict[str, int | float] = {
                    on: str_to_num(splt[idx_objectives[on]])
                    for on in objectives}
                the_bin_bounds: dict[str, int] = {
                    on: int(splt[idx_bounds[on]]) for on in bounds}
                the_objective_bounds: dict[str, int | float] = {
                    on: str_to_num(splt[idxx])
                    for on, idxx in idx_objective_bounds.items()}

                collector(PackingResult(
                    n_items=int(splt[idx_n_items]),
                    n_different_items=int(splt[idx_n_different]),
                    bin_width=int(splt[idx_bin_width]),
                    bin_height=int(splt[idx_bin_height]),
                    end_result=er,
                    objectives=the_objectives,
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


# Evaluate an experiment from the command line

# Run log files to end results if executed as script
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__,
        "Convert log files for the bin packing experiment to a CSV file.",
        "Re-evaluate all results based on different objective functions.")
    parser.add_argument(
        "source", nargs="?", default="./results",
        help="the location of the experimental results, i.e., the root folder "
             "under which to search for log files", type=Path)
    parser.add_argument(
        "dest", help="the path to the end results CSV file to be created",
        type=Path, nargs="?", default="./evaluation/end_results.txt")
    args: Final[argparse.Namespace] = parser.parse_args()

    packing_results: Final[list[PackingResult]] = []
    PackingResult.from_logs(args.source, packing_results.append)
    PackingResult.to_csv(packing_results, args.dest)
