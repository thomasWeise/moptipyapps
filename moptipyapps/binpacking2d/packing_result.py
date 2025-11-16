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
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Final, Generator, Iterable, Mapping, cast

from moptipy.api.objective import Objective
from moptipy.evaluation.base import (
    EvaluationDataElement,
)
from moptipy.evaluation.end_results import CsvReader as ErCsvReader
from moptipy.evaluation.end_results import CsvWriter as ErCsvWriter
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_results import from_logs as er_from_logs
from moptipy.evaluation.log_parser import LogParser
from pycommons.ds.immutable_map import immutable_mapping
from pycommons.ds.sequences import reiterable
from pycommons.io.console import logger
from pycommons.io.csv import (
    SCOPE_SEPARATOR,
    csv_column,
    csv_scope,
    csv_select_scope,
)
from pycommons.io.csv import CsvReader as CsvReaderBase
from pycommons.io.csv import CsvWriter as CsvWriterBase
from pycommons.io.path import Path, file_path, line_writer
from pycommons.strings.string_conv import (
    num_to_str,
    str_to_num,
)
from pycommons.types import check_int_range, type_error

from moptipyapps.binpacking2d.instance import (
    Instance,
    _lower_bound_damv,
)
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
from moptipyapps.utils.shared import (
    moptipyapps_argparser,
    motipyapps_footer_bottom_comments,
)

#: the number of items
KEY_N_ITEMS: Final[str] = "nItems"
#: the number of different items
KEY_N_DIFFERENT_ITEMS: Final[str] = "nDifferentItems"
#: the bin width
KEY_BIN_WIDTH: Final[str] = "binWidth"
#: the bin height
KEY_BIN_HEIGHT: Final[str] = "binHeight"


#: the default objective functions
DEFAULT_OBJECTIVES: Final[tuple[Callable[[Instance], Objective], ...]] = (
    BinCount, BinCountAndLastEmpty, BinCountAndLastSmall,
    BinCountAndLastSkyline, BinCountAndEmpty, BinCountAndSmall,
    BinCountAndLowestSkyline)


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
_OBJECTIVE_LOWER: Final[str] = "lowerBound"
#: the upper bound of an objective
_OBJECTIVE_UPPER: Final[str] = "upperBound"
#: the start string for bin bounds
LOWER_BOUNDS_BIN_COUNT: Final[str] = csv_scope("bins", _OBJECTIVE_LOWER)
#: the default bounds
_DEFAULT_BOUNDS: Final[Mapping[str, Callable[[Instance], int]]] = \
    immutable_mapping({
        LOWER_BOUNDS_BIN_COUNT: lambda i: i.lower_bound_bins,
        csv_scope(LOWER_BOUNDS_BIN_COUNT, "geometric"): __lb_geometric,
        csv_scope(LOWER_BOUNDS_BIN_COUNT, "damv"): lambda i: _lower_bound_damv(
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
    #: the bin height
    bin_height: int
    #: the objective values evaluated after the optimization
    objectives: Mapping[str, int | float]
    #: the bounds for the objective values (append ".lowerBound" and
    #: ".upperBound" to all objective function names)
    objective_bounds: Mapping[str, int | float]
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
            lll: str = csv_scope(name, _OBJECTIVE_LOWER)
            lower = objective_bounds[lll]
            if not isfinite(lower):
                raise ValueError(f"{lll}=={lower}.")
            uuu = csv_scope(name, _OBJECTIVE_UPPER)
            upper = objective_bounds[uuu]
            if not (lower <= value <= upper):
                raise ValueError(
                    f"it is required that {lll}<=f<={uuu}, but got "
                    f"{lower}, {value}, and {upper}.")

        bins: Final[int | None] = cast("int", objectives[BIN_COUNT_NAME]) \
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


def from_packing_and_end_result(  # pylint: disable=W0102
        end_result: EndResult, packing: Packing,
        objectives: Iterable[Callable[[Instance], Objective]] =
        DEFAULT_OBJECTIVES,
        bin_bounds: Mapping[str, Callable[[Instance], int]] =
        _DEFAULT_BOUNDS,
        cache: Mapping[str, tuple[Mapping[str, int], tuple[
            Objective, ...], Mapping[str, int | float]]] | None =
        None) -> PackingResult:
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
            obounds[csv_scope(str(objf), _OBJECTIVE_LOWER)] = \
                objf.lower_bound()
            obounds[csv_scope(str(objf), _OBJECTIVE_UPPER)] = \
                objf.upper_bound()
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


def from_single_log(  # pylint: disable=W0102
        file: str,
        objectives: Iterable[Callable[[Instance], Objective]] =
        DEFAULT_OBJECTIVES,
        bin_bounds: Mapping[str, Callable[[Instance], int]] =
        _DEFAULT_BOUNDS,
        cache: Mapping[str, tuple[Mapping[str, int], tuple[
            Objective, ...], Mapping[str, int | float]]] | None =
        None) -> PackingResult:
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
    end_result: Final[EndResult] = next(er_from_logs(the_file_path))
    packing = Packing.from_log(the_file_path)
    if not isinstance(packing, Packing):
        raise type_error(packing, f"packing from {file!r}", Packing)
    return from_packing_and_end_result(
        end_result=end_result, packing=packing,
        objectives=objectives, bin_bounds=bin_bounds, cache=cache)


def from_logs(  # pylint: disable=W0102
        directory: str,
        objectives: Iterable[Callable[[Instance], Objective]] =
        DEFAULT_OBJECTIVES,
        bin_bounds: Mapping[str, Callable[[Instance], int]]
        = _DEFAULT_BOUNDS) -> Generator[PackingResult, None, None]:
    """
    Parse a directory recursively to get all packing results.

    :param directory: the directory to parse
    :param objectives: the objective function factories
    :param bin_bounds: the bin bounds calculators
    """
    return __LogParser(objectives, bin_bounds).parse(directory)


def to_csv(results: Iterable[PackingResult], file: str) -> Path:
    """
    Write a sequence of packing results to a file in CSV format.

    :param results: the end results
    :param file: the path
    :return: the path of the file that was written
    """
    path: Final[Path] = Path(file)
    logger(f"Writing packing results to CSV file {path!r}.")
    path.ensure_parent_dir_exists()
    with path.open_for_write() as wt:
        consumer: Final[Callable[[str], None]] = line_writer(wt)
        for p in CsvWriter.write(sorted(results)):
            consumer(p)
    logger(f"Done writing packing results to CSV file {path!r}.")
    return path


def from_csv(file: str) -> Iterable[PackingResult]:
    """
    Load the packing results from a CSV file.

    :param file: the file to read from
    :returns: the iterable with the packing result
    """
    path: Final[Path] = file_path(file)
    logger(f"Now reading CSV file {path!r}.")
    with path.open_for_read() as rd:
        yield from CsvReader.read(rd)
    logger(f"Done reading CSV file {path!r}.")


class CsvWriter(CsvWriterBase[PackingResult]):
    """A class for CSV writing of :class:`PackingResult`."""

    def __init__(self, data: Iterable[PackingResult],
                 scope: str | None = None) -> None:
        """
        Initialize the csv writer.

        :param data: the data to write
        :param scope: the prefix to be pre-pended to all columns
        """
        data = reiterable(data)
        super().__init__(data, scope)
        #: the end result writer
        self.__er: Final[ErCsvWriter] = ErCsvWriter((
            pr.end_result for pr in data), scope)

        bin_bounds_set: Final[set[str]] = set()
        objectives_set: Final[set[str]] = set()
        for pr in data:
            bin_bounds_set.update(pr.bin_bounds.keys())
            objectives_set.update(pr.objectives.keys())
        #: the bin bounds
        self.__bin_bounds: Final[list[str] | None] = None \
            if set.__len__(bin_bounds_set) <= 0 else sorted(bin_bounds_set)
        #: the objectives
        self.__objectives: Final[list[str] | None] = None \
            if set.__len__(objectives_set) <= 0 else sorted(objectives_set)

    def get_column_titles(self) -> Iterable[str]:
        """
        Get the column titles.

        :returns: the column titles
        """
        p: Final[str] = self.scope
        yield from self.__er.get_column_titles()

        yield csv_scope(p, KEY_BIN_HEIGHT)
        yield csv_scope(p, KEY_BIN_WIDTH)
        yield csv_scope(p, KEY_N_ITEMS)
        yield csv_scope(p, KEY_N_DIFFERENT_ITEMS)
        if self.__bin_bounds:
            for b in self.__bin_bounds:
                yield csv_scope(p, b)
        if self.__objectives:
            for o in self.__objectives:
                oo: str = csv_scope(p, o)
                yield csv_scope(oo, _OBJECTIVE_LOWER)
                yield oo
                yield csv_scope(oo, _OBJECTIVE_UPPER)

    def get_row(self, data: PackingResult) -> Iterable[str]:
        """
        Render a single packing result record to a CSV row.

        :param data: the end result record
        :returns: the iterable with the row data
        """
        yield from self.__er.get_row(data.end_result)
        yield repr(data.bin_height)
        yield repr(data.bin_width)
        yield repr(data.n_items)
        yield repr(data.n_different_items)
        if self.__bin_bounds:
            for bb in self.__bin_bounds:
                yield (repr(data.bin_bounds[bb])
                       if bb in data.bin_bounds else "")
        if self.__objectives:
            for ob in self.__objectives:
                ox = csv_scope(ob, _OBJECTIVE_LOWER)
                yield (num_to_str(data.objective_bounds[ox])
                       if ox in data.objective_bounds else "")
                yield (num_to_str(data.objectives[ob])
                       if ob in data.objectives else "")
                ox = csv_scope(ob, _OBJECTIVE_UPPER)
                yield (num_to_str(data.objective_bounds[ox])
                       if ox in data.objective_bounds else "")

    def get_header_comments(self) -> Iterable[str]:
        """
        Get any possible header comments.

        :returns: the header comments
        """
        return ("End Results of Bin Packing Experiments",
                "See the description at the bottom of the file.")

    def get_footer_comments(self) -> Iterable[str]:
        """
        Get any possible footer comments.

        :return: the footer comments
        """
        yield from self.__er.get_footer_comments()
        yield ""
        p: Final[str | None] = self.scope
        if self.__bin_bounds:
            for bb in self.__bin_bounds:
                yield (f"{csv_scope(p, bb)} is a lower bound "
                       f"for the number of bins.")
        if self.__objectives:
            for obb in self.__objectives:
                ob: str = csv_scope(p, obb)
                ox: str = csv_scope(ob, _OBJECTIVE_LOWER)
                yield f"{ox}: a lower bound of the {ob} objective function."
                yield (f"{ob}: one of the possible objective functions for "
                       "the two-dimensional bin packing problem.")
                ox = csv_scope(ob, _OBJECTIVE_UPPER)
                yield f"{ox}: an upper bound of the {ob} objective function."

    def get_footer_bottom_comments(self) -> Iterable[str]:
        """Get the bottom footer comments."""
        yield from motipyapps_footer_bottom_comments(
            self, "The packing data is assembled using module "
                  "moptipyapps.binpacking2d.packing_statistics.")
        yield from ErCsvWriter.get_footer_bottom_comments(self.__er)


class CsvReader(CsvReaderBase):
    """A class for CSV reading of :class:`PackingResult` instances."""

    def __init__(self, columns: dict[str, int]) -> None:
        """
        Create a CSV parser for :class:`EndResult`.

        :param columns: the columns
        """
        super().__init__(columns)
        #: the end result csv reader
        self.__er: Final[ErCsvReader] = ErCsvReader(columns)
        #: the index of the n-items column
        self.__idx_n_items: Final[int] = csv_column(columns, KEY_N_ITEMS)
        #: the index of the n different items column
        self.__idx_n_different: Final[int] = csv_column(
            columns, KEY_N_DIFFERENT_ITEMS)
        #: the index of the bin width column
        self.__idx_bin_width: Final[int] = csv_column(columns, KEY_BIN_WIDTH)
        #: the index of the bin height  column
        self.__idx_bin_height: Final[int] = csv_column(
            columns, KEY_BIN_HEIGHT)
        #: the indices for the objective bounds
        self.__bin_bounds: Final[tuple[tuple[str, int], ...]] = \
            csv_select_scope(
                lambda x: tuple(sorted(((k, v) for k, v in x.items()))),
                columns, LOWER_BOUNDS_BIN_COUNT)
        #: the objective bounds
        self.__objective_bounds: Final[tuple[tuple[str, int], ...]] = \
            csv_select_scope(
                lambda x: tuple(sorted(((k, v) for k, v in x.items()))),
                columns, None,
                skip_orig_key=lambda s: not str.endswith(
                    s, (_OBJECTIVE_LOWER, _OBJECTIVE_UPPER)))
        n_bounds: Final[int] = tuple.__len__(self.__objective_bounds)
        if n_bounds <= 0:
            raise ValueError("No objective function bounds found?")
        if (n_bounds & 1) != 0:
            raise ValueError(f"Number of bounds {n_bounds} should be even.")
        #: the parsers for the objective values
        self.__objectives: Final[tuple[tuple[str, int], ...]] = \
            tuple((ss, csv_column(columns, ss))
                  for ss in sorted({s[0] for s in (str.split(
                      kk[0], SCOPE_SEPARATOR) for kk in
                      self.__objective_bounds) if (list.__len__(s) > 1)
                      and (str.__len__(s[0]) > 0)}))
        n_objectives: Final[int] = tuple.__len__(self.__objectives)
        if n_objectives <= 0:
            raise ValueError("No objectives found?")
        if (2 * n_objectives) != n_bounds:
            raise ValueError(
                f"Number {n_objectives} of objectives "
                f"inconsistent with number {n_bounds} of bounds.")

    def parse_row(self, data: list[str]) -> PackingResult:
        """
        Parse a row of data.

        :param data: the data row
        :return: the end result statistics
        """
        return PackingResult(
            self.__er.parse_row(data),
            int(data[self.__idx_n_items]),
            int(data[self.__idx_n_different]),
            int(data[self.__idx_bin_width]),
            int(data[self.__idx_bin_height]),
            {n: str_to_num(data[i]) for n, i in self.__objectives
             if str.__len__(data[i]) > 0},
            {n: str_to_num(data[i]) for n, i in self.__objective_bounds
             if str.__len__(data[i]) > 0},
            {n: int(data[i]) for n, i in self.__bin_bounds
             if str.__len__(data[i]) > 0})


class __LogParser(LogParser[PackingResult]):
    """The internal log parser class."""

    def __init__(self, objectives: Iterable[Callable[[Instance], Objective]],
                 bin_bounds: Mapping[str, Callable[[Instance], int]]) -> None:
        """
        Parse a directory recursively to get all packing results.

        :param objectives: the objective function factories
        :param bin_bounds: the bin bounds calculators
        """
        super().__init__()
        if not isinstance(objectives, Iterable):
            raise type_error(objectives, "objectives", Iterable)
        if not isinstance(bin_bounds, Mapping):
            raise type_error(bin_bounds, "bin_bounds", Mapping)
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

    def _parse_file(self, file: Path) -> PackingResult:
        """
        Parse a log file.

        :param file: the file path
        :return: the parsed result
        """
        return from_single_log(file, self.__objectives, self.__bin_bounds,
                               self.__cache)


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

    to_csv(from_logs(args.source), args.dest)
