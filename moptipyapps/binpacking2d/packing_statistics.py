"""An extended end result statistics record to represent packings."""
import argparse
import os.path
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Final, Generator, Iterable, Mapping, cast

from moptipy.evaluation.base import (
    KEY_N,
    EvaluationDataElement,
)
from moptipy.evaluation.end_results import EndResult
from moptipy.evaluation.end_statistics import CsvReader as EsCsvReader
from moptipy.evaluation.end_statistics import CsvWriter as EsCsvWriter
from moptipy.evaluation.end_statistics import (
    EndStatistics,
)
from moptipy.evaluation.end_statistics import (
    from_end_results as es_from_end_results,
)
from moptipy.utils.strings import (
    num_to_str,
)
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
from pycommons.math.sample_statistics import CsvReader as SsCsvReader
from pycommons.math.sample_statistics import CsvWriter as SsCsvWriter
from pycommons.math.sample_statistics import SampleStatistics
from pycommons.math.sample_statistics import from_samples as ss_from_samples
from pycommons.strings.string_conv import str_to_num
from pycommons.types import check_int_range, type_error

from moptipyapps.binpacking2d.objectives.bin_count import BIN_COUNT_NAME
from moptipyapps.binpacking2d.packing_result import (
    _OBJECTIVE_LOWER,
    _OBJECTIVE_UPPER,
    KEY_BIN_HEIGHT,
    KEY_BIN_WIDTH,
    KEY_N_DIFFERENT_ITEMS,
    KEY_N_ITEMS,
    LOWER_BOUNDS_BIN_COUNT,
    PackingResult,
)
from moptipyapps.binpacking2d.packing_result import from_csv as pr_from_csv
from moptipyapps.binpacking2d.packing_result import from_logs as pr_from_logs
from moptipyapps.utils.shared import (
    moptipyapps_argparser,
    motipyapps_footer_bottom_comments,
)


@dataclass(frozen=True, init=False, order=False, eq=False)
class PackingStatistics(EvaluationDataElement):
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
    #: the bin height
    bin_height: int
    #: the objective values evaluated after the optimization
    objectives: Mapping[str, SampleStatistics]
    #: the bounds for the objective values (append ".lowerBound" and
    #: ".upperBound" to all objective function names)
    objective_bounds: Mapping[str, int | float]
    #: the bounds for the minimum number of bins of the instance
    bin_bounds: Mapping[str, int]

    def __init__(self,
                 end_statistics: EndStatistics,
                 n_items: int,
                 n_different_items: int,
                 bin_width: int,
                 bin_height: int,
                 objectives: Mapping[str, SampleStatistics],
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
        :param bin_bounds: the different bounds for the number of bins
        :param objective_bounds: the bounds for the objective functions
        :raises TypeError: if any parameter has a wrong type
        :raises ValueError: if the parameter values are inconsistent
        """
        super().__init__()
        if not isinstance(end_statistics, EndStatistics):
            raise type_error(end_statistics, "end_statistics", EndResult)
        if end_statistics.best_f != objectives[end_statistics.objective]:
            raise ValueError(
                f"end_statistics.best_f={end_statistics.best_f}, but "
                f"objectives[{end_statistics.objective!r}]="
                f"{objectives[end_statistics.objective]}.")
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

        for name, stat in objectives.items():
            if not isinstance(name, str):
                raise type_error(
                    name, f"name of evaluation[{name!r}]={stat!r}", str)
            if not isinstance(stat, int | float | SampleStatistics):
                raise type_error(
                    stat, f"value of evaluation[{name!r}]={stat!r}",
                    (int, float, SampleStatistics))
            lll: str = csv_scope(name, _OBJECTIVE_LOWER)
            lower = objective_bounds[lll]
            if not isfinite(lower):
                raise ValueError(f"{lll}=={lower}.")
            uuu = csv_scope(name, _OBJECTIVE_UPPER)
            upper = objective_bounds[uuu]
            for value in (stat.minimum, stat.maximum) \
                    if isinstance(stat, SampleStatistics) else (stat, ):
                if not isfinite(value):
                    raise ValueError(
                        f"non-finite value of evaluation[{name!r}]={value!r}")
                if not (lower <= value <= upper):
                    raise ValueError(
                        f"it is required that {lll}<=f<={uuu}, but got "
                        f"{lower}, {value}, and {upper}.")
        bins: Final[SampleStatistics | None] = cast(
            "SampleStatistics", objectives[BIN_COUNT_NAME]) \
            if BIN_COUNT_NAME in objectives else None
        for name2, value2 in bin_bounds.items():
            if not isinstance(name2, str):
                raise type_error(
                    name2, f"name of bounds[{name2!r}]={value2!r}", str)
            check_int_range(value2, f"bounds[{name2!r}]", 1, 1_000_000_000)
            if (bins is not None) and (bins.minimum < value2):
                raise ValueError(
                    f"number of bins={bins} is inconsistent with "
                    f"bound {name2!r}={value2}.")

        object.__setattr__(self, "end_statistics", end_statistics)
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
        return self.end_statistics._tuple()


def from_packing_results(results: Iterable[PackingResult]) \
        -> Generator[PackingStatistics, None, None]:
    """
    Create packing statistics from a sequence of packing results.

    :param results: the packing results
    :returns: a sequence of packing statistics
    """
    if not isinstance(results, Iterable):
        raise type_error(results, "results", Iterable)
    groups: Final[dict[tuple[str, str, str, str], list[PackingResult]]] \
        = {}
    objectives_set: set[str] = set()
    for i, pr in enumerate(results):
        if not isinstance(pr, PackingResult):
            raise type_error(pr, f"end_results[{i}]", PackingResult)
        setting: tuple[str, str, str, str] = \
            (pr.end_result.algorithm, pr.end_result.instance,
             pr.end_result.objective, "" if pr.end_result.encoding is None
             else pr.end_result.encoding)
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
        used_objective: str = pr0.end_result.objective
        encoding: str | None = pr0.end_result.encoding
        if used_objective not in objectives_set:
            raise ValueError(
                f"{used_objective!r} not in {objectives_set!r}.")
        if used_objective != key[2]:
            raise ValueError(
                f"used objective={used_objective!r} different "
                f"from key[2]={key[2]}!?")
        if (encoding is not None) and (encoding != key[3]):
            raise ValueError(
                f"used encoding={encoding!r} different "
                f"from key[3]={key[3]}!?")
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
            if used_objective != pr.end_result.objective:
                raise ValueError(
                    f"used objective={used_objective!r} for data[0] "
                    f"but {pr.end_result.objective!r} for data[{i}]?")
            if objective_bounds != pr.objective_bounds:
                raise ValueError(
                    f"objective_bounds={objective_bounds!r} for data[0] "
                    f"but {pr.objective_bounds!r} for data[{i}]?")
            if bin_bounds != pr.bin_bounds:
                raise ValueError(
                    f"bin_bounds={bin_bounds!r} for data[0] "
                    f"but {pr.bin_bounds!r} for data[{i}]?")

        end_stats.extend(es_from_end_results(pr.end_result for pr in data))
        if len(end_stats) != 1:
            raise ValueError(f"got {end_stats} from {data}?")

        yield PackingStatistics(
            end_statistics=end_stats[0],
            n_items=n_items,
            n_different_items=n_different_items,
            bin_width=bin_width,
            bin_height=bin_height,
            objectives={
                o: ss_from_samples(pr.objectives[o] for pr in data)
                for o in objectives
            },
            objective_bounds=objective_bounds,
            bin_bounds=bin_bounds,
        )
        end_stats.clear()


def to_csv(results: Iterable[PackingStatistics], file: str) -> Path:
    """
    Write a sequence of packing statistics to a file in CSV format.

    :param results: the end statistics
    :param file: the path
    :return: the path of the file that was written
    """
    path: Final[Path] = Path(file)
    logger(f"Writing packing statistics to CSV file {path!r}.")
    path.ensure_parent_dir_exists()
    with path.open_for_write() as wt:
        consumer: Final[Callable[[str], None]] = line_writer(wt)
        for p in CsvWriter.write(sorted(results)):
            consumer(p)
    logger(f"Done writing packing statistics to CSV file {path!r}.")
    return path


def from_csv(file: str) -> Iterable[PackingStatistics]:
    """
    Load the packing statistics from a CSV file.

    :param file: the file to read from
    :returns: the iterable with the packing statistics
    """
    path: Final[Path] = file_path(file)
    logger(f"Now reading CSV file {path!r}.")
    with path.open_for_read() as rd:
        yield from CsvReader.read(rd)
    logger(f"Done reading CSV file {path!r}.")


class CsvWriter(CsvWriterBase[PackingStatistics]):
    """A class for CSV writing of :class:`PackingStatistics`."""

    def __init__(self, data: Iterable[PackingStatistics],
                 scope: str | None = None) -> None:
        """
        Initialize the csv writer.

        :param data: the data to write
        :param scope: the prefix to be pre-pended to all columns
        """
        data = reiterable(data)
        super().__init__(data, scope)
        #: the end statistics writer
        self.__es: Final[EsCsvWriter] = EsCsvWriter((
            pr.end_statistics for pr in data), scope)

        bin_bounds_set: Final[set[str]] = set()
        objectives_set: Final[set[str]] = set()
        for pr in data:
            bin_bounds_set.update(pr.bin_bounds.keys())
            objectives_set.update(pr.objectives.keys())

        #: the bin bounds
        self.__bin_bounds: list[str] | None = None \
            if set.__len__(bin_bounds_set) <= 0 else sorted(bin_bounds_set)

        #: the objectives
        objectives: list[SsCsvWriter] | None = None
        #: the objective names
        objective_names: tuple[str, ...] | None = None
        #: the lower bound names
        objective_lb_names: tuple[str, ...] | None = None
        #: the upper bound names
        objective_ub_names: tuple[str, ...] | None = None
        if set.__len__(objectives_set) > 0:
            p: Final[str | None] = self.scope
            objective_names = tuple(sorted(objectives_set))
            objective_lb_names = tuple(csv_scope(
                oxx, _OBJECTIVE_LOWER) for oxx in objective_names)
            objective_ub_names = tuple(csv_scope(
                oxx, _OBJECTIVE_UPPER) for oxx in objective_names)
            objectives = [SsCsvWriter(
                data=(ddd.objectives[k] for ddd in data),
                scope=csv_scope(p, k), n_not_needed=True, what_short=k,
                what_long=f"objective function {k}") for k in objective_names]

        #: the objectives
        self.__objectives: Final[list[SsCsvWriter] | None] = objectives
        #: the objective names
        self.__objective_names: Final[tuple[str, ...] | None] \
            = objective_names
        #: the lower bound names
        self.__objective_lb_names: Final[tuple[str, ...] | None] \
            = objective_lb_names
        #: the upper bound names
        self.__objective_ub_names: Final[tuple[str, ...] | None] \
            = objective_ub_names

    def get_column_titles(self) -> Iterable[str]:
        """Get the column titles."""
        p: Final[str | None] = self.scope
        yield from self.__es.get_column_titles()

        yield csv_scope(p, KEY_BIN_HEIGHT)
        yield csv_scope(p, KEY_BIN_WIDTH)
        yield csv_scope(p, KEY_N_ITEMS)
        yield csv_scope(p, KEY_N_DIFFERENT_ITEMS)
        if self.__bin_bounds:
            for b in self.__bin_bounds:
                yield csv_scope(p, b)
        if self.__objective_names and self.__objectives:
            for i, o in enumerate(self.__objectives):
                yield csv_scope(p, self.__objective_lb_names[i])
                yield from o.get_column_titles()
                yield csv_scope(p, self.__objective_ub_names[i])

    def get_row(self, data: PackingStatistics) -> Iterable[str]:
        """
        Render a single packing result record to a CSV row.

        :param data: the end result record
        :returns: the iterable with the row text
        """
        yield from self.__es.get_row(data.end_statistics)
        yield repr(data.bin_height)
        yield repr(data.bin_width)
        yield repr(data.n_items)
        yield repr(data.n_different_items)
        if self.__bin_bounds:
            for bb in self.__bin_bounds:
                yield (repr(data.bin_bounds[bb])
                       if bb in data.bin_bounds else "")
        if self.__objective_names and self.__objectives:
            lb: Final[tuple[str, ...] | None] = self.__objective_lb_names
            ub: Final[tuple[str, ...] | None] = self.__objective_ub_names
            for i, ob in enumerate(self.__objective_names):
                if lb is not None:
                    ox = lb[i]
                    yield (num_to_str(data.objective_bounds[ox])
                           if ox in data.objective_bounds else "")
                yield from SsCsvWriter.get_optional_row(
                    self.__objectives[i], data.objectives.get(ob))
                if ub is not None:
                    ox = ub[i]
                    yield (num_to_str(data.objective_bounds[ox])
                           if ox in data.objective_bounds else "")

    def get_header_comments(self) -> Iterable[str]:
        """
        Get any possible header comments.

        :returns: the header comments
        """
        return ("End Statistics of Bin Packing Experiments",
                "See the description at the bottom of the file.")

    def get_footer_comments(self) -> Iterable[str]:
        """
        Get any possible footer comments.

        :returns: the footer comments
        """
        yield from self.__es.get_footer_comments()
        yield ""
        p: Final[str | None] = self.scope
        if self.__bin_bounds:
            for bb in self.__bin_bounds:
                yield (f"{csv_scope(p, bb)} is a lower "
                       "bound for the number of bins.")
        if self.__objectives and self.__objective_names:
            for i, obb in enumerate(self.__objective_names):
                ob: str = csv_scope(p, obb)
                ox: str = csv_scope(ob, _OBJECTIVE_LOWER)
                yield f"{ox}: a lower bound of the {ob} objective function."
                yield from self.__objectives[i].get_footer_comments()
                ox = csv_scope(ob, _OBJECTIVE_UPPER)
                yield f"{ox}: an upper bound of the {ob} objective function."

    def get_footer_bottom_comments(self) -> Iterable[str]:
        """Get the bottom footer comments."""
        yield from motipyapps_footer_bottom_comments(
            self, "The packing data is assembled using module "
                  "moptipyapps.binpacking2d.packing_statistics.")
        yield from EsCsvWriter.get_footer_bottom_comments(self.__es)


class CsvReader(CsvReaderBase[PackingStatistics]):
    """A class for CSV parsing to get :class:`PackingStatistics`."""

    def __init__(self, columns: dict[str, int]) -> None:
        """
        Create a CSV parser for :class:`EndResult`.

        :param columns: the columns
        """
        super().__init__(columns)
        #: the end result csv reader
        self.__es: Final[EsCsvReader] = EsCsvReader(columns)
        #: the index of the n-items column
        self.__idx_n_items: Final[int] = csv_column(columns, KEY_N_ITEMS)
        #: the index of the n different items column
        self.__idx_n_different: Final[int] = csv_column(
            columns, KEY_N_DIFFERENT_ITEMS)
        #: the index of the bin width column
        self.__idx_bin_width: Final[int] = csv_column(
            columns, KEY_BIN_WIDTH)
        #: the index of the bin height column
        self.__idx_bin_height: Final[int] = csv_column(
            columns, KEY_BIN_HEIGHT)
        #: the indices for the objective bounds
        self.__bin_bounds: Final[tuple[tuple[str, int], ...]] = \
            csv_select_scope(
                lambda x: tuple(sorted(((k, v) for k, v in x.items()))),
                columns, LOWER_BOUNDS_BIN_COUNT)
        if tuple.__len__(self.__bin_bounds) <= 0:
            raise ValueError("No bin bounds found?")
        #: the objective bounds columns
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
        n_val: Final[tuple[tuple[str, int]]] = ((KEY_N, self.__es.idx_n), )
        #: the parsers for the per-objective statistics
        self.__objectives: Final[tuple[tuple[str, SsCsvReader], ...]] = \
            tuple((ss, csv_select_scope(SsCsvReader, columns, ss, n_val))
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

    def parse_row(self, data: list[str]) -> PackingStatistics:
        """
        Parse a row of data.

        :param data: the data row
        :return: the end result statistics
        """
        return PackingStatistics(
            self.__es.parse_row(data),
            int(data[self.__idx_n_items]),
            int(data[self.__idx_n_different]),
            int(data[self.__idx_bin_width]),
            int(data[self.__idx_bin_height]),
            {o: v.parse_row(data) for o, v in self.__objectives},
            {o: str_to_num(data[v]) for o, v in self.__objective_bounds},
            {o: int(data[v]) for o, v in self.__bin_bounds},
        )


# Run packing-results to stat file if executed as script
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "Build an end-results statistics CSV file.",
        "This program computes statistics over packing results")
    def_src: str = "./evaluation/end_results.txt"
    if not os.path.isfile(def_src):
        def_src = "./results"
    parser.add_argument(
        "source", nargs="?", default=def_src,
        help="either the directory with moptipy log files or the path to the "
             "end-results CSV file", type=Path)
    parser.add_argument(
        "dest", type=Path, nargs="?",
        default="./evaluation/end_statistics.txt",
        help="the path to the end results statistics CSV file to be created")
    args: Final[argparse.Namespace] = parser.parse_args()

    src_path: Final[Path] = args.source
    packing_results: Iterable[PackingResult]
    if src_path.is_file():
        logger(f"{src_path!r} identifies as file, load as end-results csv")
        packing_results = pr_from_csv(src_path)
    else:
        logger(f"{src_path!r} identifies as directory, load it as log files")
        packing_results = pr_from_logs(src_path)
    to_csv(from_packing_results(results=packing_results), args.dest)
