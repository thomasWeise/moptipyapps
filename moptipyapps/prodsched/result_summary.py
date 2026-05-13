"""Load an ROP multi-simulation summary from a log file."""

import argparse
from typing import Any, Callable, Final, Generator, Iterable

from moptipy.api.logging import (
    PREFIX_SECTION_ARCHIVE,
    SECTION_ARCHIVE_QUALITY,
    SUFFIX_SECTION_ARCHIVE_X,
    SUFFIX_SECTION_ARCHIVE_Y,
)
from moptipy.api.space import Space
from moptipy.spaces.intspace import IntSpace
from moptipy.utils.logger import SECTION_END, SECTION_START
from pycommons.io.console import logger
from pycommons.io.csv import COMMENT_START
from pycommons.io.csv import CSV_SEPARATOR as __CS
from pycommons.io.path import Path
from pycommons.math.stream_statistics import (
    StreamStatistics,
    StreamStatisticsAggregate,
)
from pycommons.strings.string_conv import num_or_none_to_str as __n
from pycommons.types import type_error

from moptipyapps.prodsched.instances import get_instances
from moptipyapps.prodsched.multistatistics import (
    MultiStatistics,
    MultiStatisticsSpace,
)
from moptipyapps.prodsched.statistics import Statistics
from moptipyapps.utils.shared import moptipyapps_argparser


def __m(v: int | float | StreamStatistics | None) -> str:
    """
    Convert a value to a string.

    :param v: the value
    :return: the string
    """
    if v is None:
        return ""
    if isinstance(v, StreamStatistics):
        return __n(v.mean_arith)
    return __n(v)


def summarize_multi_statistics_to_text(y: MultiStatistics) \
        -> Generator[str, None, None]:
    """
    Summarize multi-statistics to text.

    :param y: the multi-statistics
    :return: the text
    """
    pi: Final[tuple[Statistics, ...]] = y.per_instance

    n_inst: Final[int] = tuple.__len__(pi)
    if n_inst <= 0:
        raise ValueError("No instance.")
    n_prod: Final[int] = list.__len__(pi[0].immediate_rates)

    yield "## Per-Instance Statistics"
    yield (f"instance{__CS}fillRate{__CS}stockLevel{__CS}meanWaitingTime"
           f"{__CS}meanProductionTime")
    irs = StreamStatistics.aggregate()
    sls = StreamStatistics.aggregate()
    wts = StreamStatistics.aggregate()
    pts = StreamStatistics.aggregate()
    for i, st in enumerate(pi):
        yield (f"{i}{__CS}{__m(st.immediate_rate)}{__CS}{__m(st.stock_level)}"
               f"{__CS}{__m(st.waiting_time)}{__CS}{__m(st.production_time)}")
        if st.immediate_rate is not None:
            irs.add(st.immediate_rate)
        if st.stock_level is not None:
            sls.add(st.stock_level)
        if (st.waiting_time is not None) and (
                st.waiting_time.mean_arith is not None):
            wts.add(st.waiting_time.mean_arith)
        if (st.production_time is not None) and (
                st.production_time.mean_arith is not None):
            pts.add(st.production_time.mean_arith)
    irss = irs.result_or_none()
    slss = sls.result_or_none()
    wtss = wts.result_or_none()
    ptss = pts.result_or_none()
    yield (f"min{__CS}{'' if irss is None else __n(irss.minimum)}{__CS}"
           f"{'' if slss is None else __n(slss.minimum)}{__CS}"
           f"{'' if wtss is None else __n(wtss.minimum)}{__CS}"
           f"{'' if ptss is None else __n(ptss.minimum)}")
    yield (f"mean{__CS}{'' if irss is None else __n(irss.mean_arith)}{__CS}"
           f"{'' if slss is None else __n(slss.mean_arith)}{__CS}"
           f"{'' if wtss is None else __n(wtss.mean_arith)}{__CS}"
           f"{'' if ptss is None else __n(ptss.mean_arith)}")
    yield (f"max{__CS}{'' if irss is None else __n(irss.maximum)}{__CS}"
           f"{'' if slss is None else __n(slss.maximum)}{__CS}"
           f"{'' if wtss is None else __n(wtss.maximum)}{__CS}"
           f"{'' if ptss is None else __n(ptss.maximum)}")
    yield (f"stddev{__CS}{'' if irss is None else __n(irss.stddev)}{__CS}"
           f"{'' if slss is None else __n(slss.stddev)}{__CS}"
           f"{'' if wtss is None else __n(wtss.stddev)}{__CS}"
           f"{'' if ptss is None else __n(ptss.stddev)}")
    yield ""
    yield "## Per-Instance and Per-Product Statistics"

    s: str = "instance"
    collect: list[StreamStatisticsAggregate] = []
    for j in range(n_prod):
        js = str(j + 1)
        s = (f"{s}{__CS}fillRate[{js}]{__CS}stockLevel[{js}]{__CS}"
             f"meanWaitingTime[{js}]{__CS}meanProductionTime[{js}]")
        collect.extend((StreamStatistics.aggregate(),
                        StreamStatistics.aggregate(),
                        StreamStatistics.aggregate(),
                        StreamStatistics.aggregate()))
    yield s
    for i, st in enumerate(pi):
        s = str(i)
        ci: int = 0
        for p in range(n_prod):
            s = (f"{s}{__CS}{__m(st.immediate_rates[p])}{__CS}"
                 f"{__m(st.stock_levels[p])}{__CS}{__m(st.waiting_times[p])}"
                 f"{__CS}{__m(st.production_times[p])}")

            v = st.immediate_rates[p]
            if v is not None:
                collect[ci].add(v)
            ci += 1
            v = st.stock_levels[p]
            if v is not None:
                collect[ci].add(v)
            ci += 1
            vx = st.waiting_times[p]
            if vx is not None:
                v = vx.mean_arith
                if v is not None:
                    collect[ci].add(v)
            ci += 1
            vx = st.production_times[p]
            if vx is not None:
                v = vx.mean_arith
                if v is not None:
                    collect[ci].add(v)
            ci += 1
        yield s

    cres: list[StreamStatistics | None] = [
        cc.result_or_none() for cc in collect]
    s = "min"
    for cress in cres:
        s = f"{s}{__CS}{'' if cress is None else __n(cress.minimum)}"
    yield s
    s = "mean"
    for cress in cres:
        s = f"{s}{__CS}{'' if cress is None else __n(cress.mean_arith)}"
    yield s
    s = "max"
    for cress in cres:
        s = f"{s}{__CS}{'' if cress is None else __n(cress.maximum)}"
    yield s
    s = "stddev"
    for cress in cres:
        s = f"{s}{__CS}{'' if cress is None else __n(cress.stddev)}"
    yield s


def summarize_rop_to_text(x: Any) -> Generator[str, None, None]:
    """
    Summarize an ROP to text.

    :param x: the re-order point
    :return: the text
    """
    n: Final[int] = len(x)
    s = "product"
    for i in range(n):
        s = f"{s}{__CS}{int(i + 1)}"
    yield s
    s = "ROP"
    for v in x:
        s = f"{s}{__CS}{int(v)}"
    yield s


def __default_x_space(ms: MultiStatisticsSpace) -> Space:
    """
    Create the default integer space.

    :param ms: the multi-statistics space
    :return: the int space
    """
    return IntSpace(ms.instances[0].n_products, 0, 1_000_000_000)


def result_summary(
        source: Iterable[str],
        y_space: MultiStatisticsSpace,
        x_space: Space | Callable[[MultiStatisticsSpace], Space] =
        __default_x_space,
        index_filter: Callable[[int], bool] = lambda _: True,
        x_from_text: Callable[[Iterable[str]], Any] | None = None,
        y_from_text: Callable[[Iterable[str]], MultiStatistics] | None = None,
        x_to_text: Callable[[Any], Generator[str, None, None]] | None =
        summarize_rop_to_text,
        y_to_text: Callable[[MultiStatistics], Generator[
            str, None, None]] | None =
        summarize_multi_statistics_to_text) \
        -> Generator[str, None, None]:
    """
    Load an ROP multi-simulation summary from a log file.

    :param source: the source log file
    :param x_space: the search space
    :param y_space: the multi-statistics space
    :param x_from_text: convert text to an element of the x-space
    :param y_from_text: convert text to an element of the y-space
    :param x_to_text: convert an element of the x-space to text
    :param y_to_text: convert an element of the y-space to text
    :param index_filter: the index filter function
    :return: the generator with the summary text
    """
    if not isinstance(source, Iterable):
        raise type_error(source, "source", Iterable)
    if not isinstance(y_space, MultiStatisticsSpace):
        raise type_error(y_space, "y_space", MultiStatisticsSpace)
    if callable(x_space):
        x_space = x_space(y_space)
    if not isinstance(x_space, Space):
        raise type_error(x_space, "x_space", Space, call=True)

    if y_from_text is None:
        def __y_from_text(text, _z=y_space) -> MultiStatistics:
            return _z.create().from_stream(text)
        y_from_text = __y_from_text
    if not callable(y_from_text):
        raise type_error(y_from_text, "y_from_text", call=True)

    if x_from_text is None:
        def __x_from_text(text, _z=x_space) -> Any:
            return _z.from_str(text[0])
        x_from_text = __x_from_text
    if not callable(x_from_text):
        raise type_error(x_from_text, "x_from_text", call=True)

    if y_to_text is None:
        def __y_to_text(y: MultiStatistics, _z=y_space) \
                -> Generator[str, None, None]:
            yield _z.to_str(y)
        y_to_text = __y_to_text
    if not callable(y_to_text):
        raise type_error(y_to_text, "y_to_text", call=True)

    if x_to_text is None:
        def __x_to_text(x, _z=x_space) -> Generator[str, None, None]:
            yield _z.to_str(x)
        x_to_text = __x_to_text
    if not callable(x_to_text):
        raise type_error(x_to_text, "x_to_text", call=True)

    if not callable(index_filter):
        raise type_error(index_filter, "index_filter", call=True)

    collected: dict[int, tuple[list[str], list[str], list[str]]] = {}

    # collect the raw data
    mode: int = -1
    current_index: int | None = None
    s_arch: Final[str] = f"{SECTION_START}{PREFIX_SECTION_ARCHIVE}"
    s_ql: Final[str] = f"{SECTION_START}{SECTION_ARCHIVE_QUALITY}"
    suf_x: Final[str] = SUFFIX_SECTION_ARCHIVE_X
    suf_y: Final[str] = SUFFIX_SECTION_ARCHIVE_Y
    e_arch: Final[str] = f"{SECTION_END}{PREFIX_SECTION_ARCHIVE}"
    e_ql: Final[str] = f"{SECTION_END}{SECTION_ARCHIVE_QUALITY}"
    cur_coll: list[str] = []
    for srow in source:
        row = str.strip(srow)
        if (str.__len__(row) <= 0) or row.startswith(COMMENT_START):
            continue
        if 0 <= mode <= 1:
            if row.startswith(e_arch):
                if current_index is None:
                    mode = -1
                    continue
                if current_index not in collected:
                    collected[current_index] = ([], [], [])
                collected[current_index][mode].extend(cur_coll)
                cur_coll.clear()
                current_index = None
                mode = -1
                continue
            if current_index is not None:
                cur_coll.append(row)
            continue

        if mode == 2:
            if row == e_ql:
                mode = -1
                current_index = None
                continue
            if str.lower(row[0]) == "f":
                continue
            if current_index is not None:
                current_index += 1
                if not index_filter(current_index):
                    continue
                if current_index not in collected:
                    collected[current_index] = ([], [], [])
                collected[current_index][mode].append(row)
                continue

        if row.startswith(s_arch):
            if row.endswith(suf_x):
                mode = 0
                current_index = int(row[str.__len__(s_arch):
                                        -str.__len__(suf_x)])
                if not index_filter(current_index):
                    current_index = None
                continue
            if row.endswith(suf_y):
                mode = 1
                current_index = int(row[str.__len__(s_arch):
                                        -str.__len__(suf_y)])
                if not index_filter(current_index):
                    current_index = None
                continue
            if row == s_ql:
                mode = 2
                current_index = -1

    # now print the results
    not_first: bool = False
    counter: int = 0
    for idx in sorted(collected.keys()):
        counter += 1
        value: tuple[list[str], list[str], list[str]] = collected[idx]
        if not_first:
            yield ""
            yield ""
        not_first = True

        yield f"# ================== Solution {idx} =================="
        yield from x_to_text(x_from_text(value[0]))
        yield ""
        yield from y_to_text(y_from_text(value[1]))
        yield ""
        obvals: list[str] = value[2][0].split(__CS)
        yield f"summary objective value{__CS}{obvals[0]}"
        for i in range(1, list.__len__(obvals)):
            yield f"f{i}{__CS}{obvals[i]}"
    logger(f"Found {counter} results.")


def result_summaries(
        source: str,
        dest: str,
        y_space: MultiStatisticsSpace,
        x_space: Space | Callable[[MultiStatisticsSpace], Space] =
        __default_x_space,
        index_filter: Callable[[int], bool] = lambda _: True,
        x_from_text: Callable[[Iterable[str]], Any] | None = None,
        y_from_text: Callable[[Iterable[str]], MultiStatistics] | None = None,
        x_to_text: Callable[[Any], Generator[str, None, None]] | None =
        summarize_rop_to_text,
        y_to_text: Callable[[MultiStatistics], Generator[
            str, None, None]] | None =
        summarize_multi_statistics_to_text) -> None:
    """
    Convert one or multiple files from a source to a destination.

    :param source: the source file or directory
    :param dest: the destination directory
    :param x_space: the search space
    :param y_space: the multi-statistics space
    :param x_from_text: convert text to an element of the x-space
    :param y_from_text: convert text to an element of the y-space
    :param x_to_text: convert an element of the x-space to text
    :param y_to_text: convert an element of the y-space to text
    """
    src: Final[Path] = Path(source)
    dst: Final[Path] = Path(dest)
    dst.ensure_dir_exists()

    if src.is_dir():
        for spt in src.list_dir():
            result_summaries(spt, dst, y_space, x_space, index_filter,
                             x_from_text, y_from_text)
        return
    if not src.is_file():
        return

    dest_file: Final[Path] = dst.resolve_inside(src.basename())
    logger(f"Now processing {src!r} to {dest_file!r}.")
    with dest_file.open_for_write() as ds, src.open_for_read() as ss:
        for s in result_summary(
                ss,
                y_space=y_space,
                x_space=x_space,
                x_from_text=x_from_text,
                y_from_text=y_from_text,
                x_to_text=x_to_text,
                y_to_text=y_to_text):
            ds.write(s)
            ds.write("\n")


# Run to parse all log files and to create csv
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__,
        "Postprocess Solutions",
        "Create postprocessing results.")
    parser.add_argument(
        "source", nargs="?", default="./results",
        help="the location of the experimental results, i.e., the root folder "
             "under which to search for log files", type=Path)
    parser.add_argument(
        "dest", help="the path to the destination folder to be created",
        type=Path, nargs="?", default="./evaluation/")
    parser.add_argument(
        "insts", help="the directory with the instances",
        type=Path, nargs="?", default="./instances/")
    parser.add_argument(
        "n_inst", help="the number of instances",
        type=int, nargs="?", default=23)

    args: Final[argparse.Namespace] = parser.parse_args()

    n_insts: Final[int] = args.n_inst
    inst_path: Final[Path] = args.insts
    logger(f"Loading {n_insts} instances from {inst_path!r}.")
    insts = get_instances(n_insts, inst_path)
    logger("Done loading instances, now executing evaluation.")
    result_summaries(args.source, args.dest, MultiStatisticsSpace(insts))
