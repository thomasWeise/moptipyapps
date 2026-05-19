"""Create data for runtime distribution plots."""

import argparse
from typing import Final

from pycommons.io.csv import COMMENT_START
from pycommons.io.csv import CSV_SEPARATOR as __CS
from pycommons.io.path import Path
from pycommons.strings.string_conv import num_to_str, str_to_num
from pycommons.types import type_error

from moptipyapps.utils.shared import moptipyapps_argparser


def find_runtimes(
        source: str, runtimes: dict[str, list[float | int]]) -> None:
    """
    Load an ROP multi-simulation summary from a log file.

    :param source: the source path
    :param runtimes: the destination for runtimes
    """
    if not isinstance(runtimes, dict):
        raise type_error(runtimes, "runtimes", dict)
    spath: Final[Path] = Path(source)

    if spath.is_dir():
        for subfile in spath.list_dir():
            find_runtimes(subfile, runtimes)
        return
    if not spath.is_file():
        return

    computer_id: str = ""

    # collect the raw data
    computer_key: Final[str] = "session.node: "
    simtime_key: Final[str] = "time.s: "
    dest: list[float | int] | None = None

    with spath.open_for_read() as stream:
        for srow in stream:
            row = str.strip(srow)
            if (str.__len__(row) <= 0) or row.startswith(COMMENT_START):
                continue
            if str.startswith(row, computer_key):
                if (dest is not None) or (str.__len__(computer_id) > 0):
                    raise ValueError("Illegal State, already got computer id "
                                     f"{computer_id!r}, now found {row!r}.")
                computer_id = str.strip(row[str.__len__(computer_key):])
                if computer_id in runtimes:
                    dest = runtimes[computer_id]
                else:
                    dest = []
                    runtimes[computer_id] = dest
            elif str.startswith(row, simtime_key):
                if dest is None:
                    raise ValueError("Don't have computer id yet!")
                dest.append(str_to_num(row[str.__len__(simtime_key):]))


def runtime_summary(
        source: str,
        dest: str) -> None:
    """
    Convert one or multiple files from a source to a destination.

    :param source: the source file or directory
    :param dest: the destination file
    """
    src: Final[Path] = Path(source)
    dst_file: Final[Path] = Path(dest)

    runtimes: dict[str, list[int | float]] = {}
    find_runtimes(src, runtimes)

    data: list[list[str]] = []
    max_len: int = -1
    for key in sorted(runtimes.keys()):
        lst: list[str] = [key]
        keydata = runtimes[key]
        keydata.sort()
        lst.extend(map(num_to_str, keydata))
        max_len = max(max_len, list.__len__(lst))
        data.append(lst)

    row: list[str] = []
    with dst_file.open_for_write() as stream:
        for i in range(max_len):
            row.clear()
            for col in data:
                if i == 0:
                    row.extend((f"{col[0]}_x", f"{col[0]}_rt"))
                else:
                    ll = list.__len__(col)
                    if i < ll:
                        row.extend((num_to_str((i - 1) / (ll - 1)), col[i]))
                    else:
                        row.extend(("", ""))
            stream.write(__CS.join(row))
            stream.write("\n")


# Run to parse all log files and to create csv
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__,
        "Postprocess Runtimes",
        "Create postprocessing results.")
    parser.add_argument(
        "source", nargs="?", default="./results",
        help="the location of the experimental results, i.e., the root folder "
             "under which to search for log files", type=Path)
    parser.add_argument(
        "dest", help="the path to the destination file to be created",
        type=Path, nargs="?", default="./evaluation/runtimes.txt")

    args: Final[argparse.Namespace] = parser.parse_args()
    runtime_summary(args.source, args.dest)
