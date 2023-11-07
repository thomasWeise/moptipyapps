"""
Find a reasonable one-dimensional order for permutations.

Solve the one-dimensional ordering problem for permutations with fitness
stored in the `dat` file format as used by Thomson et al. This program
loads permutations from one or multiple files of this type and then tries
to arrange them on a horizontal axis such that their neighborhood relation
on the one-dimensional axis represents their neighborhood based one the
swap distance metric in the permutation space. It does this re-arrangement
by applying a randomized local search (RLS) to find the right permutation
of these permutations to be mapped to one-dimensional coordinates.

You can call this program as follows:

`python3 order1_from_dat.py sourceData outputFile \
[fitnessCap] [maxFEs] [fileNameRegEx]`

where
- `sourceData` is the path to a file or folder with the original data.
  If it is a file, the file will be loaded if it matches `fileNameRegEx`.
  If it is a folder, all files in this folder and all sub-folders will
  recursively be parsed for input data.
- `outputFile` is the path to a file where to store the result.
- `fitnessCap` is an upper limit for the fitness values stored in the files,
  all points with higher fitness are ignored. (default: So high that it does
  not matter.)
- `maxFEs` are the maximum number of optimization steps until done. By
  default, these are `10000`.
- `fileNameRegEx` is a regular expression. Only file names matching this
  regular expression are parsed. By default, it matches all files. But you
  could provide `tai27e01.*`, e.g., to only match files starting with
  `tai27e01` (make sure to write `"tai27e01.*"` on the command line to prevent
  the dot-start from being expanded by the shell),

The input format of this program are `dat` files of the format
```
EVALS    GENOTYPE    FITNESS
1    [22, 7, 6, 26, 27, 19, 3, 1, ... 5, 21, 8, 17, 2, 16, 9, 23]    87018
13    [20, 7, 6, 26, 18, 19, 9, 1, ... 25, 13, 23, 16, 15, 24]    85456
20    [20, 7, 18, 26, 6, 16, 9, 1,  ...  21, 13, 12, 19, 15, 17]    84152
29    [20, 11, 14, 25, 5, 16, 15, 1,  ...  21, 13, 12, 9, 19, 17]    83180
32    [20, 10, 14, 25, 5, 12, 15, 1,  ... 17, 13, 16, 9, 19, 21]    82846
34    [20, 15, 14, 25, 5, 12, 10, 1,  ...  6, 17, 13, 16, 9, 19, 21]    78204
```

`fitnessCap` reflects an upper limit to the `FITNESS` column. Setting it to
`80000` would, for example, lead to ignoring all but the last line in the
above example.
"""

import argparse
from os import listdir
from os.path import basename, isdir, isfile, join
from re import Pattern
from re import compile as re_compile
from typing import Any, Callable, Final

import numpy as np
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swap2 import Op1Swap2
from moptipy.utils.console import logger
from moptipy.utils.help import argparser
from moptipy.utils.path import Path
from moptipy.utils.types import check_to_int_range

from moptipyapps.order1d.distances import swap_distance
from moptipyapps.order1d.instance import Instance
from moptipyapps.order1d.space import OrderingSpace
from moptipyapps.qap.objective import QAPObjective

#: the impact of rank differences
POWER: Final[float] = 2.0
#: ignore everything that is farther away than the 64th nearest neighbor
HORIZON: Final[int] = 64
#: the fitness tag
TAG_FITNESS: Final[str] = "fitness"
#: the evaluation tag
TAG_EVAL: Final[str] = "evaluations"
#: the permutation
TAG_PERM: Final[str] = "permutation"
#: the file
TAG_FILE: Final[str] = "file"


def parse_data(path: str, collector: Callable[[
        tuple[str, int, int, np.ndarray]], Any],
        fitness_limit: int, pattern: Pattern) -> None:
    """
    Parse a dat file.

    :param path: the path
    :param collector: the collector function to invoke when loading data
    :param fitness_limit: the minimum acceptable fitness
    :param pattern: the file name pattern
    """
    the_path: Final[Path] = Path.path(path)
    if isdir(the_path):  # recursively parse directories
        logger(f"recursing into directory '{the_path}'.")
        for subpath in listdir(the_path):
            parse_data(join(the_path, subpath), collector, fitness_limit,
                       pattern)
        return

    if not isfile(the_path):
        return  # if it is not a file, we quit
    the_name: Final[str] = basename(the_path)
    if not pattern.match(the_name):
        return  # file does not match

    # parse the file
    for oline in the_path.open_for_read():
        line = oline.strip()
        if len(line) <= 0:
            continue
        bracket_open: int = line.find("[")
        if bracket_open <= 0:
            continue
        bracket_close: int = line.find("]", bracket_open + 1)
        if bracket_close <= bracket_open:
            continue
        f: int = check_to_int_range(line[bracket_close + 1:],
                                    TAG_FITNESS, 0, 1_000_000_000_000)
        if f > fitness_limit:
            continue
        evals: int = check_to_int_range(line[:bracket_open].strip(),
                                        TAG_EVAL, 1, 1_000_000_000_000_000)
        perm: list[int] = [
            check_to_int_range(s, TAG_PERM, 1, 1_000_000_000) - 1
            for s in line[bracket_open + 1:bracket_close].split(",")]
        collector((the_name, evals, f, np.array(perm)))


def get_tags(data: tuple[str, int, int, np.ndarray]) \
        -> tuple[str, str, str, str]:
    """
    Get the tags to store along with the data.

    :param data: the data
    :return: the tags
    """
    return (data[0], str(data[1]), str(data[2]),
            f"\"[{','.join(str(a + 1) for a in data[3])}]\"")


def get_distance(a: tuple[str, int, int, np.ndarray],
                 b: tuple[str, int, int, np.ndarray]) -> int:
    """
    Get the distance between two data elements.

    The distance here is the swap distance.

    :param a: the first element
    :param b: the second element
    :return: the swap distance
    """
    return swap_distance(a[3], b[3])


def run(source: str, dest: str, max_fes: int, fitness_limit: int,
        file_name_regex: str) -> None:
    """
    Run the RLS algorithm to optimize a horizontal layout permutation.

    :param source: the source file or directory
    :param dest: the destination file
    :param max_fes: the maximum FEs
    :param fitness_limit: the minimum acceptable fitness
    :param file_name_regex: the file name regular expression
    """
    logger(f"invoked program with source='{source}', dest='{dest}', "
           f"max_fes={max_fes}, fitness_limit={fitness_limit}, and "
           f"file_name_regex='{file_name_regex}'.")
    # first, we load all the data to construct a distance rank matrix
    pattern: Final[Pattern] = re_compile(file_name_regex)
    logger(f"now loading data from '{source}' matching to '{pattern}'.")

    data: list[tuple[str, int, int, np.ndarray]] = []
    parse_data(source, data.append, fitness_limit, pattern)
    logger(f"finished loading {len(data)} rows of data, "
           "now constructing distance rank matrix.")
    instance: Final[Instance] = Instance.from_sequence_and_distance(
        data, get_distance, POWER, HORIZON,
        (TAG_FILE, TAG_EVAL, TAG_FITNESS, TAG_PERM), get_tags)
    del data  # free the now useless data

    # run the algorithm
    logger(f"finished constructing matrix with {instance.n} rows, "
           "now doing optimization for "
           f"{max_fes} FEs and, afterwards, writing result to '{dest}'.")
    space: Final[OrderingSpace] = OrderingSpace(instance)
    with (Execution().set_solution_space(space)
          .set_objective(QAPObjective(instance))
          .set_algorithm(RLS(Op0Shuffle(space), Op1Swap2()))
          .set_max_fes(max_fes)
          .set_log_improvements(True)
          .set_log_file(dest).execute()):
        pass
    logger("all done.")


# Perform the optimization
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = argparser(
        __file__, "One-Dimensional Ordering of Permutations",
        "Run the one-dimensional order of permutations experiment.")
    parser.add_argument(
        "source", help="the directory or file with the input data",
        type=Path.path, nargs="?", default="./")
    parser.add_argument(
        "dest", help="the file to write the output to",
        type=Path.path, nargs="?", default="./result.txt")
    parser.add_argument("fitnessLimit", help="the minimum acceptable fitness",
                        type=int, nargs="?", default=1_000_000_000)
    parser.add_argument("maxFEs", help="the maximum FEs to perform",
                        type=int, nargs="?", default=10_000)
    parser.add_argument(
        "fileNameRegEx",
        help="a regular expression that file names must match",
        type=str, nargs="?", default=".*")
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.source, args.dest, args.maxFEs, args.fitnessLimit,
        args.fileNameRegEx)
