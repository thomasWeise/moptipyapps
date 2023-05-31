"""
A set of tools to load known optima in the TSPLib format.

The TSPLib provides benchmark instances for the Traveling Salesperson
Problem (TSP). We provide a subset of these instances with up to 2000 cities
as resources in :mod:`~moptipyapps.tsp.instance`. Additionally, TSPLib
provides the optimal solutions for a subset of these instances. Within this
module here, we provide methods for reading the optimal solution files
in the TSPLib format. We also include as resources the optimal solutions to
the instances that our package provide as resources as well.

You can get the list of instance names for which the optimal tours are
included in this package via :func:`list_resource_tours` and then load a tour
from a resource via :func:`opt_tour_from_resource`. If you want to read a tour
from an external file that obeys the TSPLib format, you can do so via
:func:`opt_tour_from_file`.

1. Gerhard Reinelt. TSPLIB - A Traveling Salesman Problem Library.
   *ORSA Journal on Computing* 3(4):376-384. November 1991.
   https://doi.org/10.1287/ijoc.3.4.376.
   http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
2. Gerhard Reinelt. *TSPLIB95.* Heidelberg, Germany: Universität
   Heidelberg, Institut für Angewandte Mathematik. 1995.
   http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
"""

from typing import Final, TextIO, cast

import moptipy.utils.nputils as npu
import numpy as np
from moptipy.utils.path import Path
from moptipy.utils.types import check_to_int_range, type_error

from moptipyapps.tsp.tsplib import open_resource_stream

#: the set of known optimal tours
_TOURS: Final[tuple[str, ...]] = (
    "a280", "att48", "bayg29", "bays29", "berlin52", "brg180", "ch130",
    "ch150", "eil101", "eil51", "eil76", "fri26", "gr120", "gr202", "gr24",
    "gr48", "gr666", "gr96", "kroA100", "kroC100", "kroD100", "lin105",
    "pcb442", "pr1002", "pr76", "rd100", "st70", "tsp225", "ulysses16",
    "ulysses22")


def _from_stream(stream: TextIO) -> np.ndarray:
    """
    Read an optimal tour from a stream.

    :param stream: the text stream
    :return: the tour
    """
    nodes: list[int] = []
    in_tour: bool = False
    done_nodes: set[int] = set()
    max_node: int = -1

    for the_line in stream:
        if not isinstance(the_line, str):
            raise type_error(the_line, "line", str)
        line = the_line.strip().upper()
        if len(line) <= 0:
            continue
        if line == "TOUR_SECTION":
            in_tour = True
            continue
        if line in ("-1", "EOF"):
            break
        if in_tour:
            for node_str in line.rsplit():
                node = check_to_int_range(
                    node_str, "node", 1, 1_000_000_000_000)
                if node in done_nodes:
                    raise ValueError(f"encountered node {node} twice")
                done_nodes.add(node)
                if node > max_node:
                    max_node = node
                nodes.append(node - 1)
    if len(nodes) != max_node:
        raise ValueError(
            f"expected {max_node} nodes, got {len(nodes)} instead.")
    return np.array(nodes, npu.int_range_to_dtype(0, max_node - 1))


def opt_tour_from_file(path: str) -> np.ndarray:
    """
    Read a known optimal tour from a file.

    :param path: the path to the file
    :return: the tour
    """
    file: Final[Path] = Path.file(path)
    with file.open_for_read() as stream:
        try:
            return _from_stream(cast(TextIO, stream))
        except (TypeError, ValueError) as err:
            raise ValueError(f"error when parsing file {file!r}") from err


def opt_tour_from_resource(name: str) -> np.ndarray:
    """
    Load an optimal tour from a resource.

    >>> np.array2string(opt_tour_from_resource("ulysses16"))
    '[ 0 13 12 11  6  5 14  4 10  8  9 15  2  1  3  7]'

    :param name: the name string
    :return: the optimal tour
    """
    if not isinstance(name, str):
        raise type_error(name, "name", str)
    with open_resource_stream(f"{name}.opt.tour") as stream:
        return _from_stream(stream)


def list_resource_tours() -> tuple[str, ...]:
    """
    Get a tuple of the names of the optimal tours in the resources.

    >>> list_resource_tours()
    ('a280', 'att48', 'bayg29', 'bays29', 'berlin52', 'brg180', 'ch130', \
'ch150', 'eil101', 'eil51', 'eil76', 'fri26', 'gr120', 'gr202', 'gr24', \
'gr48', 'gr666', 'gr96', 'kroA100', 'kroC100', 'kroD100', 'lin105', \
'pcb442', 'pr1002', 'pr76', 'rd100', 'st70', 'tsp225', 'ulysses16', \
'ulysses22')

    :return: the tuple
    """
    return _TOURS
