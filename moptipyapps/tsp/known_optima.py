"""A set of tools to load known optima in the TSPLib format."""

from typing import Final, TextIO, cast

import moptipy.utils.nputils as npu
import numpy as np
from moptipy.utils.path import Path
from moptipy.utils.types import check_to_int_range, type_error

from moptipyapps.tsp.tsplib import open_resource_stream

#: the set of known optimal tours
_TOURS: Final[tuple[str, ...]] = (
    "ulysses22", "ulysses16", "tsp225", "st70", "rd100", "pr1002",
    "pr76", "pcb442", "lin105", "kroD100", "kroC100", "kroA100", "eil101",
    "eil76", "eil51", "ch150", "ch130", "berlin52", "att48", "a280")


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

    :return: the tuple
    """
    return _TOURS
