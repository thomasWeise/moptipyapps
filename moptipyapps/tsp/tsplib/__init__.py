"""
The TSPLib example data for the Traveling Salesperson Problem (TSP).

This package does not offer anything useful except for holding the TSPLib
files. You can find the documentation and actual classes for solving and
playing around with the TSP in package :mod:`~moptipyapps.tsp`.

The original data of TSPLib can be found at
<http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>. Before doing
anything with these data directly, you should make sure to read the FAQ
<http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html> and the
documentation
<http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf>.

1. Gerhard Reinelt. TSPLIB - A Traveling Salesman Problem Library.
   *ORSA Journal on Computing* 3(4):376-384. November 1991.
   https://doi.org/10.1287/ijoc.3.4.376.
   http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
2. Gerhard Reinelt. *TSPLIB95.* Heidelberg, Germany: Universität
   Heidelberg, Institut für Angewandte Mathematik. 1995.
   http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
"""

from importlib import resources  # nosem
from typing import TextIO


def open_resource_stream(file_name: str) -> TextIO:
    """
    Open a TSPLib resource stream.

    :param file_name: the file name of the resource
    :return: the stream
    """
    return resources.open_text(package=f"{__package__}", resource=file_name)
