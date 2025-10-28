"""
QAPLIB -- A Quadratic Assignment Problem Library.

1. QAPLIB - A Quadratic Assignment Problem Library. The Websites
   <https://qaplib.mgi.polymtl.ca/> (updated 2018) and
   <https://coral.ise.lehigh.edu/data-sets/qaplib/> (updated 2011), including
   the benchmark instances, on visited 2023-10-21.
2. Rainer E. Burkard, Stefan E. Karisch, and Franz Rendl. QAPLIB - A Quadratic
   Assignment Problem Library. Journal of Global Optimization. 10:391-403,
   1997. https://doi.org/10.1023/A:1008293323270.
"""

from importlib import resources  # nosem
from typing import TextIO, cast

from pycommons.io.path import UTF8


def open_resource_stream(file_name: str) -> TextIO:
    """
    Open a QAPLib resource stream.

    :param file_name: the file name of the resource
    :return: the stream
    """
    return cast("TextIO", resources.files(__package__).joinpath(
        file_name).open("r", encoding=UTF8))
