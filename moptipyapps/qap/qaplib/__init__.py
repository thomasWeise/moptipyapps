"""
QAPLIB -- A Quadratic Assignment Problem Library.

The original data  can be found at <https://qaplib.mgi.polymtl.ca> and
<https://coral.ise.lehigh.edu/data-sets/qaplib>.
"""

from importlib import resources  # nosem
from typing import TextIO


def open_resource_stream(file_name: str) -> TextIO:
    """
    Open a QAPLib resource stream.

    :param file_name: the file name of the resource
    :return: the stream
    """
    return resources.open_text(package=f"{__package__}", resource=file_name)
