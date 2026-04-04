"""
The Lunar Tomato Logistics beginner problem data.

See <https://github.com/esa/SpOC4> for the original source of the data.
"""

from importlib import resources  # nosem  # noqa: RUF067
from typing import TextIO, cast  # noqa: RUF067

from pycommons.io.path import UTF8  # noqa: RUF067


def open_resource_stream(file_name: str) -> TextIO:  # noqa: RUF067
    """
    Open a matching resource stream.

    :param file_name: the file name of the resource
    :return: the stream
    """
    return cast("TextIO", resources.files(__package__).joinpath(
        file_name).open("r", encoding=UTF8))
