"""
The `RobinX` example data for the Traveling Tournament Problem (TTP).

David Van Bulck of the Sports Scheduling Research group, part of the
Faculty of Economics and Business Administration at Ghent University, Belgium,
maintains "RobinX: An XML-driven Classification for Round-Robin Sports
Timetabling", a set of benchmark data instances and results of the TTP.
Here we include some of these TTP instances into our package.

This package does not offer anything useful except for holding the TTP
files. You can find the documentation and actual classes for solving and
playing around with the TSP in package :mod:`~moptipyapps.ttp`.

The original data of `robinX` can be found at
<https://robinxval.ugent.be/RobinX/travelRepo.php>
"""

from importlib import resources  # nosem
from typing import TextIO, cast

from pycommons.io.path import UTF8


def open_resource_stream(file_name: str) -> TextIO:
    """
    Open a RobinX resource stream.

    :param file_name: the file name of the resource
    :return: the stream
    """
    return cast("TextIO", resources.files(__package__).joinpath(
        file_name).open("r", encoding=UTF8))
