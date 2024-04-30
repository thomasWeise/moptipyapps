"""Some shared variables and constants."""

import argparse
from typing import Any, Callable, Final

import moptipy.examples.jssp.instance as ins
from pycommons.io.arguments import make_argparser, make_epilog

from moptipyapps.version import __version__ as moptipyapps_version

#: the instance scope
SCOPE_INSTANCE: Final[str] = ins.SCOPE_INSTANCE


def moptipyapps_argparser(file: str, description: str,
                          epilog: str) -> argparse.ArgumentParser:
    """
    Create an argument parser with default settings.

    :param file: the `__file__` special variable of the calling script
    :param description: the description string
    :param epilog: the epilogue string
    :returns: the argument parser

    >>> ap = moptipyapps_argparser(
    ...     __file__, "This is a test program.", "This is a test.")
    >>> isinstance(ap, argparse.ArgumentParser)
    True
    >>> "Copyright" in ap.epilog
    True
    """
    return make_argparser(
        file, description,
        make_epilog(epilog, 2023, 2024, "Thomas Weise",
                    url="https://thomasweise.github.io/moptipyapps",
                    email="tweise@hfuu.edu.cn, tweise@ustc.edu.cn"),
        moptipyapps_version)


def motipyapps_footer_bottom_comments(
        _: Any, dest: Callable[[str], Any],
        additional: str | None = None) -> None:
    """
    Print the standard csv footer for moptipyapps.

    :param _: the setup object, ignored
    :param dest: the destination callable
    :param dest: the destination to write to
    :param additional: any additional output string

    >>> def __qpt(s: str):
    ...     print(s[:49])
    >>> motipyapps_footer_bottom_comments(None, __qpt, "bla")
    This data has been generated with moptipyapps ver
    bla
    You can find moptipyapps at https://thomasweise.g

    >>> motipyapps_footer_bottom_comments(None, __qpt, None)
    This data has been generated with moptipyapps ver
    You can find moptipyapps at https://thomasweise.g
    """
    dest("This data has been generated with moptipyapps version "
         f"{moptipyapps_version}.")
    if (additional is not None) and (str.__len__(additional) > 0):
        dest(additional)
    dest("You can find moptipyapps at "
         "https://thomasweise.github.io/moptipyapps.")
