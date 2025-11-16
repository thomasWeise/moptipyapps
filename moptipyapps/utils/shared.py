"""Some shared variables and constants."""

import argparse
from typing import Any, Final, Iterable

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
        _: Any, additional: "str | None" = None) -> Iterable[str]:
    """
    Print the standard csv footer for moptipyapps.

    :param _: the setup object, ignored
    :param additional: any additional output string
    :returns: the comments

    >>> for s in motipyapps_footer_bottom_comments(None, "bla"):
    ...     print(s[:49])
    This data has been generated with moptipyapps ver
    bla
    You can find moptipyapps at https://thomasweise.g

    >>> for s in motipyapps_footer_bottom_comments(None, None):
    ...     print(s[:49])
    This data has been generated with moptipyapps ver
    You can find moptipyapps at https://thomasweise.g
    """
    yield ("This data has been generated with moptipyapps version "
           f"{moptipyapps_version}.")
    if (additional is not None) and (str.__len__(additional) > 0):
        yield additional
    yield ("You can find moptipyapps at "
           "https://thomasweise.github.io/moptipyapps.")
