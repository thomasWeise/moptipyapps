"""Some shared variables and constants."""

import argparse
from typing import Final

import moptipy.examples.jssp.instance as ins
from pycommons.io.arguments import make_argparser, make_epilog

from moptipyapps.version import __version__

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
        __version__)
