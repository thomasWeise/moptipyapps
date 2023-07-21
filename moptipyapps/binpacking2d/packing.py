"""A two-dimensional packing."""

from typing import Final

import numpy as np
from moptipy.api.component import Component
from moptipy.api.logging import SECTION_RESULT_Y, SECTION_SETUP
from moptipy.evaluation.log_parser import LogParser
from moptipy.utils.types import type_error

from moptipyapps.binpacking2d.instance import Instance

#: the index of the ID in a :class:`Packing` row
IDX_ID: Final[int] = 0
#: the index of the bin in a :class:`Packing` row
IDX_BIN: Final[int] = 1
#: the index of the left x coordinate in a :class:`Packing` row
IDX_LEFT_X: Final[int] = 2
#: the index of the bottom y coordinate in a :class:`Packing` row
IDX_BOTTOM_Y: Final[int] = 3
#: the index of the right x coordinate in a :class:`Packing` row
IDX_RIGHT_X: Final[int] = 4
#: the index of the top y coordinate in a :class:`Packing` row
IDX_TOP_Y: Final[int] = 5


class Packing(Component, np.ndarray):
    """
    A packing, i.e., a solution to an 2D bin packing instance.

    A packing is a two-dimensional numpy array. In each row, the position of
    one item is stored: 1. the item ID (starts at 1), 2. the bin into which
    the item is packaged (starts at 1), 3. the left x coordinate, 4. the
    bottom y coordinate, 5. the right x coordinate, 6. the top y coordinate.
    """

    #: the 2d bin packing instance
    instance: Instance
    #: the number of bins
    n_bins: int

    def __new__(cls, instance: Instance) -> "Packing":
        """
        Create an solution record for the 2D bin packing problem.

        :param cls: the class
        :param instance: the solution record
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        obj: Final[Packing] = super().__new__(
            cls, (instance.n_items, 6), instance.dtype)
        #: the 2d bin packing instance
        obj.instance = instance
        #: the number of bins
        obj.n_bins = -1
        return obj

    def __str__(self):
        """
        Convert the packing to a compact string.

        :return: the compact string
        """
        return "\n".join(";".join(str(self[i, j]) for j in range(6))
                         for i in range(self.shape[0]))

    @staticmethod
    def from_log(file: str, instance: Instance | None = None) -> "Packing":
        """
        Load a packing from a log file.

        :param file: the log file path
        :param instance: the optional Packing instance: if `None` is provided,
            we try to load it from the resources
        :returns: the Packing
        """
        parser: Final[_PackingParser] = _PackingParser(instance)
        parser.parse_file(file)
        # noinspection PyProtectedMember
        res = parser._result
        if not isinstance(res, Packing):
            raise type_error(res, f"packing from {file!r}", Packing)
        return res


class _PackingParser(LogParser):
    """The log parser for loading packings."""

    def __init__(self, instance: Instance | None = None):
        """
        Create the packing parser.

        :param instance: the optional packing instance: if `None` is provided,
            we try to load it from the resources
        """
        super().__init__()
        if (instance is not None) and (not isinstance(instance, Instance)):
            raise type_error(instance, "instance", Instance)
        #: the internal instance
        self.__instance: Instance | None = instance
        #: the internal section mode: 0=none, 1=setup, 2=y
        self.__sec_mode: int = 0
        #: the packing string
        self.__packing_str: str | None = None
        #: the result packing
        self._result: Packing | None = None
        #: the used objective, this is used for packing result parsing
        self._used_objective: str | None = None

    def start_section(self, title: str) -> bool:
        """Start a section."""
        super().start_section(title)
        self.__sec_mode = 0
        if title == SECTION_SETUP:
            if self.__instance is None:
                self.__sec_mode = 1
                return True
            return False
        if title == SECTION_RESULT_Y:
            self.__sec_mode = 2
            return True
        return False

    def lines(self, lines: list[str]) -> bool:
        """Parse the lines."""
        if self.__sec_mode == 1:
            if self.__instance is not None:
                raise ValueError(
                    f"instance is already set to {self.__instance}.")
            key_1: Final[str] = "y.inst.name: "
            key_2: Final[str] = "f.name: "
            for line in lines:
                if line.startswith(key_1):
                    self.__instance = Instance.from_resource(
                        line[len(key_1):].strip())
                elif line.startswith(key_2):
                    self._used_objective = line[len(key_2):].strip()
            if self.__instance is None:
                raise ValueError(f"Did not find instance key {key_1!r} "
                                 f"in section {SECTION_SETUP}!")
            if self._used_objective is None:
                raise ValueError(f"Did not find instance key {key_2!r} "
                                 f"in section {SECTION_SETUP}!")
        elif self.__sec_mode == 2:
            self.__packing_str = " ".join(lines).strip()
        else:
            raise ValueError("Should not be in section?")
        return (self.__instance is None) or (self.__packing_str is None)

    def end_file(self) -> bool:
        """End the file."""
        if self.__packing_str is None:
            raise ValueError(f"Section {SECTION_RESULT_Y} missing!")
        if self.__instance is None:
            raise ValueError(f"Section {SECTION_SETUP} missing or empty!")
        if self._result is not None:
            raise ValueError("Applied parser to more than one log file?")
        # pylint: disable=C0415,R0401
        from moptipyapps.binpacking2d.packing_space import (
            PackingSpace,  # pylint: disable=C0415,R0401
        )

        self._result = PackingSpace(self.__instance)\
            .from_str(self.__packing_str)
        self.__packing_str = None
        return False
