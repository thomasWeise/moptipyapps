"""A simple code generator."""

from io import StringIO
from typing import Any, Callable, Final

import numba  # type: ignore
import numpy as np
from moptipy.utils.types import type_error


class CodeGenerator:
    """A simple code generator."""

    def __init__(self, args: str = "", retval: str = "None",
                 fastmath: bool = True) -> None:
        """
        Initialize the code generator.

        :param args: the command line arguments
        :param retval: the return type
        :param fastmath: do we use numba fast math?
        """
        if not isinstance(args, str):
            raise type_error(args, "args", str)
        if not isinstance(retval, str):
            raise type_error(retval, "retval", str)
        if not isinstance(fastmath, bool):
            raise type_error(fastmath, "fastmath", bool)

        io: Final[StringIO] = StringIO()
        wrt: Final[Callable[[str], int]] = io.write

        #: the callable for writing
        self.__write: Final[Callable[[str], int]] = wrt
        wrt("@numba.njit(cache=False, inline='always', fastmath"
            f"={fastmath}, boundscheck=False)\n")
        wrt(f"def ____func({args}) -> {retval}:\n")

        #: the result callable
        self.__res: Final[Callable[[], str]] = io.getvalue
        #: the current indent
        self.__indent: int = 1
        #: are we at the start of a line?
        self.__start: bool = True

    def write(self, text: str) -> None:
        """
        Write some code.

        :param text: the code text
        """
        if self.__start:
            self.__write(self.__indent * "    ")
            self.__start = False
        self.__write(text)

    def indent(self) -> None:
        """Increase the indent."""
        self.__indent += 1

    def unindent(self) -> None:
        """Increase the indent."""
        self.__indent -= 1
        if self.__indent <= 0:
            raise ValueError("indent becomes 0, not allowed")

    def writeln(self, text: str = "") -> None:
        """
        End a line.

        :param text: the text to be written
        """
        self.write(text)
        self.endline()

    def endline(self) -> None:
        """End a line."""
        if not self.__start:
            self.__write("\n")
            self.__start = True

    def build(self) -> Callable:
        """
        Compile the generated code.

        :returns: the generated function
        """
        loca: Final[dict[str, Any]] = {}
        self.endline()
        code: Final[str] = self.__res()
        globs: Final[dict[str, Any]] = {"numba": numba, "np": np,
                                        "Final": Final}
        # pylint: disable=W0122
        exec(code, globs, loca)  # nosec # nosemgrep # noqa
        res = loca["____func"]
        if not callable(res):
            raise type_error(res, f"compiled {code!r}", call=True)
        return res
