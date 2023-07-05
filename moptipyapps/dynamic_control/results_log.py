"""A logger for results gathered from ODE integration to a text file."""

from contextlib import AbstractContextManager
from io import TextIOBase
from typing import Callable, Final

import numpy as np
from moptipy.utils.console import logger
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.path import Path
from moptipy.utils.strings import float_to_str


class ResultsLog(AbstractContextManager):
    """
    A class for logging results via `multi_run_ode`.

    Function :func:`moptipyapps.dynamic_control.ode.multi_run_ode` can pass
    its results to various output generating procedures. This class here
    offers a procedure for writing them to a log file.

    >>> def projectile(position, ttime, ctrl, out):
    ...     out[0] = 70.71067811865474
    ...     out[1] = 70.71067811865474 - ttime * 9.80665
    >>> param: np.ndarray = np.array([1])   # ignore
    >>> def contrl(position, ttime, params, dest):
    ...     dest[0] = params[0]  #  controller that returns param
    >>> from io import StringIO
    >>> from moptipyapps.dynamic_control.ode import multi_run_ode
    >>> with StringIO() as sio:
    ...     with ResultsLog(2, sio) as log:
    ...         multi_run_ode([np.array([0.0, 1.0])],
    ...                       [np.array([1.0, 1.0])],
    ...                       log.collector, projectile, contrl, param,
    ...                       1, 10000, 10000)
    ...         x=sio.getvalue()
    >>> tt = x.split()
    >>> print(tt[0])
    figureOfMerit;totalTime;nSteps;start0;start1;end0;end1
    >>> for y in tt[1:]:
    ...     print(";".join(f"{float(v):.3f}" for v in y.split(";")))
    403386.481;14.890;10000.000;0.000;1.000;1052.875;-33.125
    407847.896;14.961;10000.000;1.000;1.000;1058.933;-38.536
    """

    def __init__(self, state_dim: int, out: TextIOBase | str) -> None:
        """
        Create the test results logger.

        :param state_dim: the state dimension
        :param out: the output destination
        """
        super().__init__()
        if isinstance(out, str):
            pp = Path.path(out)
            logger(f"logging data to file {pp!r}.")
            out = pp.open_for_write()
        #: the internal output destination
        self.__out: TextIOBase = out
        #: the state dimension
        self.__state_dim: Final[int] = state_dim

    def collector(self, index: int, ode: np.ndarray,
                  j: float, time: float) -> None:
        """
        Log the result of a multi-ode run.

        :param index: the index of the result
        :param ode: the ode result matrix
        :param j: the figure of merit
        :param time: the time value
        """
        if self.__out is None:
            raise ValueError("Already closed output destination!")
        out_str: Final[Callable[[str], int]] = self.__out.write

        state_dim: Final[int] = self.__state_dim
        if index <= 0:
            out_str(
                f"figureOfMerit{CSV_SEPARATOR}totalTime{CSV_SEPARATOR}nSteps")
            for i in range(state_dim):
                out_str(f"{CSV_SEPARATOR}start{i}")
            for i in range(state_dim):
                out_str(f"{CSV_SEPARATOR}end{i}")
            out_str("\n")

        out_str(float_to_str(float(j)))
        out_str(CSV_SEPARATOR)
        out_str(float_to_str(float(time)))
        out_str(CSV_SEPARATOR)
        out_str(str(len(ode)))
        start = ode[0]
        for i in range(state_dim):
            out_str(f"{CSV_SEPARATOR}{float_to_str(float(start[i]))}")
        end = ode[-1]
        for i in range(state_dim):
            out_str(f"{CSV_SEPARATOR}{float_to_str(float(end[i]))}")
        out_str("\n")

    def __exit__(self, _, __, ___) -> None:
        """
        Close this context manager.

        :param _: the exception type; ignored
        :param __: the exception value; ignored
        :param ___: the exception whatever; ignored
        """
        if self.__out is not None:
            try:
                self.__out.close()
            finally:
                self.__out = None
