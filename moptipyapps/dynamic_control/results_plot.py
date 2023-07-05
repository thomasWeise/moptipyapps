"""
An illustrator for results gathered from ODE integration to multi-plots.

This evaluator will create figures where each sub-plot corresponds to one
evolution of the system over time. The starting state is marked with a blue
cross, the final state with a red one. Both states are connected with a line
that marks the progress over time. The line begins with blue color and changes
its color over violet/pinkish towards yellow the farther in time the system
progresses.
"""

from contextlib import AbstractContextManager
from math import isfinite, sqrt
from os.path import basename, dirname
from typing import Any, Collection, Final

import matplotlib as mpl  # type: ignore
import moptipy.utils.plot_utils as pu
import numpy as np
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.lines import Line2D  # type: ignore
from moptipy.utils.console import logger
from moptipy.utils.path import Path
from moptipy.utils.strings import float_to_str
from moptipy.utils.types import check_to_int_range
from mpl_toolkits.mplot3d.art3d import Line3D  # type: ignore


def _str(f: float) -> str:
    """
    Convert a floating point number to a short string.

    :param f: the number
    :return: the string
    """
    f = float(f)
    a1 = float_to_str(f)
    a2 = f"{f:.2f}"
    while a2.endswith("0"):
        a2 = a2[:-1]
    if a2.endswith("."):
        a2 = a2[:-1]
    if (a1 == "-0") or (a2 == "-0"):
        return "0"
    return a1 if len(a1) < len(a2) else a2


def _vec(v: np.ndarray) -> str:
    """
    Convert a vector to a quick string.

    :param v: the vector
    :return: the string
    """
    s = ",".join(_str(vv) for vv in v)
    return f"({s})"


def _get_colors(alen: int, cm: str = "plasma") -> list:
    """
    Get a given number of colors.

    :param alen: the number of colors
    :param cm: the color map name
    :return: the colors
    """
    obj: Final = _get_colors
    if hasattr(obj, cm):
        the_colors: dict[int, list] = getattr(obj, cm)
    else:
        the_colors = {}
        setattr(obj, cm, the_colors)
    if alen in the_colors:
        return the_colors[alen]
    colors = mpl.colormaps[cm].resampled(alen).colors
    the_colors[alen] = colors
    return colors


#: the start point color
START_COLOR: Final[str] = "blue"
#: the end point color
END_COLOR: Final[str] = "red"


def _plot_2d(state: np.ndarray, axes: Axes,
             colors: list, z: int) -> int:
    """
    Plot a figure in 2D.

    :param state: the state matrix
    :param axes: the plot
    :param colors: the colors to use
    :param z: the z index
    :returns: the new z index
    """
    for i in range(len(state) - 1):
        z += 1
        axes.add_artist(Line2D(state[i:i + 2, 0],
                               state[i:i + 2, 1],
                        color=colors[i], zorder=z))
    z += 1
    axes.scatter(state[0, 0], state[0, 1], color=START_COLOR,
                 zorder=z, marker="+")
    z += 1
    axes.scatter(state[-1, 0], state[-1, 1], color=END_COLOR,
                 zorder=z, marker="+")
    return z + 1


def _plot_3d(state: np.ndarray, axes: Axes,
             colors: list, z: int, ranges: list[float]) -> int:
    """
    Plot a figure in 2D.

    :param state: the state matrix
    :param axes: the plot
    :param colors: the colors to use
    :param z: the z index
    :param ranges: the axes ranges
    :returns: the new z index
    """
    axes.force_zorder = True

    for i in range(len(state) - 1):
        z += 1
        axes.add_artist(Line3D(state[i:i + 2, 0],
                               state[i:i + 2, 1],
                               state[i:i + 2, 2],
                        color=colors[i], zorder=z))

    col: str = START_COLOR
    for p in [state[0], state[-1]]:
        w: float = 0.04 * ranges[0]
        z += 1
        axes.add_artist(Line3D((p[0] - w, p[0] + w),
                               (p[1], p[1]),
                               (p[2], p[2]),
                        color=col, zorder=z))
        z += 1
        w = 0.04 * ranges[1]
        axes.add_artist(Line3D((p[0], p[0]),
                               (p[1] - w, p[1] + w),
                               (p[2], p[2]),
                        color=col, zorder=z))
        z += 1
        w = 0.04 * ranges[2]
        axes.add_artist(Line3D((p[0], p[0]),
                               (p[1], p[1]),
                               (p[2] - w, p[2] + w),
                        color=col, zorder=z))
        col = END_COLOR
    return z + 1


class ResultsPlot(AbstractContextManager):
    """
    A class for plotting results via `multi_run_ode`.

    Function :func:`moptipyapps.dynamic_control.ode.multi_run_ode` can pass
    its results to various output generating procedures. This class here
    offers a procedure for plotting them to a file.
    """

    def __init__(self, dest_file: str, sup_title: str | None,
                 state_dims: int,
                 plot_indices: Collection[int] = (0, )) -> None:
        """
        Create the ODE plotter.

        :param dest_file: the path to the destination file
        :param sup_title: the title for the figure
        :param state_dims: the state dimensions
        :param plot_indices: the indices that are supposed to be plotted
        """
        super().__init__()
        #: the destination file
        self.__dest_file: Final[Path] = Path.path(dest_file)
        logger(f"plotting data to file {self.__dest_file!r}.")
        #: the state dimensions
        self.__state_dims: Final[int] = check_to_int_range(
            state_dims, "state_dimes", 2, 3)

        total_plots: Final[int] = len(plot_indices)
        #: the plot indexes
        if not (0 < total_plots <= 9):
            raise ValueError(f"Invalid plot indices {plot_indices!r}, "
                             f"contains {total_plots} elements.")
        #: the plot indices
        self.__plot_indices: Final[Collection[int]] = plot_indices

        srt: Final[int] = max(3, int(round(1 + sqrt(total_plots))))
        args: Final[dict[str, Any]] = {
            "items": total_plots,
            "max_items_per_plot": 1,
            "max_rows": srt,
            "max_cols": srt,
        }
        if state_dims >= 3:
            args["plot_config"] = {"projection": "3d"}

        figure, plots = pu.create_figure_with_subplots(**args)
        #: the figure
        self.__figure: Figure | None = figure
        #: the plots
        self.__subplots: None | tuple[tuple[
            Axes | Figure, int, int, int, int, int], ...] = plots
        #: the current plot
        self.__cur_plot: int = 0
        #: the z-order
        self.__z: int = 0

        if sup_title is not None:
            figure.suptitle(sup_title)

    def collector(self, index: int, ode: np.ndarray,
                  j: float, time: float) -> None:
        """
        Plot the result of a multi-ode run.

        :param index: the index of the result
        :param ode: the ode result matrix
        :param j: the figure of merit
        :param time: the time value
        """
        if self.__figure is None:
            raise ValueError("Already closed output figure!")
        if index not in self.__plot_indices:
            return

        axes: Final[Axes] = pu.get_axes(
            self.__subplots[self.__cur_plot][0])
        state_dim: Final[int] = self.__state_dims

        title: str = (f"{_vec(ode[0, 0:state_dim])}\u2192"
                      f"{_vec(ode[-1, 0:state_dim])}\n"
                      f"J={j}, T={time:.4f}")

        state_min: Final[list[float]] = [
            ode[:, i].min() for i in range(state_dim)]
        state_max: Final[list[float]] = [
            ode[:, i].max() for i in range(state_dim)]
        ranges: Final[list[float]] = [
            state_max[i] - state_min[i] for i in range(state_dim)]

        can_plot: bool = all(
            isfinite(state_min[i]) and (state_min[i] > -1e20)
            and isfinite(state_max[i]) and (state_max[i] < 1e20)
            and (state_min[i] < state_max[i]) and isfinite(ranges[i])
            and (0 < ranges[i] < 1e20) for i in range(state_dim))

        if can_plot:
            if (max(1e-20, min(ranges)) / max(ranges)) > 0.25:
                axes.set_aspect("equal")
            axes.set_xlim(state_min[0], state_max[0])
            axes.set_ylim(state_min[1], state_max[1])
            if state_dim > 2:
                axes.set_zlim(state_min[2], state_max[2])
            colors: Final[list] = _get_colors(len(ode) - 1)

            if state_dim == 2:
                self.__z = _plot_2d(ode, axes, colors, self.__z)
            elif state_dim == 3:
                self.__z = _plot_3d(ode, axes, colors, self.__z, ranges)
            else:
                raise ValueError(f"Huh? state_dim={state_dim}??")
        else:
            title = f"{title}\n<cannot plot>"

        axes.set_title(title)
        self.__cur_plot += 1

    def __exit__(self, _, __, ___) -> None:
        """
        Close this context manager.

        :param _: the exception type; ignored
        :param __: the exception value; ignored
        :param ___: the exception whatever; ignored
        """
        if self.__figure is not None:
            try:
                the_dir: Final[str] = dirname(self.__dest_file)
                the_file: Final[str] = basename(self.__dest_file)
                dot: Final[int] = the_file.rindex(".")
                prefix: Final[str] = the_file[:dot]
                suffix: Final[str] = the_file[dot + 1:]
                pu.save_figure(self.__figure, prefix, the_dir, suffix)
                logger(f"finished plotting data to {self.__dest_file!r}.")
            finally:
                self.__figure = None
                self.__subplots = None
