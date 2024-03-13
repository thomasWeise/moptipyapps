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
from typing import Any, Callable, Collection, Final

import matplotlib as mpl  # type: ignore
import moptipy.utils.plot_utils as pu
import numpy as np
from matplotlib.axes import Axes  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.lines import Line2D  # type: ignore
from moptipy.utils.strings import float_to_str
from mpl_toolkits.mplot3d.art3d import Line3D  # type: ignore
from pycommons.io.console import logger
from pycommons.io.path import Path
from pycommons.types import check_int_range


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


#: the color maps to use
__COLOR_MAPS: Final[tuple[str, str, str, str, str, str, str, str, str]] = (
    "spring", "winter", "copper", "plasma", "cool", "hsv", "winter",
    "coolwarm", "BrBG")


def _get_colors(alen: int, cmi: int = 0, multi: bool = False) -> Callable[
        [float], tuple[float, float, float, float]]:
    """
    Get a given color map with the given number of columns.

    For a given color map index (`cmi`), this function returns a callable that
    maps values from `[0,1]` to colors. If this is supposed to be the only
    color map in use (`m̀ulti == False`), then it will return the standard
    `spring` color map. Otherwise, it will return different color maps
    depending on the color map index `cmi`.

    :param alen: the number of colors
    :param cmi: the color map name index
    :param multi: are there multiple colors
    :return: the colors
    """
    if multi:
        if cmi == 0:  # pink
            return lambda x: (0.18 + 0.8 * x, 0.0, 0.18 + 0.8 * x, 1.0)
        if cmi == 1:  # green
            return lambda x: (0.0, 0.18 + 0.8 * x, 0.0, 1.0)
        if cmi == 2:  # blue
            return lambda x: (0.0, 0.0, 0.18 + 0.8 * x, 1.0)
        cmi -= 2

    cm: Final[str] = __COLOR_MAPS[
        check_int_range(cmi, "color_map_index", 0, 10000) % len(__COLOR_MAPS)]
    obj: Final = _get_colors

    if hasattr(obj, cm):
        the_colors: dict[int, Callable[[
            float], tuple[float, float, float, float]]] = getattr(obj, cm)
    else:
        the_colors = {}
        setattr(obj, cm, the_colors)
    if alen in the_colors:
        return the_colors[alen]
    colors: Callable[[float], tuple[float, float, float, float]] \
        = mpl.colormaps[cm].resampled(alen)
    the_colors[alen] = colors
    return colors


#: the start point color
START_COLOR: Final[str] = "blue"
#: the start marker
START_MARKER: Final[str] = "++"
#: the end point color
END_COLOR: Final[str] = "red"
#: the end marker
END_MARKER: Final[str] = "o\u25CF"


def _plot_2d(state: np.ndarray, axes: Axes,
             xi: int, yi: int,
             colors: Callable[[float], tuple[float, float, float, float]],
             z: int) -> int:
    """
    Plot a figure in 2D.

    :param state: the state matrix
    :param axes: the plot
    :param xi: the x-index
    :param yi: the y-index
    :param colors: the colors to use
    :param z: the z index
    :returns: the new z index
    """
    size: Final[int] = len(state) - 1
    for i in range(size):
        z += 1
        axes.add_artist(Line2D(state[i:i + 2, xi],
                               state[i:i + 2, yi],
                        color=colors(i / size), zorder=z))
    z += 1
    axes.scatter(state[0, xi], state[0, yi], color=START_COLOR,
                 zorder=z, marker=START_MARKER[0])
    z += 1
    axes.scatter(state[-1, xi], state[-1, yi], color=END_COLOR,
                 zorder=z, marker=END_MARKER[0])
    return z + 1


def _plot_3d(state: np.ndarray, axes: Axes,
             xi: int, yi: int, zi: int,
             colors: Callable[[float], tuple[float, float, float, float]],
             z: int, ranges: list[float]) -> int:
    """
    Plot a figure in 3D.

    :param state: the state matrix
    :param axes: the plot
    :param xi: the x-index
    :param yi: the y-index
    :param yi: the z-index
    :param colors: the colors to use
    :param z: the z index
    :param ranges: the axes ranges
    :returns: the new z index
    """
    if hasattr(axes, "force_zorder"):
        axes.force_zorder = True

    size: Final[int] = len(state) - 1
    for i in range(size):
        z += 1
        axes.add_artist(Line3D(state[i:i + 2, xi],
                               state[i:i + 2, yi],
                               state[i:i + 2, zi],
                        color=colors(i / size), zorder=z))

    col: str = START_COLOR
    for p in [state[0], state[-1]]:
        w: float = 0.04 * ranges[0]
        z += 1
        axes.add_artist(Line3D((p[xi] - w, p[xi] + w),
                               (p[yi], p[yi]),
                               (p[zi], p[zi]),
                        color=col, zorder=z))
        z += 1
        w = 0.04 * ranges[1]
        axes.add_artist(Line3D((p[xi], p[xi]),
                               (p[yi] - w, p[yi] + w),
                               (p[zi], p[zi]),
                        color=col, zorder=z))
        z += 1
        w = 0.04 * ranges[2]
        axes.add_artist(Line3D((p[xi], p[xi]),
                               (p[yi], p[yi]),
                               (p[zi] - w, p[zi] + w),
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
                 plot_indices: Collection[int] = (0, ),
                 state_dim_mod: int = 0) -> None:
        """
        Create the ODE plotter.

        :param dest_file: the path to the destination file
        :param sup_title: the title for the figure
        :param state_dims: the state dimensions
        :param plot_indices: the indices that are supposed to be plotted
        :param state_dim_mod: the modulus for the state dimensions
        """
        super().__init__()
        #: the destination file
        self.__dest_file: Final[Path] = Path(dest_file)
        logger(f"plotting data to file {self.__dest_file!r}.")
        #: the state dimensions
        self.__state_dims: Final[int] = check_int_range(
            state_dims, "state_dimes", 2, 1_000_000)
        #: The modulus for the state dimensions for plotting
        self.__state_dim_mod: Final[int] = check_int_range(
            state_dim_mod, "state_dim_mod", 0, state_dims)

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
        use_dims: Final[int] = state_dims if state_dim_mod <= 0 \
            else state_dim_mod
        if use_dims >= 3:
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
        state_dims: Final[int] = self.__state_dims
        state_dim_mod: Final[int] = self.__state_dim_mod
        use_dims: Final[int] = state_dims if state_dim_mod <= 0 \
            else state_dim_mod

        # make the appropriate title
        v1: str = _vec(ode[0, 0:state_dims])
        v2: str = _vec(ode[-1, 0:state_dims])
        if use_dims == 2:
            v1 = f"{START_MARKER[1]}{v1}"
            v2 = f"{END_MARKER[1]}{v2}"
        lv1: Final[int] = len(v1)
        lv2: Final[int] = len(v2)
        title: str = f"{v1}\u2192{v2}" if ((lv1 + lv2) < 40) else (
            f"{v1}\u2192\n{v2}" if lv1 <= lv2 else f"{v1}\n\u2192{v2}")
        title = f"{title}\nJ={j}, T={time:.4f}"

        state_min: Final[list[float]] = [
            ode[:, i:state_dims:state_dim_mod].min() for i in range(use_dims)]
        state_max: Final[list[float]] = [
            ode[:, i:state_dims:state_dim_mod].max() for i in range(use_dims)]
        ranges: Final[list[float]] = [
            state_max[i] - state_min[i] for i in range(use_dims)]

        can_plot: bool = all(
            isfinite(state_min[i]) and (state_min[i] > -1e20)
            and isfinite(state_max[i]) and (state_max[i] < 1e20)
            and (state_min[i] < state_max[i]) and isfinite(ranges[i])
            and (0 < ranges[i] < 1e20) for i in range(use_dims))

        if can_plot:
            if (max(1e-20, min(ranges)) / max(ranges)) > 0.25:
                axes.set_aspect("equal")
            axes.set_xlim(state_min[0], state_max[0])
            axes.set_ylim(state_min[1], state_max[1])
            if (use_dims > 2) and hasattr(axes, "set_zlim"):
                axes.set_zlim(state_min[2], state_max[2])

            size: Final[int] = len(ode) - 2
            for ci, i in enumerate(range(0, state_dims, use_dims)):
                colors: Callable[[float], tuple[
                    float, float, float, float]] = _get_colors(
                    size, ci, 0 < state_dim_mod < state_dims)
                if use_dims == 3:
                    self.__z = _plot_3d(ode, axes, i, i + 1, i + 2, colors,
                                        self.__z, ranges)
                elif use_dims == 2:
                    self.__z = _plot_2d(ode, axes, i, i + 1, colors,
                                        self.__z)
                else:
                    raise ValueError(f"Huh? state_dim={state_dims}, "
                                     f"state_dim_mod={state_dim_mod}??")
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
