"""Plot a packing into one figure."""
from collections import Counter
from typing import Callable, Final, Iterable

import moptipy.utils.plot_defaults as pd
import moptipy.utils.plot_utils as pu
from matplotlib.artist import Artist  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
from matplotlib.text import Text  # type: ignore
from moptipy.utils.types import type_error

from moptipyapps.binpacking2d.packing import (
    IDX_BIN,
    IDX_BOTTOM_Y,
    IDX_ID,
    IDX_LEFT_X,
    IDX_RIGHT_X,
    IDX_TOP_Y,
    Packing,
)


def default_packing_item_str(item_id: int, item_index: int,
                             item_in_bin_index: int) -> Iterable[str]:
    """
    Get a packing item string(s).

    The default idea is to include the item id, the index of the item in the
    bin, and the overall index of the item. If the space is insufficient,
    we remove the latter or the latter two. Hence, this function returns a
    tuple of three strings.

    :param item_id: the ID of the packing item
    :param item_index: the item index
    :param item_in_bin_index: the index of the item in its bin
    :return: the string
    """
    return (f"{item_id}/{item_in_bin_index}/{item_index}",
            f"{item_id}/{item_in_bin_index}", str(item_id))


def plot_packing(packing: Packing | str,
                 max_rows: int = 3,
                 max_bins_per_row: int = 3,
                 default_width_per_bin: float | int | None = 8.6,
                 max_width: float | int | None = 8.6,
                 default_height_per_bin: float | int | None =
                 5.315092303249095,
                 max_height: float | int | None = 9,
                 packing_item_str: Callable[
                     [int, int, int], str | Iterable[str]] =
                 default_packing_item_str,
                 importance_to_font_size_func: Callable[[int], float] =
                 pd.importance_to_font_size,
                 dpi: float | int | None = 384.0) -> Figure:
    """
    Plot a packing.

    Each item is drawn in a different color. Each item rectangle includes, if
    there is enough space, the item-ID. If there is more space, also the index
    of the item inside the bin (starting at 1) is included. If there is yet
    more space, even the overall index of the item is included.

    :param packing: the packing or the file to load it from
    :param max_rows: the maximum number of rows
    :param max_bins_per_row: the maximum number of bins per row
    :param default_width_per_bin: the optional default width of a column
    :param max_height: the maximum height
    :param default_height_per_bin: the optional default height per row
    :param max_width: the maximum width
    :param packing_item_str: the function converting an item id,
        item index, and item-in-bin index to a string or sequence of strings
        (of decreasing length)
    :param importance_to_font_size_func: the function converting
        importance values to font sizes
    :param dpi: the dpi value
    :returns: the Figure object to allow you to add further plot elements
    """
    if isinstance(packing, str):
        packing = Packing.from_log(packing)
    if not isinstance(packing, Packing):
        raise type_error(packing, "packing", (Packing, str))
    if not callable(packing_item_str):
        raise type_error(packing_item_str, "packing_item_str", call=True)

    # allocate the figure ... this is hacky for now
    figure, bin_figures = pu.create_figure_with_subplots(
        items=packing.n_bins, max_items_per_plot=1, max_rows=max_rows,
        max_cols=max_bins_per_row, min_rows=1, min_cols=1,
        default_width_per_col=default_width_per_bin, max_width=max_width,
        default_height_per_row=default_height_per_bin, max_height=max_height,
        dpi=dpi)

    # initialize the different plots
    bin_width: Final[int] = packing.instance.bin_width
    bin_height: Final[int] = packing.instance.bin_height
    for the_axes, _, _, _, _, _ in bin_figures:
        axes = pu.get_axes(the_axes)
        axes.set_ylim(0, bin_width)  # pylint: disable=E1101
        axes.set_ybound(0, bin_height)  # pylint: disable=E1101
        axes.set_xlim(0, bin_width)  # pylint: disable=E1101
        axes.set_xbound(0, bin_width)  # pylint: disable=E1101
        axes.set_aspect("equal", None, "C")  # pylint: disable=E1101
        axes.tick_params(  # pylint: disable=E1101
            left=False, bottom=False, labelleft=False, labelbottom=False)

    # get the color and font styles
    colors: Final[tuple] = pd.distinct_colors(
        packing.instance.n_different_items)
    font_size: Final[float] = importance_to_font_size_func(-1)

    # get the transforms needed to obtain text dimensions
    renderers: Final[list] = [
        pu.get_renderer(axes) for axes, _, _, _, _, _ in bin_figures]
    inverse: Final[list] = [axes.transData.inverted()  # pylint: disable=E1101
                            for axes, _, _, _, _, _ in bin_figures]

    z_order: int = 0  # the z-order of all drawing elements

    # we now plot the items one-by-one
    bin_counters: Counter[int] = Counter()
    for item_index in range(packing.instance.n_items):
        item_id: int = packing[item_index, IDX_ID]
        item_bin: int = packing[item_index, IDX_BIN]
        x_left: int = packing[item_index, IDX_LEFT_X]
        y_bottom: int = packing[item_index, IDX_BOTTOM_Y]
        x_right: int = packing[item_index, IDX_RIGHT_X]
        y_top: int = packing[item_index, IDX_TOP_Y]
        item_in_bin_index: int = bin_counters[item_bin] + 1
        bin_counters[item_bin] = item_in_bin_index
        width: int = x_right - x_left
        height: int = y_top - y_bottom

        background = colors[item_id - 1]
        foreground = pd.text_color_for_background(colors[item_id - 1])

        axes = pu.get_axes(bin_figures[item_bin - 1][0])
        rend = renderers[item_bin - 1]
        inv = inverse[item_bin - 1]

        axes.add_artist(Rectangle(  # paint the item's rectangle
            xy=(x_left, y_bottom), width=width, height=height,
            facecolor=background, linewidth=0.75, zorder=z_order,
            edgecolor="black"))
        z_order += 1
        x_center: float = 0.5 * (x_left + x_right)
        y_center: float = 0.5 * (y_bottom + y_top)

# get the box label string or string sequence
        strs = packing_item_str(item_id, item_index + 1, item_in_bin_index)
        if isinstance(strs, str):
            strs = [strs]
        elif not isinstance(strs, Iterable):
            raise type_error(
                strs, f"packing_item_str({item_id}, {item_index}, "
                      f"{item_in_bin_index})", (str, Iterable))

# iterate over the possible box label strings
        for i, item_str in enumerate(strs):
            if not isinstance(item_str, str):
                raise type_error(
                    str, f"packing_item_str({item_id}, {item_index}, "
                         f"{item_in_bin_index})[{i}]", str)
# Get the size of the text using a temporary text  that gets immediately
# deleted again.
            tmp: Text = axes.text(x=x_center, y=y_center, s=item_str,
                                  fontsize=font_size,
                                  color=foreground,
                                  horizontalalignment="center",
                                  verticalalignment="baseline")
            bb_bl = inv.transform_bbox(tmp.get_window_extent(
                renderer=rend))
            Artist.set_visible(tmp, False)
            Artist.remove(tmp)
            del tmp

# Check if this text did fit into the rectangle.
            if (bb_bl.width < (0.97 * width)) and \
                    (bb_bl.height < (0.97 * height)):
                # OK, there is enough space. Let's re-compute the y offset
                # to do proper alignment using another temporary text.
                tmp = axes.text(x=x_center, y=y_center, s=item_str,
                                fontsize=font_size,
                                color=foreground,
                                horizontalalignment="center",
                                verticalalignment="bottom")
                bb_bt = inv.transform_bbox(tmp.get_window_extent(
                    renderer=rend))
                Artist.set_visible(tmp, False)
                Artist.remove(tmp)
                del tmp

# Now we can really print the actual text with a more or less  nice vertical
# alignment.
                adj = bb_bl.y0 - bb_bt.y0
                if adj < 0:
                    y_center += adj / 3

                axes.text(x=x_center, y=y_center, s=item_str,
                          fontsize=font_size,
                          color=foreground,
                          horizontalalignment="center",
                          verticalalignment="center",
                          zorder=z_order)
                z_order += 1
                break  # We found a text that fits, so we can quit.
    return figure
