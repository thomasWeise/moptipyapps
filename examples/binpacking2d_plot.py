"""
Plot a 2-dimensional packing.

Here we try out the two different variants of the improved-bottom-left
encoding. `ImprovedBottomLeftEncoding1` inserts objects one by one and moves
on to a new bin once an object doesn't fit into the current bin. It will then
never consider the filled bin again. `ImprovedBottomLeftEncoding2` always
tries all bins for any object.

We try both encodings for instance `a10` and fixed (but randomly chosen)
signed permutation as input string.

You will see that `ImprovedBottomLeftEncoding1` needs four bins, whereas
`ImprovedBottomLeftEncoding2` needs three.

In the file `binpacking2d_rls.py`, we plug `ImprovedBottomLeftEncoding2` into
a simple randomized local search. This search runs for 1024 steps and finds a
packing that only needs two bins.
"""
import os
from time import sleep
from webbrowser import open_new_tab

import numpy as np
import psutil
from moptipy.utils.nputils import rand_generator
from moptipy.utils.plot_utils import save_figure
from moptipy.utils.temp import TempDir

from moptipyapps.binpacking2d.ibl_encoding_1 import (
    ImprovedBottomLeftEncoding1,
)
from moptipyapps.binpacking2d.ibl_encoding_2 import (
    ImprovedBottomLeftEncoding2,
)
from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.packing_space import PackingSpace
from moptipyapps.binpacking2d.plot_packing import plot_packing

# We do not show the generated graphics in the browser if this script is
# called from a "make" build. This small lambda checks whether there is any
# process with "make" in its name anywhere in the parent hierarchy of the
# current process.
ns = lambda prc: False if prc is None else (  # noqa: E731
    "make" in prc.name() or ns(prc.parent()))

# should we show the plots?
SHOW_PLOTS_IN_BROWSER = not ns(psutil.Process(os.getppid()))

random = rand_generator(3)  # get a random number generator
instance = Instance.from_resource("a10")  # pick instance a10
space = PackingSpace(instance)  # create the packing space

# We test two versions of the Improved Bottom Left Encoding
encodings = [ImprovedBottomLeftEncoding1(instance),  # the 1st encoding
             ImprovedBottomLeftEncoding2(instance)]  # the 2nd encoding

# The data that the encoding can convert to a packing is an array with
# the same length as the total number n_items of objects to pack. Now a
# packing instance has n_different_items different items, each of which
# can occur multiple times, i.e., may occur repeatedly.
# So we include each of n_different_items item IDs exactly as often as
# it should be repeated. We sometimes include it directly and sometimes
# in the negated form, which is interpreted as a 90° rotation by the encoding.
# Then, we convert the list of item IDs to a numpy array and, finally, shuffle
# the array. The encoding then inserts the items in the same sequence they
# appear in the array into bins.

# generate the data for a random packing
x_data = instance.get_standard_item_sequence()
for i, e in enumerate(x_data):
    if random.integers(2) != 0:
        x_data[i] = -e
x = np.array(x_data, instance.dtype)  # convert the data to a numpy array
random.shuffle(x)  # shuffle the data, i.e., the insertion sequence

# The packing is a two-dimensional matrix of items.
# Each row corresponds to one item. It stores the item ID, the ID of the bin
# into which the item is to be inserted, as well as the left-x, bottom-y,
# right-x, and top-y coordinates of the item.
# We now allocate one packing record for each encoding and apply the
# encodings.
ys = []
for encoding in encodings:
    y = space.create()  # allocate the packing
    encoding.decode(x, y)  # perform the encoding
    space.validate(y)  # check
    ys.append(y)


# We can now plot the packing.  We create the figures in a temp directory.
# To keep the figures, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    files = []  # the collection of files

# Plot the packings. The default plotting includes the item ID into each
# rectangle. If enough space is there, it will also include the index of the
# item in the bin (starting at 1), and, if then there is still enough space,
# also the overall index of the item in the encoding (starting at 1).
    for i, y in enumerate(ys):
        fig = plot_packing(y, max_bins_per_row=2, max_rows=2)

        # Save the image as svg and png.
        files.extend(save_figure(
            fig=fig,  # store fig to a file
            file_name=f"packing_plot_a10_ibl{i + 1}",  # base name
            dir_name=td,  # store graphic in temp directory
            formats=("svg", "png")))  # file types: svg and png
        del fig  # dispose figure

# OK, we have now generated and saved the plot in a file.
# We will open it in the web browser if we are not in a make build.
    if SHOW_PLOTS_IN_BROWSER:
        for file in files:  # for each file we generated
            open_new_tab(f"file://{file}")  # open a browser tab
        sleep(10)  # sleep 10 seconds (enough time for the browser to load)
# The temp directory is deleted as soon as we leave the `with` block.
# Hence, all the figures generated above as well as the experimental results
# now have disappeared.
