"""
Run a small experiment applying RLS to one 2d bin packing instance.

We plug the second variant (`ImprovedBottomLeftEncoding2`) of the
improved-bottom-left encoding into a simple randomized local search (RLS).
The encoding processes a signed permutation from the beginning to the end and
places objects iteratively into bins. For each object, it will always try all
bins.

We apply the algorithm to instance `a10`. The result of a short run with 1024
steps of the algorithm is a packing that needs two bins only. You can compare
this with the example file `binpacking2d_plot.py`, where three bins are needed
by the same encoding (or four by `ImprovedBottomLeftEncoding1`).
"""
import os
from time import sleep
from webbrowser import open_new_tab

import psutil
from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.operators.signed_permutations.op0_shuffle_and_flip import (
    Op0ShuffleAndFlip,
)
from moptipy.operators.signed_permutations.op1_swap_2_or_flip import (
    Op1Swap2OrFlip,
)
from moptipy.spaces.signed_permutations import SignedPermutations
from moptipy.utils.plot_utils import save_figure
from moptipy.utils.temp import TempDir

from moptipyapps.binpacking2d.bin_count_and_last_empty import (
    BinCountAndLastEmpty,
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

# load the problem instance
instance = Instance.from_resource("a10")  # pick instance a10

search_space = SignedPermutations(
    instance.get_standard_item_sequence())  # Create the search space.
solution_space = PackingSpace(instance)  # Create the space of packings.
y = solution_space.create()  # Here we will put the best solution.

# Build a single execution of a single run of a single algorithm, execute it,
# and store the best solution discovered in y.
with Execution()\
        .set_rand_seed(1)\
        .set_search_space(search_space)\
        .set_solution_space(solution_space)\
        .set_encoding(ImprovedBottomLeftEncoding2(instance))\
        .set_algorithm(  # This is the algorithm: Randomized Local Search.
            RLS(Op0ShuffleAndFlip(search_space), Op1Swap2OrFlip()))\
        .set_objective(BinCountAndLastEmpty(instance))\
        .set_max_fes(1024)\
        .execute() as process:
    process.get_copy_of_best_y(y)

# We can now plot the best packing. We create the figures in a temp directory.
# To keep the figures, you would put an existing directory path into `td`
# by doing `from moptipy.utils.path import Path; td = Path.directory("mydir")`
# and not use the `with` block.
with TempDir.create() as td:  # create temporary directory `td`
    files = []  # the collection of files

# Plot the packing. The default plotting includes the item ID into each
# rectangle. If enough space is there, it will also include the index of the
# item in the bin (starting at 1), and, if then there is still enough space,
# also the overall index of the item in the encoding (starting at 1).
    fig = plot_packing(y, max_bins_per_row=2, max_rows=2)

    # Save the image as svg and png.
    files.extend(save_figure(
        fig=fig,  # store fig to a file
        file_name="packing_plot_a10",  # base name
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
