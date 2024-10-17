"""
A decoder for 2D BPP instances.

The goal of developing this decoding procedure is that we need a deterministic
mapping of some easy-to-process data structure to an instance of the
two-dimensional bin packing problem. The instance produced by the mapping
should use a pre-defined bin width, bin height, and number of items. It should
also have a pre-defined lower bound for the number of bins required and it
must be ensured that this lower bound can also be reached, i.e., that at least
one solution exists that can indeed pack all items into this number of bins.

As source data structure to be mapped, we choose the real vectors of a fixed
length (discussed later on).

The idea is that we take the bin width, bin height, and lower bound of the
number of bins (let's call it `lb`) from a template instance. We also take
the number items (let's call it `n`) from that instance.

Now we begin with `lb` items, each of which exactly of the size and dimensions
of a bin. At the end, we want to have `n` items. To get there, in each step of
our decoding, we split one existing item into two items. This means that each
step will create one additional item for the instance (while making one
existing item smaller). This, in turn, means that we have to do `n - lb`
decoding steps, as we start with `lb` items and, after `n - lb` steps, will
have `lb + n - lb = n` items.

So far so good.
But how do we split?

Each split that we want to perform be defined by four features:

1. the index of the item that we are going to split,
2. whether we split it horizontally or vertically,
3. where we are going to split it,
4. and how to continue if the proposed split is not possible, e.g., because
   it would lead to a zero-width or zero-height item.

Now we can encode this in two real numbers `selector` and `cutter` from the
interval `[-1,1]`.

First, we multiply the absolute value of the `selector` with the current
number of items that we have. This is initially `2`, will then be `3` in the
next iteration, then `4`, and so on.
Converted to an int, the result of this multiplication gives us the index of
the item to split.

Then, if `cutter >= 0`, we will cut the item horizontally. Otherwise, i.e., if
`cutter < 0`, we cut vertically.
Where to cut is then decided by multiplying the absolute value of `cutter`
with the length of the item in the selected cutting dimension.

If that is not possible, we move to the next item and try again.
If `selector < 0`, we move towards smaller indices and wrap after `0`.
Otherwise, we move towards higher indices and wrap at the end of the item
list.
If we arrive back at the first object, this means that the split was not
possible for any of the existing items.
We now rotate the split by 90°, i.e., if we tried horizontal splits, we now
try vertical ones and vice versa.

It is easy to see that it must always be possible to split at least one item
in at least one direction. Since we took the bin dimensions and numbers of
items from an existing instance of the benchmark set, it must be possible to
divide the bins into `n` items in one way or another. Therefore, the loop
will eventually terminate and yield the right amount of items.

This means that with `2 * (n - lb)` floating point numbers, we can describe an
instance whose result is a perfect packing, without any wasted space.

Now the benchmark instances are not just instances that can only be solved by
perfect packings. Instead, they have some wasted space.

We now want to add some wasted space to our instance. So far, our items occupy
exactly `lb * bin_width * bin_height` space units. We can cut at most
`bin_width * bin_height - 1` of these units without changing the required bins
of the packing: We would end up with `(lb - 1) * bin_width * bin_height + 1`
space units required by our items, which is `1` too large to fit into `lb - 1`
bins.

So we just do the same thing again:
We again use two real numbers to describe each split.
Like before, we loop through these numbers, pick the object to split, and
compute where to split it. Just now we throw away the piece that was cut off.
(Of course, we compute the split positions such that we never dip under the
area requirement discussed above).

This allows us to use additional real variables to define how the space should
be reduced. `2 * (n - lb)` variables, we get instances requiring perfect
packing. With every additional pair of variables, we cut some more space.
If we would use `2 * (n - lb) + 10` variables, then we would try to select
five items from which we can cut off a bit. This number of additional
variables can be chosen by the user.

Finally, we merge all items that have the same dimension into groups, as is
the case in some of the original instances. We then shuffle these groups
randomly, to give the instances a bit of a more unordered texture.
The random shuffling is seeded with the binary representation of the input
vectors.

In the end, we have translated a real vector to a two-dimensional bin packing
instance. Hurray.

>>> space = InstanceSpace(Instance.from_resource("a04"))
>>> print(f"{space.inst_name!r} with {space.n_different_items}/"
...       f"{space.n_items} items with area {space.total_item_area} "
...       f"in {space.min_bins} bins of "
...       f"size {space.bin_width}*{space.bin_height}.")
'a04n' with 2/16 items with area 7305688 in 3 bins of size 2750*1220.

>>> decoder = InstanceDecoder(space)
>>> import numpy as np
>>> x = np.array([ 0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, -0.1,  0.3,  0.5, -0.6, -0.7,  0.9,
...                0.0,  0.2, ])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 15/16 items with area 10065000 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;15;2750;1220;1101,1098;2750,244;2750,98;1101,171;1649,171;2750,976;\
441,122;1649,122;2750,10;2750,1,2;2750,3;1649,1098;2750,878;2750,58;660,122


>>> x = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, ])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 3/16 items with area 10065000 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;3;2750;1220;2750,1216,2;2750,1,13;2750,1215

>>> from math import nextafter
>>> a1 = nextafter(1.0, -10)
>>> x = np.array([ a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, ])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 4/16 items with area 10065000 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;4;2750;1220;2750,1208;2750,1219;2750,1220;2750,1,13

>>> from math import nextafter
>>> a1 = nextafter(-1.0, 10)
>>> x = np.array([ a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, ])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 5/16 items with area 10065000 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;5;2750;1220;2750,1220;2730,1220;2748,1220;1,1220,4;2,1220,9

>>> from math import nextafter
>>> a1 = nextafter(-1.0, 10)
>>> x = np.array([ a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, 0.3, 0.7 ])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 6/16 items with area 10064146 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;6;2750;1220;2,1220,9;1,1220,3;2748,1220;1,366;2750,1220;2730,1220

>>> from math import nextafter
>>> a1 = nextafter(-1.0, 10)
>>> x = np.array([ a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, a1, a1, a1, a1, a1, a1,
...                a1, a1, 0.3, 0.7, -0.2, -0.3,
...                0.5, -0.3])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 6/16 items with area 10061706 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;6;2750;1220;2,1220,7;2750,1220;2730,1220;1,1220,5;1,366;2748,1220


>>> x = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
...                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
...                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
...                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
...                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
...                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
...                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
...                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
...                0.0, 0.0, ])
>>> y = space.create()
>>> decoder.decode(x, y)
>>> space.validate(y)
>>> res: Instance = y[0]
>>> print(f"{res.name!r} with {res.n_different_items}/"
...       f"{res.n_items} items with area {res.total_item_area} "
...       f"in {res.lower_bound_bins} bins of "
...       f"size {res.bin_width}*{res.bin_height}.")
'a04n' with 5/16 items with area 9910948 in 3 bins of size 2750*1220.
>>> print(space.to_str(y))
a04n;5;2750;1220;2698,1;2750,1,12;2750,1216;2750,1215;2750,1160
"""
from math import isfinite
from typing import Final

import numpy as np
from moptipy.api.encoding import Encoding
from numpy.random import default_rng
from pycommons.types import check_int_range, type_error

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.instgen.instance_space import InstanceSpace


class InstanceDecoder(Encoding):
    """Decode a string of `n` real values in `[0,1]` to an instance."""

    def __init__(self, space: InstanceSpace) -> None:
        """
        Create the instance decoder.

        :param space: the instance description and space
        """
        super().__init__()
        if not isinstance(space, InstanceSpace):
            raise type_error(space, "space", InstanceSpace)
        #: the instance description
        self.space: Final[InstanceSpace] = space

    def get_x_dim(self, slack: float | int = 0) -> int:
        """
        Get the minimum dimension that a real vector must have.

        :param slack: a parameter denoting the amount of slack for reducing
            the item size
        :return: the minimum dimension
        """
        if not isinstance(slack, float | int):
            raise type_error(slack, "slack", (float, int))
        if (not isfinite(slack)) or (slack < 0):
            raise ValueError(f"slack={slack} is invalid")
        base: Final[int] = check_int_range(
            self.space.n_items - self.space.min_bins, "base", 1, 1_000_000)
        added: Final[int] = int(slack * base + 0.5)
        if (added < 0) or (added > 1_000_000):
            raise ValueError(f"invalid result {added} based "
                             f"on slack={slack} and base={base}")
        return 2 * (base + added)

    def decode(self, x: np.ndarray, y: list[Instance]) -> None:
        """
        Decode the real-valued array to a 2D BPP instance.

        :param x: an array with values in [-1, 1]
        :param y: the instance receiver
        """
        # We start by using the prescribed number of bins as items.
        # There will be n_bins items, each of which has exactly the size of a
        # bin. Therefore, we know that all items fit exactly into n_bins bins.
        bin_width: Final[int] = self.space.bin_width
        bin_height: Final[int] = self.space.bin_height
        n_bins: Final[int] = self.space.min_bins
        items: list[list[int]] = [
            [bin_width, bin_height] for _ in range(n_bins)]

        # Now we need to make n_items - n_bins cuts. Each cut will divide one
        # item into two items. Therefore, each cut yields exactly one new
        # item. Therefore, after n_items - n_bins cuts we get n_items - n_bins
        # new items. Since we start with n_bins items, this means that we get
        # n_items - n_bins + n_bins items at the end, which are exactly
        # n_items.
        #
        # Each cut cuts an item into two items along either the horizontal or
        # vertical dimension. If we would put the two new, smaller items back
        # together, they would exactly result in the item that we just cut.
        # Therefore, they do fit into the same area and their area sum is also
        # the same. Therefore, the overall area and the overall number of
        # required bins will also stay the same.
        x_idx: int = 0
        for cur_n_items in range(n_bins, self.space.n_items):
            # In each iteration of this loop, we perform one cut.
            # This means, that we add exactly one item to the item list.
            # Our counter variable `cur_n_items`, which starts at `n_bins`
            # and iterates until just before the target number of items
            # `self.desc.n_items` represents the current number of items
            # in our list `items`.
            # It thus always holds that `cur_n_items == len(items)`.

            # Each cut is described by four properties:
            # 1. The index of the item that we want to cut.
            # 2. The direction (horizontal or vertical) into which we cut.
            # 3. The location where we will cut along this dimension.
            # 4. The direction into which we will search for the next item
            #    if the current item cannot be cut this way.
            # These four properties are encoded in two real numbers `selector`
            # and `cutter` from `[-1, 1]`.

            # The first number is `sel`. It encodes (1) and (4).
            # First, we multiply `selector` with `cur_n_items` and take this
            # modulo `cur_n_items`. This gives us a value `t` with
            # `-cur_n_items < t < cur_n_items`. By adding `cur_n_items`
            # and repeating the modulo division, we get
            # `0 <= sel_i < cur_n_items`.
            # `sel_i` is the index of the item that we want to cut.
            selector: float = x[x_idx]
            x_idx += 1

            sel_i: int = ((int(cur_n_items * selector) % cur_n_items)
                          + cur_n_items) % cur_n_items
            orig_sel_i: int = sel_i  # the original selection index.

            # Now, it may not be possible to cut this item.
            # The idea is that if we cannot cut the item, we simply try to cut
            # the next item.
            # The next item could be the one at the next-higher index or the
            # one at the next lower index.
            # If `selector < 0.0`, then we will next try the item at the next
            # lower index. If `selector >= 0.0`, we will move to the next
            # higher index.
            # Of course, we will wrap our search when reaching either end of
            # the list.
            #
            # If we arrive back at `orig_sel_i`, then we have tried to cut
            # each item in the list in the prescribed cutting direction,
            # however none could be cut.
            # In this case, we will change the cutting direction.
            #
            # It must be possible to cut at least one item in at least one
            # direction, or otherwise the problem would not be solvable.
            # Therefore, we know that this way, trying all items in all
            # directions in the worst case, we will definitely succeed.
            sel_dir: int = -1 if selector < 0.0 else 1

            # `cutter` tells us where to cut and in which direction.
            # If `cutter >= 0`, then we will cut horizontally and
            # if `cutter < 0`, we cut vertically.
            # Each item is described by list `[width, height]`, so cutting
            # horizontally means picking a vertical height coordinate and
            # cutting horizontally along it. This means that for horizontal
            # cuts, the `cut_dimension` should be `1` and for vertical cuts,
            # it must be `0`.
            cutter: float = x[x_idx]
            x_idx += 1
            cut_dimension: int = 1 if cutter >= 0.0 else 0

            while True:
                cur_item: list[int] = items[sel_i]  # Get the item to cut.

                item_size_in_dim: int = cur_item[cut_dimension]

                # We define the `cut_modulus` as the modulus for the cutting
                # operation. We will definitely get a value for `cut_position`
                # such that `0 < cut_position <= cut_modulus`. This means that
                # we will get `0 < cut_position < item_size_in_dim`.
                # Therefore, if `cut_modulus > 1`, then we know that there
                # definitely is a `cut_position` at which we can cut the
                # current item and obtain two new items that have both
                # non-zero width and height.
                cut_modulus: int = item_size_in_dim - 1
                if cut_modulus > 0:  # Otherwise, we cannot cut the item.
                    cut_position: int = (((int(
                        cut_modulus * cutter) % cut_modulus) + cut_modulus)
                        % cut_modulus) + 1

                    if 0 < cut_position < item_size_in_dim:  # Sanity check...
                        # Now we perform the actual cut.
                        # The original item now gets `cut_position` as the
                        # size in the `cut_dimension`, the other one gets
                        # `item_size_in_dim - cut_position`. Therefore, the
                        # overall size remains the same.
                        cur_item[cut_dimension] = cut_position
                        cur_item = cur_item.copy()
                        cur_item[cut_dimension] = (
                            item_size_in_dim - cut_position)
                        items.append(cur_item)
                        break  # we cut one item and can stop

                sel_i = ((((sel_i + sel_dir) % cur_n_items) + cur_n_items)
                         % cur_n_items)
                if sel_i == orig_sel_i:
                    cut_dimension = 1 - cut_dimension

        # At this stage, our instance can only be solved with a perfect
        # packing. The items need to be placed perfectly together and they
        # will cover the complete `current_area`.
        bin_area: Final[int] = bin_width * bin_height
        current_area: int = n_bins * bin_area

        # However, requiring a perfect packing may add hardness to the problem
        # that does not exist in the original problem.
        # We now want to touch some of the items and make them a bit smaller.
        # However, we must never make them smaller as
        # `current_area - bin_area`, because then we could create a situation
        # where less than `n_bins` bins are required.
        # Also, we never want to slide under the `total_item_area` prescribed
        # by the original problem. If the original problem prescribed a
        # perfect packing, then we will create a perfect packing.
        min_area: Final[int] = current_area - bin_area + 1

        # If and only if the input array still has information left and if we
        # still have area that we can cut, then let's continue.
        # We will keep cutting items, but this time, we throw away the piece
        # that we cut instead of adding it as item. Of course, we never cut
        # more than what we are permitted to.
        max_x_idx: Final[int] = len(x)
        cur_n_items = list.__len__(items)
        while (x_idx < max_x_idx) and (current_area > min_area):
            # We perform the same selection and cutting choice.
            selector = x[x_idx]
            x_idx += 1
            sel_i = ((int(cur_n_items * selector) % cur_n_items)
                     + cur_n_items) % cur_n_items
            orig_sel_i = sel_i  # the original selection index.
            sel_dir = -1 if selector < 0.0 else 1
            cutter = x[x_idx]
            x_idx += 1
            cut_dimension = 1 if cutter >= 0.0 else 0

            # We have selected the item and got the cutter value, too.
            # Now we try to perform the cut. If we cannot cut, we try
            # the next item. If we could not cut any item along the
            # cut dimension, we switch the cut dimension.
            # However, we may fail to cut anything (which means that
            # we arrive at `step == 2`). In this case, we  just skip
            # this cut.
            step: int = 0
            while step < 2:  # This time, we may actually fail to cut.
                cur_item = items[sel_i]  # Get the item to cut.

                item_size_in_dim = cur_item[cut_dimension]
                item_size_in_other_dim: int = cur_item[1 - cut_dimension]
                # This time, the `cut_modulus` is also limited by the area
                # that we can actually cut. This is `current_area - min_area`.
                # Now each cut along the `cut_dimension` will cost us
                # `item_size_in_other_dim` area units.
                cut_modulus = min(
                    (current_area - min_area) // item_size_in_other_dim,
                    item_size_in_dim) - 1
                if cut_modulus > 0:
                    cut_position = (((int(
                        cut_modulus * cutter) % cut_modulus) + cut_modulus)
                        % cut_modulus) + 1

                    if 0 < cut_position < item_size_in_dim:
                        # We cut away cut_position and do not add the area at
                        # the end.
                        cur_item[cut_dimension] = \
                            item_size_in_dim - cut_position
                        break  # we cut one item and can stop

                sel_i = ((((sel_i + sel_dir) % cur_n_items) + cur_n_items)
                         % cur_n_items)
                if sel_i == orig_sel_i:
                    cut_dimension = 1 - cut_dimension
                    step += 1  # If we tried everything, this enforces a stop.

        # Finally, we sort the items in order to merge items of the
        # same dimension.
        items.sort()
        lo: int = 0
        n_items: int = list.__len__(items)
        while lo < n_items:
            # For each item in the list, we try to find all items of the same
            # dimension. Since the list is sorted, these items must all be
            # located together.
            hi: int = lo
            cur_item = items[lo]
            while (hi < n_items) and (items[hi] == cur_item):
                hi += 1
            cur_item.append(hi - lo)  # We now have the item multiplicity.
            # We delete all items with the same dimension (which must come
            # directly afterward).
            hi -= 1
            while lo < hi:
                del items[hi]
                hi -= 1
                n_items -= 1
            lo += 1  # Move to the next item

        # Now all items are sorted. This may or may not be a problem:
        # Some heuristics may either benefit or suffer if the item list
        # has a pre-defined structure. Therefore, we want to try to at least
        # somewhat remove the order and make the item list order more random.
        default_rng(int.from_bytes(x.tobytes())).shuffle(items)

        # And now we can fill in the result as the output of our encoding.
        res: Instance = Instance(  # Generate the instance
            self.space.inst_name, bin_width, bin_height, items)
        if list.__len__(y) > 0:  # If the destination has length > 0...
            y[0] = res  # ...store the instance at index 0,
        else:  # otherwise
            y.append(res)  # add the instance to it.
