"""Test that downloading and making the instances is consistent."""

from moptipy.utils.temp import TempFile

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.make_instances import (
    make_2dpacklib_resource,
)


def test_make_instances() -> None:
    """Test that the instance resource is the same as the current download."""
    insts = list(Instance.list_resources())

    with TempFile.create() as tf:
        assert len(list(make_2dpacklib_resource(dest_file=tf)[1])) \
               == len(insts)
        for i, row in enumerate(tf.read_all_list()):
            i1 = Instance.from_compact_str(row)
            i2 = Instance.from_resource(insts[i])
            assert i1.name == i2.name
            assert i1.n_items == i2.n_items
            assert i1.n_different_items == i2.n_different_items
            assert i1.lower_bound_bins == i2.lower_bound_bins
            assert i1.bin_width == i2.bin_width
            assert i1.bin_height == i2.bin_height
            assert all(i1.flatten() == i2.flatten())
