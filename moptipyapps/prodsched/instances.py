"""
Generate instances for training and testing.

In this package, we provide a function for generating instances, i.e., objects
of type :class:`~moptipyapps.prodsched.instance.Instance`, in a
deterministic fashion for training and testing of MFC scenarios.

The function :func:`get_instances` will return a fixed set of instances
for a given instance number. It allows you to store and retrieve compatible
instance sets of different sizes from a given directory.

This is necessary when doing repeatable experiments that average performance
metrics over multiple :class:`~moptipyapps.prodsched.instance.Instance`
objects. We do not just want to be able to generate the instances, but we also
need to store them and to re-load them. Storing them is easy, function
:func:`~moptipyapps.prodsched.instance.store_instances` can do it.
Loading instances is easy, too, because for that we have function
:func:`~moptipyapps.prodsched.instance.load_instances`.

However, what do you do if you generated 10 instances in a deterministic
fashion, but for your next experiment you only want to use five of them?
How do you decide which to use?
Or what if you want to use 15 now. How do you make sure that the previous
ten instances are part of the set of 15 instances?
:func:`get_instances` does all of that for you.
It creates the random seeds for the instance creation in the good old
deterministic "moptipy" style, using
:func:`~moptipy.utils.nputils.rand_seeds_from_str`.
It then checks the instance directory for instances to use that comply
with the seeds and generates (and stores) additional instances if need be.
For this, we use the Thürer-style instance synthesis implemented as
:func:`~moptipyapps.prodsched.mfc_generator.sample_mfc_instance`.
Thus, we have a consistent way of generating, storing, and loading instances
in a transparent way.

(The implementation is not overly efficient, but it will do.)

>>> from pycommons.io.temp import temp_dir
>>> with temp_dir() as td:
...     inst_1 = get_instances(3, td)
...     inst_2 = get_instances(1, td)
...     inst_3 = get_instances(5, td)

>>> len(inst_1)
3
>>> len(inst_2)
1
>>> len(inst_3)
5

>>> all(ix in inst_1 for ix in inst_2)
True
>>> all(ix in inst_3 for ix in inst_1)
True
"""

from typing import Final

from moptipy.utils.nputils import rand_seeds_from_str
from pycommons.io.path import Path
from pycommons.types import check_int_range

from moptipyapps.prodsched.instance import (
    Instance,
    instance_sort_key,
    load_instances,
    store_instances,
)
from moptipyapps.prodsched.mfc_generator import (
    INFO_RAND_SEED,
    sample_mfc_instance,
)


def get_instances(n: int, inst_dir: str) -> tuple[Instance, ...]:
    """
    Get the instances for the experiment.

    :param n: the expected number of instances
    :param inst_dir: the instance directory
    """
    check_int_range(n, "n", 1, 1_000_000)
    use_dir: Final[Path] = Path(inst_dir)

    has_instances: bool = False
    if use_dir.exists():
        try:
            has_instances = next(use_dir.list_dir(True, False)).endswith(
                ".txt")
        except StopIteration:
            has_instances = False

    seeds: Final[list[int]] = rand_seeds_from_str("mfc", n)
    insts: Final[list[Instance]] = []
    newly: Final[list[Instance]] = []

    if has_instances:
        def __filter(p: Path, seedstrs=tuple(map(
                str.casefold, map(hex, seeds)))) -> bool:
            """
            Filter the file names.

            :param p: the path
            :param seedstrs: the internal seed array
            :return: `True` if the file name matches, else `False`
            """
            return any(map(str.casefold(p.basename()).__contains__, seedstrs))

        for inst in load_instances(use_dir, __filter):
            if INFO_RAND_SEED in inst.infos:
                seed = int(inst.infos[INFO_RAND_SEED], 16)
                if seed in seeds:
                    insts.append(inst)
                    seeds.remove(seed)

    for seed in seeds:
        inst = sample_mfc_instance(seed=seed)
        newly.append(inst)
        insts.append(inst)

    insts.sort(key=instance_sort_key)
    if list.__len__(newly) > 0:
        newly.sort(key=instance_sort_key)
        store_instances(use_dir, newly)
    return tuple(insts)
