"""
Generate instances for training and testing.

In this package, we provide a function for generating instances in a
deterministic fashion for training and testing of MFC scenarios.

The function :func:`get_instances` will return a fixed set of instances
for a given instance number. It allows you to store and retrieve compatible
instance sets of different sizes from a given directory.
It is not very efficient, but it will do.

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
