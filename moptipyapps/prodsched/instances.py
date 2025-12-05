"""Generate instances for training and testing."""

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
        has_instances = True
        try:
            next(use_dir.list_dir(True, False))
        except StopIteration:
            has_instances = False

    if has_instances:
        loaded: Final[tuple[Instance, ...]] = load_instances(use_dir)
        loaded_len: Final[int] = tuple.__len__(loaded)
        if loaded_len != n:
            raise ValueError(f"Excepted {n} instances, got {loaded_len}.")
        return loaded

    insts: Final[list[Instance]] = []
    for seed in rand_seeds_from_str("mfc", n):
        insts.append(sample_mfc_instance(seed=seed))

    insts.sort(key=instance_sort_key)
    store_instances(use_dir, insts)
    return tuple(insts)
