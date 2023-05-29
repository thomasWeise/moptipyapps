"""Test loading and validity of bin packing 2D instances."""

from moptipy.utils.types import type_error

from moptipyapps.binpacking2d.instance import Instance


def check_instance(inst: Instance, name: str | None = None) -> None:
    """
    Check an instance, throw an error if it is wrong.

    :param inst: the instance
    :param name: the instance name, optional
    """
    if not isinstance(inst, Instance):
        raise type_error(inst, "inst", Instance)
    if name is not None:
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        if inst.name != name:
            raise ValueError(f"inst.name={inst.name!r} != {name!r}!")


def test_load_all_from_resources() -> None:
    """Test loading the all the instances from resources."""
    count: int = 0
    for name in Instance.list_resources():
        check_instance(Instance.from_resource(name), name)
        count += 1
    if count != 553:
        raise ValueError(f"Excepted 553 instances, got {count}.")
