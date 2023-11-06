"""Test loading and validity of QAP Instances."""
from typing import Final

from moptipyapps.qap.instance import Instance


def __check_resource_instance(name: str) -> None:
    """
    Check an instance from a resource, throw an error if it is wrong.

    :param name: the instance name
    """
    instance: Final[Instance] = Instance.from_resource(name)
    assert isinstance(instance, Instance)
    assert (0 <= instance.lower_bound <= instance.upper_bound
            < 1_000_000_000_000_000)
    assert instance.name == name
    assert instance.n > 1


def test_resource_instances() -> None:
    """Test all the QAPLib instances provided as resources."""
    for name in Instance.list_resources():
        __check_resource_instance(name)
