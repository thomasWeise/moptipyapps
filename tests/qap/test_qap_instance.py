"""Test loading and validity of QAP Instances."""
from typing import Final

from moptipyapps.qap.instance import _BKS, _BOUNDS, Instance


def __check_resource_instance(name: str) -> None:
    """
    Check an instance from a resource, throw an error if it is wrong.

    :param name: the instance name
    """
    instance: Final[Instance] = Instance.from_resource(name)
    assert isinstance(instance, Instance)
    assert instance.name == name
    assert instance.n > 1
    lb: Final[int] = instance.lower_bound
    ub: Final[int] = instance.upper_bound
    assert (0 <= lb <= ub < 1_000_000_000_000_000)
    opt: Final[tuple[str, int]] = instance.bks()
    assert tuple.__len__(opt) == 2
    assert str.__len__(opt[0]) > 0
    assert isinstance(opt[1], int)
    assert lb <= opt[1] <= ub
    assert (opt[0] in ("LB", "OPT")) == (lb == opt[1])
    assert (opt[0] not in ("LB", "OPT")) == (opt[1] > lb)


def test_resource_instances() -> None:
    """Test all the QAPLib instances provided as resources."""
    all_res: Final[tuple[str, ...]] = Instance.list_resources()
    assert tuple.__len__(all_res) > 0

    for keys in (_BKS.keys(), _BOUNDS.keys()):
        assert len(keys) > 0
        for k in keys:
            assert k in all_res

    for name in all_res:
        __check_resource_instance(name)
