"""Test loading and validity of TSP Instances."""
from typing import Final

import numpy as np
from pycommons.io.path import line_writer
from pycommons.io.temp import temp_file

from moptipyapps.tsp.instance import Instance
from moptipyapps.tsp.known_optima import (
    list_resource_tours,
    opt_tour_from_resource,
)
from moptipyapps.tsp.tour_length import tour_length
from moptipyapps.ttp.instance import Instance as TTPInstance

#: the resource tours
__AVAILABLE_TOURS: Final[set[str]] = set(list_resource_tours())


def __check_resource_instance(name: str, root_class) -> None:
    """
    Check an instance from a resource, throw an error if it is wrong.

    :param name: the instance name
    :param root_class: the root class
    """
    instance: Final[Instance] = root_class.from_resource(name)
    try:
        assert isinstance(instance, Instance)
        assert 0 <= instance.tour_length_lower_bound <= \
               instance.tour_length_upper_bound < 1_000_000_000_000_000
        assert instance.name == name
        assert instance.n_cities > 1
        if instance.name.startswith("ftv"):
            # for some reason, the ftv name pattern is different...
            assert instance.name.endswith(str(instance.n_cities - 1))
        elif instance.name == "kro124p":  # and so is kro124p
            assert instance.n_cities == 100
        else:
            assert instance.name.index(str(instance.n_cities)) > 0

        if name in __AVAILABLE_TOURS:
            tour: Final[np.ndarray] = opt_tour_from_resource(name)
            assert isinstance(tour, np.ndarray)
            assert len(tour) == instance.n_cities
            assert tour_length(instance, tour) == \
                   instance.tour_length_lower_bound

        lb: int = 0
        ub: int = 0
        is_symmetric: bool = True
        for i in range(instance.n_cities):
            nearest: int = 1_000_000_000_000_000_000_000
            farthest: int = -1
            for j in range(instance.n_cities):
                if i == j:
                    assert instance[i, j] == 0
                dij = instance[i, j]
                if dij < nearest:
                    nearest = dij
                if dij > farthest:
                    farthest = dij
                is_symmetric = is_symmetric and (dij == instance[j, i])
            assert 0 <= nearest <= farthest <= 1_000_000_000_000_000
            lb += int(nearest)
            ub += int(farthest)
        assert 0 <= lb <= instance.tour_length_lower_bound <= ub
        assert ub == instance.tour_length_upper_bound
        assert is_symmetric == instance.is_symmetric

        with temp_file() as tf:
            with tf.open_for_write() as stream:
                instance.to_stream(line_writer(stream))
            inst_read = Instance.from_file(
                tf, lambda _: instance.tour_length_lower_bound)

        assert instance.n_cities == inst_read.n_cities
        assert instance.name == inst_read.name
        assert instance.tour_length_lower_bound \
            == inst_read.tour_length_lower_bound
        assert instance.tour_length_upper_bound \
            == inst_read.tour_length_upper_bound
        assert list(instance.flatten()) == list(inst_read.flatten())
    except AssertionError as ae:
        raise ValueError(f"error when testing {name!r}") from ae


def test_resource_instances_and_tours() -> None:
    """Test all the TSPLib instances provided as resources."""
    for name in Instance.list_resources():
        __check_resource_instance(name, Instance)


def test_ttp_instances() -> None:
    """Test all the TTP instances provided as resources."""
    for name in TTPInstance.list_resources():
        __check_resource_instance(name, TTPInstance)
