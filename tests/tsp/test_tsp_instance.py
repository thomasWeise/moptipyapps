"""Test loading and validity of TSP Instances."""
from typing import Final

import numpy as np

from moptipyapps.tsp.instance import Instance
from moptipyapps.tsp.known_optima import (
    list_resource_tours,
    opt_tour_from_resource,
)
from moptipyapps.tsp.tour_length import tour_length

#: the resource tours
__AVAIABLE_TOURS: Final[set[str]] = set(list_resource_tours())


def __check_resource_instance(name: str) -> None:
    """
    Check an instance from a resource, throw an error if it is wrong.

    :param name: the instance name
    """
    instance: Final[Instance] = Instance.from_resource(name)
    try:
        assert isinstance(instance, Instance)
        assert 0 <= instance.tour_length_lower_bound < \
               instance.tour_length_upper_bound < 1_000_000_000_000_000
        assert instance.name == name
        assert instance.n_cities > 1

        if name in __AVAIABLE_TOURS:
            tour: Final[np.ndarray] = opt_tour_from_resource(name)
            assert isinstance(tour, np.ndarray)
            assert len(tour) == instance.n_cities
            assert tour_length(instance, tour) == \
                   instance.tour_length_lower_bound
    except AssertionError as ae:
        raise ValueError(f"error when testing {name!r}") from ae


def test_resource_tours() -> None:
    """Test all the instances provided as resources."""
    for name in Instance.list_resources():
        __check_resource_instance(name)
