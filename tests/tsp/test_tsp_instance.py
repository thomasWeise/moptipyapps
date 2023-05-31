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
        if instance.name.startswith("ftv"):
            # for some reason, the ftv name pattern is different...
            assert instance.name.endswith(str(instance.n_cities - 1))
        elif instance.name == "kro124p":  # and so is kro124p
            assert instance.n_cities == 100
        else:
            assert instance.name.index(str(instance.n_cities)) > 0

        if name in __AVAIABLE_TOURS:
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
    except AssertionError as ae:
        raise ValueError(f"error when testing {name!r}") from ae


def test_resource_tours() -> None:
    """Test all the instances provided as resources."""
    for name in Instance.list_resources():
        __check_resource_instance(name)
