"""Test TTP instances."""

from typing import Final

import numpy as np
from moptipy.utils.nputils import is_np_int
from pycommons.types import check_int_range

from moptipyapps.tsp.instance import Instance as TSPInstance
from moptipyapps.ttp.instance import Instance as TTPInstance
from moptipyapps.ttp.plan_length import GamePlanLength


def __check_instance(inst: TTPInstance) -> None:
    """
    Check a single instance.

    :param inst: the TTP instance
    """
    assert isinstance(inst, TTPInstance)
    assert isinstance(inst, TSPInstance)
    assert isinstance(inst, np.ndarray)
    check_int_range(inst.n_cities, "n_cities", 2, 1_000_000)
    assert is_np_int(inst.dtype)
    for v in inst.flatten():
        assert 0 <= v <= 1_000_000_000_000_000
    check_int_range(inst.rounds, "rounds", 1, 1_000_000)
    check_int_range(inst.away_streak_min, "away_streak_min", 1, 1_000_000)
    check_int_range(inst.away_streak_max, "away_streak_max",
                    inst.away_streak_min, 1_000_000)
    check_int_range(inst.home_streak_min, "home_streak_min", 1, 1_000_000)
    check_int_range(inst.home_streak_max, "home_streak_max",
                    inst.home_streak_min, 1_000_000)
    check_int_range(inst.separation_min, "separation_min", 0, 1_000_000)
    check_int_range(inst.separation_max, "separation_max",
                    inst.separation_min, 1_000_000)
    f: Final[GamePlanLength] = GamePlanLength(inst)
    opt_bounds: Final[tuple[int, int]] = inst.get_optimal_plan_length_bounds()
    assert tuple.__len__(opt_bounds) == 2
    assert isinstance(opt_bounds[0], int)
    assert isinstance(opt_bounds[1], int)
    f_lb: Final[int] = f.lower_bound()
    assert isinstance(f_lb, int)
    f_ub: Final[int] = f.upper_bound()
    assert isinstance(f_ub, int)
    assert 0 <= f_lb <= opt_bounds[0] <= opt_bounds[1] <= f_ub
    assert 0 <= inst.tour_length_lower_bound <= opt_bounds[0]
    assert 0 <= inst.tour_length_upper_bound <= f_ub


def test_instances() -> None:
    """Test all the instances in the resources."""
    for r in TTPInstance.list_resources(True, True):
        __check_instance(TTPInstance.from_resource(r))
