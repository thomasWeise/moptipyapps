"""
Here we provide a :class:`~moptipy.api.space.Space` of bin game plans.

The bin game plans object is defined in module
:mod:`~moptipyapps.ttp.game_plan`. Basically, it is a
two-dimensional numpy array holding, for each day (or time slot) for each team
the opposing team.
"""
from typing import Final

import numpy as np
from moptipy.api.space import Space
from moptipy.utils.logger import CSV_SEPARATOR, KeyValueLogSection
from pycommons.types import type_error

from moptipyapps.ttp.game_plan import GamePlan
from moptipyapps.ttp.instance import Instance
from moptipyapps.utils.shared import SCOPE_INSTANCE


class GamePlanSpace(Space):
    """An implementation of the `Space` API of for game plans."""

    def __init__(self, instance: Instance) -> None:
        """
        Create a 2D packing space.

        :param instance: the 2d bin packing instance

        >>> inst = Instance.from_resource("circ4")
        >>> space = GamePlanSpace(inst)
        >>> space.instance is inst
        True
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: The instance to which the packings apply.
        self.instance: Final[Instance] = instance
        self.copy = np.copyto  # type: ignore
        self.to_str = GamePlan.__str__  # type: ignore

    def create(self) -> GamePlan:
        """
        Create a game plan without assigning items to locations.

        :return: the (empty, uninitialized) packing object

        >>> inst = Instance.from_resource("circ8")
        >>> space = GamePlanSpace(inst)
        >>> x = space.create()
        >>> print(inst.rounds)
        2
        >>> print(inst.n_cities)
        8
        >>> x.shape
        (14, 8)
        >>> x.instance is inst
        True
        >>> type(x)
        <class 'moptipyapps.ttp.game_plan.GamePlan'>
        """
        return GamePlan(self.instance)

    def is_equal(self, x1: GamePlan, x2: GamePlan) -> bool:
        """
        Check if two bin game plans have the same contents.

        :param x1: the first game plan
        :param x2: the second game plan
        :return: `True` if both game plans are for the same instance and have
            the same structure

        >>> inst = Instance.from_resource("circ4")
        >>> space = GamePlanSpace(inst)
        >>> y1 = space.create()
        >>> y1.fill(0)
        >>> y2 = space.create()
        >>> y2.fill(0)
        >>> space.is_equal(y1, y2)
        True
        >>> y1 is y2
        False
        >>> y1[0, 0] = 1
        >>> space.is_equal(y1, y2)
        False
        """
        return (x1.instance is x2.instance) and np.array_equal(x1, x2)

    def from_str(self, text: str) -> GamePlan:
        """
        Convert a string to a packing.

        :param text: the string
        :return: the packing

        >>> inst = Instance.from_resource("circ6")
        >>> space = GamePlanSpace(inst)
        >>> y1 = space.create()
        >>> y1.fill(0)
        >>> y2 = space.from_str(space.to_str(y1))
        >>> space.is_equal(y1, y2)
        True
        >>> y1 is y2
        False
        """
        if not isinstance(text, str):
            raise type_error(text, "packing text", str)

        # we only want the very first line
        text = text.lstrip()
        lb: int = text.find("\n")
        if lb > 0:
            text = text[:lb].rstrip()

        x: Final[GamePlan] = self.create()
        np.copyto(x, np.fromstring(text, dtype=x.dtype, sep=CSV_SEPARATOR)
                  .reshape(x.shape))
        self.validate(x)
        return x

    def validate(self, x: GamePlan) -> None:
        """
        Check if a game plan is an instance of the right object.

        This method performs a superficial feasibility check, as in the TTP,
        we try to find feasible game plans and may have infeasible ones. All
        we check here is that the object is of the right type and dimensions
        and that it does not contain some out-of-bounds value.

        :param x: the game plan
        :raises TypeError: if any component of the game plan is of the wrong
            type
        :raises ValueError: if the game plan is not feasible
        """
        if not isinstance(x, GamePlan):
            raise type_error(x, "x", GamePlan)
        inst: Final[Instance] = self.instance
        if inst is not x.instance:
            raise ValueError(
                f"x.instance must be {inst} but is {x.instance}.")
        if inst.game_plan_dtype is not x.dtype:
            raise ValueError(f"inst.game_plan_dtype = {inst.game_plan_dtype}"
                             f" but x.dtype={x.dtype}")

        n: Final[int] = inst.n_cities  # the number of teams
        # each team plays every other team 'rounds'  times
        n_days: Final[int] = (n - 1) * inst.rounds

        needed_shape: Final[tuple[int, int]] = (n_days, n)
        if x.shape != needed_shape:
            raise ValueError(f"x.shape={x.shape}, but must be {needed_shape}.")
        min_id: Final[int] = -n

        for i in range(n_days):
            for j in range(n):
                v = x[i, j]
                if not (min_id <= v <= n):
                    raise ValueError(f"value {v} at x[{i}, {j}] should be in "
                                     f"{min_id}...{n}, but is not.")

    def n_points(self) -> int:
        """
        Get the number of game plans.

        The values in a game plan go from `-n..n`, including zero, and we have
        `days*n` values. This gives `(2n + 1) ** (days * n)`, where `days`
        equals `(n - 1) * rounds` and `rounds` is the number of rounds. In
        total, this gives `(2n + 1) ** ((n - 1) * rounds * n)`.

        :return: the number of possible game plans

        >>> space = GamePlanSpace(Instance.from_resource("circ6"))
        >>> print((2 * 6 + 1) ** ((6 - 1) * 2 * 6))
        6864377172744689378196133203444067624537070830997366604446306636401
        >>> space.n_points()
        6864377172744689378196133203444067624537070830997366604446306636401
        >>> space = GamePlanSpace(Instance.from_resource("circ4"))
        >>> space.n_points()
        79766443076872509863361
        >>> print((2 * 4 + 1) ** ((4 - 1) * 2 * 4))
        79766443076872509863361
        """
        inst: Final[Instance] = self.instance
        n: Final[int] = inst.n_cities
        n_days: Final[int] = (n - 1) * inst.rounds
        total_values: Final[int] = 2 * n + 1
        return total_values ** (n_days * n)

    def __str__(self) -> str:
        """
        Get the name of the game plan space.

        :return: the name, simply `gp_` + the instance name

        >>> print(GamePlanSpace(Instance.from_resource("bra24")))
        gp_bra24
        """
        return f"gp_{self.instance}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
