"""A two-dimensional packing."""

from io import StringIO
from typing import Final

import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import CSV_SEPARATOR
from moptipy.utils.types import type_error

from moptipyapps.ttp.instance import Instance


class GamePlan(Component, np.ndarray):
    """A game plan, i.e., a solution to the Traveling Tournament Problem."""

    #: the TTP instance
    instance: Instance

    def __new__(cls, instance: Instance) -> "GamePlan":
        """
        Create a solution record for the Traveling Tournament Problem.

        :param cls: the class
        :param instance: the solution record
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)

        n: Final[int] = instance.n_cities  # the number of teams
        # each team plays every other team 'rounds'  times
        n_days: Final[int] = (n - 1) * instance.rounds
        obj: Final[GamePlan] = super().__new__(
            cls, (n_days, n), instance.game_plan_dtype)
        #: the TTP instance
        obj.instance = instance
        return obj

    def __str__(self):
        """
        Convert the game plan to a compact string.

        :return: the compact string
        """
        csv: Final[str] = CSV_SEPARATOR
        sep: str = ""
        teams: Final[tuple[str, ...]] = self.instance.teams
        len(teams)

        with StringIO() as sio:
            for k in self.flatten():
                sio.write(sep)
                sio.write(str(k))
                sep = csv

            sio.write("\n\n")

            sep = ""
            for t in teams:
                sio.write(sep)
                sio.write(t)
                sep = " "

            for row in self:
                sio.write("\n")
                sep = ""
                for d in row:
                    sio.write(sep)
                    if d < 0:
                        idx = -d - 1
                        sio.write("@")
                    else:
                        idx = d - 1
                    sio.write(teams[idx])
                    sep = " "

            return sio.getvalue()
