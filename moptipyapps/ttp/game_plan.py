"""
A game plan assigns teams to games.

A game plan is a two-dimensional matrix `G`. The rows are the time slots.
There is one column for each time. If `G` has value `v` at row `i` and
column `j`, then this means:

- at the time slot `i` ...
- the team with name `j+1` plays
    + no team if `v == 0`,
    + *at home* against the team `v` if `v > 0`, i.e., team `v` travels
      to the home stadium of team `j+1`
    + *away* against the team `-v` if `v < 0`, i.e., team `j+1` travels
      to the home stadium of team `-v` and plays against them there

Indices in matrices are zero-based, i.e., the lowest index for a row `i` is
`0` and the lowest index for a column `j` is also `0`. However, team names
are one-based, i.e., that with `1`. Therefore, we need to translate the
zero-based column index `j` to a team name by adding `1` to it.

This is just a numerical variant of the game plan representation given at
<https://robinxval.ugent.be/RobinX/travelRepo.php>. Indeed, the `str(...)`
representation of a game plan is exactly the table shown there.

Of course, if `G[i, j] = v`, then `G[i, v - 1] = -(j + 1)` should hold if
`v > 0`, for example. Vice versa, if `v < 0` and `G[i, j] = v`, then
`G[i, (-v) - 1] = j + 1` should hold. Such constraints are checked by the
:mod:`~moptipyapps.ttp.errors` objective function.

The corresponding space implementation,
:mod:`~moptipyapps.ttp.game_plan_space`, offers the functionality to convert
strings to game plans as well as to instantiate them in a black-box algorithm.
"""

from io import StringIO
from typing import Final

import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import CSV_SEPARATOR
from pycommons.types import type_error

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

        The first line of the output is a flattened version of this matrix
        with the values being separated by `;`. Then we place an empty line.

        We then put a more easy-to-read representation and follow the pattern
        given at https://robinxval.ugent.be/RobinX/travelRepo.php, which is
        based upon the notation by Easton et al. Here, first, a row with the
        team names separated by spaces is generated. Then, each row contains
        the opponents of these teams, again separated by spaces. If an
        opponent plays at their home, this is denoted by an `@`.
        If a team has no scheduled opponent, then this is denoted as `-`.

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
                        sio.write(f"@{teams[-d - 1]}")
                    elif d > 0:
                        sio.write(teams[d - 1])
                    else:
                        sio.write("-")
                    sep = " "

            return sio.getvalue()
