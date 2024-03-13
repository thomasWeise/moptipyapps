"""
An objective computing the total travel length of all teams in a game plan.

This objective function takes a game plan as argument and computes the total
travel length for all teams.
A :mod:`~moptipyapps.ttp.game_plan` is basically a matrix that, for each day
(first dimension) stores against which each team (second dimension) plays.
If a team plays at home (has a home game), its opponent is stored as a
positive number.
If the team has an away game, i.e., needs to visit the opponent, then this
opponent is stored as a negative number.
Team IDs go from `1` to `n`, i.e., a value in `1..n` indicates a home game
and a value in `-n..-1` indicates an away game.
The value `0` denotes a `bye`, i.e., that no game is scheduled for a team
at the given day.

The total game plan length is computed as follows:

1. the total length = 0
2. for each team,
    1. start at the current location = home
    2. for each day,
        1. if the opponent number is negative, the next location is the
           opponent's hometown;
        2. else if the opponent number is positive, the next location is the
           own hometown;
        3. else (if the opponent number is 0): add the `bye penalty` to the
           total length and jump to the next iteration
        4. add the distance from the current to the next location to the total
           length
        5. set the current location = the next location
    3. if the current location != own hometown, add the travel distance back
       from the current location to the hometown to the total travel length.

As penalty for the `bye` situation where no game is scheduled, we use twice
the maximum distance between any two teams plus 1.
The logic is that if a `bye` (i.e., a `0`) inserted into a game plan, it
replaces one game. Since it replaces one game, it affects up to two travels,
namely from the previous location to the game location and from the game
location to the next location.
So the optimization process could sneakily try to cut longer legs of the
tournament by inserting a `bye`.
The longest imaginable travel would be between the two cities that are
farthest away from each other and back.
By making the penalty for a `bye` exactly one distance unit longer than
this longest imaginable distance, we ensure that the travel length can
never be reduced by inserting a `bye`.
Thus, having a `bye` is always more costly than any other travel it could
replace.
"""


from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from pycommons.types import type_error

from moptipyapps.ttp.game_plan import GamePlan
from moptipyapps.ttp.instance import Instance


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def game_plan_length(
        y: np.ndarray, distances: np.ndarray, bye_penalty: int) -> int:
    """
    Compute the total travel length of a game plan.

    :param y: the game plan
    :param distances: the distance matrix
    :param bye_penalty: the penalty for `bye = 0` entries, i.e., days where
        no game is scheduled
    :returns: the total plan length

    >>> yy = np.array([[ 2, -1,  4, -3],
    ...                [-2,  1, -4,  3],
    ...                [ 3,  4, -1, -2],
    ...                [-3, -4,  1,  2],
    ...                [ 4,  3, -2, -1],
    ...                [-4, -3,  2,  1]], int)
    >>> dd = np.array([[ 0,  1,  2,  3],
    ...                [ 7,  0,  4,  5],
    ...                [ 8, 10,  0,  6],
    ...                [ 9, 11, 12,  0]], int)
    >>> 0 + 1 + 7 + 2 + 8 + 3 + 9  # team 1
    30
    >>> 7 + 1 + 0 + 5 + 11 + 4 + 10  # team 2
    38
    >>> 0 + 6 + 9 + 2 + 10 + 4  # team 3
    31
    >>> 12 + 6 + 11 + 5 + 9 + 3  # team 4
    46
    >>> 30 + 38 + 31 + 46  # total sum
    145
    >>> game_plan_length(yy, dd, 0)
    145

    >>> yy[1, 0] = 0  # add a bye
    >>> 0 + 25 + 0 + 2 + 8 + 3 + 9  # team 1
    47
    >>> game_plan_length(yy, dd, 2 * 12 + 1)
    162
    """
    days, teams = y.shape  # get the number of days and teams
    length: int = 0

    for team in range(teams):  # for each team
        current_location: int = team  # start at home
        for day in range(days):  # for each day
            next_location: int = y[day, team]
            if next_location < 0:  # away game at other team
                next_location = (-next_location) - 1
            elif next_location > 0:  # home game at home
                next_location = team
            else:  # bye = no game = penalty
                length += bye_penalty
                continue  # team stays in current location
            if current_location == next_location:
                continue  # no move
            length += distances[current_location, next_location]
            current_location = next_location

        if current_location != team:  # go back home
            length += distances[current_location, team]

    return int(length)


class GamePlanLength(Objective):
    """
    Compute the total travel length of a game plan.

    This objective function sums up all the travel lengths over all teams.
    Days without game (`bye`) are penalized.
    """

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the game plan length objective function.

        :param instance: the TTP instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        super().__init__()

        #: the TTP instance
        self.instance: Final[Instance] = instance
        #: the bye penalty
        self.bye_penalty: Final[int] = (2 * int(instance.max())) + 1

    def evaluate(self, x: GamePlan) -> int:
        """
        Count the errors in a game plan as objective value.

        :param x: the game plan
        :return: the number of errors in the plan
        """
        return game_plan_length(x, x.instance, self.bye_penalty)

    def lower_bound(self) -> int:
        """
        Obtain the lower bound for the travel length.

        :return: `0`
        """
        return 0

    def upper_bound(self) -> int:
        """
        Compute upper bound for the travel length: All `n*days*bye_penalty`.

        :returns: `n * days * self.bye_penalty`
        """
        n: Final[int] = self.instance.n_cities
        rounds: Final[int] = self.instance.rounds
        days: Final[int] = (n - 1) * rounds
        return n * days * self.bye_penalty

    def is_always_integer(self) -> bool:
        """
        State that this objective function is always integer-valued.

        :return: `True`
        """
        return True

    def __str__(self) -> str:
        """Get the name of this objective function."""
        return "planLength"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the instance to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("byePenalty", self.bye_penalty)
