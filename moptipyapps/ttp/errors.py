"""
An objective that counts constraint violations.

The idea is that we will probably not be able to always produce game plans
that adhere to all the constraints imposed by a Traveling Tournament Problem
:mod:`~moptipyapps.ttp` :mod:`~moptipyapps.ttp.instance`, so we will instead
probably usually generate game plans that may contain errors.

We will hope that optimization can take care of this by applying this
objective function here to get rid of them. In the documentation of function
:func:`~moptipyapps.ttp.errors.count_errors`, we explain the different types
of errors that may occur and that are counted.

This objective function plays thus well with encodings that produce infeasible
schedules, such as the very simple :mod:`~moptipyapps.ttp.game_encoding`.
"""


from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.nputils import int_range_to_dtype
from pycommons.types import type_error

from moptipyapps.ttp.game_plan import GamePlan
from moptipyapps.ttp.instance import Instance


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def count_errors(y: np.ndarray, home_streak_min: int,
                 home_streak_max: int, away_streak_min: int,
                 away_streak_max: int, separation_min: int,
                 separation_max: int, temp_1: np.ndarray,
                 temp_2: np.ndarray) -> int:
    """
    Compute the number of errors in a game plan.

    This method counts the total number of the violations of any of the
    following constraints over `D = (n - 1) * rounds` days for `n`-team
    tournaments, where `rounds == 2` for double-round robin. The following
    kinds of errors are counted:

    1.  If a team `A` plays a team `B` on a given day, then `B` must play
        against `A` on that day and if `A` plays at home, then `B` must play
        away. If not, then that's an error. This can result in at most
        `D * n` errors, because each of the `n` teams has (at most) one game
        on each of the `D` days and if the *other* team could play against
        the wrong team (or does not play at all), then that's one error.
    2.  If a team has no other team assigned to play with, this is designated
        by value `0` and causes 1 error. This error also ends all ongoing
        streaks, i.e., may additionally lead to a streak violation of at
        most `max(home_streak_min, away_streak_min) - 1`. However, this
        cannot be more then two errors in sum per day (minus 1, for the
        first day). Also, this error is mutually exclusive with error 1.
        This can result in at most `D * n + (D - 1) * n = (2*D - 1) * n`
        errors. Since this error is also mutually exclusive with the errors
        from constraints 3 to 8 below, this gives us an upper bound of
        `(2*D - 1) * n` errors for all of the constraints (1-8) together.
    3.  A team has a home streak shorter than `home_streak_min`. No such error
        can occur on the first day. This error is mutually exclusive with
        error 2, as the streak violation is already counted there.
        This can result in at most `(D - 1) * n` errors, but this number is
        shared with the following constraints (4-6), because a streak can only
        either be a home streak or an away streak but not both and it can
        either be too short or too long, but not both.
    4.  A team has a home streak longer than `home_streak_max`. No such error
        can occur on the first day. This error is mutually exclusive with
        error 2.
    5.  A team has an away streak shorter than `away_streak_min`. No such
        error can occur on the first day. This error is mutually exclusive
        with error 2, as the streak violation is already counted there.
    6.  A team has an away streak longer than `away_streak_max`. No such
        error can occur on the first day. This error is mutually exclusive
        with error 2.
    7.  Team `A` plays team `B` *again* before the team has played at least
        `separation_min` games against other teams. This error cannot occur
        on the first day and is mutually exclusive with error 2.
        There can be most 1 such error per day for any pairing of teams and
        there are `n//2` pairings per day, giving us an upper bound of
        `D * n//2` errors in total. This error is mutually exclusive with
        the next constraint 8 (and constraint 2).
    8.  Team `A` plays team `B` *again* after the team has played more than
        `separation_max` games against other teams. This error cannot occur
        on the first day and is mutually exclusive with error 2.
        There can be most 1 such error per day for any pairing of teams and
        there are `n//2` pairings per day, giving us an upper bound of
        `D * n//2` errors in total. This error is mutually exclusive with
        the previous constraint 7 (and constraint 2).
    9.  If team `A` plays team `B` at home `a` times, then team `B` must play
        team `A` at home at least `a-1` and at most `a+1` times.
        In total, we have `D*n` games. There cannot be more than `(D*n) - 1`
        such errors. Notice that this kind of error can never occur if we use
        our :mod:`~moptipyapps.ttp.game_encoding` as representation.
    10. Each pairing of teams occurs as same as often, namely `rounds` times,
        with `rounds = 2` for double-round robin.
        In total, we have `D*n` games. There cannot be more than `D*n` such
        errors. Notice that this kind of error can never occur if we use
        our :mod:`~moptipyapps.ttp.game_encoding` as representation.

    The violations are counted on a per-day basis. For example, if
    `home_streak_max` is `3` and a team has a home streak of length `5`, then
    this counts as `2` errors. However, the errors are also counted in a
    non-redundant fashion: If a team `A` violates `separation_min` by, say,
    two days, then this counts as two errors. However, in this case, its
    opposing team `B` would have necessarily incured the exactly same
    violations. These are then not counted.

    As upper bound for the number of errors, we therefore have to add those of
    constraints 2, 9, and 10 and get `(2*D - 1) * n + D*n - 1 + D*n`, which
    gives us `(4*D - 1) * n - 1, where `D = (n - 1) * rounds`.
    The lower bound is obviously `0`.

    :param y: the game plan
    :param home_streak_min: the minimum permitted home streak length
    :param home_streak_max: the maximum permitted home streak length
    :param away_streak_min: the minimum permitted away streak length
    :param away_streak_max: the maximum permitted away streak length
    :param separation_min: the minimum number of games between a repetition
    :param separation_max: the maximum number games between a repetition
    :param temp_1: a temporary `n*(n-1)/2` integer array, which is used to
        hold, for each pairing, when the last game was played
    :param temp_2: a temporary `n,n` integer array, which is used to hold,
        how often each team played each other team
    :returns: the total number of errors. `0` if the game plan is feasible

    >>> count_errors(np.array([[-2, 1], [2, -1]], int),
    ...              1, 3, 1, 3, 1, 2, np.empty(1, int),
    ...              np.empty((2, 2), int))
    1
    >>> count_errors(np.array([[2, -1], [-2, 1]], int),
    ...              1, 3, 1, 3, 1, 2, np.empty(1, int),
    ...              np.empty((2, 2), int))
    1
    >>> count_errors(np.array([[ 2, -1,  4, -3],
    ...                        [ 4,  3, -2, -1],
    ...                        [-2,  1, -4,  3],
    ...                        [-4, -3,  2,  1],
    ...                        [ 3,  4, -1, -2],
    ...                        [-3, -4,  1,  2]], int),
    ...              1, 3, 1, 3, 1, 2, np.empty(6, int),
    ...              np.empty((4, 4), int))
    2
    >>> count_errors(np.array([[ 2, -1,  4, -3],
    ...                        [ 4,  3, -2, -1],
    ...                        [-2,  1, -4,  3],
    ...                        [ 3,  4, -1, -2],
    ...                        [-4, -3,  2,  1],
    ...                        [-3, -4,  1,  2]], int),
    ...              1, 3, 1, 3, 1, 2, np.empty(6, int),
    ...              np.empty((4, 4), int))
    0
    >>> count_errors(np.array([[ 2, -1,  4, -3],
    ...                        [ 4,  3, -2, -1],
    ...                        [ 3,  4, -1, -2],
    ...                        [-2,  1, -4,  3],
    ...                        [-4, -3,  2,  1],
    ...                        [-3, -4,  1,  2]], int),
    ...              1, 3, 1, 3, 1, 2, np.empty(6, int),
    ...              np.empty((4, 4), int))
    0
    >>> count_errors(np.array([[ 2, -1,  4, -3],
    ...                        [ 4,  3, -2, -1],
    ...                        [ 3,  4, -1, -2],
    ...                        [-2,  1, -4,  3],
    ...                        [-4, -3,  2,  1],
    ...                        [-3, -4,  1,  2]], int),
    ...              1, 2, 1, 3, 1, 2, np.empty(6, int),
    ...              np.empty((4, 4), int))
    3
    >>> count_errors(np.array([[ 2, -1,  4, -3],
    ...                        [ 4,  3, -2, -1],
    ...                        [ 3,  4, -1, -2],
    ...                        [-2,  1, -4,  3],
    ...                        [-4, -3,  2,  1],
    ...                        [-3, -4,  1,  2]], int),
    ...              1, 2, 1, 2, 1, 2, np.empty(6, int),
    ...              np.empty((4, 4), int))
    6
    >>> count_errors(np.array([[ 2, -1,  4, -3],
    ...                        [ 4,  3, -2, -1],
    ...                        [ 3,  4, -1, -2],
    ...                        [-2,  1, -4,  3],
    ...                        [-4, -3,  2,  1],
    ...                        [-3, -4,  1,  2]], int),
    ...              1, 3, 1, 3, 1, 1, np.empty(6, int),
    ...              np.empty((4, 4), int))
    6
    """
    days, teams = y.shape  # get the number of days and teams
    errors: int = 0  # the error counter
    temp_1.fill(-1)  # last time the teams played each other
    temp_2.fill(0)
    for team_1 in range(teams):
        col = y[:, team_1]
        team_1_id: int = team_1 + 1
        is_in_home_streak: bool = False
        home_streak_len: int = -1
        is_in_away_streak: bool = False
        away_streak_len: int = -1

        for day, team_2_id in enumerate(col):
            if team_2_id == 0:  # is there no game on this day?
                errors += 1  # no game on this day == 1 error

                if is_in_away_streak:  # this ends away streaks
                    is_in_away_streak = False
                    if away_streak_len < away_streak_min:
                        errors += (away_streak_min - away_streak_len)
                    away_streak_len = -1
                elif is_in_home_streak:  # this ends home streaks, too
                    is_in_home_streak = False
                    if home_streak_len < home_streak_min:
                        errors += (home_streak_min - home_streak_len)
                    home_streak_len = -1
                continue  # nothing more to do here

            if team_2_id > 0:  # our team plays a home game

                # If team_2 > 0, this is a home game. So the other team
                # must have the corresponding index.
                team_2 = team_2_id - 1
                if y[day, team_2] != -team_1_id:
                    errors += 1
                # increase number of home games of team 1 against team 2
                temp_2[team_1, team_2] += 1

                if is_in_home_streak:  # if we are in a home streak...
                    home_streak_len += 1  # ...it continues
                    if home_streak_len > home_streak_max:
                        errors += 1  # too long? add to errors
                else:  # if we are not in home streak, it begins
                    is_in_home_streak = True
                    home_streak_len = 1
                    # if a home streak begins, any away streak ends
                    if is_in_away_streak:
                        if away_streak_len < away_streak_min:
                            errors += (away_streak_min - away_streak_len)
                        away_streak_len = -1
                        is_in_away_streak = False
            else:  # else: team_2 < 0 --> team_2 at home, team_1 away
                team_2 = (-team_2_id) - 1

                # This is an away game, so the other team must have the
                # corresponding id
                if y[day, team_2] != team_1_id:
                    errors += 1

                if is_in_away_streak:  # if we are in an away streak...
                    away_streak_len += 1  # ...the streak continues
                    if away_streak_len > away_streak_max:
                        errors += 1  # away streak too long? add to error
                else:  # team_1 away, but not in away streak
                    is_in_away_streak = True
                    away_streak_len = 1
                    if is_in_home_streak:
                        if home_streak_len < home_streak_min:
                            errors += (home_streak_min - home_streak_len)
                        is_in_home_streak = False
                        home_streak_len = 0

            # now we need to check for the game separation difference
            idx: int = ((team_1 * (team_1 - 1) // 2) + team_2) \
                if team_1 > team_2 \
                else ((team_2 * (team_2 - 1) // 2) + team_1)
            last_time: int = temp_1[idx]
            if last_time >= 0:
                if last_time < day:
                    difference = day - last_time - 1
                    if difference < separation_min:
                        errors += (separation_min - difference)
                    elif difference > separation_max:
                        errors += (difference - separation_max)
                else:
                    continue
            temp_1[idx] = day

    # sum up the team games
    games_per_combo: Final[int] = days // (teams - 1)
    for i in range(teams):
        for j in range(i):
            ij = temp_2[i, j]
            ji = temp_2[j, i]
            errors += abs(ij + ji - games_per_combo)
            diff = abs(ij - ji)
            if diff > 1:
                errors += diff - 1

    return int(errors)


class Errors(Objective):
    """
    Compute the errors in a game plan.

    This objective function encompasses all the constraints imposed on
    standard TTP instances in one summarizing number. See the documentation
    of :func:`count_errors` for more information.
    """

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the errors objective function.

        :param instance: the TTP instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        super().__init__()

        #: the TTP instance
        self.instance: Final[Instance] = instance
        n: Final[int] = instance.n_cities
        # the data type for the temporary arrays
        dtype: Final[np.dtype] = int_range_to_dtype(
            -1, (n - 1) * instance.rounds)
        #: the internal temporary array 1
        self.__temp_1: Final[np.ndarray] = np.empty(n * (n - 1) // 2, dtype)
        #: the internal temporary array 2
        self.__temp_2: Final[np.ndarray] = np.empty((n, n), dtype)

    def evaluate(self, x: GamePlan) -> int:
        """
        Count the errors in a game plan as objective value.

        :param x: the game plan
        :return: the number of errors in the plan
        """
        inst: Final[Instance] = x.instance
        return count_errors(x, inst.home_streak_min, inst.home_streak_max,
                            inst.away_streak_min, inst.away_streak_max,
                            inst.separation_min, inst.separation_max,
                            self.__temp_1, self.__temp_2)

    def lower_bound(self) -> int:
        """
        Obtain the lower bound for errors: `0`, which means error-free.

        :return: `0`
        """
        return 0

    def upper_bound(self) -> int:
        """
        Compute upper bound for errors: `(4*D - 1) * n - 1`.

        Here `D` is the number of days, `n` is the number of teams, and
        `D = (n - 1) * rounds`. See the documentation of :func:`count_errors`.

        :return: `(4*D - 1) * n - 1`
        """
        n: Final[int] = self.instance.n_cities
        rounds: Final[int] = self.instance.rounds
        days: Final[int] = (n - 1) * rounds
        return (4 * days - 1) * n - 1

    def is_always_integer(self) -> bool:
        """
        State that this objective function is always integer-valued.

        :return: `True`
        """
        return True

    def __str__(self) -> str:
        """Get the name of this objective function."""
        return "errors"
