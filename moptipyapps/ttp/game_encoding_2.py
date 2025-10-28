"""
A permutation-with-repetition-based encoding based on games with rescheduling.

This game-based encoding method is an extension of the method presented in
:mod:`~moptipyapps.ttp.game_encoding`. The method presented in
:mod:`~moptipyapps.ttp.game_encoding` takes a permutation of games and fills
these games into a game plan one-by-one. For each game it tries to find the
earliest possible slot where it can be scheduled, i.e., the earliest slot
where both involved teams do not yet have a game scheduled.

During this process, it can happen that a game cannot be added to the game
plan, because on each day, at least one of the two teams has already a game
scheduled. In such a case, the original encoding will simply drop the game.
This results in "byes" in the game plan.

This new encoding tries to reduce the number of such byes. It does this as
follows: It still processes the game permutations from front to end and
adds the games to the schedule. If one game cannot be added to the schedule
due to the above reason, it will go through game plan again from beginning
to end. This time, it will look for a day on which only *one* of the two teams
has a game scheduled already. It will then take this already-scheduled game
and try to schedule it on a *later* day. This is a recursive process: If no
later slot is found where both teams involved have no game scheduled, we will
again look for a day on which only one of the teams have scheduled a game and
try to move that to a later slot. Since this will be an even later slot, the
recursion will never go too deep. Either way, if a later slot is found, the
game can be moved, meaning that the original game blocking our "current" game
can be moved as well. Then the new game can be inserted.

If no such slot can be found, then we try to insert the game by searching
from the back end and moving other games forward in the same recursive
fashion. The hope is that, this way, the number of "byes" in the schedule can
be reduced. If the number of "byes" is reduced, then there will automatically
fewer constraint violations. This makes it more likely that we discover a
feasible schedule earlier in the search.

Furthermore, the process might even make it easier for a search to move from
one feasible schedule to another one. There would hopefully be fewer
permutations leading to infeasible schedules and fewer infeasible schedules
between two feasible ones.

Of course, all of that comes at the cost of increased runtime.
"""


from typing import Final

import numba  # type: ignore
import numpy as np

from moptipyapps.ttp.game_encoding import GameEncoding
from moptipyapps.ttp.game_plan_space import GamePlanSpace
from moptipyapps.ttp.instance import Instance


# Due to the recursion, caching must be disabled or we get a segfault.
@numba.njit(cache=False, nogil=True, fastmath=True, boundscheck=True)
def _re_schedule(y: np.ndarray, index: int | np.ndarray,
                 day: int, end: int, direction: int) -> bool:
    """
    Try to re-schedule a game at the given index.

    This function tries to take the game at the given index and given day and
    attempts to move it to a later day. If it succeeds, it returns `True`, if
    rescheduling was not possible, it returns `False`.

    :param y: the schedule
    :param index: the index
    :param day: the day
    :param end: the end day
    :return: `True` if the game was successfully re-scheduled, `False`
        otherwise

    >>> dest = np.array([
    ...     [ 2, -1,  4, -3],
    ...     [ 3,  4, -1, -2],
    ...     [ 4,  3, -2, -1],
    ...     [-2,  1, -4,  3],
    ...     [-3, -4,  1,  2],
    ...     [-4, -3,  2,  1]])
    >>> _re_schedule(dest, 0, 2, 6, 1)
    False
    >>> dest = np.array([
    ...     [ 2, -1,  4, -3],
    ...     [ 3,  4, -1, -2],
    ...     [ 4,  3, -2, -1],
    ...     [ 0,  1, -4,  0],
    ...     [-3, -4,  1,  2],
    ...     [-4, -3,  2,  1]])
    >>> _re_schedule(dest, 0, 2, 6, 1)
    True
    >>> print(dest)
    [[ 2 -1  4 -3]
     [ 3  4 -1 -2]
     [ 0  3 -2  0]
     [ 4  1 -4 -1]
     [-3 -4  1  2]
     [-4 -3  2  1]]
    >>> dest = np.array([
    ...     [ 2, -1,  4, -3],
    ...     [ 3,  4, -1, -2],
    ...     [ 0,  3, -2, -1],
    ...     [-2,  0,  0,  3],
    ...     [-3, -4,  1,  2],
    ...     [-4, -3,  2,  1]])
    >>> _re_schedule(dest, 2, 1, 6, 1)
    True
    >>> print(dest)
    [[ 2 -1  4 -3]
     [ 0  4  0 -2]
     [ 3  0 -1 -1]
     [-2  3 -2  3]
     [-3 -4  1  2]
     [-4 -3  2  1]]
    """
    home_idx = int(index)  # We first guess that this is the home team
    away_idx = int(y[day, index])  # and that this is the away team.
    if away_idx < 0:  # The home team actually plays away.
        away_idx, home_idx = home_idx, (-away_idx) - 1  # Swap.
    else:  # We were right.
        away_idx -= 1  # So the ID of the away team -1 is its index.
    if _schedule(y, home_idx, away_idx, day + direction,
                 end, direction, True):  # Try to re-schedule the game.
        y[day, home_idx] = 0  # Set first slot to empty.
        y[day, away_idx] = 0  # Set last slot to empty.
        return True  # We successfully rescheduled the game.
    return False  # We failed to recursively reschedule.


# Due to the recursion, caching must be disabled or we get a segfault.
@numba.njit(cache=False, nogil=True, fastmath=True, boundscheck=True)
def _schedule(y: np.ndarray, home_idx: int, away_idx: int, start: int,
              end: int, direction: int, is_recursive: bool) -> bool:
    """
    Try to find a slot in the day range where the given game can be placed.

    :param y: the destination array
    :param home_idx: the home index
    :param away_idx: the away index
    :param start: the starting day
    :param end: the ending day
    :return: `True` if the game could be placed, `False` otherwise.

    >>> dest = np.array([
    ...     [ 2, -1,  4, -3],
    ...     [ 3,  4, -1, -2],
    ...     [ 4,  3, -2, -1],
    ...     [-2,  1, -4,  3],
    ...     [-3, -4,  1,  2],
    ...     [-4, -3,  2,  1]])
    >>> _schedule(dest, 1, 2, 0, 6, 1, True)
    False
    >>> dest = np.array([
    ...     [ 2, -1,  4, -3],
    ...     [ 3,  4, -1, -2],
    ...     [ 4,  0,  0, -1],
    ...     [-2,  1, -4,  3],
    ...     [-3, -4,  1,  2],
    ...     [-4, -3,  2,  1]])
    >>> _schedule(dest, 1, 2, 0, 6, 1, True)
    True
    >>> print(dest)
    [[ 2 -1  4 -3]
     [ 3  4 -1 -2]
     [ 4  3 -2 -1]
     [-2  1 -4  3]
     [-3 -4  1  2]
     [-4 -3  2  1]]
    >>> dest = np.array([
    ...     [ 2, -1,  4, -3],
    ...     [ 3,  4, -1, -2],
    ...     [ 0,  3, -2, -1],
    ...     [-2,  0,  0,  3],
    ...     [-3, -4,  1,  2],
    ...     [-4, -3,  2,  1]])
    >>> _schedule(dest, 0, 2, 0, 6, 1, True)
    True
    >>> print(dest)
    [[ 2 -1  4 -3]
     [ 3  4 -1 -2]
     [ 3  0 -1 -1]
     [-2  3 -2  3]
     [-3 -4  1  2]
     [-4 -3  2  1]]
    """
    empty_slots_needed = 2  # Number of required open slots for a game.
    while True:
        for day in range(start, end, direction):  # iterate over all days
            can_use_home: bool = y[day, home_idx] == 0  # Check for empty slot
            can_use_away: bool = y[day, away_idx] == 0  # Check for empty slot
            if can_use_home + can_use_away < empty_slots_needed:
                continue  # Too few empty slots: Try with next day.
            if ((can_use_home or _re_schedule(  # At least one empty slot?
                    y, home_idx, day, end, direction)) and (
                    can_use_away or _re_schedule(
                    y, away_idx, day, end, direction))):
                # There either were two empty slots, or one and the other game
                # was rescheduled. So we can schedule the game.
                y[day, home_idx] = away_idx + 1
                y[day, away_idx] = -(home_idx + 1)
                return True  # Success!

        # After we tried finding a perfect slot, we now try with just one slot.
        if empty_slots_needed >= 2:
            empty_slots_needed = 1  # Try requiring just one slot.
            continue  # And execute scheduling loop again.
        if is_recursive:  # Only if this is the root call, we can go one
            return False  # after the attempt requiring 1 slot failed.
        is_recursive = True  # But we can only do this once.
        # If we got here, we are in the root call.
        # We first tried to find two open slots and failed.
        # Then we tried to find a single open slot and tried to re-schedule
        # the other game to a later point in time ... an failed.
        # Now we try rescheduling games towards an early time.
        start = end - 1
        end = direction = -1


# Due to the recursion, caching must be disabled or we get a segfault.
@numba.njit(cache=False, fastmath=True, boundscheck=True)
def map_games_2(x: np.ndarray, y: np.ndarray) -> None:
    """
    Translate a permutation of games to a game plan.

    This is a straightforward decoding that places the games into the map one
    by one. Each game is placed at the earliest slot in which it can be
    placed. If a game cannot be placed, it is ignored. This will lead to many
    errors, which can be counted via the :mod:`~moptipyapps.ttp.errors`
    objective.

    :param x: the source permutation
    :param y: the destination game plan

    >>> from moptipy.utils.nputils import int_range_to_dtype
    >>> from moptipyapps.ttp.game_encoding import (
    ...     search_space_for_n_and_rounds)
    >>> teams = 2
    >>> rounds = 2
    >>> perm = search_space_for_n_and_rounds(teams, rounds).blueprint
    >>> dest = np.empty((rounds * (teams - 1), teams),
    ...                 int_range_to_dtype(-teams, teams))
    >>> map_games_2(perm, dest)
    >>> print(dest)
    [[ 2 -1]
     [-2  1]]
    >>> teams = 4
    >>> rounds = 2
    >>> perm = search_space_for_n_and_rounds(teams, rounds).blueprint
    >>> dest = np.empty((rounds * (teams - 1), teams),
    ...                 int_range_to_dtype(-teams, teams))
    >>> map_games_2(perm, dest)
    >>> print(dest)
    [[ 2 -1  4 -3]
     [ 3  4 -1 -2]
     [ 4  3 -2 -1]
     [-2  1 -4  3]
     [-3 -4  1  2]
     [-4 -3  2  1]]
     >>> from moptipyapps.ttp.instance import Instance
     >>> inst = Instance.from_resource("circ10")
     >>> perm = np.array([73,77,55,74,21,20,3,11,63,19,38,8,27,47,88,16,75,
     ...     45,89,36,24,80,17,40,0,32,53,82,31,13,12,66,71,6,87,84,2,60,61,
     ...     10,30,58,49,57,54,23,26,46,59,42,29,62,22,18,50,78,37,34,65,43,
     ...     48,85,86,83,56,41,5,81,52,15,51,9,4,76,7,69,67,33,79,72,35,25,
     ...     1,14,70,44,68,64,28,39])
    >>> teams = 10
    >>> rounds = 2
    >>> dest = np.empty((rounds * (teams - 1), teams),
    ...                 int_range_to_dtype(-teams, teams))
    >>> map_games_2(perm, dest)
    >>> print(dest)
    [[ -8  -9   5   7  -3  10  -4   1   2  -6]
     [  5  -7   4  -3  -1  -9   2 -10   6   8]
     [ 10   4  -9  -2   6  -5   8  -7   3  -1]
     [ -4  -3   2   1  -7   8   5  -6 -10   9]
     [ -6   9  -5  -8   3   1 -10   4  -2   7]
     [ -5  10  -6  -9   1   3  -8   7   4  -2]
     [  2  -1   8   6   7  -4  -5  -3  10  -9]
     [  8 -10   6   5  -4  -3   9  -1  -7   2]
     [  4   6   7  -1   9  -2  -3  10  -5  -8]
     [ -7   5  -8 -10  -2   9   1   3  -6   4]
     [-10   3  -2  -7  -6   5   4  -9   8   1]
     [  3  -4  -1   2  10  -8  -9   6   7  -5]
     [  9  -8  10  -5   4  -7   6   2  -1  -3]
     [ -3  -6   1   9   8   2  10  -5  -4  -7]
     [ -2   1   9   8 -10   7  -6  -4  -3   5]
     [  7   8 -10  -6  -9   4  -1  -2   5   3]
     [ -9   7  -4   3  -8 -10  -2   5   1   6]
     [  6  -5  -7  10   2  -1   3   9  -8  -4]]
    >>> int(dest.shape[0] * dest.shape[1] - np.count_nonzero(dest))
    0
    >>> from moptipyapps.ttp.game_encoding import re_encode
    >>> re_encode(perm, dest)
    >>> print(perm)
    [21 32 53 63 73  3 20 55 77 88  8 11 40 60 74 19 27 51 58 89 16 38 45 66
     87 17 36 47 69 75  0 24 31 41 80  6 22 30 61 82  2 13 23 43 71 12 52 54
     65 84 10 49 57 79 81  1 28 44 68 78  7 26 39 59 64 18 34 42 46 62  9 25
     33 50 85  5 15 48 76 83 14 29 67 72 86  4 35 37 56 70]
    >>> map_games_2(perm, dest)
    >>> print(dest)
    [[ -8  -9   5   7  -3  10  -4   1   2  -6]
     [  5  -7   4  -3  -1  -9   2 -10   6   8]
     [ 10   4  -9  -2   6  -5   8  -7   3  -1]
     [ -4  -3   2   1  -7   8   5  -6 -10   9]
     [ -6   9  -5  -8   3   1 -10   4  -2   7]
     [ -5  10  -6  -9   1   3  -8   7   4  -2]
     [  2  -1   8   6   7  -4  -5  -3  10  -9]
     [  8 -10   6   5  -4  -3   9  -1  -7   2]
     [  4   6   7  -1   9  -2  -3  10  -5  -8]
     [ -7   5  -8 -10  -2   9   1   3  -6   4]
     [-10   3  -2  -7  -6   5   4  -9   8   1]
     [  3  -4  -1   2  10  -8  -9   6   7  -5]
     [  9  -8  10  -5   4  -7   6   2  -1  -3]
     [ -3  -6   1   9   8   2  10  -5  -4  -7]
     [ -2   1   9   8 -10   7  -6  -4  -3   5]
     [  7   8 -10  -6  -9   4  -1  -2   5   3]
     [ -9   7  -4   3  -8 -10  -2   5   1   6]
     [  6  -5  -7  10   2  -1   3   9  -8  -4]]
     >>> inst = Instance.from_resource("con14")
     >>> perm = np.array([70,114,54,51,98,31,49,46,148,169,174,151,155,110,
     ...     75,118,128,81,171,19,133,93,5,91,47,1,84,109,142,121,7,40,163,
     ...     61,20,96,165,177,160,39,73,37,101,108,119,65,2,117,164,106,60,
     ...     145,105,90,170,74,8,26,94,10,22,86,120,154,89,66,77,178,140,48,
     ...     0,52,159,25,176,135,63,57,76,161,68,124,162,29,55,80,24,45,85,
     ...     115,131,100,17,53,27,38,95,158,104,175,87,123,167,97,79,112,
     ...     172,127,136,12,44,50,102,69,144,143,113,4,181,88,166,150,59,
     ...     141,72,125,116,132,14,147,153,129,111,92,62,6,32,82,16,179,21,
     ...     3,126,134,122,149,146,36,107,34,103,42,41,18,35,78,28,137,43,
     ...     33,71,99,139,56,23,138,67,168,130,30,180,152,83,9,13,173,11,
     ...     157,156,64,15,58])
    >>> teams = 14
    >>> rounds = 2
    >>> from moptipy.utils.nputils import int_range_to_dtype
    >>> dest = np.empty((rounds * (teams - 1), teams),
    ...                 int_range_to_dtype(-teams, teams))
    >>> from moptipyapps.ttp.game_encoding_2 import map_games_2
    >>> map_games_2(perm, dest)
    >>> print(dest)
    [[ -8 -10  -5  14   3   7  -6   1  12   2 -13  -9  11  -4]
     [-14  -6   7  12  11   2  -3   9  -8  13  -5  -4 -10   1]
     [  7   8 -14   9 -10 -12  -1  -2  -4   5  13   6 -11   3]
     [ -5  11  -8  -7   1 -14   4   3 -12 -13  -2   9  10   6]
     [  3  -5  -1 -11   2  10  -9 -13   7  -6   4  14   8 -12]
     [  9  -3   2  10 -13  12   8  -7  -1  -4  14  -6   5 -11]
     [-10  -4  13   2 -11  -9  14  12   6   1   5  -8  -3  -7]
     [ -4   9 -10   1   7  -8  -5   6  -2   3 -14  13 -12  11]
     [ -6 -11 -12  -8  10   1  13   4 -14  -5   2   3  -7   9]
     [  4 -14 -13  -1  -9  11  10 -12   5  -7  -6   8   3   2]
     [ 10  -7   5   8  -3  14   2  -4 -13  -1  12 -11   9  -6]
     [ 12  14  -9 -10  13 -11  -8   7   3   4   6  -1  -5  -2]
     [ -3  -9   1  11  -8  13  12   5   2 -14  -4  -7  -6  10]
     [  2  -1  -7 -13  -6   5   3 -14  10  -9 -12  11   4   8]
     [-11 -12  14  -5   4 -13   9 -10  -7   8   1   2   6  -3]
     [ -9   3  -2  -6 -14   4 -13  11   1  12  -8 -10   7   5]
     [-12  13   8   5  -4 -10 -14  -3  11   6  -9   1  -2   7]
     [  8   6  10 -14 -12  -2  11  -1  13  -3  -7   5  -9   4]
     [ 14  -8 -11   6   9  -4 -10   2  -5   7   3 -13  12  -1]
     [ -2   1  -6  13  12   3 -11  14 -10   9   7  -5  -4  -8]
     [ 11   5  12   7  -2   9  -4  13  -6  14  -1  -3  -8 -10]
     [  6  10  11 -12  -7  -1   5  -9   8  -2  -3   4 -14  13]
     [  5 -13  -4   3  -1   8 -12  -6  14  11 -10   7   2  -9]
     [-13   7   6  -9  14  -3  -2 -11   4 -12   8  10   1  -5]
     [ -7  12   4  -3   6  -5   1  10 -11  -8   9  -2  14 -13]
     [ 13   4   9  -2   8  -7   6  -5  -3 -11  10 -14  -1  12]]
    """
    y.fill(0)  # first zero the output matrix
    days, n = y.shape  # the number of days and teams to be scheduled
    div: Final[int] = n - 1  # the divisor for permutation values -> teams

    for game in x:
        g = int(game)  # make sure that the indices are Python integers
        home_idx: int = (g // div) % n  # home idx is in 0..n-1
        away_idx: int = g % div  # away index in 0..n-2
        if away_idx >= home_idx:  # "A vs. A" games impossible
            away_idx += 1  # away index in 0..n-1, but != home_idx
        _schedule(y, home_idx, away_idx, 0, days, 1, False)


class GameEncoding2(GameEncoding):
    """A second encoding that transforms strings of games to game plans."""

    def __init__(self, instance: Instance) -> None:
        """
        Create the game-based encoding that can reschedule games.

        :param instance: the instance
        """
        super().__init__(instance)
        self.decode = map_games_2  # type: ignore

        # We invoke all the functions in order to enforce proper compilation.
        # Numba has some problems with recursive functions.
        # Therefore, we try to enforce that all functions used here are
        # invoked in an order where their parameters can be fully inferred.
        y: np.ndarray = GamePlanSpace(instance).create()
        y.fill(1)
        _re_schedule(y, 0, 0, 1, 1)
        _schedule(y, 0, 1, 0, 1, 1, False)
        y[0, 0] = 0
        _schedule(y, 0, 1, 0, 1, 1, True)
        y.fill(0)
        _re_schedule(y, 0, 0, 1, 1)
        _schedule(y, 0, 1, 0, 1, 1, False)
        x: np.ndarray = self.search_space().blueprint
        x.sort()
        map_games_2(x, y)
