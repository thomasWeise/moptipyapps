"""
A permutation-with-repetition-based encoding based on games.

A point in the search space is a permutation (potentially with repetitions)
that can be translated to a :class:`~moptipyapps.ttp.game_plan.GamePlan`.
Each value `v` in the permutation represents a game to be played by two of
the `n` teams. There are `n(n-1)` possible games between `n` teams,
distinguishing home and away teams. Given a value `v` from `0..n(n-1)-1`,
we can get the zero-based index of the home team as
`home_idx = (game // (n - 1)) % n`. The away index is computed in two steps,
first we set `away_idx = game % (n - 1)` and if `away_idx >= home_idx`, we
do `away_idx = away_idy + 1`. (Because a team can never play against itself,
the situation that `home_idx == away_idx` does not need to be represented, so
we can "skip" over this possible value by doing the `away_idx = away_idy + 1`
and thus get a more "compact" numeric range for the permutation elements.)

A game schedule for any round-robin tournament with any given number of rounds
can then be represented as permutation (potentially with repetitions) of these
game values. In the decoding procedure, it is processed from beginning to end
each game is then placed into the earliest slot not already occupied by
another game. In other words, it is placed at the earliest day at which both
involved teams do not yet have other games. If no such slot is available, this
game is not placed at all. In this case, there will be some zeros in the game
plan after the encoding. No other constraint is considered at this stage.

In other words, this encoding may produce game plans that violate constraints.
It does not care about the streak length constraints.
It does not ensure that each team always has a game.
Therefore, it should only be used in conjunction with objective functions that
force the search towards feasible solutions, such as the
:mod:`~moptipyapps.ttp.errors` objective.
"""


from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.encoding import Encoding
from moptipy.spaces.permutations import Permutations
from pycommons.types import check_int_range

from moptipyapps.ttp.instance import Instance


def search_space_for_n_and_rounds(n: int, rounds: int) -> Permutations:
    """
    Create a proper search space for the given number of teams and rounds.

    If the instance prescribes a double-round robin tournament, then this
    is just the :meth:`~moptipy.spaces.permutations.Permutations.standard`
    permutations set. Otherwise, it will be a permutation where some
    elements are omitted (for
    :attr:`~moptipyapps.ttp.instance.Instance.rounds` == 1) or duplicated
    (if :attr:`~moptipyapps.ttp.instance.Instance.rounds` > 2).

    If an odd number of rounds is played, then it is not possible that all
    teams have the same number of games at home and away. Then, the
    permutation is generated such that, if the highest numbers of games at
    home for any team is `k`, no other team has less than `k - 1` games at
    home. If the number of rounds is even, then all teams will have the
    same number of home and away games, that is, the number of teams
    divided by two and multiplied by the number of rounds.

    :param n: the number of teams
    :param rounds: the number of rounds
    :return: the search space

    >>> ";".join(map(str, search_space_for_n_and_rounds(2, 2).blueprint))
    '0;1'
    >>> ";".join(map(str, search_space_for_n_and_rounds(2, 3).blueprint))
    '0;1;1'
    >>> ";".join(map(str, search_space_for_n_and_rounds(2, 4).blueprint))
    '0;0;1;1'
    >>> ";".join(map(str, search_space_for_n_and_rounds(2, 5).blueprint))
    '0;0;1;1;1'
    >>> ";".join(map(str, search_space_for_n_and_rounds(3, 1).blueprint))
    '1;2;5'
    >>> ";".join(map(str, search_space_for_n_and_rounds(3, 2).blueprint))
    '0;1;2;3;4;5'
    >>> ";".join(map(str, search_space_for_n_and_rounds(3, 3).blueprint))
    '0;1;1;2;2;3;4;5;5'
    >>> ";".join(map(str, search_space_for_n_and_rounds(4, 1).blueprint))
    '1;2;3;7;8;10'
    >>> ";".join(map(str, search_space_for_n_and_rounds(4, 2).blueprint))
    '0;1;2;3;4;5;6;7;8;9;10;11'
    >>> ";".join(map(str, search_space_for_n_and_rounds(4, 3).blueprint))
    '0;1;1;2;2;3;3;4;5;6;7;7;8;8;9;10;10;11'
    >>> ";".join(map(str, search_space_for_n_and_rounds(4, 4).blueprint))
    '0;0;1;1;2;2;3;3;4;4;5;5;6;6;7;7;8;8;9;9;10;10;11;11'
    >>> ";".join(map(str, search_space_for_n_and_rounds(4, 5).blueprint))
    '0;0;1;1;1;2;2;2;3;3;3;4;4;5;5;6;6;7;7;7;8;8;8;9;9;10;10;10;11;11'
    >>> ";".join(map(str, search_space_for_n_and_rounds(5, 1).blueprint))
    '1;2;4;7;9;10;13;15;16;18'
    >>> ";".join(map(str, search_space_for_n_and_rounds(5, 2).blueprint))
    '0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19'
    >>> ";".join(map(str, search_space_for_n_and_rounds(5, 3).blueprint))
    '0;1;1;2;2;3;4;4;5;6;7;7;8;9;9;10;10;11;12;13;13;14;15;15;16;16;17;18;\
18;19'
    """
    check_int_range(n, "n", 2, 100000)
    check_int_range(n, "rounds", 1, 100000)
    div: Final[int] = n - 1
    games: Final[list[int]] = []
    order: bool = False
    for r in range(rounds):
        # If we have an odd round of games, the very last round needs
        # to be treated differently to ensure that the home-away games
        # distribution is fair.
        normal: bool = (r < (rounds - 1)) or ((rounds % 2) == 0)
        for i in range(n):  # for each city
            for j in range(i):  # for each other city
                order = ((r % 2) == 0) if normal else (not order)
                m1 = i if order else j  # determine home city
                m2 = j if order else i  # determine away city
                if m2 > m1:
                    m2 -= 1
                games.append(m1 * div + m2)  # add encoded game tuple
    games.sort()
    return Permutations(games)  # create permutations with repetition


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def map_games(x: np.ndarray, y: np.ndarray) -> None:
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
    >>> teams = 2
    >>> rounds = 2
    >>> perm = search_space_for_n_and_rounds(teams, rounds).blueprint
    >>> dest = np.empty((rounds * (teams - 1), teams),
    ...                 int_range_to_dtype(-teams, teams))
    >>> map_games(perm, dest)
    >>> print(dest)
    [[ 2 -1]
     [-2  1]]
    >>> teams = 4
    >>> rounds = 2
    >>> perm = search_space_for_n_and_rounds(teams, rounds).blueprint
    >>> dest = np.empty((rounds * (teams - 1), teams),
    ...                 int_range_to_dtype(-teams, teams))
    >>> map_games(perm, dest)
    >>> print(dest)
    [[ 2 -1  4 -3]
     [ 3  4 -1 -2]
     [ 4  3 -2 -1]
     [-2  1 -4  3]
     [-3 -4  1  2]
     [-4 -3  2  1]]
    """
    y.fill(0)  # first zero the output matrix
    days, n = y.shape  # the number of days and teams to be scheduled
    div: Final[int] = n - 1  # the divisor for permutation values -> teams

    for game in x:
        home_idx: int = (game // div) % n  # home idx is in 0..n-1
        away_idx: int = game % div  # away index in 0..n-2
        if away_idx >= home_idx:  # "A vs. A" games impossible
            away_idx += 1  # away index in 0..n-1, but != home_idx

        for day in range(days):  # iterate over all possible rows for game
            if (y[day, home_idx] != 0) or (y[day, away_idx] != 0):
                continue  # day already blocked
            y[day, home_idx] = away_idx + 1
            y[day, away_idx] = -(home_idx + 1)
            break


class GameEncoding(Encoding):
    """An encoding that transforms strings of games to game plans."""

    def __init__(self, instance: Instance) -> None:
        """
        Create the game-based encoding.

        :param instance: the instance
        """
        super().__init__()
        #: the instance
        self.instance: Final[Instance] = instance
        self.decode = map_games  # type: ignore

    def search_space(self) -> Permutations:
        """
        Create a proper search space for this game-based encoding.

        The search space is a set of :mod:`~moptipy.spaces.permutations` that
        represents all the games that can take place in the tournament.
        Depending on the number of
        :attr:`~moptipyapps.ttp.instance.Instance.rounds` in the tournament,
        some games may appear multiple times. Home and away games are
        distributed in a fair and deterministic mannner between the teams.

        :return: the search space

        >>> inst = Instance.from_resource("circ4")
        >>> inst.n_cities
        4
        >>> inst.rounds
        2
        >>> ";".join(map(str, GameEncoding(inst).search_space().blueprint))
        '0;1;2;3;4;5;6;7;8;9;10;11'
        >>> inst = Instance(inst.name, inst, inst.teams, inst.rounds,
        ...                 inst.home_streak_min, inst.home_streak_max,
        ...                 inst.away_streak_min, inst.away_streak_max,
        ...                 inst.separation_min, inst.separation_max)
        >>> inst.rounds = 1  # modify number of rounds for copied instance
        >>> ";".join(map(str, GameEncoding(inst).search_space().blueprint))
        '1;2;3;7;8;10'
        >>> inst.rounds = 3  # modify number of rounds for copied instance
        >>> ";".join(map(str, GameEncoding(inst).search_space().blueprint))
        '0;1;1;2;2;3;3;4;5;6;7;7;8;8;9;10;10;11'
        >>> inst.rounds = 4  # modify number of rounds for copied instance
        >>> ";".join(map(str, GameEncoding(inst).search_space().blueprint))
        '0;0;1;1;2;2;3;3;4;4;5;5;6;6;7;7;8;8;9;9;10;10;11;11'
        >>> inst.rounds = 5  # modify number of rounds for copied instance
        >>> ";".join(map(str, GameEncoding(inst).search_space().blueprint))
        '0;0;1;1;1;2;2;2;3;3;3;4;4;5;5;6;6;7;7;7;8;8;8;9;9;10;10;10;11;11'
        """
        return search_space_for_n_and_rounds(
            self.instance.n_cities, self.instance.rounds)
