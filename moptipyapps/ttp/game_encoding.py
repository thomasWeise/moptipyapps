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
do `away_idx = away_idx + 1`. (Because a team can never play against itself,
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

In :mod:`~moptipyapps.ttp.game_encoding_2`, we present a modified version of
this encoding procedure that tries to reduce the number of zeros, i.e.,
"byes," in the resulting schedule.
"""


from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.encoding import Encoding
from moptipy.spaces.permutations import Permutations
from pycommons.types import check_int_range

from moptipyapps.ttp.instance import Instance


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def game_to_id(home: int | np.ndarray, away: int | np.ndarray, n: int) -> int:
    """
    Encode a game to a game ID.

    :param home: the home team
    :param away: the away team
    :param n: the total number of teams
    :return: the game ID

    >>> game_to_id(0, 1, 2)
    0
    >>> game_to_id(1, 0, 2)
    1

    >>> game_to_id(0, 1, 4)
    0
    >>> game_to_id(0, 2, 4)
    1
    >>> game_to_id(0, 3, 4)
    2
    >>> game_to_id(1, 0, 4)
    3
    >>> game_to_id(1, 2, 4)
    4
    >>> game_to_id(1, 3, 4)
    5
    >>> game_to_id(2, 0, 4)
    6
    >>> game_to_id(2, 1, 4)
    7
    >>> game_to_id(2, 3, 4)
    8
    >>> game_to_id(3, 0, 4)
    9
    >>> game_to_id(3, 1, 4)
    10
    >>> game_to_id(3, 2, 4)
    11
    >>> game_to_id(5, 4, 10)
    49
    """
    a = int(home)
    b = int(away)
    if b > a:
        b -= 1
    return (a * (n - 1)) + b


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

    >>> ",".join(map(str, search_space_for_n_and_rounds(2, 2).blueprint))
    '0,1'
    >>> ",".join(map(str, search_space_for_n_and_rounds(2, 3).blueprint))
    '0,1,1'
    >>> ",".join(map(str, search_space_for_n_and_rounds(2, 4).blueprint))
    '0,0,1,1'
    >>> ",".join(map(str, search_space_for_n_and_rounds(2, 5).blueprint))
    '0,0,1,1,1'
    >>> ",".join(map(str, search_space_for_n_and_rounds(3, 1).blueprint))
    '1,2,5'
    >>> ",".join(map(str, search_space_for_n_and_rounds(3, 2).blueprint))
    '0,1,2,3,4,5'
    >>> ",".join(map(str, search_space_for_n_and_rounds(3, 3).blueprint))
    '0,1,1,2,2,3,4,5,5'
    >>> ",".join(map(str, search_space_for_n_and_rounds(4, 1).blueprint))
    '1,2,3,7,8,10'
    >>> ",".join(map(str, search_space_for_n_and_rounds(4, 2).blueprint))
    '0,1,2,3,4,5,6,7,8,9,10,11'
    >>> ",".join(map(str, search_space_for_n_and_rounds(4, 3).blueprint))
    '0,1,1,2,2,3,3,4,5,6,7,7,8,8,9,10,10,11'
    >>> ",".join(map(str, search_space_for_n_and_rounds(4, 4).blueprint))
    '0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11'
    >>> ",".join(map(str, search_space_for_n_and_rounds(4, 5).blueprint))
    '0,0,1,1,1,2,2,2,3,3,3,4,4,5,5,6,6,7,7,7,8,8,8,9,9,10,10,10,11,11'
    >>> ",".join(map(str, search_space_for_n_and_rounds(5, 1).blueprint))
    '1,2,4,7,9,10,13,15,16,18'
    >>> ",".join(map(str, search_space_for_n_and_rounds(5, 2).blueprint))
    '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19'
    >>> ",".join(map(str, search_space_for_n_and_rounds(5, 3).blueprint))
    '0,1,1,2,2,3,4,4,5,6,7,7,8,9,9,10,10,11,12,13,13,14,15,15,16,16,17,18,\
18,19'
    """
    check_int_range(n, "n", 2, 100000)
    check_int_range(n, "rounds", 1, 100000)
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
                games.append(game_to_id(
                    i if order else j, j if order else i, n))
    games.sort()
    return Permutations(games)  # create permutations with repetition


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def re_encode(x: np.ndarray, y: np.ndarray) -> None:
    """
    Re-encode a game plan to a permutation.

    `x` must be a valid permutation. It is then transformed based on the game
    plan `y` to represent `y` in a straightforward manner. The general
    contract of this function is that:

    - `y` can be a game plan of any kind, with all sorts of game scheduling
       error except one: If team A plays team B in one slot, then team B must
       also play against team A in that slot and exactly one of them is the
       home team and one of them is the away team. Apart from that, there can
       be arbitrary byes or streak violations.
    - `x` must a valid game permutation, but can be entirely unrelated
       to `y`.
    - This function will then transform `x` such that `map_games(x, z)` will
      result in a game plan such that `z = y`.

    Due to the possibility of byes, it can happen that after
    `map_games(x1, z)` and `re_encode(x2, z)` it may be that `x1 != x2`.
    The reason is that the order of game IDs n `x2` for games that were not
    scheduled in `z` will be undefined.

    :param x: the permutation
    :param y: the game plan

    >>> from moptipyapps.ttp.instance import Instance
    >>> inst = Instance.from_resource("con14")
    >>> perm = np.array([70,114,54,51,98,31,49,46,148,169,174,151,155,110,
    ...     75,118,128,81,171,19,133,93,5,91,47,1,84,109,142,121,7,40,163,
    ...     61,20,96,165,177,160,39,73,37,101,108,119,65,2,117,164,106,60,
    ...     145,105,90,170,74,8,26,94,10,22,86,120,154,89,66,77,178,140,48,
    ...     0,52,159,25,176,135,63,57,76,161,68,124,162,29,55,80,24,45,85,
    ...     115,131,100,17,53,27,38,95,158,104,175,87,123,167,97,79,112,172,
    ...     127,136,12,44,50,102,69,144,143,113,4,181,88,166,150,59,141,72,
    ...     125,116,132,14,147,153,129,111,92,62,6,32,82,16,179,21,3,126,
    ...     134,122,149,146,36,107,34,103,42,41,18,35,78,28,137,43,33,71,99,
    ...     139,56,23,138,67,168,130,30,180,152,83,9,13,173,11,157,156,64,
    ...     15,58])
    >>> teams = 14
    >>> rounds = 2
    >>> from moptipy.utils.nputils import int_range_to_dtype
    >>> dest = np.empty((rounds * (teams - 1), teams),
    ...                 int_range_to_dtype(-teams, teams))
    >>> map_games(perm, dest)
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
     [  6   0   0  13  12  -1 -11  14 -10   9   7  -5  -4  -8]
     [ 11   5  12   7  -2   9  -4  13  -6  14  -1  -3  -8 -10]
     [  0  10  11 -12  -7   0   5  -9   8  -2  -3   4 -14  13]
     [  5 -13  -4   3  -1   8 -12  -6  14  11 -10   7   2  -9]
     [  0   7   0  -9   6  -5  -2 -11   4 -12   8  10  14 -13]
     [ -7  12   4  -3  14   0   1  10 -11  -8   9  -2   0  -5]
     [ -2   1   9   0   8  -7   6  -5  -3 -11  10 -14   0  12]]
    >>> from moptipyapps.ttp.game_encoding_2 import map_games_2
    >>> dest.fill(0)
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
    >>> re_encode(perm, dest)
    >>> print(perm)
    [ 51  54  70  91 114 118 166  31  49  61  66  98 128 169   5  19  46 121
     141 148 171  22  52  81  93 151 165 174   1  53  73 110 133 155 163   7
      27  47  75  84 142 160  37  40  90 101 109 117 134  20  39  57  96 119
     154 179  60  65  89  94 131 145 177   2  74  86 108 150 158 170   8  29
      45  77  79 140 164  10  25  63  97 106 120 135  26  48  76  88  95 105
     178   0  69  80 112 153 159 176  38  55  85 124 130 144 161  14  68 100
     104 127 162 173  24  32  42 113 122 143 175   6  17  34  87 115 147 172
      12  43  59  92 123 132 167  13  50  62  67 103 125 136   9  16  36  44
      72 102 129   4  21  35  82 111 146 181   3  41  71 116 126 149 157  18
      30  64 107 137 152 156  23  28  56  78  99 138 168  11  15  33  58  83
     139 180]
    >>> dest.fill(0)
    >>> map_games(perm, dest)
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
    days, n = y.shape  # the number of days and teams to be scheduled
    total: Final[int] = len(x)
    index: int = 0
    for day in range(days):
        for slot in range(n):
            other = int(y[day, slot])
            if other > 0:
                game_id = game_to_id(slot, other - 1, n)
                for swap in range(index, total):
                    if x[swap] == game_id:
                        x[swap] = x[index]
                        x[index] = game_id
                        break
                index += 1


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
    >>> print(perm)
    [0 1]
    >>> dest = np.empty((rounds * (teams - 1), teams),
    ...                 int_range_to_dtype(-teams, teams))
    >>> map_games(perm, dest)
    >>> print(dest)
    [[ 2 -1]
     [-2  1]]
    >>> perm[0] = 1
    >>> perm[1] = 0
    >>> re_encode(perm, dest)
    >>> print(perm)
    [0 1]
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
    >>> re_encode(perm, dest)
    >>> print(perm)
    [ 0  8  1  5  2  4  3 11  6 10  7  9]
    >>> map_games(perm, dest)
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
    >>> map_games(perm, dest)
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
     [  0  -6  10   0   8   2  -9  -5   7  -3]
     [  9  -5  -4   3   2  -7   6   0  -1   0]
     [ -3   8   1   9   0   0  10  -2  -4  -7]
     [ -2   1   9   8 -10   7  -6  -4  -3   5]
     [  7  -8 -10  -6  -9   4  -1   2   5   3]
     [ -9  -4  -7   2  -8 -10   3   5   1   6]
     [  6   7   0  10   0  -1  -2   9  -8  -4]]
    >>> int(dest.shape[0] * dest.shape[1] - np.count_nonzero(dest))
    8
    >>> perm = np.array([21,32,53,63,73,3,20,55,77,88,8,11,40,60,74,19,27,51,
    ...     58,89,16,38,45,66,87,17,36,47,69,75,0,24,31,41,80,6,22,30,61,82,
    ...     2,13,23,43,71,12,52,54,65,84,10,49,57,79,81,1,28,44,68,78,7,26,
    ...     39,59,64,18,34,42,46,62,9,25,33,50,85,5,15,48,76,83,14,29,67,72,
    ...     86,4,35,37,56,70])
    >>> dest.fill(0)
    >>> map_games(perm, dest)
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
    """
    y.fill(0)  # first zero the output matrix
    days, n = y.shape  # the number of days and teams to be scheduled
    div: Final[int] = n - 1  # the divisor for permutation values -> teams

    for game in x:
        g = int(game)  # make sure that we have the full integer range.
        home_idx: int = (g // div) % n  # home idx is in 0..n-1
        away_idx: int = g % div  # away index in 0..n-2
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
        >>> ",".join(map(str, GameEncoding(inst).search_space().blueprint))
        '0,1,2,3,4,5,6,7,8,9,10,11'
        >>> inst = Instance(inst.name, inst, inst.teams, inst.rounds,
        ...                 inst.home_streak_min, inst.home_streak_max,
        ...                 inst.away_streak_min, inst.away_streak_max,
        ...                 inst.separation_min, inst.separation_max)
        >>> inst.rounds = 1  # modify number of rounds for copied instance
        >>> ",".join(map(str, GameEncoding(inst).search_space().blueprint))
        '1,2,3,7,8,10'
        >>> inst.rounds = 3  # modify number of rounds for copied instance
        >>> ",".join(map(str, GameEncoding(inst).search_space().blueprint))
        '0,1,1,2,2,3,3,4,5,6,7,7,8,8,9,10,10,11'
        >>> inst.rounds = 4  # modify number of rounds for copied instance
        >>> ",".join(map(str, GameEncoding(inst).search_space().blueprint))
        '0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11'
        >>> inst.rounds = 5  # modify number of rounds for copied instance
        >>> ",".join(map(str, GameEncoding(inst).search_space().blueprint))
        '0,0,1,1,1,2,2,2,3,3,3,4,4,5,5,6,6,7,7,7,8,8,8,9,9,10,10,10,11,11'
        """
        return search_space_for_n_and_rounds(
            self.instance.n_cities, self.instance.rounds)
