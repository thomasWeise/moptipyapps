"""Test the game-based encoding."""

from itertools import permutations
from typing import Final

import numpy as np
from moptipy.spaces.permutations import Permutations
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.types import type_error
from numpy.random import Generator, default_rng

from moptipyapps.ttp.game_encoding import (
    map_games,
    search_space_for_n_and_rounds,
)


def __test_for_n_and_rounds(n: int, rounds: int,
                            random: Generator = default_rng()) -> None:
    """
    Test the encoding.

    :param n: the number of teams
    :param rounds: the number of rounds
    :param random: the default generator
    """
    space: Final[Permutations] = search_space_for_n_and_rounds(n, rounds)
    perm: Final[np.ndarray] = space.blueprint
    if not isinstance(perm, np.ndarray):
        raise type_error(perm, "perm", np.ndarray)
    expected_games: Final[int] = (n * (n - 1) * rounds) // 2
    actual_games: Final[int] = len(perm)
    if actual_games != expected_games:
        raise ValueError(
            f"expected {expected_games} games, but got {actual_games}")
    expected_diff_games: Final[int] = (n * (n - 1) * min(2, rounds)) // 2
    actual_diff_games: Final[int] = len(set(perm))
    if actual_diff_games != expected_diff_games:
        raise ValueError(f"expected {expected_diff_games} different games, "
                         f"but got {actual_diff_games}")
    days: Final[int] = (n - 1) * rounds

    y: Final[np.ndarray] = np.empty((days, n), int_range_to_dtype(-n, n))
    max_rounds: int = 1_000
    if space.n_points() <= max_rounds:
        t = np.copy(perm)
        for p in permutations(perm):
            t[:] = p
            map_games(t, y)
    else:
        for _ in range(max_rounds):
            map_games(perm, y)
            random.shuffle(perm)


def test_2_2() -> None:
    """Test 2 teams, 2 rounds combinations."""
    __test_for_n_and_rounds(2, 2)


def test_4_1() -> None:
    """Test 4 teams, 1 round combinations."""
    __test_for_n_and_rounds(4, 1)


def test_4_3() -> None:
    """Test 4 teams, 3 round combinations."""
    __test_for_n_and_rounds(4, 3)


def test_4_2() -> None:
    """Test 4 teams, 2 round combinations."""
    __test_for_n_and_rounds(4, 2)


def test_6_1() -> None:
    """Test 6 teams, 1 round combinations."""
    __test_for_n_and_rounds(6, 1)


def test_6_2() -> None:
    """Test 6 teams, 2 round combinations."""
    __test_for_n_and_rounds(6, 2)


def test_8_1() -> None:
    """Test 8 teams,  round combinations."""
    __test_for_n_and_rounds(8, 1)


def test_8_2() -> None:
    """Test 8 teams, 2 round combinations."""
    __test_for_n_and_rounds(8, 2)


def test_10_1() -> None:
    """Test 10 teams, 1 round combinations."""
    __test_for_n_and_rounds(10, 1)


def test_10_2() -> None:
    """Test 10 teams, 2 round combinations."""
    __test_for_n_and_rounds(10, 2)
