"""
A (1+1) FEA for the TSP using the reversal move.

A (1+1) FEA is the same as the (1+1) EA with Frequency Fitness Assignment.
The (1+1) EA using the same search operator as this algorithm here is
implemented in module :mod:`~moptipyapps.tsp.ea1p1_revn`.
The algorithm implemented here is the same as the basic (1+1) FEA with `rev`
operator in the paper [1].

The original version of this code has been contributed by Mr. Tianyu LIANG
(梁天宇), <liangty@stu.hfuu.edu.cn> a Master's student at the Institute of
Applied Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School
of Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei
University (合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the
supervision of Prof. Dr. Thomas Weise (汤卫思教授).

1. Tianyu Liang, Zhize Wu, Jörg Lässig, Daan van den Berg, and Thomas Weise.
   Solving the Traveling Salesperson Problem using Frequency Fitness
   Assignment. In Hisao Ishibuchi, Chee-Keong Kwoh, Ah-Hwee Tan, Dipti
   Srinivasan, Chunyan Miao, Anupam Trivedi, and Keeley A. Crockett, editors,
   *Proceedings of the IEEE Symposium on Foundations of Computational
   Intelligence (IEEE FOCI'22)*, part of the *IEEE Symposium Series on
   Computational Intelligence (SSCI 2022)*. December 4-7, 2022, Singapore,
   pages 360-367. IEEE. https://doi.org/10.1109/SSCI51031.2022.10022296.
2. Pedro Larrañaga, Cindy M. H. Kuijpers, Roberto H. Murga, Iñaki Inza, and
   S. Dizdarevic. Genetic Algorithms for the Travelling Salesman Problem: A
   Review of Representations and Operators. *Artificial Intelligence Review,*
   13(2):129-170, April 1999. Kluwer Academic Publishers, The Netherlands.
   https://doi.org/10.1023/A:1006529012972.
"""

from typing import Callable, Final, cast

import numba  # type: ignore
import numpy as np
from moptipy.algorithms.so.fea1plus1 import log_h
from moptipy.api.algorithm import Algorithm
from moptipy.api.process import Process
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import DEFAULT_INT
from moptipy.utils.types import type_error
from numpy.random import Generator

from moptipyapps.shared import SCOPE_INSTANCE
from moptipyapps.tsp.instance import Instance


@numba.njit(nogil=True, cache=True, inline="always")
def rev_if_h_not_worse(i: int, j: int, n_cities: int, dist: np.ndarray,
                       h: np.ndarray, x: np.ndarray, y: int) -> int:
    """
    Apply a reversal move if its tour length frequency is not worse.

    :param i: the first, smaller index
    :param j: the second, larger index
    :param n_cities: the number of cities
    :param dist: the problem instance
    :param h: the frequency table
    :param x: the candidate solution
    :param y: the tour length
    """
    xi: Final[int] = x[i]  # the value of x at index i
    xim1: Final[int] = x[((i - 1) + n_cities) % n_cities]  # x[i - 1]
    xj: Final[int] = x[j]  # the value of x at index j
    xjp1: Final[int] = x[(j + 1) % n_cities]  # x[j + 1], but index wrapped

    # compute the difference in tour length if we would apply the move
    dy: Final[int] = (dist[xim1, xj] + dist[xi, xjp1]
                      - dist[xim1, xi] - dist[xj, xjp1])
    y2: Final[int] = int(y + dy)  # and compute the new tour length
    h[y] += 1  # update frequency of the tour length of the current solution
    h[y2] += 1  # update frequency of the tour length of the new solution
    if h[y2] <= h[y]:  # this move does not make the frequency worse?
        # so we reverse the sequence from i to j in the solution
        if i == 0:  # deal with the special case that i==0
            x[0:j + 1:1] = x[j::-1]
        else:  # the normal case that i > 0
            x[i:j + 1:1] = x[j:i - 1:-1]
        return y2  # return new tour length
    return y  # return old tour length


class TSPFEA1p1revn(Algorithm):
    """A (1+1) FEA using the reversal operator for the TSP."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the RLS algorithm for the TSP with reversal move.

        :param instance: the problem instance
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        self.instance: Final[Instance] = instance

    def solve(self, process: Process) -> None:
        """
        Apply a (1+1) FEA optimization process with reversing operator.

        :param process: the process instance which provides random numbers,
            functions for creating, copying, and evaluating solutions, as well
            as the termination criterion
        """
        random: Final[Generator] = process.get_random()
        # set up the fast calls
        register: Final[Callable[[np.ndarray, int], int]] =\
            cast(Callable[[np.ndarray, int], int], process.register)
        should_terminate: Final[Callable[[], bool]] = process.should_terminate
        ri: Final[Callable[[int], int]] = random.integers

        instance: Final[Instance] = self.instance  # get the instance
        h: Final[np.ndarray] = np.zeros(  # allocate the frequency table
            instance.tour_length_upper_bound + 1, DEFAULT_INT)
        n: Final[int] = instance.n_cities  # get the number of cities
        x: Final[np.ndarray] = process.create()  # create the solution
        x[:] = range(n)  # fill array with 0..n
        random.shuffle(x)  # randomly generate an initial solution

        y: int = cast(int, process.evaluate(x))  # get length of first tour
        nm1: Final[int] = n - 1  # need n-1 in the loop for the random numbers
        nm2: Final[int] = n - 12  # we need this to check the move indices
        while not should_terminate():
            i = ri(nm1)  # get the first index
            j = ri(nm1)  # get the second index
            if i > j:  # ensure that i <= j
                i, j = j, i  # swap indices if i > j
            if (i == j) or ((i == 0) and (j == nm2)):
                continue  # either a nop or a complete reversal
            y = rev_if_h_not_worse(i, j, n, instance, h, x, y)  # move
            register(x, y)  # register the objective value

        # we log the frequency table at the very end of the run
        if h[y] == 0:
            h[y] = 1
        log_h(process, range(len(h)),
              cast(Callable[[int | float], int], h.__getitem__), str)

    def __str__(self):
        """
        Get the name of this algorithm.

        This name is then used in the directory path and file name of the
        log files.

        :returns: "tsp_fea1p1_revn"
        :retval "tsp_fea1p1_revn": always
        """
        return "tsp_fea1p1_revn"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the algorithm to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
