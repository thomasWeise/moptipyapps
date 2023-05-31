"""
The tour length objective function for tours in path representation.

A Traveling Salesperson Problem (TSP) instance  is defined as a
fully-connected graph with
:attr:`~moptipyapps.tsp.instance.Instance.n_cities` nodes. Each edge in the
graph has a weight, which identifies the distance between the nodes. The goal
is to find the *shortest* tour that visits every single node in the graph
exactly once and then returns back to its starting node. Then nodes are
usually called cities. In this file, we present methods for loading instances
of the TSP as distance matrices `A`. In other words, the value at `A[i, j]`
identifies the travel distance from `i` to `j`.

A tour can be represented in path representation, which means that it is
stored as a permutation of the numbers `0` to `n_cities-1`. The number at
index `k` identifies that `k`-th city to visit. So the first number in the
permutation identifies the first city, the second number the second city,
and so on.

The length of the tour can be computed by summing up the distances from the
`k`-th city to the `k+1`-st city, for `k` in `0..n_cities-2` and then adding
the distance from the last city to the first city. This is what the function
:func:`tour_length` is doing. This function is then wrapped as objective
function object in :class:`TourLength`.

Important initial work on this code has been contributed by Mr. Tianyu LIANG
(梁天宇), <liangty@stu.hfuu.edu.cn> a Master's student at the Institute of
Applied Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School
of Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei
University (合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the
supervision of Prof. Dr. Thomas Weise (汤卫思教授).

1. Gerhard Reinelt. TSPLIB - A Traveling Salesman Problem Library.
   *ORSA Journal on Computing* 3(4):376-384. November 1991.
   https://doi.org/10.1287/ijoc.3.4.376.
   http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
2. Gerhard Reinelt. *TSPLIB95.* Heidelberg, Germany: Universität
   Heidelberg, Institut für Angewandte Mathematik. 1995.
   http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
3. David Lee Applegate, Robert E. Bixby, Vašek Chvátal, and William John Cook.
   *The Traveling Salesman Problem: A Computational Study.* Second Edition,
   2007. Princeton, NJ, USA: Princeton University Press. Volume 17 of
   Princeton Series in Applied Mathematics. ISBN: 0-691-12993-2.
4. Gregory Z. Gutin and Abraham P. Punnen. *The Traveling Salesman Problem and
   its Variations.* 2002. Kluwer Academic Publishers. Volume 12 of
   Combinatorial Optimization. https://doi.org/10.1007/b101971.
5. Pedro Larrañaga, Cindy M. H. Kuijpers, Roberto H. Murga, Iñaki Inza, and
   Sejla Dizdarevic. Genetic Algorithms for the Travelling Salesman Problem: A
   Review of Representations and Operators. *Artificial Intelligence Review*
   13(2):129-170. April 1999. https://doi.org/10.1023/A:1006529012972.
6. Eugene Leighton Lawler, Jan Karel Lenstra, Alexander Hendrik George Rinnooy
   Kan, and David B. Shmoys. *The Traveling Salesman Problem: A Guided Tour of
   Combinatorial Optimization.* September 1985. Chichester, West Sussex, UK:
   Wiley Interscience. In Estimation, Simulation, and
   Control - Wiley-Interscience Series in Discrete Mathematics and
   Optimization. ISBN: 0-471-90413-9
7. Tianyu Liang, Zhize Wu, Jörg Lässig, Daan van den Berg, and Thomas Weise.
   Solving the Traveling Salesperson Problem using Frequency Fitness
   Assignment. In *Proceedings of the IEEE Symposium on Foundations of
   Computational Intelligence (IEEE FOCI'22),* part of the *IEEE Symposium
   Series on Computational Intelligence (SSCI'22),* December 4-7, 2022,
   Singapore. Pages 360-367. IEEE.
   https://doi.org/10.1109/SSCI51031.2022.10022296.
8. Thomas Weise, Raymond Chiong, Ke Tang, Jörg Lässig, Shigeyoshi Tsutsui,
   Wenxiang Chen, Zbigniew Michalewicz, and Xin Yao. Benchmarking Optimization
   Algorithms: An Open Source Framework for the Traveling Salesman Problem.
   *IEEE Computational Intelligence Magazine.* 9(3):40-52. August 2014.
   https://doi.org/10.1109/MCI.2014.2326101.
"""

from typing import Final

import numba  # type: ignore
import numpy as np
from moptipy.api.objective import Objective
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error

from moptipyapps.shared import SCOPE_INSTANCE
from moptipyapps.tsp.instance import Instance


@numba.njit(cache=True, inline="always")
def tour_length(instance: np.ndarray, x: np.ndarray) -> int:
    """
    Compute the length of a tour.

    :param instance: the distance matrix
    :param x: the tour
    :return: the length of the tour `x`
    """
    result: int = 0
    last: int = x[-1]
    for cur in x:
        result = result + instance[last, cur]
        last = cur
    return result


class TourLength(Objective):
    """The tour length objective function."""

    def __init__(self, instance: Instance) -> None:
        """
        Initialize the tour length objective function.

        :param instance: the tsp instance
        """
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the TSP instance we are trying to solve
        self.instance: Final[Instance] = instance

    def evaluate(self, x) -> int:
        """
        Compute the length of a tour in path representation.

        :param x: the tour in path representation
        :return: the tour length
        """
        return tour_length(self.instance, x)

    def lower_bound(self) -> int:
        """
        Get the lower bound of the tour length.

        :return: the lower bound of the tour length
        """
        return self.instance.tour_length_lower_bound

    def upper_bound(self) -> int:
        """
        Get the upper bound of the tour length.

        :return: the upper bound of the tour length
        """
        return self.instance.tour_length_upper_bound

    def __str__(self):
        """
        Get the name of this objective function: always "tourLength".

        :return: "tourLength"
        :retval "tourLength": always
        """
        return "tourLength"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of the space to the given logger.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope(SCOPE_INSTANCE) as kv:
            self.instance.log_parameters_to(kv)
