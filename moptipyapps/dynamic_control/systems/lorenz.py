"""
The three-dimensional Lorenz system.

The initial starting point of the work here were conversations with
Prof. Dr. Bernd NOACK and Guy Yoslan CORNEJO MACEDA of the Harbin Institute
of Technology in Shenzhen, China (哈尔滨工业大学(深圳)) as well as the
following two MSc theses and book:

1. Yuxiang LI (李宇翔). Jet Mixing Enhancement using Deep Reinforcement
   Learning (基于深度强化学习的射流混合增强控制). MSc Thesis. Harbin Institute
   of Technology in Shenzhen, China (哈尔滨工业大学(深圳)).
   January 2023.
2. Wei SUN (孙伟). Wake Control of 1-2-3 Fluidic Pinball using Deep
   Reinforcement Learning (基于深度强化学习方法的 1-2-3 流体弹球尾流控制).
   MSc Thesis. Harbin Institute of Technology in Shenzhen, China
   (哈尔滨工业大学(深圳)). January 2023.
3. Guy Yoslan CORNEJO MACEDA, François LUSSEYRAN, and Bernd R. NOACK.
   xMLC: A Toolkit for Machine Learning Control, First Edition.
   Machine Learning Tools in Fluid Mechanics, Vol 2.
   Shenzhen & Paris; Universitätsbibliothek der Technischen Universität
   Braunschweig. 2022 https://doi.org/10.24355/dbbs.084-202208220937-0
"""

from typing import Final

import numba  # type: ignore
import numpy as np
from pycommons.types import check_int_range

from moptipyapps.dynamic_control.starting_points import (
    make_interesting_starting_points,
)
from moptipyapps.dynamic_control.system import System


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __lorenz_equations(state: np.ndarray, _: float,
                       control: np.ndarray, out: np.ndarray) -> None:
    """
    Compute the differential equations of a controlled 3D Lorenz system.

    :param state: the state of the system
    :param _: the time index, which is ignored
    :param control: the output of the controller
    :param out: the differential, i.e., the output of this function
    """
    x: Final[float] = state[0]
    y: Final[float] = state[1]
    z: Final[float] = state[2]

    # compute the differential system
    out[0] = 10.0 * (y - x)
    out[1] = 28.0 * x - y - (x * z) + control[0]
    out[2] = x * y - (2.6666666666666665 * z)


def make_lorenz(n_points: int) -> System:
    """
    Create the Lorenz system.

    :param n_points: the number of training points
    :return: the Lorenz system
    """
    check_int_range(n_points, "n_points", 1, 1_000)
    tests: Final[np.ndarray] = \
        np.array([[-1.0, 1.0, 3.0], [0.0, 1.0, 22.0]], float)
    # similar to training = make_interesting_starting_points(111, tests)
    training: Final[np.ndarray] = np.array(
        [[-2.31837616e-1, -5.24709723e-2, -1.28671997e-1],
         [5.24425115e-1, -4.36886027e-2, 1.23705573e-1],
         [-3.34157047e-1, 6.24361532e-1, 3.95015241e-1],
         [-4.55212674e-2, -9.84129335e-1, -4.45366280e-1],
         [3.63753119e-1, 5.10336447e-1, -1.19737193],
         [-2.09587174e-1, -6.89638430e-1, 1.45278120],
         [-1.83653628, 3.19596142e-1, -3.23809729e-1],
         [1.15939504, 1.50893034, 1.02697020],
         [1.97903772, -1.41455814, 1.27529901e-2],
         [-8.62639452e-1, 2.43142041, -8.06154187e-1],
         [-1.69664567, -2.39320251, -4.83774211e-1],
         [2.73331044, 8.57709170e-1, -1.52110281],
         [5.33078125e-2, -1.31842985, -3.25664933],
         [-8.07769100e-2, -2.97754587, 2.33390990],
         [2.35072569, -1.43006329e-1, 3.30026821],
         [-4.03232742, -8.31922303e-1, 1.32332254],
         [-3.24522964, -1.89918862e-2, -3.25299734],
         [1.55208382, 4.52665580, 8.78256040e-1],
         [2.22089300, -4.58186003, -6.69524512e-1],
         [1.00280557, 2.54311243, -4.66372240],
         [-4.04118291, 3.70295973, 1.47501504],
         [5.67173423, 1.44565665, 1.04964605],
         [-2.59523889, -5.26245489, -2.05406409],
         [2.94504196, -2.45896263, -5.23085552],
         [-3.28897946, 4.83504710, -3.38620915],
         [-2.58817813, -6.25659365e-1, 6.50364050],
         [5.72357009, -3.49153015, 2.88262233],
         [5.22299795, 4.53644041, -3.06899979],
         [-4.87825184, -5.29004044, 3.10808896],
         [4.27501083, 3.63860787, 5.85126863],
         [-8.34000452, 4.25671605e-1, -6.87181237e-1],
         [-1.08540415, 8.10559431, 2.81655916],
         [1.94788569, -5.30894243, 6.89792956],
         [8.45383153, -1.55041688, -3.25336920],
         [-1.03169607, 1.91837732, -9.20608115],
         [-6.59584225, -4.08814508, -5.87073907],
         [5.15768660e-2, -7.31844351, -6.81576316],
         [1.10412724, 9.35404100, -4.09623004],
         [-9.24919988e-1, -1.04001359e+1, 1.45033454],
         [-8.42728582, 4.69942357, 4.87746425],
         [6.56849521, -8.92524514, 7.73503710e-2],
         [-7.84621150, 3.63654240, -7.35441923],
         [-1.76374875, 5.20695220, 1.02402030e+1],
         [6.59956499, 2.63679539, -9.53593460],
         [1.11970782e+1, 4.34954429e-1, 4.73069196],
         [8.21196486, 9.31990914, 5.42613335e-1],
         [-6.86424196, 1.06609923e+1, -7.81891961e-1],
         [-4.09122371, -5.09173842, 1.12099315e+1],
         [-1.12617383e+1, -3.36092053, 6.10660311],
         [4.40559872, 1.77756117e-1, 1.27751719e+1],
         [-8.84888642, -1.05672816e+1, 2.33623289e-1],
         [3.10452705, 1.18665218e+1, 6.86275419],
         [7.72028081, -7.37696538, -9.54979561],
         [-2.05792588, -3.26013517, -1.40774520e+1],
         [-1.43317322e+1, -2.67395566, -2.90739935],
         [9.70967854, -7.60857398, 8.77170340],
         [1.33595404e+1, 5.64205226, -5.20159173],
         [1.84184161, 9.60079349, -1.22556406e+1],
         [1.45763613e+1, -6.18659546, -1.88994719],
         [3.57766009, -1.50015844e+1, -5.01626682],
         [9.17805707e-1, -1.39139085e+1, 8.79834996],
         [1.06480184e+1, 6.70055614, 1.10706193e+1],
         [-1.56128381e+1, 6.46573099, -2.09815734],
         [-9.87640379, -8.30988969, -1.15173542e+1],
         [-9.95414070, 4.28155228, 1.38294971e+1],
         [-6.89378192, 1.36522586e+1, -9.18334631],
         [1.76686982, 1.79486554e+1, -1.63796984],
         [7.05633216, -1.36984415, -1.69160748e+1],
         [-6.42179549, -1.65952281e+1, -5.58478539],
         [-7.41816999, 1.54645465e+1, 7.98777508],
         [-8.42681552, 4.69511699, -1.65901139e+1],
         [-9.78324506, -1.39304328e+1, 9.43216455],
         [1.94440370e+1, 2.31090199, 2.43252766],
         [3.15058476, -9.05446229, 1.75544337e+1],
         [1.06459182e+1, 1.47218781e+1, -8.99355553],
         [1.22190707e+1, -1.64529457e+1, 1.40712921],
         [-1.90636227e+1, 2.16964624, 8.06436472],
         [-2.34494082, 1.15135585e+1, 1.75050371e+1],
         [1.78508769e+1, -2.19686564, -1.15098756e+1],
         [2.31504343e-1, -1.38773263e+1, -1.65813076e+1],
         [1.50443571e+1, 1.49229413e+1, 5.50536850],
         [-1.79414521e+1, -6.05704638e-1, -1.29993027e+1],
         [-1.90893740e+1, -1.14789797e+1, -2.66981417],
         [-1.09824863e+1, -4.56662192, 1.93398756e+1],
         [1.27613899e+1, -2.31578801, 1.89639329e+1],
         [1.06885845, -2.27641276e+1, 4.58167139],
         [-1.78853615e+1, 1.41595978e+1, 5.70950161],
         [2.03291895e+1, -8.39726474, 9.05394871],
         [1.93958349, 7.21737732, -2.28657560e+1],
         [1.90522968, 2.16902183e+1, 1.08479166e+1],
         [1.45272901e+1, -1.63102458e+1, -1.13106168e+1],
         [-1.87965003e+1, 1.38830549e+1, -8.50402100],
         [-8.67486188, 2.35641803e+1, -1.16532740],
         [-2.08371573e+1, -9.29064237, 1.11821538e+1],
         [-7.40930680, -5.51574155, -2.39589341e+1],
         [1.55046343e+1, 8.72083349, -1.88907194e+1],
         [9.62045596, 1.31513890e+1, 2.05400575e+1],
         [8.89447741e-1, 2.16932725e+1, -1.51744310e+1],
         [1.20103755e+1, -1.90498725e+1, 1.44533139e+1],
         [2.47165331, -2.52776137e+1, -9.24753127],
         [2.43672486e+1, 1.09659183e+1, -5.59054774],
         [1.05756254e+1, 2.54424857e+1, -9.67114390e-1],
         [-1.36887388e+1, -2.40892890e+1, 2.72043969],
         [-2.80062395e+1, 8.56852541e-1, -2.26177776],
         [-5.19799304, -1.81696491e+1, 2.11734080e+1],
         [2.40714873e+1, 6.51484929, 1.41068791e+1],
         [1.09672511e+1, -8.86272444, -2.52510743e+1],
         [-1.66896882e+1, -1.74937806e+1, -1.63577116e+1],
         [2.72013507e+1, -1.05512055e+1, -4.09451211],
         [-1.68644649e+1, 1.27734462e+1, 2.08910298e+1],
         [-1.36658517e+1, 1.36662584e+1, -2.29484206e+1]]) \
        if n_points == 111 else (
            np.array([[-0.75900911, 3.40732601, -6.63879075],
                      [10.95019756, -9.9362743, -2.53014984],
                      [-17.32322703, -10.12101153, -10.18854763],
                      [13.282736, 14.93163678, -22.37782999]])
            if n_points == 4 else make_interesting_starting_points(
                n_points, tests))

    lorenz: Final[System] = System(
        "lorenz", 3, 1, 3, 3, 0.1, tests, training, 5000, 50.0, 5000, 50.0,
        (0, 1, 2, len(training) + len(tests) - 1))
    lorenz.equations = __lorenz_equations  # type: ignore
    return lorenz


#: The Lorenz system with 111 training points.
LORENZ_111: Final[System] = make_lorenz(111)

#: The Lorenz system with 4 training points.
LORENZ_4: Final[System] = make_lorenz(4)
