"""
The two-dimensional Stuart-Landau system.

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

from math import sqrt
from typing import Final

import numba  # type: ignore
import numpy as np

from moptipyapps.dynamic_control.system import System


@numba.njit(numba.none(numba.float64[:], numba.float64, numba.float64[:],
                       numba.float64[:]),
            cache=True, inline="always", fastmath=True, boundscheck=False)
def __stuart_landau_equations(state: np.ndarray, _: float,
                              control: np.ndarray, out: np.ndarray) -> None:
    """
    Compute the differential equations of a controlled 2D Stuart-Landau model.

    :param state: the state of the system
    :param _: the time index, which is ignored
    :param control: the output of the controller
    :param out: the differential, i.e., the output of this function
    """
    sigma: Final[float] = 0.1 - state[0]**2 - state[1]**2
    out[0] = sigma * state[0] - state[1]
    out[1] = sigma * state[1] + state[0] + control[0]


def __beautify(f: float) -> float:
    """
    Beautify a floating point number.

    :param f: the original value
    :return: the beautified one.
    """
    if abs(f) < 1e-16:
        return 0.0
    return f


def __make_stuart_landau() -> System:
    """
    Create the Stuart-Landau system.

    :return: the Stuart-Landau system
    """
    tests: Final[np.ndarray] = np.array([[0.1, 0.0], [sqrt(0.1), 0.0]], float)
    # similar to training = make_interesting_starting_points(111, tests)
    training: Final[np.ndarray] = np.array(
        [[-0.0038209, -0.00240577],
         [0.00307569, 0.00849048],
         [0.00976623, -0.00938637],
         [-0.01182026, 0.01365555],
         [0.01777127, 0.01392328],
         [0.0006177, -0.02708416],
         [-0.03153332, -0.00214806],
         [0.02792395, -0.02291339],
         [-0.02847394, 0.02899283],
         [0.01350566, 0.04308481],
         [-0.02991208, -0.0396497],
         [0.0540025, 0.00441161],
         [-0.05551053, -0.01907852],
         [-0.01080416, -0.06228265],
         [-0.03100233, 0.06021576],
         [-0.06858837, 0.02268735],
         [0.04134416, 0.06467235],
         [0.02084729, -0.07855437],
         [0.07834551, -0.03495282],
         [0.08049101, 0.04093911],
         [-0.08926819, -0.03196673],
         [-0.02208376, 0.0968485],
         [0.07379551, -0.07306821],
         [-0.08376945, 0.06874307],
         [0.0275525, 0.10946577],
         [-0.05169461, -0.10540068],
         [-0.11942003, 0.02451537],
         [-0.09709783, -0.08096571],
         [0.10400721, 0.0795487],
         [0.0636276, -0.11958202],
         [-0.13503465, -0.03684538],
         [-0.06982037, 0.12649679],
         [0.07510267, 0.12868982],
         [-0.03055115, -0.15044613],
         [0.00066117, 0.15803063],
         [0.15246865, -0.05634631],
         [0.0232859, -0.1654316],
         [0.17156619, 0.00197981],
         [-0.16413595, 0.06378141],
         [0.16890575, 0.06395391],
         [0.14766927, -0.11164403],
         [-0.08647114, -0.16877639],
         [-0.12327999, 0.14999224],
         [-0.19863449, -0.00369282],
         [0.11207882, -0.1694759],
         [-0.06793134, 0.19627607],
         [0.16954787, 0.12762632],
         [-0.14915337, -0.15724184],
         [0.04633135, -0.21633926],
         [0.06125791, 0.21729025],
         [-0.18584921, 0.13596598],
         [-0.1999492, -0.12307257],
         [0.23850607, -0.01954563],
         [0.1439136, 0.19681835],
         [-0.02521609, -0.24705248],
         [-0.00891478, 0.25269401],
         [-0.24709608, -0.07197915],
         [-0.10863885, -0.23828467],
         [0.25404676, 0.08017174],
         [0.1665829, -0.2136433],
         [0.25571773, -0.10231616],
         [-0.27547718, 0.04980043],
         [-0.09155915, 0.26931962],
         [0.0781761, -0.27819739],
         [-0.26285954, 0.13053765],
         [0.23770086, -0.17973375],
         [0.12078667, 0.27735893],
         [-0.30500318, -0.03525198],
         [0.20619554, 0.23355099],
         [-0.19712815, 0.24705659],
         [-0.18688532, -0.26047056],
         [0.26924503, 0.18219083],
         [0.04358426, 0.32671534],
         [-0.29933575, -0.14845035],
         [-0.10118493, -0.32316973],
         [0.15974011, -0.30370809],
         [0.33188974, -0.10355639],
         [0.0401967, -0.34988418],
         [-0.14267712, 0.32692311],
         [-0.27812549, -0.23048476],
         [-0.3127193, 0.18964695],
         [-0.36944776, 0.02430573],
         [-0.03954588, 0.37266929],
         [0.34413746, 0.15943751],
         [-0.27511481, 0.26759701],
         [0.14876924, 0.35867843],
         [0.38719748, 0.06623872],
         [0.25924229, -0.30111563],
         [-0.03399581, -0.40041226],
         [-0.38191229, -0.13884515],
         [-0.38558806, 0.14193967],
         [0.25389894, 0.32877223],
         [-0.27063978, -0.32106318],
         [0.36693718, -0.21330011],
         [-0.23322584, 0.35999818],
         [0.0372525, 0.43185548],
         [-0.11945503, 0.42136931],
         [0.0834364, -0.434552],
         [0.44623645, -0.02619827],
         [-0.44718354, -0.06242773],
         [-0.2100912, -0.40475896],
         [0.35704647, 0.29090293],
         [0.20490764, -0.4174912],
         [-0.46428805, 0.07030482],
         [-0.39615467, -0.26043909],
         [-0.10245864, -0.46751571],
         [0.35093411, -0.33204879],
         [0.45711698, 0.16981881],
         [0.17358843, 0.46052731],
         [-0.39398933, 0.30241613],
         [0.47669529, -0.15475867]])

    # x n_rotations: Final[int] = 32
    # x training: Final[np.ndarray] = np.empty(
    # x     (n_rotations * len(tests), 2), float)
    # x n_interesting: Final[int] = len(tests)
    # x total_angles: Final[int] = n_interesting * n_rotations
    # x dest_i: int = 0
    # x for index, a_raw in enumerate(tests):
    # x     a = complex(a_raw[0], a_raw[1])
    # x     for i in range(n_rotations):
    # x             * pi / total_angles
    # x         point: complex = a * complex(cos(angle), sin(angle))
    # x         training[dest_i, 0] = __beautify(point.real)
    # x         training[dest_i, 1] = __beautify(point.imag)
    # x         dest_i = dest_i + 1

    system: Final[System] = System(
        "stuart_landau", 2, 1, tests, training, 5000, 5000, (0, 1))
    system.equations = __stuart_landau_equations  # type: ignore
    return system


#: The Stuart-Landau system.
STUART_LANDAU: Final[System] = __make_stuart_landau()
