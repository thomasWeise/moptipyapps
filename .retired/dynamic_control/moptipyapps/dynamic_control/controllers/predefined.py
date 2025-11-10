"""
A set of pre-defined controllers.

In this module, we provide a set of pre-defined controllers taken from the
works of NOACK, CORNEJO MACEDA, LI, and SUN of the Harbin Institute of
Technology in Shenzhen, China (哈尔滨工业大学(深圳)). We ignore the
parameterizations offered in the original works and instead synthesize the
parameter values by ourselves.

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

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __cornejo_maceda(state: np.ndarray, _: float,
                     params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute a parameterized version of Cornejo Maceda's Law.

    See page 16, chapter 3.3.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    z = np.tanh(state[0] - state[1])
    b = params[0]
    z = np.tanh(1.0 if b == 0 else z / b)
    b = params[1]
    z = np.tanh(1.0 if b == 0 else z / b)
    b = params[2]
    out[0] = np.tanh(1.0 if b == 0 else z / b)


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __table_3_1_ga(state: np.ndarray, _: float,
                   params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute the law evolved by genetic algorithm in Table 3-1.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    out[0] = state[0] * params[0] + state[1] * params[1]


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def __table_3_1_lgpc(state: np.ndarray, _: float,
                     params: np.ndarray, out: np.ndarray) -> None:
    """
    Compute the law evolved by LGPC algorithm in Table 3-1.

    :param state: the current state of the system
    :param params: the parameters of the polynomial.
    :param out: the control vector, receiving one single element
    """
    a: Final[float] = state[0] * params[0] + params[1]
    out[0] = params[2] * np.sin((params[3] / a) if (a != 0.0) else 1.0)


def predefined(system: System) -> tuple[Controller, ...]:
    """
    Create a set of pre-defined controllers for the given equations object.

    :param system: the equations object
    :return: the linear controller
    """
    if system.control_dims != 1:
        raise ValueError("invalid controller dimensions "
                         f"{system.control_dims} for {system!r}.")
    if system.state_dims == 2:
        return (Controller("cornejo_maceda", 2, 1, 3, __cornejo_maceda), )
    if system.state_dims == 3:
        return (Controller("table_3_1_ga", 3, 1, 2, __table_3_1_ga),
                Controller("table_3_1_lgpc", 3, 1, 4, __table_3_1_lgpc))
    raise ValueError("invalid state dimensions "
                     f"{system.state_dims} for {system!r}.")
