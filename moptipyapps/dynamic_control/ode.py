"""
A primitive integrator for systems of ordinary differential equations.

Many dynamic systems can be modeled as systems of ordinary differential
equations that govern their progress over time. Trying to find out in
which state such systems are at a given point in time means to integrate
these equations until that point in time (starting from a starting state).
Now there are several methods for doing such a thing, including, e.g.,
:func:`scipy.integrate.odeint`.

What we want to play around with, however, is synthesizing controllers.
In this case, the differential equations also merge the output of the
controller with the current state. If the controller behaves inappropriately,
this may make the system diverge, i.e., some of its state variables go to
infinity over time or sometimes rather quickly.

Using ODE integrators that compute the system state at pre-defined time steps
is thus cumbersome, as the system may have already exploded at these goal
times. Therefore, we cooked our own, simple ODE integrator. Our integrator
dynamically chooses the length of the time step based on the output of the
differential equations. If the output is large, then we will automatically use
shorter time slices. A large differential applied for a small time slice still
leads to a reasonably-large change in the system state. More specifically, the
system cannot diverge to infinity: The larger the differential step, the
shorter the time slice we use, so everything balances out.

Thus, we can simulate a well-behaved system over a long time and an
ill-behaved system for a shorter time period. Neither system will diverge.
"""

from math import fsum, sqrt
from typing import Callable, Final, Iterable, TypeVar

import numba  # type: ignore
import numpy as np


@numba.njit(numba.float64(numba.float64[:], numba.float64[:],
                          numba.float64[:], numba.float64[:], numba.float64),
            cache=True, inline="always", fastmath=False, boundscheck=False)
def __diff_update(cur_diff: np.ndarray, cur_row: np.ndarray,
                  last_diff: np.ndarray, next_row: np.ndarray,
                  max_dist: float) -> float:
    """
    Add the current differential to the current state to get the next state.

    The input of this function is the current differential vector `cur_diff`,
    the current row of the ode matrix `cur_row`, the last differential vector
    `last_diff`, the next row of the ode matrix `next_row`, and the maximum
    permitted step length `max_dist`.
    In an ideal world, we would set the weight value of the current row, i.e.,
    `cur_row[-1]` to `1.0` and set `next_row[0:s] = cur_row[0:s] + cur_diff`,
    where `s` be the dimension of the state. In other words, we would just add
    the differential vector to the current state and get the next state.
    However, we do not live in an ideal world. The absolute value
    `z = |cur_diff| = np.sqrt(np.square(cur_diff).sum())`
    may be too large. This can have two reasons:

    1. Maybe the differential equation system diverges, i.e., the system
       moves towards infinity.
    2. Maybe we are close to some state towards the system may converge, but
       the step length does not become small enough to actually converge and
       the system jumps around.

    So we put several mechanisms in place to limit the step lengths:

    1. We never use the full step length `z` but pre-scale it to
       `log(1 + z) * 0.6180339887498948`.
    2. We limit the step length to the 1/128th of the distance of the current
       system state to the center of coordinates.
    3. We multiply the step length with a factor that exponentially decreases
       with the angle between the current and the last differential.
    4. We impose a hard limit on the step length based on the dimensionality
       of the system.

    All of this leads to a smaller step length `t <= z`. The weight of the
    step is then `t / z` (if `z > 0`, otherwise `1`)

    :param cur_diff: the differential vector
    :param cur_row: the current row
    :param last_diff: the last differential vector
    :param next_row: the state to update
    :param max_dist: the maximum distance to move
    :returns: the current weight
    :returns: the length of the time slice

    >>> c_row = np.array([0.0, 0.0, 0.0, 0.0], float)
    >>> c_diff = np.array([1.0, 1.0, 1.0], float)
    >>> l_diff = np.zeros_like(c_diff)
    >>> n_row = np.zeros_like(c_row)
    >>> _ = __diff_update(c_diff, c_row, l_diff, n_row, 100.0)  # log-limit
    >>> print(";".join(f"{s:.10f}" for s in n_row))
    0.3586249472;0.3586249472;0.3586249472;0.0000000000
    >>> print(f"{c_row[-1]:.10f}")
    0.3586249472
    >>> print((np.log1p(np.sqrt(3.0)) * 0.6180339887498948) / np.sqrt(3.0))
    0.3586249472058058
    >>> _ = __diff_update(c_diff, c_row, l_diff, n_row, 0.05)  # hard limit
    >>> print(";".join(f"{s:.10f}" for s in n_row))
    0.0288675135;0.0288675135;0.0288675135;0.0000000000
    >>> print(f"{c_row[-1]:.10f}")
    0.0288675135
    >>> print(0.05 / np.sqrt(3.0))
    0.02886751345948129
    >>> l_diff = np.array([-1.0, -1.0, -1.0], float)
    >>> _ = __diff_update(c_diff, c_row, l_diff, n_row, 100.0)  # angle-limit
    >>> print(";".join(f"{s:.10f}" for s in n_row))
    0.0000289409;0.0000289409;0.0000289409;0.0000000000
    >>> print(f"{c_row[-1]:.10f}")
    0.0000289409
    >>> a = (np.log1p(np.sqrt(3.0)) * 0.6180339887498948
    ...      * np.exp(-3 * np.pi)) / np.sqrt(3.0)
    >>> print(f"{a:.10f}")
    0.0000289409
    >>> l_diff = np.array([-1.0, -1.0, 1.0], float)
    >>> _ = __diff_update(c_diff, c_row, l_diff, n_row, 100.0)  # angle-limit
    >>> print(";".join(f"{s:.10f}" for s in n_row))
    0.0011622728;0.0011622728;0.0011622728;0.0000000000
    >>> print(f"{c_row[-1]:.10f}")
    0.0011622728
    >>> a = (np.log1p(np.sqrt(3.0)) * 0.6180339887498948
    ...      * np.exp(-3 * np.arccos(-1 / 3))) / np.sqrt(3.0)
    >>> print(f"{a:.10f}")
    0.0011622728
    >>> c_row = np.array([0.1, 0.1, 0.1, 0.0], float)
    >>> l_diff = np.zeros_like(c_diff)
    >>> _ = __diff_update(c_diff, c_row, l_diff, n_row, 100.0)  # center limit
    >>> print(";".join(f"{s:.10f}" for s in n_row))
    0.1007812500;0.1007812500;0.1007812500;0.0000000000
    >>> print(f"{c_row[-1]:.10f}")
    0.0007812500
    >>> print((np.sqrt(0.03) / 128) / np.sqrt(3))
    0.00078125
    """
# Compute the square sum of the current differential.
    dsum: float = 0.0
    c: float = 0.0
    failure: bool = False
    for dx in cur_diff:
        if -1e100 < dx < 1e100:
            y = (dx * dx) - c
            t = dsum + y
            c = (t - dsum) - y
            dsum = t
        else:
            failure = True
            break

# If the differential grew out of bounds, we just add a logarithmic step.
    if failure or (dsum > 1e100):
        for i, d in enumerate(cur_diff):
            next_row[i] = cur_row[i] + (
                -np.log1p(-d) if d < 0.0 else np.log1p(d))
        cur_row[-1] = 1.0
        return 1.0

    dist: Final[float] = np.sqrt(dsum)
    dim: Final[int] = len(cur_diff)
    if dist <= 0.0:  # If the step length if 0, we can quit directly.
        next_row[0:dim] = cur_row[0:dim]
        next_row[-1] = 1.0
        return 1.0

# the first adjustment of the step length is always log(1+dist) / golden ratio
    use_dist: float = np.log1p(dist) * 0.6180339887498948

# now compute the angle-based limit
# The angle between two vectors A and B is arccos(A*B / (|A| * |B|)).
    p_dsum: float = 0.0  # this will be used to compute |B|
    prod: float = 0.0  # this is used to compute A*B
    c2: float = 0.0
    for i, dd in enumerate(last_diff):
        if -1e100 < dd < 1e100:
            y = (dd * dd) - c
            t = p_dsum + y
            c = (t - p_dsum) - y
            p_dsum = t

            y = (dd * cur_diff[i]) - c2
            t = prod + y
            c2 = (t - prod) - y
            prod = t
        else:
            failure = True
            break

    if (not failure) and (p_dsum > 0.0):
        p_dsum = np.sqrt(p_dsum) * dist
        if p_dsum > 0.0:
            prod /= p_dsum
            if prod < 1.0:
                use_dist *= np.exp(
                    -3.0 * (np.arccos(prod) if prod > -1.0 else np.pi))

# now limit the step length to 1/128th of the distance to the center
    center_dist: float = 0.0
    c = 0.0
    for i in range(dim):
        dd = cur_row[i]
        if -1e100 < dd < 1e100:
            y = (dd * dd) - c
            t = center_dist + y
            c = (t - center_dist) - y
            center_dist = t
        else:
            center_dist = max_dist
            break
    center_dist = 0.0078125 * np.sqrt(center_dist)
    if 0.0 < center_dist < use_dist:
        use_dist = center_dist

# now apply the hard, dimension-based limit
    if max_dist < use_dist:
        use_dist = max_dist

# store the weight in the current row and update the next row
    cur_row[-1] = weight = use_dist / dist
    next_row[0:dim] = cur_row[0:dim] + (weight * cur_diff)
    return weight


#: the type variable for ODE controller parameters
T = TypeVar("T")


def run_ode(starting_state: np.ndarray,
            equations: Callable[[
                np.ndarray, float, np.ndarray, np.ndarray], None],
            controller: Callable[[np.ndarray, float, T, np.ndarray], None],
            parameters: T, controller_dim: int = 1,
            steps: int = 5000) -> np.ndarray:
    """
    Simulate a set of controlled differential system.

    The system begins in the starting state stored in the vector
    `starting_state`. In each time step, first, the `controller` is invoked.
    It receives the current system state vector as input, the current time
    `t`, its parameters (`parameters`), and an output array to store its
    computed control values into. This array has dimension `controller_dim`,
    which usually is `1`. Then, the function `system` will be called and
    receives the current state vector, the time step `t`, and the controller
    output as input, as well as an output array to store the result of the
    differential into. This output array has the same dimension as the state
    vector.

    The system will be simulated for exactly `steps` time steps. The system
    will automatically determine the right length of the time steps. If the
    differential equations tell our system to make a step of length `z`, then:

    1. We never use the full step length `z` but pre-scale it to
       `log(1 + z) * 0.6180339887498948`.
    2. We limit the step length to the 1/128th of the distance of the current
       system state to the center of coordinates.
    3. We multiply the step length with a factor that exponentially decreases
       with the angle between the current and the last differential.
    4. We impose a hard limit on the step length based on the dimensionality
       of the system.

    As result, this function returns a tuple of three components: First, the
    matrix `steps` rows, where each row corresponds to one system state
    vector. The first row will be identical tp `starting_state`. The second
    element of the return tuple is the time vector, holding `steps` strictly
    increasing time value. The third and last return component is the control
    matrix. It has one row for each time step, i.e., `steps` in total, and
    each row corresponds to the controller output at that time step. If
    `controller_dim==1`, this third tuple element is a plain vector and not a
    matrix.

    The last time slice (at index `-1,-1`) will always be set to zero.

    :param starting_state: the starting
    :param equations: the differential system
    :param controller: the controller function
    :param parameters: the controller parameters
    :param controller_dim: the dimension of the controller result
    :param steps: the number of steps to simulate
    :returns: a matrix where each row represents a point in time, composed of
        the current state, the controller output, and the length of the time
        slice

    If we simulate the flight of a projectile with our ODE execution, then
    both the flight time as well as the flight length are about 0.12% off from
    what the mathematical solution of the flight system prescribe. That's
    actually not bad for a crude and fast integration method...

    >>> v = 100.0
    >>> angle = np.deg2rad(45.0)
    >>> v_x = v * np.cos(angle)
    >>> print(f"{v_x:.10f}")
    70.7106781187
    >>> v_y = v * np.sin(angle)
    >>> print(f"{v_y:.10f}")
    70.7106781187
    >>> def projectile(position, ttime, ctrl, out):
    ...     out[0] = 70.71067811865474
    ...     out[1] = 70.71067811865474 - ttime * 9.80665
    >>> param = 0.0   # ignore
    >>> def contrl(position, ttime, params, dest):
    ...     dest[0] = 0.0  #  controller that does nothing
    >>> strt = np.array([0.0, 1.0])
    >>> ode = run_ode(strt, projectile, contrl, param, 1, 10000)
    >>> time_of_flight = 2 * v_y / 9.80665
    >>> print(f"{time_of_flight:.10f}")
    14.4209649817
    >>> travel_distance_x = time_of_flight * v_x
    >>> print(f"{travel_distance_x:.10f}")
    1019.7162129779
    >>> idx = np.argwhere(ode[:, 1] <= 0.0)[0][0]
    >>> print(idx)
    9631
    >>> print(f"{ode[idx - 1, 0]:.10f}")
    1020.8026525593
    >>> print(f"{ode[idx, 0]:.10f}")
    1020.8909467853
    >>> print(f"{fsum(ode[0:idx, -1]):.10f}")
    14.4375782265
    >>> print(f"{fsum(ode[0:idx + 1, -1]):.10f}")
    14.4388267873
    >>> print(ode[-1, -1])
    0.0
    """
    state_dim: Final[int] = len(starting_state)
    result: Final[np.ndarray] = np.zeros((
        steps, state_dim + controller_dim + 1))
    max_dist: Final[float] = sqrt(
        (state_dim - 1) / 64.0)  # the maximum distance

    point: np.ndarray = np.empty(state_dim)
    cur_diff: np.ndarray = np.empty(state_dim)
    last_diff: np.ndarray = np.zeros(state_dim)
    c_out: Final[np.ndarray] = np.empty(controller_dim)
    time: float = 0.0
    c: float = 0.0
    c_end_dim: Final[int] = state_dim + controller_dim

    next_row: np.ndarray = result[0]
    next_row[0:state_dim] = starting_state

    # compute the first row
    for i in range(1, steps):
        cur_row = next_row
        next_row = result[i]
        point[:] = cur_row[0:state_dim]  # get point part of state
        controller(point, time, parameters, c_out)  # invoke controller
        equations(point, time, c_out, cur_diff)  # invoke equations

# invoke the differential update and update time as well
        y = __diff_update(cur_diff, cur_row, last_diff,
                          next_row, max_dist) - c
        t = time + y
        c = (t - time) - y
        time = t

        cur_row[state_dim:c_end_dim] = c_out  # store controller output
        last_diff, cur_diff = cur_diff, last_diff

    point[:] = next_row[0:state_dim]
    controller(point, time, parameters, c_out)
    next_row[state_dim:c_end_dim] = c_out
    return result


@numba.njit(cache=True, inline="always", fastmath=False, boundscheck=False)
def __j_from_ode_compute(ode: np.ndarray, state_dim: int,
                         dest: np.ndarray) -> None:
    """
    Prepare the input array for the figure of merit computation.

    The `ode` matrix contains one row for each time step.
    The row is composed of the current state, the current controller output,
    and the length of the time slice.
    However, all systems of a given simulation start in the same initial
    state, so it makes little sense to include this initial state into the
    figure of merit computation. It does make sense to include the control
    output at that moment, though, because it contributes to the next state.
    It also makes no sense to count in the last row of the ODE computation,
    because this is the final system state whose weight will not actually
    be computed.

    :param ode: the output array from the ODE simulation
    :param state_dim: the state dimension
    :param dest: the destination array

    >>> od = np.array([[1, 2, 3, 4, 0.1],
    ...                [5, 6, 7, 8, 0.2],
    ...                [9, 6, 4, 3, 0.3],
    ...                [7, 4, 2, 1, 0.4]])
    >>> sta_dim = 3
    >>> dst = np.empty((od.shape[0] - 1) * (od.shape[1] - 1) - sta_dim)
    >>> __j_from_ode_compute(od, sta_dim, dst)
    >>> print(dst)
    [24.3  10.8   4.8   0.27  5.    7.2   9.8   1.28  0.16]
    >>> rs = np.array([9 * 9 * 0.3, 6 * 6 * 0.3, 4 * 4 * 0.3, 3 * 3 * 0.03,
    ...                5 * 5 * 0.2, 6 * 6 * 0.2, 7 * 7 * 0.2, 8 * 8 * 0.02,
    ...                4 * 4 * 0.01])
    >>> print(rs)
    [24.3  10.8   4.8   0.27  5.    7.2   9.8   1.28  0.16]
    """
    index: int = len(dest)
    start: Final[int] = ode.shape[1] - 2

    inner: int = start
    row = ode[0]
    weight_01: float = row[-1] * 0.1
    while inner >= state_dim:
        v = row[inner]
        inner -= 1
        index -= 1
        dest[index] = (v * v) * weight_01 if -1e100 < v < 1e100 else 1e100
    for row in ode[1:-1]:
        inner = start
        weight: float = row[-1]
        weight_01 = weight * 0.1
        while inner >= state_dim:
            v = row[inner]
            inner -= 1
            index -= 1
            dest[index] = (v * v) * weight_01 if -1e100 < v < 1e100 else 1e100
        while inner >= 0:
            v = row[inner]
            inner -= 1
            index -= 1
            dest[index] = (v * v) * weight if -1e100 < v < 1e100 else 1e100


def t_from_ode(ode: np.ndarray) -> float:
    """
    Get the time sum from an ODE solution.

    The total time that we simulate a system depends on the behavior of the
    system. If the system makes small steps (has small differential values)
    and does not change its direction, it can be simulated for a long time.
    If the system diverges, e.g., if some state variables would go to
    infinity, or if it often changes direction or has some larger steps, then
    the total simulation time shortens automatically. This prevents the system
    from actually diverging. Notice that the time slice length of the very
    last step in the ODE solution is ignored.

    :param ode: the ODE solution, as return from :func:`run_ode`.
    :return: the time sum

    This adds up the last column in the ODE solution, ignoring the last time
    slice. This means we get 0.1 + 0.2 + 0.3 = 0.6.

    >>> od = np.array([[1, 2, 3, 4, 0.1],
    ...                [5, 6, 7, 8, 0.2],
    ...                [9, 6, 4, 3, 0.3],
    ...                [7, 4, 2, 1, 0.4]])
    >>> print(t_from_ode(od))
    0.6
    """
    return fsum(ode[0:-1, -1])


def j_from_ode(ode: np.ndarray, state_dim: int) -> float:
    """
    Compute the original figure of merit from an ODE array.

    The figure of merit is the sum of state variable squares plus 0.1 times
    the control variable squares. We disregard the state variable values of
    the starting states (because they are the same for all controllers on
    a given training case and because the control cannot influence them) and
    we also disregard the final state and final controller output (as there is
    no time slice associated with them, i.e., we only "arrive" in them but
    basically spent 0 time in them in our simulation).

    :param ode: the array returned by the ODE function, i.e., :func:`run_ode`
    :param state_dim: the state dimension
    :return: the figure of merit

    >>> od = np.array([[1, 2, 3, 4, 0.1],
    ...                [5, 6, 7, 8, 0.2],
    ...                [9, 6, 4, 3, 0.3],
    ...                [7, 4, 2, 1, 0.4]])
    >>> sta_dim = 3
    >>> print(f"{j_from_ode(od, sta_dim):.10f}")
    106.0166666667
    >>> print((24.3 + 10.8 + 4.8 + 0.27 + 5. + 7.2 + 9.8 + 1.28 + 0.16) / 0.6)
    106.01666666666667
    """
    dest: Final[np.ndarray] = np.empty(
        (ode.shape[0] - 1) * (ode.shape[1] - 1) - state_dim)
    __j_from_ode_compute(ode, state_dim, dest)
    return fsum(dest) / t_from_ode(ode)


def multi_run_ode(
        test_starting_states: Iterable[np.ndarray],
        training_starting_states: Iterable[np.ndarray],
        collector: Callable[[int, np.ndarray, float, float], None] | Iterable[
            Callable[[int, np.ndarray, float, float], None]],
        equations: Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None],
        controller: Callable[[np.ndarray, float, T, np.ndarray], None],
        parameters: T, controller_dim: int = 1,
        test_steps: int = 5000, training_steps: int = 5000) -> None:
    """
    Invoke :func:`run_ode` multiple times and pass the result to `collector`.

    This function allows us to perform multiple runs of the differential
    equation simulator, using different starting points. It also allows us to
    distinguish training and test points and to assign them different numbers
    of steps. For each of them, :func:`run_ode` will be applied and the
    returned matrix is passed to the `collector` function.

    :param test_starting_states: the iterable of test starting states
    :param training_starting_states: the iterable of training starting states
    :param collector: the destination to receive the results
    :param equations: the differential system
    :param controller: the controller function
    :param parameters: the controller parameters
    :param controller_dim: the dimension of the controller result
    :param test_steps: the number of test steps to simulate
    :param training_steps: the number of training steps to simulate
    """
    if not isinstance(collector, Iterable):
        collector = (collector, )
    index: int = 0
    for sp in test_starting_states:
        ode = run_ode(sp, equations, controller, parameters, controller_dim,
                      test_steps)
        for c in collector:
            c(index, ode, j_from_ode(ode, len(sp)), t_from_ode(ode))
        index += 1
    for sp in training_starting_states:
        ode = run_ode(sp, equations, controller, parameters, controller_dim,
                      training_steps)
        for c in collector:
            c(index, ode, j_from_ode(ode, len(sp)), t_from_ode(ode))
        index += 1
