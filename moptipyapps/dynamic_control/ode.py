"""
A primitive integrator for systems of ordinary differential equations.

Many dynamic systems can be modeled as systems of ordinary differential
equations that govern their progress over time. Trying to find out in
which state such systems are at a given point in time means to integrate
these equations until that point in time (starting from a starting state).

What we want to play around with, however, is synthesizing controllers.
In this case, the differential equations also merge the output of the
controller with the current state. If the controller behaves inappropriately,
this may make the system diverge, i.e., some of its state variables go to
infinity over time or sometimes rather quickly.

Using ODE integrators that compute the system state at pre-defined time steps
is thus cumbersome, as the system may have already exploded at these goal
times. Therefore, we perform ODE integration in several steps. First, we try
it the "normal" way. However, as soon as the system escapes the sane parameter
range, we stop. We then use the last point where the system was stable and the
first point where it escaped the reasonable range to estimate a new reasonable
end time for the integration. We do this until we finally succeed.

Thus, we can simulate a well-behaved system over a long time and an
ill-behaved system for a shorter time period. Neither system will diverge.

The following functions are provided:

- :func:`run_ode` executes a set of differential system equations and
  controller equations and returns an array with the system state, controller
  output, and time at the different interpolation steps.
- :func:`t_from_ode` returns the total time over which the result of
  :func:`run_ode` was simulated.
- :func:`j_from_ode` returns a figure of merit, i.e., a weighted sum of (a
  part of) the system state and the controller output from the result of
  :func:`run_ode`.
- :func:`diff_from_ode` extracts state difference information from the result
  of :func:`run_ode`.
"""

from math import fsum, inf
from typing import Callable, Final, Iterable, TypeVar

import numba  # type: ignore
import numpy as np
from scipy.integrate import RK45, DenseOutput  # type: ignore

#: the type variable for ODE controller parameters
T = TypeVar("T")


@numba.njit(cache=True, inline="always", fastmath=True, boundscheck=False)
def _is_ok(x: np.ndarray) -> bool:
    """
    Check whether all values in a vector are acceptable.

    A vector is "ok" if all of its elements are from the finite range
    `(-1e10, 1e10)`.  Anything else indicates that we are somehow moving
    out of the reasonable bounds.

    :param x: the vector
    :return: `True` if all values are OK, `False` otherwise
    """
    for xx in x:  # noqa: SIM110
        if not -1e10 < xx < 1e10:
            return False
    return True


class __IntegrationState:
    """
    The internal integrator state class.

    This class serves two purposes. First, it encapsulates the system
    equations and the controller equations such that they can be called as a
    unit. Second, it raises an alert if the system escapes the "OK" state,
    i.e., diverge towards infinity. As long as everything is OK,
    :attr:`is_ok` will remain `True`. But if either the controller or the
    system state escapes the acceptable interval, it becomes `False`.
    In that case, :attr:`max_ok_t` holds the highest `t` value at which
    the system and controller were in an acceptable state and
    :attr:`min_error_t` holds the smallest `t` value at which that was not
    the case. We can assume that somewhere inbetween lies the last moment we
    can find the system in a sane state.
    """

    def __init__(self,
                 equations: Callable[[
                     np.ndarray, float, np.ndarray, np.ndarray], None],
                 controller: Callable[[
                     np.ndarray, float, T, np.ndarray], None],
                 parameters: T, controller_dim: int) -> None:
        """
        Create the integrator.

        :param equations: the differential system
        :param controller: the controller function
        :param parameters: the controller parameters
        :param controller_dim: the dimension of the controller result
        """
        self.__equations: Final[Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None]] = equations
        self.__controller: Final[Callable[[
            np.ndarray, float, T, np.ndarray], None]] = controller
        self.__parameters: Final[T] = parameters
        self.__ctrl: Final[np.ndarray] = np.empty(controller_dim)
        self.max_ok_t: float = -inf
        self.min_error_t: float = inf
        self.is_ok: bool = True

    def init(self) -> None:
        """Prepare the system for integration."""
        self.max_ok_t = -inf
        self.min_error_t = inf
        self.is_ok = True

    def f(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute the differential at the given state and time.

        First, we invoke the controller function at the state and time. Then
        we pass the controller vector to the differential equations to update
        the system state.
        This function also checks if the state or control go out of bounds
        and if they do, it sets the :attr:`is_ok` to `False`.

        :param t: the time
        :param state: the state
        :return: the differential
        """
        out: Final[np.ndarray] = np.zeros_like(state)  # allocate output vec
        ctrl: Final[np.ndarray] = self.__ctrl  # the controller vector

        # invoke the controller
        self.__controller(state, t, self.__parameters, ctrl)
        ok: bool = _is_ok(ctrl)  # is the controller vector ok?
        if ok:  # if yes, let's invoke the state update equations
            self.__equations(state, t, ctrl, out)
            ok = _is_ok(out)  # and check if their result is ok
        if ok:  # is it ok?
            if self.max_ok_t < t < self.min_error_t:
                self.max_ok_t = t
        else:  # no: there was some error, either in state or controller
            self.is_ok = False  # then we are no longer OK
            if t < self.min_error_t:  # update the minimum error time
                self.min_error_t = t
            if self.max_ok_t > t:  # what? an earlier error?
                self.max_ok_t = np.nextafter(t, -inf)  # ok, reset ok time

        return out


def run_ode(starting_state: np.ndarray,
            equations: Callable[[
                np.ndarray, float, np.ndarray, np.ndarray], None],
            controller: Callable[[np.ndarray, float, T, np.ndarray], None],
            parameters: T, controller_dim: int = 1,
            steps: int = 5000, max_time: float = 50.0) -> np.ndarray:
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

    Now the `run_ode` function simulates such a system over `steps` time steps
    over the closed interval `[0, max_time]`. If both the system and the
    controller are well-behaved, then the output array will contain `steps`
    rows with the state, controller, and time information of each step. If
    the system diverges at some point in time but we can simulate it
    reasonably well before that, then we try to simulate `steps`, but on a
    shorter time frame. If even that fails, you will get a single row output
    with `1e100` as the controller value.

    This function returns a matrix where each row corresponds to a simulated
    time step. Each row contains three components in a concatenated fashion:
    1. the state vector,
    2. the control vector,
    3. the time value

    :param starting_state: the starting
    :param equations: the differential system
    :param controller: the controller function
    :param parameters: the controller parameters
    :param controller_dim: the dimension of the controller result
    :param steps: the number of steps to simulate
    :param max_time: the maximum time to simulate for
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
    >>> print(len(ode))
    10000
    >>> time_of_flight = 2 * v_y / 9.80665
    >>> print(f"{time_of_flight:.10f}")
    14.4209649817
    >>> travel_distance_x = time_of_flight * v_x
    >>> print(f"{travel_distance_x:.10f}")
    1019.7162129779
    >>> idx = np.argwhere(ode[:, 1] <= 0.0)[0][0]
    >>> print(idx)
    2887
    >>> print(f"{ode[idx - 1, 0]:.10f}")
    1020.4571309653
    >>> print(f"{ode[idx, 0]:.10f}")
    1020.8107197148
    >>> print(f"{ode[idx - 1, -1]:.10f}")
    14.4314431443
    >>> print(f"{ode[idx, -1]:.10f}")
    14.4364436444
    >>> print(ode[-1, -1])
    50.0

    >>> def contrl2(position, ttime, params, dest):
    ...     dest[0] = 1e50  #  controller that is ill-behaved
    >>> run_ode(strt, projectile, contrl2, param, 1, 10000)
    array([[0.e+000, 1.e+000, 1.e+100, 0.e+000]])

    >>> def contrl3(position, ttime, params, dest):
    ...     dest[0] = 1e50 if ttime > 10 else 0.0  # diverging controller
    >>> ode = run_ode(strt, projectile, contrl3, param, 1, 10000)
    >>> print(len(ode))
    10000
    >>> print(ode[-1])
    [690.10677249 224.06765771   0.           9.75958357]

    >>> def projectile2(position, ttime, ctrl, out):
    ...     out[:] = 0
    >>> ode = run_ode(strt, projectile2, contrl, param, 1, 10000)
    >>> print(len(ode))
    10000
    >>> print(ode[-1])
    [ 0.  1.  0. 50.]
    >>> ode = run_ode(strt, projectile2, contrl3, param, 1, 10000)
    >>> print(len(ode))
    10000
    >>> print(ode[-1])
    [0.         1.         0.         9.41234557]
    """
    func_state: Final[__IntegrationState] = __IntegrationState(
        equations, controller, parameters, controller_dim)
    func_call: Final[Callable[[float, np.ndarray], np.ndarray]] = func_state.f
    denses: Final[list[DenseOutput]] = []
    n: Final[int] = len(starting_state)
    dim: Final[int] = n + controller_dim + 1

    cycle: int = 0
    while True:  # loop until we have a sane integration over a sane range
        cycle += 1
        # first, we reset all the state information
        func_state.init()  # reset the function state
        denses.clear()  # always discard all interpolators, if there are any

        # then we create the integrator for the time range that we simulate
        integration = RK45(
            fun=func_call, t0=0.0, y0=starting_state, t_bound=max_time,
            max_step=steps)
        is_finished: bool = False

        # perform the integration and collect all the points at which stuff
        # was computed
        while True:  # iteratively add interpolation points
            integration.step()  # do the integration step
            if not func_state.is_ok:
                break  # some out-of-bounds thing happened! quit!
            is_finished = integration.status == "finished"
            is_running: bool = integration.status == "running"
            if is_finished or is_running:  # keep taking interpolator
                denses.append(integration.dense_output())
                if is_finished:
                    break  # we are finished, so we quit and build output
                continue  # more integration to do, so we go on
            break  # if we get here, there was an error: quit

        if is_finished and func_state.is_ok:
            # if we get here, everything looks fine so far, so we can try
            # to build the output
            result: np.ndarray = np.zeros((steps, dim))
            point: np.ndarray = result[0]
            point[0:n] = starting_state

            # we now compute all the points by using the interpolation
            # we start with the first point
            controller(starting_state, 0.0, parameters, point[n:-1])
            if not _is_ok(point):
                break
            result[:, -1] = np.linspace(0.0, max_time, steps)

            j: int = 0  # the index of the dense interpolator to use
            n_dense: int = len(denses)  # the number of interpolators
            t: float = 0.0  # the current time
            dense: DenseOutput = denses[j]  # the current interpolator
            for point in result[1:]:  # for each of the remaining points...
                last_time: float = t  # remember the last successful point
                t = point[-1]  # get the time value

                # we now need to find the right interpolator if we have left
                # the range of the current interpolator
                while not (dense.t_min <= t <= dense.t_max):
                    j += 1  # step counter
                    if j >= n_dense:   # we have left the interpolation range??
                        func_state.max_ok_t = last_time  # so we need to adjust
                        func_state.min_error_t = t
                        is_finished = False  # and try the whole thing again
                        break
                    dense = denses[j]  # pick next interpolator

                if is_finished:  # if we get here, we got a right interpolator
                    point[0:n] = dense(t)  # so we can interpolate the state
                    controller(point[0:n], t, parameters, point[n:-1])
                    if not _is_ok(point):  # is there an error in the vector?
                        func_state.max_ok_t = last_time  # last ok time
                        func_state.min_error_t = t  # error time
                        is_finished = False  # we need to quit and try again
                        break  # stop inner loop

            if is_finished:  # did we succeed?
                return result  # yes! return the result.

        # if we arrive here, things went wrong somehow.
        # this means that we should reduce the maximum runtime
        if (cycle < 3) and (func_state.max_ok_t < func_state.min_error_t):
            max_time = np.nextafter(min(func_state.min_error_t, (
                0.8 * func_state.max_ok_t) + (0.2 * func_state.min_error_t)),
                -inf)
        else:  # the small reductions did not work out well ... reduce rapidly
            max_time = np.nextafter(0.7 * min(func_state.max_ok_t,
                                              max_time), -inf)

        if (cycle > 4) or (max_time <= 1e-10):
            break  # if we get here, everything seems so pointless...

    # the default error result
    result = np.zeros((1, dim))
    result[0, 0:n] = starting_state
    result[0, n:-1] = 1e100
    result[0, -1] = 0.0
    return result


@numba.njit(cache=True, inline="always", fastmath=False, boundscheck=False)
def __j_from_ode_compute(ode: np.ndarray, state_dim: int,
                         use_state_dims: int,
                         gamma: float,
                         dest: np.ndarray) -> None:
    """
    Prepare the input array for the figure of merit computation.

    The `ode` matrix contains one row for each time step.
    The row is composed of the current state, the current controller output,
    and the current time.
    However, all systems of a given simulation start in the same initial
    state, so it makes little sense to include this initial state into the
    figure of merit computation. It does make sense to include the control
    output at that moment, though, because it contributes to the next state.
    It also makes no sense to count in the last row of the ODE computation,
    because this is the final system state and the system will not spend any
    time in it in the simulation.

    :param ode: the output array from the ODE simulation
    :param state_dim: the state dimension
    :param use_state_dims: the dimension until which the state is used
    :param gamma: the weight for the control variable
    :param dest: the destination array

    >>> od = np.array([[1, 2, 3, 4, 0],
    ...                [5, 6, 7, 8, 1],
    ...                [9, 6, 4, 3, 3],
    ...                [7, 4, 2, 1, 7]])
    >>> sta_dim = 3
    >>> dst = np.empty((od.shape[0] - 1) * (od.shape[1] - 1) - sta_dim)
    >>> __j_from_ode_compute(od, sta_dim, sta_dim, 0.1, dst)
    >>> print(dst)
    [  1.6  12.8  98.   72.   50.    3.6  64.  144.  324. ]
    >>> rs = np.array([0.1 * 1 * 4 * 4,
    ...                0.1 * 2 * 8 * 8, 2 * 7 * 7, 2 * 6 * 6, 2 * 5 * 5,
    ...                0.1 * 4 * 3 * 3, 4 * 4 * 4, 4 * 6 * 6, 4 * 9 * 9])
    >>> print(rs)
    [  1.6  12.8  98.   72.   50.    3.6  64.  144.  324. ]
    >>> u_sta_dim = 2
    >>> dst = np.empty((od.shape[0] - 1) * (
    ...     od.shape[1] - 1 - sta_dim + u_sta_dim) - u_sta_dim)
    >>> __j_from_ode_compute(od, sta_dim, u_sta_dim, 1.0, dst)
    >>> print(dst)
    [ 16. 128.  72.  50.  36. 144. 324.]
    """
    index: int = 0
    start: Final[int] = ode.shape[1] - 2

    # from now on, we compute the impact of the state and the controller
    add_state: bool = False
    last_row: np.ndarray = ode[0]
    for i in range(1, len(ode)):
        next_row: np.ndarray = ode[i]
        weight: float = next_row[-1] - last_row[-1]
        inner = start
        weight_01: float = weight * gamma
        while inner >= state_dim:
            v = last_row[inner]
            inner -= 1
            dest[index] = (v * v) * weight_01 if -1e100 < v < 1e100 else 1e100
            index += 1

        if add_state:
            inner = use_state_dims  # jump to the used state
            while inner > 0:
                inner -= 1
                v = last_row[inner]
                dest[index] = (v * v) * weight if -1e100 < v < 1e100 else 1e100
                index += 1

        add_state = True
        last_row = next_row


def t_from_ode(ode: np.ndarray) -> float:
    """
    Get the time sum from an ODE solution.

    The total time that we simulate a system depends on the behavior of the
    system.

    :param ode: the ODE solution, as return from :func:`run_ode`.
    :return: the total consumed time

    >>> od = np.array([[1, 2, 3, 4, 0.1],
    ...                [5, 6, 7, 8, 0.2],
    ...                [9, 6, 4, 3, 0.3],
    ...                [7, 4, 2, 1, 0.4]])
    >>> print(t_from_ode(od))
    0.4
    """
    return ode[-1, -1]


def j_from_ode(ode: np.ndarray, state_dim: int,
               use_state_dims: int = -1,
               gamma: float = 0.1) -> float:
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
    :param use_state_dims: the dimension until which the state is used,
        `-1` for using the complete state
    :param gamma: the weight of the controller input
    :return: the figure of merit

    >>> od = np.array([[1, 2, 3, 4, 0],
    ...                [5, 6, 7, 8, 1],
    ...                [9, 6, 4, 3, 3],
    ...                [7, 4, 2, 1, 7]])
    >>> sta_dim = 3
    >>> print(f"{j_from_ode(od, sta_dim):.10f}")
    110.0000000000
    >>> print((1.6 + 12.8 + 98 + 72 + 50 + 3.6 + 64 + 144 + 324) / 7)
    110.0
    >>> sta_dim = 3
    >>> print(f"{j_from_ode(od, 3, 2, 0.5):.10f}")
    97.1428571429
    >>> print((8 + 64 + 72 + 50 + 18 + 144 + 324) / 7)
    97.14285714285714
    """
    if len(ode) <= 1:
        return 1e200
    # The used state dimension could be equal to the state dimension or less.
    # If it is <= 0, then we use the complete state vector
    if use_state_dims <= 0:
        use_state_dims = state_dim
    dest: Final[np.ndarray] = np.empty((ode.shape[0] - 1) * (
        ode.shape[1] - 1 - state_dim + use_state_dims) - use_state_dims)
    __j_from_ode_compute(ode, state_dim, use_state_dims, gamma, dest)
    return fsum(dest) / ode[-1, -1]


def multi_run_ode(
        test_starting_states: Iterable[np.ndarray],
        training_starting_states: Iterable[np.ndarray],
        collector: Callable[[int, np.ndarray, float, float], None] | Iterable[
            Callable[[int, np.ndarray, float, float], None]],
        equations: Callable[[
            np.ndarray, float, np.ndarray, np.ndarray], None],
        controller: Callable[[np.ndarray, float, T, np.ndarray], None],
        parameters: T, controller_dim: int = 1,
        test_steps: int = 5000,
        test_time: float = 50.0,
        training_steps: int = 5000,
        training_time: float = 50.0,
        use_state_dims: int = -1, gamma: float = 0.1) -> None:
    """
    Invoke :func:`run_ode` multiple times and pass the result to `collector`.

    This function allows us to perform multiple runs of the differential
    equation simulator, using different starting points. It also allows us to
    distinguish training and test points and to assign them different numbers
    of steps. For each of them, :func:`run_ode` will be applied and the
    returned matrix is passed to the `collector` function.

    :param test_starting_states: the iterable of test starting states
    :param training_starting_states: the iterable of training starting states
    :param collector: the destination to receive the results, in the form of
        index, ode array, j, and t.
    :param equations: the differential system
    :param controller: the controller function
    :param parameters: the controller parameters
    :param controller_dim: the dimension of the controller result
    :param test_steps: the number of test steps to simulate
    :param test_time: the time limit for tests
    :param training_steps: the number of training steps to simulate
    :param training_time: the time limit for training
    :param use_state_dims: the dimension until which the state is used,
        `-1` for using the complete state
    :param gamma: the weight of the controller input
    """
    if not isinstance(collector, Iterable):
        collector = (collector, )
    index: int = 0
    for sp in test_starting_states:
        ode = run_ode(sp, equations, controller, parameters, controller_dim,
                      test_steps, test_time)
        for c in collector:
            c(index, ode, j_from_ode(ode, len(sp), use_state_dims, gamma),
              t_from_ode(ode))
        index += 1
    for sp in training_starting_states:
        ode = run_ode(sp, equations, controller, parameters, controller_dim,
                      training_steps, training_time)
        for c in collector:
            c(index, ode, j_from_ode(ode, len(sp), use_state_dims, gamma),
              t_from_ode(ode))
        index += 1


def diff_from_ode(ode: np.ndarray, state_dim: int) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Compute all the state+control vectors and the resulting differentials.

    This function returns two matrices. Each row of both matrices corresponds
    to a time slot. Each row in the first matrix holds the state vector and
    the control vector (that was computed by the controller). The
    corresponding row in the second matrix then holds the state differential
    resulting from the control vector being applied in the differential
    equations that govern the system state change.

    The idea is that this function basically provides the data that we would
    like to learn when training a surrogate model for a system: From the
    current state and the computed control vector, we want that our model can
    give us the resulting system differential. If we have such a model and it
    works reasonably well, then we could essentially plug this model into
    :func:`run_ode` instead of the original `equations` parameter.

    What this function does to compute the differential is to basically
    "invert" the dynamic weighting done by :func:`run_ode`. :func:`run_ode`
    starts in a given starting state `s`. It then computes the control vector
    `c` as a function of `s`, i.e., `c(s)`. Then, the equations of the dynamic
    system (see module :mod:`~moptipyapps.dynamic_control.system`) to compute
    the state differential `D=ds/dt` as a function of `c(s)` and `s`, i.e., as
    something like `D(s, c(s))`. The next step would be to update the state,
    i.e., to set `s=s+D(s, c(s))`. Unfortunately, this can make `s` go to
    infinity. So :func:`run_ode` will compute a dynamic weight `w` and do
    `s=s+w*D(s, c(s))`, where `w` is chosen such that the state vector `s`
    does not grow unboundedly. While `s` and `c(s)` and `w` are stored in one
    row of the result matrix of :func:`run_ode`, `s+w*D(s,c(s))` is stored as
    state `s` in the next row. So what this function here basically does is to
    subtract the old state from the next state and divide the result by `w` to
    get `D(s, c(s))`. `s` and `c(s)` are already available directly in the ODE
    result and `w` is not needed anymore.

    We then get the rows `s, c(s)` and `D(s, c(s))` in the first and second
    result matrix, respectively. This can then be used to train a system model
    as proposed in model :mod:`~moptipyapps.dynamic_control.system_model`.

    :param ode: the result of :func:`run_ode`
    :param state_dim: the state dimensions
    :returns: a tuple of the state+control vectors and the resulting
        state differential vectors

    >>> od = np.array([
    ...     [0, 0, 0, 0, 0, 0],  # state 0,0,0; control 0,0; time 0
    ...     [1, 2, 3, 4, 5, 1],  # state 1,2,3; control 4,5; time 1
    ...     [2, 3, 4, 5, 6, 3],  # state 2,3,4; control 5,6; time 3
    ...     [4, 6, 8, 7, 7, 7]])    # state 4,6,8; control 7,7, time 7
    >>> res = diff_from_ode(od, 3)
    >>> res[0]  # state and control vectors, time col and last row removed
    array([[0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6]])
    >>> res[1]  # (state[i + 1] - state[i]) / (time[i + 1] / time[i])
    array([[1.  , 2.  , 3.  ],
           [0.5 , 0.5 , 0.5 ],
           [0.5 , 0.75, 1.  ]])
    """
    return (ode[0:-1, 0:-1],
            np.diff(ode[:, 0:state_dim], 1, 0)
            / np.diff(ode[:, -1])[:, None])
