"""
Poor man's Artificial Neural Networks.

Here, artificial neural networks (ANNs) are defined as plain mathematical
functions which are parameterized by their weights. The weights are subject
to black-box optimization and all together put into a single vector.
In other words, we do not use proper back-propagation learning or any other
sophisticated neural network specific training strategy. Instead, we treat the
neural networks as black boxes that can be parameterized using the weight
vector. Different ANN architectures have different weight vectors.
As activation functions, we use `arctan`.

The neural networks here are automatically generated via code generation
provided by module :mod:`~moptipyapps.dynamic_control.controllers.codegen`.
This allows us to have networks of arbitrary shape for arbitrary input and
output dimensions. Since code generation followed by on-the-fly compilation
via numba is time and memory consuming, we cache all neural network-based
instances of :class:`~moptipyapps.dynamic_control.controller.Controller`.
"""

from typing import Callable, Final, Iterable, cast

import numpy as np
from pycommons.types import check_int_range, type_error

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.controllers.codegen import CodeGenerator
from moptipyapps.dynamic_control.system import System


def make_ann(state_dims: int, control_dims: int, layers: list[int]) \
        -> Controller:
    """
    Dynamically create an ANN.

    :param state_dims: the state or input dimension
    :param control_dims: the output dimension
    :param layers: the sizes of the hidden layers
    :returns: the controller
    """
    state_dims = check_int_range(state_dims, "state_dims", 1, 100)
    control_dims = check_int_range(control_dims, "state_dims", 1, 100)
    if not isinstance(layers, list):
        raise type_error(layers, "layers", list)
    for layer in layers:
        check_int_range(layer, "layer", 1, 64)

    # we also try to cache the generated controllers
    description = "_".join(map(str, ([state_dims, control_dims, *layers])))
    description = f"__cache_{description}"
    if hasattr(make_ann, description):
        return cast(Controller, getattr(make_ann, description))

    code: Final[CodeGenerator] = CodeGenerator(
        "state: np.ndarray, _: float, params: np.ndarray, out: np.ndarray")
    params: int = 0  # the number of parameters
    var_count: int = 0  # the number of variables
    vars_in: list[str] = []  # the variables forming the current layer input
    vars_out: list[str] = []  # the variables forming the current layer output
    vars_cached: list[str] = []  # the variables cached for re-use
    write: Final[Callable[[str], None]] = code.write  # fast call
    writeln: Final[Callable[[str], None]] = code.writeln  # fast call

    # first, we cache the state vector into local variables
    for i in range(state_dims):
        vv = f"s{i}"
        vars_in.append(vv)
        writeln(f"{vv} = state[{i}]")

    # now we build the hidden layers of the network
    for layer in layers:
        for _ in range(layer):
            # allocate a variable for storing the current neuron's output
            if len(vars_cached) > 0:
                var = vars_cached.pop(-1)
            else:
                var_count += 1
                var = f"v{var_count}"
            vars_out.append(var)  # remember the variable
            write(f"{var} = np.arctan(params[{params}]")  # the bias
            params += 1
            for vv in vars_in:
                write(f" + params[{params}] * {vv}")  # input * weight
                params += 1
            writeln(")")
        vars_cached.extend(vars_in)  # old inputs ready for reuse
        vars_in.clear()  # inputs are no longer used
        vars_in, vars_out = vars_out, vars_in  # outputs become inputs

    # now we construct the output layer
    for i in range(control_dims):
        write(f"out[{i}] = params[{params}] * ")  # the multiplier
        params += 1
        write(f"np.arctan(params[{params}]")  # the bias
        params += 1
        for vv in vars_in:
            write(f" + params[{params}] * {vv}")  # input * weight
            params += 1
        writeln(")")

    result: Final[Controller] = Controller(
        f"ann_{'_'.join(map(str, layers))}" if len(layers) > 0 else "ann",
        state_dims, control_dims, params, code.build())
    # perform one test invocation
    result.controller(np.zeros(state_dims), 0.0, np.zeros(params),
                      np.empty(control_dims))
    setattr(make_ann, description, result)  # cache the controller
    return result


def anns(system: System) -> Iterable[Controller]:
    """
    Create poor man's ANNs fitting to a given system.

    Based on the dimensionality of the state space, we generate a set of ANNs
    with different numbers of layers and neurons. The weights of the neurons
    can then directly be optimized by a numerical optimization algorithm.
    This is, of course, probably much less efficient that doing some proper
    learning like back-propagation. However, it allows us to easily plug the
    ANNs into the same optimization routines as other controllers.

    :param system: the equations object
    :return: the ANNs
    """
    state_dims: Final[int] = system.state_dims
    control_dims: Final[int] = system.control_dims
    return (make_ann(state_dims, control_dims, []),
            make_ann(state_dims, control_dims, [1]),
            make_ann(state_dims, control_dims, [2]),
            make_ann(state_dims, control_dims, [3]),
            make_ann(state_dims, control_dims, [2, 2]),
            make_ann(state_dims, control_dims, [3, 2]))
