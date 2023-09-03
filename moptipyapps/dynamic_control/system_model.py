"""
An extended dynamic control problem `Instance` with a model for the dynamics.

An :class:`~moptipyapps.dynamic_control.instance.Instance` is a combination of

- a set of differential equations
  (:mod:`~moptipyapps.dynamic_control.system`) that govern the change
  `D=ds/dt` of the state `s` of a dynamic system (based on its current
  state `s` and the controller output `c(s)`), i.e., `ds/dt=D(s,c(s))` and
- a blueprint `c(s, p)` of the controller `c(s)`
  (:mod:`~moptipyapps.dynamic_control.controller`).

The blueprint of the controller is basically a function that can be
parameterized, so it is actually a function `c(s, p)` and the goal of
optimization is to parameterize it in such a way that the figure of merit,
i.e., the objective value (:mod:`~moptipyapps.dynamic_control.objective`)
of the system (usually the sum of squared state values and squared controller
outputs) is minimized. Parameterization means finding good values `p` such
that the above goal is reached. In other words, we want to synthesize a
controller (by finding good values of `p`) in such a way that the state
equations drive the system into a stable state, usually ideally the origin of
the coordinate system.

Now here this :class:`~moptipyapps.dynamic_control.instance.Instance` is
extended to a :class:`~moptipyapps.dynamic_control.system_model.SystemModel`
by adding a parameterized model `M(s, c(s), q)` to the mix. The idea is to
develop the parameterization `q` of the model `M` that can replace the actual
system equations `D`. Such a model receives as input the current system state
vector `s` and the controller output `c(s)` for the state vector `s`. It will
return the differential `D` of the system state, i.e., `ds/dt`. In other
words, a properly constructed model can replace the `equations` parameter in
the ODE integrator :func:`~moptipyapps.dynamic_control.ode.run_ode`. The
input used for training is provided by
:func:`~moptipyapps.dynamic_control.ode.diff_from_ode`.

What we do here is to re-use the same function models as used in controllers
(:mod:`~moptipyapps.dynamic_control.controller`) and learn their
parameterizations from the observed data. If successful, we can  wrap
everything together into a `Callable` and plug it into the system instead of
the original equations.

The thing that :class:`SystemModel` offers is thus a blueprint of the model
`M`. Obviously, we can conceive many different such blueprints. We could have
a linear model, a quadratic model, or maybe neural network (in which case, we
need to decide about the number of layers and layer sizes). So an instance of
this surrogate model based approach has an equation system, a controller
blueprint, and a model blueprint.

An example implementation of the concept of synthesizing models for a dynamic
system in order to synthesize controllers is given as the
:mod:`~moptipyapps.dynamic_control.surrogate_cma` algorithm.
Examples for different dynamic systems controllers (which we here also use to
model the systems themselves) are given in package
:mod:`~moptipyapps.dynamic_control.controllers`.

The starting points of the work here were conversations with Prof. Dr. Bernd
NOACK and Guy Yoslan CORNEJO MACEDA of the Harbin Institute of Technology in
Shenzhen, China (哈尔滨工业大学(深圳)).
"""
from typing import Final

from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.system import System


class SystemModel(Instance):
    """A dynamic system control `Instance` with a system model blueprint."""

    def __init__(self, system: System, controller: Controller,
                 model: Controller) -> None:
        """
        Initialize the system model.

        :param system: the system that we want to model
        :param controller: the controller that we want to use
        :param model: a system model blueprint (also in the shape of a
            controller)
        """
        if not isinstance(model, Controller):
            raise type_error(model, "model", Controller)
        super().__init__(system, controller, model.name)
        in_dims: Final[int] = system.state_dims + controller.control_dims
        if model.state_dims != in_dims:
            raise ValueError(
                f"model.state_dims={model.state_dims}, but system.state_dims"
                f"={system.state_dims} and controller.control_dims="
                f"{controller.control_dims}, so we expected {in_dims} for "
                f"controller {str(controller)!r}, system {str(system)!r}, and"
                f" model {str(model)!r}.")
        if model.control_dims != system.state_dims:
            raise ValueError(
                f"model.control_dims={model.control_dims} must be the same as"
                f" system.state_dims={system.state_dims}, but is not, for "
                f"controller {str(controller)!r}, system {str(system)!r}, and"
                f" model {str(model)!r}.")
        #: the model blueprint that can be trained to hopefully replace the
        #: :attr:`~moptipyapps.dynamic_control.instance.Instance.system` in the
        #: ODE integration / system simulation procedure
        self.model: Final[Controller] = model

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("model") as scope:
            self.model.log_parameters_to(scope)
