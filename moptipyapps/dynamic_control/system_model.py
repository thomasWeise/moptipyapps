"""
A system model tries to approximate how a controller output impacts a system.

The idea is to develop a model that can replace the actual system. Such a
model receives as input the current system state vector `state` and the
controller output `control` for the state vector `state`. It will return
the differential of the system state, i.e., `dstate/dT`. In other words,
the constructed model can replace the `equations` parameter in
:func:`~moptipyapps.dynamic_control.ode.run_ode`. The idea is here to
re-use the same function models as used in controllers
(:mod:`~moptipyapps.dynamic_control.controller`), learn their
parameterizations from the observed data, and wrap everything together
into a callable.
"""
from typing import Final

from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import type_error

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.instance import Instance
from moptipyapps.dynamic_control.system import System


class SystemModel(Instance):
    """A class that attempts to construct a system model."""

    def __init__(self, system: System, controller: Controller,
                 model: Controller) -> None:
        """
        Initialize the system model.

        :param system: the system that we want to model
        :param controller: the controller that we want to use
        :param model: the controller used as model
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
        #: the model controller
        self.model: Final[Controller] = model

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("model") as scope:
            self.model.log_parameters_to(scope)
