"""An instance of the dynamic control problem."""

from typing import Final

import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.path import Path
from moptipy.utils.types import type_error

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


class Instance(Component):
    """An instance of the dynamic control problem."""

    def __init__(self, system: System, controller: Controller) -> None:
        """
        Create an instance of the dynamic control problem.

        :param system: the system of equations governing the dynamic system
        :param controller: the controller applied to the system
        """
        super().__init__()
        if not isinstance(system, System):
            raise type_error(system, "system", System)
        if not isinstance(controller, Controller):
            raise type_error(controller, "controller", Controller)
        if controller.state_dims != system.state_dims:
            raise ValueError(
                f"controller.state_dims={controller.state_dims}, but "
                f"system.state_dims={system.state_dims} for controller "
                f"{str(controller)!r} and system {str(system)!r}.")
        if controller.control_dims != system.control_dims:
            raise ValueError(
                f"controller.control_dims={controller.control_dims}, but "
                f"system.control_dims={system.control_dims} for controller "
                f"{str(controller)!r} and system {str(system)!r}.")
        #: the system governing the dynamic system
        self.system: Final[System] = system
        #: the controller applied to the system
        self.controller: Final[Controller] = controller

    def __str__(self) -> str:
        """
        Get the name of this instance.

        :return: a combination of the equation name and the controller name
        """
        return f"{self.system}_{self.controller}"

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        with logger.scope("sys") as scope:
            self.system.log_parameters_to(scope)
        with logger.scope("ctrl") as scope:
            self.controller.log_parameters_to(scope)

    def describe_parameterization(
            self, title: str | None,
            parameters: np.ndarray, base_name: str,
            dest_dir: str) -> tuple[Path, Path]:
        """
        Describe the performance of a given system of system.

        :param title: the optional title
        :param parameters: the controller parameters
        :param base_name: the base name of the file to produce
        :param dest_dir: the destination directory
        :returns: the paths of the generated files
        """
        the_title = self.controller.name
        if title is not None:
            the_title = f"{the_title}\n{title}"
        return self.system.describe_system(
            the_title, self.controller.controller, parameters,
            base_name, dest_dir)
