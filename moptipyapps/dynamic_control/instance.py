"""
An instance of the dynamic control synthesis problem.

An instance of the dynamic control synthesis problem is comprised of two
components: a :mod:`~moptipyapps.dynamic_control.system` of differential
equations governing how the state of a system changes over time and a
:mod:`~moptipyapps.dynamic_control.controller` that uses the current system
state as input and computes a controller value as output that influences the
state change. Notice that the controller here is a parametric function. The
goal of the dynamic system control is to find the right parameterization of
the controller such that an :mod:`~moptipyapps.dynamic_control.objective` is
minimized. The objective here usually has the goal to bring the dynamic system
into a stable state while using as little controller "energy" as possible.

An instance of the simultaneous control and model synthesis problem is an
instance of the class :class:`~moptipyapps.dynamic_control.system_\
model.SystemModel`, which is a subclass of :class:`~moptipyapps.\
dynamic_control.instance.Instance`. It also adds a controller blueprint for
modelling the systems response (state differential) based on the system state
and controller output.

The starting point of the work here were conversations with Prof. Dr. Bernd
NOACK and Guy Yoslan CORNEJO MACEDA of the Harbin Institute of Technology in
Shenzhen, China (哈尔滨工业大学(深圳)).
"""

from typing import Final

import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.strings import sanitize_name
from pycommons.io.path import Path
from pycommons.types import type_error

from moptipyapps.dynamic_control.controller import Controller
from moptipyapps.dynamic_control.system import System


class Instance(Component):
    """An instance of the dynamic control problem."""

    def __init__(self, system: System, controller: Controller,
                 name_base: str | None = None) -> None:
        """
        Create an instance of the dynamic control problem.

        :param system: the system of equations governing the dynamic system
        :param controller: the controller applied to the system
        :param name_base: the name base
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

        name: str = f"{self.system}_{self.controller}"
        if name_base is not None:
            if not isinstance(name_base, str):
                raise type_error(name_base, "name_base", (str, None))
            nn = sanitize_name(name_base)
            if nn != name_base:
                raise ValueError(f"sanitized name base {nn!r} different "
                                 f"from original {name_base!r}.")
            name = f"{name}_{name_base}"
        #: the name of this instance
        self.name: Final[str] = name

    def __str__(self) -> str:
        """
        Get the name of this instance.

        :return: a combination of the equation name and the controller name
        """
        return self.name

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
            dest_dir: str) -> tuple[Path, ...]:
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
