"""A base class for implementing controllers."""


from typing import Callable, Final

import numpy as np
from moptipy.api.component import Component
from moptipy.spaces.vectorspace import VectorSpace
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.types import check_to_int_range, type_error


class Controller(Component):
    """A class for governing a system via differential system."""

    def __init__(self, name: str,
                 state_dims: int, control_dims: int, param_dims: int,
                 func: Callable[[np.ndarray, float, np.ndarray,
                                 np.ndarray], None] | None = None) -> None:
        """
        Initialize the system.

        :param name: the name of the system.
        :param state_dims: the state dimensions
        :param control_dims: the control dimensions
        :param param_dims: the parameter dimensions
        """
        super().__init__()
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        #: the controller name
        self.name: Final[str] = name
        #: the dimensions of the state variable
        self.state_dims: Final[int] = check_to_int_range(
            state_dims, "state_dims", 2, 3)
        #: the dimensions of the controller output
        self.control_dims: Final[int] = check_to_int_range(
            control_dims, "control_dims", 1, 100)
        #: the dimensions of the controller parameter
        self.param_dims: Final[int] = check_to_int_range(
            param_dims, "param_dims", 1, 1_000)
        if func is not None:
            if not callable(func):
                raise type_error(func, "func", None, call=True)
            self.controller = func  # type: ignore

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("stateDims", self.state_dims)
        logger.key_value("controlDims", self.control_dims)
        logger.key_value("paramDims", self.param_dims)

    def __str__(self):
        """
        Get the name of this controller.

        :return: the name of this controller
        """
        return self.name

    def controller(self, state: np.ndarray,  # pylint: disable=E0202
                   time: float,  # pylint: disable=E0202
                   params: np.ndarray,  # pylint: disable=E0202
                   out: np.ndarray) -> None:  # pylint: disable=E0202
        """
        Compute the control value and store it in `out`.

        :param state: the state vector
        :param time: the time value
        :param params: the controller variables
        :param out: the output array to receive the controller values
        """

    def parameter_space(self) -> VectorSpace:
        """
        Create a vector space to represent the possible parameterizations.

        :return: a vector space for the possible parameterizations of this
            controller.
        """
        return VectorSpace(self.param_dims, -32.0, 32.0)
