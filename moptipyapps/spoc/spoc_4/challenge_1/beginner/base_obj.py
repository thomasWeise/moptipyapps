"""A base object with stuff for handling the beginner problem."""

from typing import Final

import numpy as np
from pycommons.types import type_error

from moptipyapps.spoc.spoc_4.challenge_1.beginner.instance import Instance


class BaseObjectWithArrays:
    """The objective function of the beginner problem."""

    def __init__(self, instance: Instance) -> None:
        """
        Create the objective function of the beginner problem.

        :param instance: the instance of the objective function.
        """
        super().__init__()
        if not isinstance(instance, Instance):
            raise type_error(instance, "instance", Instance)
        #: the instance
        self.instance: Final[Instance] = instance

        #: the earth orbit usage
        self.earth = np.ndarray(instance.lengths[0], dtype=np.bool)
        #: the lunar orbit usage
        self.lunar = np.ndarray(instance.lengths[1], dtype=np.bool)
        #: the destination orbit usage
        self.dest = np.ndarray(instance.lengths[2], dtype=np.bool)
