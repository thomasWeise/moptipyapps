"""An instance of the instance generation problem."""

from typing import Final

from moptipy.api.component import Component
from moptipy.spaces.vectorspace import VectorSpace

from moptipyapps.binpacking2d.instance import Instance
from moptipyapps.binpacking2d.instgen.inst_decoding import InstanceDecoder
from moptipyapps.binpacking2d.instgen.instance_space import InstanceSpace


class Problem(Component):
    """An instance of the 2D Bin Packing Instance Generation Problem."""

    def __init__(self, name: str, slack: float) -> None:
        """
        Create an instance of the 2D bin packing instance generation problem.

        :param name: the name of the instance to select
        :param slack: the additional fraction of slack
        """
        super().__init__()
        #: the instance to be used as template
        self.solution_space: Final[InstanceSpace] = InstanceSpace(
            Instance.from_resource(name))
        #: the internal decoding
        self.encoding: Final[InstanceDecoder] = InstanceDecoder(
            self.solution_space)
        n: Final[int] = self.encoding.get_x_dim(slack)
        #: the search space
        self.search_space: Final[VectorSpace] = VectorSpace(n, -1.0, 1.0)

    def __str__(self) -> str:
        """
        Get the string representation of this problem.

        :return: the string representation of this problem
        """
        return (f"{self.solution_space.inst_name}_"
                f"{self.search_space.dimension}")
