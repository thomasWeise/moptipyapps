"""An encoding that is inspired by a given instance."""
from typing import Final

from moptipy.api.space import Space
from moptipy.utils.logger import KeyValueLogSection
from pycommons.types import check_int_range, type_error

from moptipyapps.binpacking2d.instance import IDX_HEIGHT, IDX_WIDTH, Instance


class InstanceSpace(Space):
    """An space structure for the instance generation problem and space."""

    def __init__(self, source: Instance) -> None:
        """
        Create the space structure for the instance generation problem.

        :param source: the source instances whose date we take as an example
        """
        super().__init__()
        #: the instance name
        self.inst_name: Final[str] = str.strip(source.name) + "n"
        #: the target number of unique items
        self.n_different_items: Final[int] = check_int_range(
            source.n_different_items, "n_different_items",
            1, 100_000)
        #: the target number of items (including repetitions)
        self.n_items: Final[int] = check_int_range(
            source.n_items, "n_items", source.n_different_items,
            1_000_000_000)
        #: the bin width
        self.bin_width: Final[int] = check_int_range(
            source.bin_width, "bin_width", 1, 1_000_000_000)
        #: the bin height
        self.bin_height: Final[int] = check_int_range(
            source.bin_height, "bin_height", 1, 1_000_000_000)
        #: the minimum number of bins that this instance requires
        self.min_bins: Final[int] = check_int_range(
            min(source.lower_bound_bins, self.n_items), "min_bins",
            1, 1_000_000_000)
        #: the minimum item width
        self.item_width_min: Final[int] = check_int_range(int(min(
            source[:, IDX_WIDTH])), "item_width_min", 1, self.bin_width)
        #: the maximum item width
        self.item_width_max: Final[int] = check_int_range(int(max(
            source[:, IDX_WIDTH])), "item_width_max", self.item_width_min,
            self.bin_width)
        #: the minimum item height
        self.item_height_min: Final[int] = check_int_range(int(min(
            source[:, IDX_HEIGHT])), "item_height_min", 1, self.bin_height)
        #: the maximum item height
        self.item_height_max: Final[int] = check_int_range(int(max(
            source[:, IDX_HEIGHT])), "item_height_max", self.item_height_min,
            self.bin_height)
        #: the total item area
        self.total_item_area: Final[int] = check_int_range(
            source.total_item_area, "total_item_area", 1, 1_000_000_000)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters of this instance.

        :param logger: the logger
        """
        super().log_parameters_to(logger)
        logger.key_value("instName", self.inst_name)
        logger.key_value("nItems", self.n_items)
        logger.key_value("nDifferentItems", self.n_different_items)
        logger.key_value("binWidth", self.bin_width)
        logger.key_value("binHeight", self.bin_height)
        logger.key_value("minBins", self.min_bins)
        logger.key_value("itemWidthMin", self.item_width_min)
        logger.key_value("itemWidthMax", self.item_width_max)
        logger.key_value("itemHeightMin", self.item_height_min)
        logger.key_value("itemHeightMax", self.item_height_max)
        logger.key_value("totalItemArea", self.total_item_area)

    def create(self) -> list[Instance]:
        """
        Generate a list for receiving an instance.

        :return: the new instance list
        """
        return []

    def copy(self, dest: list[Instance], source: list[Instance]) -> None:
        """
        Copy the instance list.

        :param dest: the destination instance list
        :param source: the source instance list
        """
        dest.clear()
        dest.extend(source)

    def to_str(self, x: list[Instance]) -> str:  # +book
        """
        Convert an instance list to a string.

        :param x: the instance list
        :return: the string representation of x
        """
        return x[0].to_compact_str()

    def from_str(self, text: str) -> list[Instance]:
        """
        Convert an instance string to a list with an instance.

        :param text: the input string
        :return: the element in the space corresponding to `text`
        """
        return [Instance.from_compact_str(text)]

    def is_equal(self, x1: list[Instance], x2: list[Instance]) -> bool:
        """
        Check if the contents of two instances of the data structure are equal.

        :param x1: the first instance
        :param x2: the second instance
        :return: `True` if the contents are equal, `False` otherwise
        """
        return x1 == x2

    def validate(self, x: list[Instance]) -> None:
        """
        Check whether a given point in the space is valid.

        :param x: the point
        :raises TypeError: if the point `x` (or one of its elements, if
            applicable) has the wrong data type
        :raises ValueError: if the point `x` is invalid and/or simply is not
            an element of this space
        """
        if list.__len__(x) != 1:
            raise ValueError("There must be exactly one instance in x.")
        inst: Instance = x[0]
        if not isinstance(inst, Instance):
            raise type_error(inst, "x[0]", Instance)
        if inst.name != self.inst_name:
            raise ValueError(
                f"instance name {inst.name!r} != {self.inst_name!r}.")
        if inst.bin_width != self.bin_width:
            raise ValueError(
                f"instance bin width={inst.bin_width} != {self.bin_width}.")
        if inst.bin_height != self.bin_height:
            raise ValueError(
                "instance bin height="
                f"{inst.bin_height} != {self.bin_height}.")
        if inst.n_items != self.n_items:
            raise ValueError(
                f"instance n_items={inst.n_items} != {self.n_items}.")

    def n_points(self) -> int:
        """
        Get the approximate number of different elements in the space.

        :return: the approximate scale of the space
        """
        return (self.bin_height * self.bin_width) ** self.n_items
