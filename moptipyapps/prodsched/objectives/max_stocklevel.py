"""An objective function for minimizing the maximal stocklevel."""

from math import inf

from moptipy.api.objective import Objective

from moptipyapps.prodsched.multistatistics import MultiStatistics


class MaxStockLevel(Objective):
    """Compute the worst stock level."""

    def evaluate(self, x: MultiStatistics) -> int | float:
        """
        Get the worst (largest) stocklevel.

        :param x: the multi-statistics
        :return: the worst stock level
        """
        max_sl: int | float = 0
        for stat in x.per_instance:
            sl = stat.stock_level
            if (sl is None) or not (0 <= sl <= 1_000_000_000):
                return inf
            max_sl = max(max_sl, sl)
        return max_sl

    def lower_bound(self) -> int:
        """
        Get the lower bound of the maximum storage level.

        :retval 0: always
        """
        return 0

    def __str__(self) -> str:
        """
        Get the name of the objective function.

        :return: `maxStorageLevel`
        :retval "maxStorageLevel": always
        """
        return "maxStorageLevel"
