"""An objective function for maximizing the worst immediate rate."""


from moptipy.api.objective import Objective

from moptipyapps.prodsched.multistatistics import MultiStatistics


class WorstFillRate(Objective):
    """Compute the worst immediate rate and return `1 -` of it."""

    def evaluate(self, x: MultiStatistics) -> int | float:
        """
        Get the negated worst immediate rate.

        :param x: the multi-statistics
        :return: the worst stock level
        """
        min_imm: int | float = 1
        for stat in x.per_instance:
            for sl in stat.immediate_rates:
                if (sl is None) or not (0 < sl <= 1):
                    return 1
                min_imm = min(min_imm, sl)
        return 1 - min_imm

    def lower_bound(self) -> int:
        """
        Get the lower bound of the inverted minimum immediate rate.

        :retval 0: always
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the inverted minimum immediate rate.

        :retval 1: always
        """
        return 1

    def __str__(self) -> str:
        """
        Get the name of the objective function.

        :return: `worstFillRate`
        :retval "worstFillRate": always
        """
        return "worstFillRate"
