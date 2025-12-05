"""Maximize the worst and average immediate rates."""


from moptipy.api.objective import Objective

from moptipyapps.prodsched.multistatistics import MultiStatistics


class WorstAndMeanFillRate(Objective):
    """Compute the worst immediate rate and return `1 -` of it."""

    def evaluate(self, x: MultiStatistics) -> int | float:
        """
        Get the negated worst immediate rate.

        :param x: the multi-statistics
        :return: the worst stock level
        """
        min_imm: int | float = 1
        avg_imm: int | float = 1
        for stat in x.per_instance:
            for sl in stat.immediate_rates:
                min_imm = 0 if sl is None else min(min_imm, sl)
            avg_imm = 0 if stat.immediate_rate is None else min(
                avg_imm, stat.immediate_rate)
        return (1 - min_imm) * 100 + (1 - avg_imm)

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
        return 101

    def __str__(self) -> str:
        """
        Get the name of the objective function.

        :return: `worstMinAndMeanFillRate`
        :retval "worstMinAndMeanFillRate": always
        """
        return "worstAndMeanFillRate"
