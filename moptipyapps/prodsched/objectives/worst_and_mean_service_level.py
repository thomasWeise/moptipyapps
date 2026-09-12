"""
Maximize the worst and average service level.

This objective function tries to find solutions which have very robust
and also good service levels.
The service level is the fraction of customers that can get served directly,
i.e., the fraction of customers that do not need to wait.
This means that it is the fraction of customers whose demands can directly be
satisfied from the stock.

Service levels are between 0 and 1.
Of course, high service levels are good and should therefore be subject to
maximization.
However, since we can only *minimize*, we minimize "1 - service level".

Now, the question is:
What is a *robust* service level / solution?
Well, we simulate the solutions (such as re-order points) over multiple
instances.
A robust good service level would be high on the worst instance.
In other words, the smallest service level measured on any instance should
be as high as possible.
This means that the largest value "1 - service level" should be as small as
possible.

However, this does not consider the average performance.
A good average performance would mean that we maximize the *average*
service level over all instances, or, in terms of minimization, that we
minimize "1 - average service level".

This objective function combines both concepts, putting special emphasis
on the worst-case service level.
It minimizes

    "100+(1 - worst-case service level) + (1 - average service level)"
"""


from moptipy.api.objective import Objective

from moptipyapps.prodsched.multistatistics import MultiStatistics


class WorstAndMeanFillRate(Objective):
    """Combine and minimize worst and average service level."""

    def evaluate(self, x: MultiStatistics) -> int | float:
        """
        Get the negated worst service level combined with the average.

        :param x: the multi-statistics
        :return: the worst stock level
        """
        min_imm: int | float = 1
        avg_imm: int | float = 1
        for stat in x.per_instance:
            for sl in stat.service_levels:
                min_imm = 0 if sl is None else min(min_imm, sl)
            avg_imm = 0 if stat.service_level is None else min(
                avg_imm, stat.service_level)
        return (1 - min_imm) * 100 + (1 - avg_imm)

    def lower_bound(self) -> int:
        """
        Get the lower bound of the inverted minimum service level.

        :retval 0: always
        """
        return 0

    def upper_bound(self) -> int:
        """
        Get the upper bound of the inverted minimum service level.

        :retval 101: always
        """
        return 101

    def __str__(self) -> str:
        """
        Get the name of the objective function.

        :return: `worstMinAndMeanFillRate`
        :retval "worstMinAndMeanFillRate": always
        """
        return "worstAndMeanFillRate"
