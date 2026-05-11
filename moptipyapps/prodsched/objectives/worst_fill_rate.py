"""
Maximize the worst-case immediate rate.

This objective function tries to find solutions which have very robust fill
rates.
The fill rate is the fraction of customers that can get served directly,
i.e., the fraction of customers that do not need to wait.
This means that it is the fraction of customers whose demands can directly be
satisfied from the stock.

Fill rates are between 0 and 1.
Of course, high fill rates are good and should therefore be subject to
maximization.
However, since we can only *minimize*, we minimize "1 - fill rate".

Now, the question is:
What is a *robust* fill rate / solution?
Well, we simulate the solutions (such as re-order points) over multiple
instances.
A robust good fill rate would be high on the worst instance.
In other words, the smallest fill rate measured on any instance should
be as high as possible.
This means that the largest value "1 - fill rate" should be as small as
possible.
So we use this as result of our objective function.
"""


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
