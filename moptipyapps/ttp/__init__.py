"""
The Traveling Tournament Problem (TTP).

The Traveling Tournament Problem (TTP) models the logistics of a sports
league, where each game is defined as two teams playing against each other.
In each time slot (let's call that "day") of the tournament, each team has
one game against one other team. One of the two teams of each game will play
"at home," the other "away."

In order to have each of the `n` teams play once against each of the `n-1`
other teams, we need to have `n - 1` "days". So in one round of the
tournament, there are `n - 1` time slots so that each of the teams can play
exactly `n - 1` games, i.e., once against every other team.

If the tournament has two rounds (i.e., is a double round-robin tournament),
then each game appears twice, but with home and away team switched. There are
also constraints for the minimum and maximum number of games at home in a row
and away in a row. There are also constraints for the minimum and maximum
number in between repeating a game (with home/away team switched).

1. David Van Bulck. Minimum Travel Objective Repository. *RobinX: An
   XML-driven Classification for Round-Robin Sports Timetabling.* Faculty of
   Economics and Business Administration at Ghent University, Belgium.
   https://robinxval.ugent.be/
2. Kelly Easton, George L. Nemhauser, and Michael K. Trick. The Traveling
   Tournament Problem Description and Benchmarks. In *Principles and Practice
   of Constraint Programming (CP'01),*  November 26 - December 1, 2001, Paphos,
   Cyprus, pages 580-584, Berlin/Heidelberg, Germany: Springer.
   ISBN: 978-3-540-42863-3. https://doi.org/10.1007/3-540-45578-7_43
   https://www.researchgate.net/publication/220270875
"""
