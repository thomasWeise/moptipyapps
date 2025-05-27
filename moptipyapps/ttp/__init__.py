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
   https://robinxval.ugent.be/. In this repository, you can also find the
   original instance data, useful descriptions, and many links to related
   works.
2. Kelly Easton, George L. Nemhauser, and Michael K. Trick. The Traveling
   Tournament Problem Description and Benchmarks. In *Principles and Practice
   of Constraint Programming (CP'01),*  November 26 - December 1, 2001, Paphos,
   Cyprus, pages 580-584, Berlin/Heidelberg, Germany: Springer.
   ISBN: 978-3-540-42863-3. https://doi.org/10.1007/3-540-45578-7_43.
   https://www.researchgate.net/publication/220270875.
3. Celso C. Ribeiro and Sebastián Urrutia. Heuristics for the Mirrored
   Traveling Tournament Problem. *European Journal of Operational Research*
   (EJOR) 179(3):775-787, June 16, 2007.
   https://doi.org/10.1016/j.ejor.2005.03.061.
   https://optimization-online.org/wp-content/uploads/2004/04/851.pdf.
4. Sebastián Urrutia and Celso C. Ribeiro. Maximizing Breaks and Bounding
   Solutions to the Mirrored Traveling Tournament Problem.
   *Discrete Applied Mathematics* 154(13):1932-1938, August 15, 2006.
   https://doi.org/10.1016/j.dam.2006.03.030.

So far, the following components have been implemented:

1. :mod:`~moptipyapps.ttp.instance` provides the data of a TTP instance,
   including the number of teams, the constraints, and the distance matrix.
   Notice that it is an extension of the Traveling Salesperson Problem
   :mod:`~moptipyapps.tsp.instance` instance data, from which it inherits the
   distance matrix. This data basically describes the starting situation and
   the input data when we try to solve a TTP instance. Also, the module
   provides several of the benchmark instances from
   <https://robinxval.ugent.be/>.
2. :mod:`~moptipyapps.ttp.game_plan` provides a class for holding one
   candidate solution to the TTP, i.e., a game plan. The game plan states, for
   each day and each team, against which other team it will plan (if any). The
   game plan may contain errors, which need to be sorted out by optimization.
3. :mod:`~moptipyapps.ttp.game_plan_space` offers the
   `moptipy` `Space` functionality for such game plans. In other words, it
   allows us to instantiate game plans in a uniform way and to convert them to
   and from strings (which is used when writing log files).
4. :mod:`~moptipyapps.ttp.game_encoding` allows us to decode a permutation
   (potentially with repetitions, see :mod:`~moptipy.spaces.permutations`)
   into a game plan. We therefore can use optimization algorithms and
   operators working on the well-understood space of permutations to produce
   game plans. However, the decoded game plans may have errors, e.g., slots
   without games or violations of the maximum or minimum streak length
   constraints.
5. :mod:`~moptipyapps.ttp.errors` offers an objective function that counts the
   number of such constraint violations in a game plan. If we can minimize it
   to zero, then the game plan is feasible.

This is code is part of the research work of Mr. Xiang CAO (曹翔), a Master's
student at the Institute of Applied Optimization (应用优化研究所) of
the School of Artificial Intelligence and Big Data
(人工智能与大数据学院) at Hefei University (合肥大学)
in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""
