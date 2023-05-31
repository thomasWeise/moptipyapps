"""
Experiments with the Traveling Salesperson Problem (TSP).

A Traveling Salesperson Problem (TSP) is defined as a fully-connected graph
with :attr:`~moptipyapps.tsp.instance.Instance.n_cities` nodes. Each edge in
the graph has a weight, which identifies the distance between the nodes. The
goal is to find the *shortest* tour that visits every single node in the graph
exactly once and then returns back to its starting node. Then nodes are
usually called cities. In this file, we present methods for loading instances
of the TSP as distance matrices `A`. In other words, the value at `A[i, j]`
identifies the travel distance from `i` to `j`. Such instance data can be
loaded via class :mod:`~moptipyapps.tsp.instance`.

In this package, we provide the following tools:

- :mod:`~moptipyapps.tsp.instance` allows you to load instance data in the
  TSPLib format and it provides several instances from TSPLib as resources.
- :mod:`~moptipyapps.tsp.known_optima` provides known optimal solutions for
  some of the TSPLib instances. These should mainly be used for testing
  purposes.
- :mod:`~moptipyapps.tsp.tour_length` is an objective function that can
  efficiently computed the length of a tour in path representation.
- :mod:`~moptipyapps.tsp.tsplib` is just a dummy package holding the actual
  TSPLib data resources.

Important initial work on this code has been contributed by Mr. Tianyu LIANG
(梁天宇), <liangty@stu.hfuu.edu.cn> a Master's student at the Institute of
Applied Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School
of Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei
University (合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the
supervision of Prof. Dr. Thomas Weise (汤卫思教授).
"""
