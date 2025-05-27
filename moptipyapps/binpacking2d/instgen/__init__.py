"""
Tools for generating 2d bin packing instances.

We want to generate instances of the two-dimensional bin packing problem.
These instances should have some pre-defined characteristics, e.g., width and
height of the bins, number of items to pack, lower bound/optimal number of
bins required by any solution, and so on.

At the same time, the instances should be hard.

We treat this whole thing as an optimization problem. Here, given are the
pre-defined instance characteristics and the goal is to find instances that
are hard to solve.

Important work on this code has been contributed by Mr. Rui ZHAO
(赵睿), <zr1329142665@163.com> a Master's student at the Institute of Applied
Optimization (应用优化研究所) of the School of
Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei University
(合肥大学) in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""
