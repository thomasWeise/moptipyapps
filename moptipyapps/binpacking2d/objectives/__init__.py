"""
Different objective functions for two-dimensional bin packing.

The following objective functions are implemented:

- :mod:`~moptipyapps.binpacking2d.objectives.bin_count` returns the number
  of bins occupied by a given packing.
- :mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_empty` returns
  a combination of the number of bins occupied by a given packing and the
  fewest number of objects located in any bin.
- :mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_last_empty`
  returns a combination of the number of bins occupied by a given packing and
  the number of objects located in the last bin.
- :mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_small` returns
  a combination of the number of bins occupied by a given packing and the
  smallest area occupied by objects in any bin.
- :mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_last_small` returns
  a combination of the number of bins occupied by a given packing and the
  area occupied by the objects in the last bin.
- :mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_lowest_skyline`
  returns a combination of the number of bins occupied by a given packing and
  the smallest area under the skyline in any bin, where the "skyline" is the
  upper border of the space occupied by objects.
- :mod:`~moptipyapps.binpacking2d.objectives.bin_count_and_last_skyline`
  returns a combination of the number of bins occupied by a given packing and
  the smallest area under the skyline in the last bin, where the "skyline" is
  the upper border of the space occupied by objects.

Important initial work on this code has been contributed by Mr. Rui ZHAO
(赵睿), <zr1329142665@163.com> a Master's student at the Institute of Applied
Optimization (应用优化研究所) of the School of
Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei University
(合肥大学) in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""
