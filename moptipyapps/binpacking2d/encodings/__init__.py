"""
Different encodings for the two-dimensional bin packing problem.

The following encodings are implemented:

- The improved bottom-left encoding 1,
  :mod:`~moptipyapps.binpacking2d.encodings.ibl_encoding_1`, processes a
  permutation of objects from beginning to end and places the objects into
  the last bin (according to the improved-bottom-left method) until that bin
  is full and then begins to put them into the next bin.
- The improved bottom-left encoding 2,
  :mod:`~moptipyapps.binpacking2d.encodings.ibl_encoding_2`, processes a
  permutation of objects from beginning to end and places the objects into
  the first bin into which they fit (according to the improved-bottom-left
  method). It is slower than
  :mod:`~moptipyapps.binpacking2d.encodings.ibl_encoding_1` but its results
  are never worse for any permutation and better for several.

Important initial work on this code has been contributed by Mr. Rui ZHAO
(赵睿), <zr1329142665@163.com> a Master's student at the Institute of Applied
Optimization (应用优化研究所) of the School of
Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei University
(合肥大学) in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""
