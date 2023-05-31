"""
Codes for the two-dimensional bin packing problem.

- :mod:`~moptipyapps.binpacking2d.instance` provides the instance data of
  two-dimensional bin packing problems.
  The instance data comprises the object sizes and multiplicities as well as
  the bin sizes.
  Several default instances from \
[2DPackLib](https://site.unibo.it/operations-research/en/research/2dpacklib)
  can be loaded from resources.
- :mod:`~moptipyapps.binpacking2d.packing` can store a packing, i.e., an
  assignment of objects to coordinates and bins.
  Such packings are solutions to the bin packing problem.
- :mod:`~moptipyapps.binpacking2d.packing_space` provides an implementation of
  the :mod:`~moptipy.api.space` interface for the
  :mod:`~moptipyapps.binpacking2d.packing` objects.
  This class allows, for example, to instantiate the packings, to verify
  whether they are correct, and to convert them to and from strings.
- :mod:`~moptipyapps.binpacking2d.plot_packing` allows you to plot a packing.
- :mod:`~moptipyapps.binpacking2d.ibl_encoding_1` is an implementation of the
  improved bottom-left encoding which closes bins immediately once an object
  does not fit.
- :mod:`~moptipyapps.binpacking2d.ibl_encoding_2` is another implementation of
  the improved bottom-left encoding which tests each bin for each object.
- :mod:`~moptipyapps.binpacking2d.bin_count_and_last_empty` provides an
  objective function that tries to minimize the number of bins and pushes
  towards decreasing the number of objects in the very last bin used.
- :mod:`~moptipyapps.binpacking2d.make_instances` offers a method to download
  the two-dimensional bin packing instances from the original sources in the \
[2DPackLib](https://site.unibo.it/operations-research/en/research/2dpacklib)
  format.

Important initial work on this code has been contributed by Mr. Rui ZHAO
(赵睿), <zr1329142665@163.com> a Master's student at the Institute of Applied
Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School of
Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei University
(合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).
"""
