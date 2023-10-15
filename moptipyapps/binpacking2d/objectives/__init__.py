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
"""
