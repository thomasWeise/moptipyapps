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
"""
