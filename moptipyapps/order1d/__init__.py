"""
A set of tools for ordering objects in 1 dimension.

Let's assume that we have `n` objects and a distance metric that can compute
the distance between two objects. We do not know and also do not care about in
how many dimension the objects exist - we just have objects and a distance
metric.

Now we want to find a one-dimensional order of the objects that reflects their
original distance-based topology. For each object `a`, we want that its
closest neighbor in the order is also its actual closest neighbor according to
the distance metric. It's second-closest neighbor should be the actual
second-closest neighbor according to the distance metric. And so on.

Since we only care about the object order and do not want to metrically map
the distances to one dimension, we can represent the solution as permutation
of natural numbers.

Of course, in a one-dimensional order, each object has exactly two closest
neighbors (the one on its left and the one on its right) unless it is situated
either at the beginning or end of the order, in which case it has exactly one
closest neighbor. Based on the actual distance metric, an object may have any
number of closest neighbors, maybe only one, or maybe three equally-far away
objects. So it is not clear whether a perfect mapping to the one-dimensional
permutations even exists.

But we can try to find one that comes as close as possible to the real deal.

Another way to describe this problem is as follows:
Imagine that you have `n` objects and only know their mutual distances.
You want to arrange them on a one-dimensional axis in a way that does sort of
reflect their neighborhood structure in whatever space they are originally
located in.

The goal of solving this one-dimensional ordering problem is then to arrange
`n` objects on a 1-dimensional (e.g., horizontal) axis given a distance matrix
describing their location in a potentially high-dimensional or unstructured
space.
The objects should be arranged in such a way that, for each object,

- the nearest neighbors on the 1-dimensional axis are also the nearest
  neighbors in the original space (according to the distance matrix
  provided),
- the second nearest neighbors on the 1-dimensional axis are also the
  second nearest neighbors in the original space (according to the distance
  matrix provided),
- the third nearest neighbors on the 1-dimensional axis are also the third
  nearest neighbors in the original space (according to the distance matrix
  provided),
- and so on; with (e.g., quadratically) decreasing weights of neighbor
  distance ranks.

We do not care about the actual precise distances (e.g., something like
"0.001") between the objects on either the one-dimensional nor the original
space. Only about the distance ranks, i.e., about "2nd nearest neighbor," but
not "0.012 distance units away." The solutions of this problem are thus
permutations (orders) of the objects. Of course, if we really want to plot the
objects, such a permutation can easily be translated to `x`-coordinates, say,
by dividing the index of an object by the number of objects, which nets values
in `[0,1]`. But basically, we reduce the task to finding permutations of
objects that reflect the neighbor structure of the original space as closely
as possible.

If such a problem is solved correctly, then the arrangement on the
one-dimensional axis should properly reflect the arrangement of the objects in
the original space. Of course, solving this problem exactly may not actually
be possible, since an object on a one-dimensional axis may either have exactly
two `i`-nearest-neighbors (if it is at least `i` slots away from either end of
the permutation) or exactly `1` such neighbor, if it is closer that `i` units.
The object directly at the start of the permutation has only 1 nearest
neighbor (the object that comes next). That next object, however, has two,
namely the first object and the third object. In the original space where the
objects come from, however, there may be any number of "nearest neighbors."
Imagine a two-dimensional space where one object sits at the center of a
circle of other objects. Then all other objects are its nearest neighbors,
whereas an object on the circle either has exactly two nearest neighbors or,
maybe, in the odd situation that the radius equals a multiple of the
distance to the neighbors on the circle, three. Such a structure cannot be
represented exactly in one dimension.

But that's OK.
Because we mainly do this for visualization purposes anyway.

Now here comes the cool thing: We can cast this problem to a Quadratic
Assignment Problem (:mod:`~moptipyapps.qap`). A QAP is defined by a distance
matrix and a flow matrix. The idea is to assign `n` facilities to locations.
The distances between the locations are given by the distance matrix. At the
same time, the flow matrix defines the flows between the facilities. The goal
is to find the assignment of facilities to locations that minimizes the
overall "flow times distance" product sum.

We can cast the one-dimensional ordering problem to a QAP as follows: The
facilities represent the original objects that we want to arrange. The indices
in the permutation of facilities be the locations and their distances their
absolute difference. The flow between two facilities be inversely proportional
to the distance rank in the original space.
"""
