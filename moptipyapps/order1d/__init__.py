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
"""
