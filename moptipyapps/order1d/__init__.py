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
permutations even exists. Probably it will not, but this shall not bother us.

Either way, we can try to find one that comes as close as possible to the real
deal.

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
space. We only care about the distance ranks, i.e., about "2nd nearest
neighbor," but not "0.012 distance units away." The solutions of this problem
are thus permutations (orders) of the objects. Of course, if we really want
to plot the objects, such a permutation can easily be translated to
`x`-coordinates, say, by dividing the index of an object by the number of
objects, which nets values in `[0,1]`. But basically, we reduce the task to
finding permutations of objects that reflect the neighbor structure of the
original space as closely as possible.

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

We can translate the "One-Dimensional Ordering Problem" to the QAP as follows:
Imagine that we have `n` unique objects and know the (symmetric) distances
between them.

Let's say that we have four numbers as objects, `1`, `2`, `3`, `4`.
>>> data = [1, 2, 3, 4]
>>> n = len(data)

The distance between two numbers `a` and `b` be `abs(a - b)`.
>>> def dist(a, b):
...     return abs(a - b)

The **original** distance matrix would be

>>> dm = [[dist(i, j) for j in range(n)] for i in range(n)]
>>> print(dm)
[[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]

Now from this matrix, we can find the nearest neighbors as follows:

>>> from scipy.stats import rankdata  # type: ignore
>>> import numpy as np
>>> rnks = rankdata(dm, axis=1, method="average") - 1
>>> print(rnks)
[[0.  1.  2.  3. ]
 [1.5 0.  1.5 3. ]
 [3.  1.5 0.  1.5]
 [3.  2.  1.  0. ]]

From the perspective of "`1`", "`2`" is the first nearest neighbor,
"`3`" is the second-nearest neighbor, and "`4`" is the third-nearest neighbor.
From the perspective of "`2`", both "`1`" and "`3`" are nearest neighbors, so
they share rank `1.5`.
And so on.

Let's multiply this by `2` to get integers:
[If no fractional ranks appear, then we do not need to multiply by `2`.]

>>> rnks = np.array(rnks * 2, dtype=int)
>>> print(rnks)
[[0 2 4 6]
 [3 0 3 6]
 [6 3 0 3]
 [6 4 2 0]]

Now we want to translate this to a flow matrix (NOT a distance matrix).
What we want is that there is a big flow from each object to its nearest
neighbor, a smaller flow from each object to its second-nearest neighbor,
a yet smaller flow to the third nearest neighbor, and so on.
So all we need to do is to subtract the elements off the diagonal from
`8 = 2*n`.
[If no fractional elements were there, then we would use `n` instead of
`2*n`.]
Anyway, we get:

>>> rnks = np.array([[0 if x == 0 else 2*n - x for x in a]
...                  for a in rnks], int)
>>> print(rnks)
[[0 6 4 2]
 [5 0 5 2]
 [2 5 0 5]
 [2 4 6 0]]

We can see:
The flow from "`1`" to "`2`" is `6`, which higher than the flow from "`1`" to
"`3`" (namely `4`), and so on.
The low from "`2`" to "`1`" is `5`, which is a bit lower than `6` but higher
than `4`, because "`1`" is the "1.5th nearest neighbor of `2`".

Just for good measures, we can weight this qudratically, so we get:

>>> rnks = rnks ** 2
>>> print(rnks)
[[ 0 36 16  4]
 [25  0 25  4]
 [ 4 25  0 25]
 [ 4 16 36  0]]

This way, we give much more importance to "getting the nearest neighbors
right" than getting "far-away neighbors" right. Nice.

OK, but how about the distance matrix for the QAP?
What's the distance?
Well, it's the distance that I would get by arranging the objects in a
specific order.
It is the difference of the indices in the permutation:

>>> print(np.array([[abs(i - j) for j in range(n)] for i in range(n)], int))
[[0 1 2 3]
 [1 0 1 2]
 [2 1 0 1]
 [3 2 1 0]]

This means:
The object that I will place at index `1` has a distance of "`1`" to the
object at index `2`.
It has a distance of "`2`" to the object at index `3`.
It has a distance of "`3`" to the object at index `4`.
The object at index `2` will have a distance of "`1`" to both the objects
at indices `1` and `3` and a distance of "`2`" to the object at index `4`.
And so on.

In other words, objects that are close in their original space have a big
flow between each other.
If we put them at directly neighboring indexes, then we will multiply this
big flow with a small number.
If we put them at distant indices, then we will multiply this big flow with
a large number.

Objects that are far away from each other in their original space have a small
flow between each other.
If we put them at indices that are far apart, we will multiply the larger
index-difference = distance with a small number.
If we put them close, then we multiply a small distance with a small flow -
but we will have to pay for this elsewhere, because then we may need to put
objects with a larger flow between each other farther apart.

>>> from moptipyapps.order1d.instance import Instance
>>> the_instance = Instance.from_sequence_and_distance(
...     [1, 2, 3, 4], dist, 2, 100, ("bla", ), lambda i: str(i))
>>> print(the_instance.flows)
[[ 0 36 16  4]
 [25  0 25  4]
 [ 4 25  0 25]
 [ 4 16 36  0]]
>>> print(the_instance.distances)
[[0 1 2 3]
 [1 0 1 2]
 [2 1 0 1]
 [3 2 1 0]]

There is one little thing that needs to be done:
If we have many objects, then the QAP objective values can get very large.
So it makes sense to define a "horizon" after which the relationship for
objects are ignored.
Above, we chose "100", which had no impact.
If we choose "2", then only the two nearest neighbors would be considered and
we get:

>>> the_instance = Instance.from_sequence_and_distance(
...     [1, 2, 3, 4], dist, 2, 2, ("bla", ), lambda i: str(i))
>>> print(the_instance.flows)
[[ 0 16  4  0]
 [ 9  0  9  0]
 [ 0  9  0  9]
 [ 0  4 16  0]]
>>> print(the_instance.distances)
[[0 1 2 3]
 [1 0 1 2]
 [2 1 0 1]
 [3 2 1 0]]

The distance matrix of the QAP stays the same, but the flow matrix now has
several zeros.
Anything farther away than the second-nearest neighbor will be ignored, i.e.,
get a flow of `0`.
"""
