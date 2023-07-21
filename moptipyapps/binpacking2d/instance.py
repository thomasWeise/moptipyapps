"""
A 2DPackLib instance.

This module provides an instance of the two-dimensional bin packing problem
as defined in 2DPackLib [1, 2].

All instances of :class:`~moptipyapps.binpacking2d.instance.Instance`
are two-dimensional numpy `ndarrays` with additional attributes.
Each instance has a :attr:`~Instance.name`. Instances also specify a
:attr:`~Instance.bin_width` and :attr:`~Instance.bin_height`.
They define the number :attr:`~Instance.n_different_items` of items with
*different* IDs. Notice that in the 2DPackLib dataset, a benchmark instance
may contain the same item multiple times. Obviously, all items of the same
ID have the exact same width and height, meaning that we only need to store
them once and remember how often they occur. (Notice that the opposite is not
true, i.e., not all items with the same width and height do have the same ID.)
Anyway, the total number :attr:`~Instance.n_items` of items, i.e., the sum of
all the repetitions of all items, is also stored.

The matrix data of the instance class is laid out as follows: There is one row
for each item. The row contains the width of the item, the height of the item,
and the number of times the item will occur. The row at index `i` stands for
the item with ID `i+1`.

Instances can be loaded directly from a 2DPackLib file via
:meth:`Instance.from_2dpacklib`. They can also be loaded from a compact string
representation (via :meth:`Instance.from_compact_str`) and can also be
converted to such a compact representation
(via :meth:`Instance.to_compact_str`). This library ships with a set of
pre-defined instances as resource which can be obtained via
:meth:`Instance.from_resource` and listed via
:meth:`Instance.list_resources`.

We provide the instances of the sets `A` [3], `BENG` [4], and `CLASS` [5].

Initial work on this code has been contributed by Mr. Rui ZHAO (赵睿),
<zr1329142665@163.com> a Master's student at the Institute of Applied
Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School of
Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei University
(合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of
Prof. Dr. Thomas Weise (汤卫思教授).

1. Manuel Iori, Vinícius Loti de Lima, Silvano Martello, and Michele Monaci.
   *2DPackLib*.
   https://site.unibo.it/operations-research/en/research/2dpacklib
2. Manuel Iori, Vinícius Loti de Lima, Silvano Martello, and Michele
   Monaci. 2DPackLib: A Two-Dimensional Cutting and Packing Library.
   *Optimization Letters* 16(2):471-480. March 2022.
   https://doi.org/10.1007/s11590-021-01808-y
3. Rita Macedo, Cláudio Alves, and José M. Valério de Carvalho. Arc-Flow
   Model for the Two-Dimensional Guillotine Cutting Stock Problem.
   *Computers & Operations Research* 37(6):991-1001. June 2010.
   https://doi.org/10.1016/j.cor.2009.08.005.
4. Bengt-Erik Bengtsson. Packing Rectangular Pieces - A Heuristic Approach.
   *The Computer Journal* 25(3):353-357, August 1982.
   https://doi.org/10.1093/comjnl/25.3.353
5. J.O. Berkey and P.Y. Wang. Two Dimensional Finite Bin Packing Algorithms.
   *Journal of the Operational Research Society* 38(5):423-429. May 1987.
   https://doi.org/10.1057/jors.1987.70

>>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 1]])
>>> ins.name
'a'
>>> ins.bin_width
100
>>> ins.bin_height
50
>>> ins.dtype
dtype('int8')
>>> ins.n_different_items
3
>>> ins.to_compact_str()
'a;3;100;50;10,5;3,3;5,5'
>>> ins = Instance("b", 100, 50, np.array([[10, 5, 1], [3, 3, 1], [3, 3, 1]]))
>>> ins.name
'b'
>>> ins.bin_width
100
>>> ins.bin_height
50
>>> ins.dtype
dtype('int8')
>>> ins.to_compact_str()
'b;3;100;50;10,5;3,3;3,3'
>>> ins = Instance.from_resource("cl02_020_06")
>>> ins.dtype
dtype('int8')
>>> ins.n_different_items
20
>>> ins.n_items
20
>>> ins = Instance.from_resource("a25")
>>> ins.dtype
dtype('int16')
>>> ins.n_different_items
75
>>> ins.n_items
156
"""

from importlib import resources  # nosem
from os.path import basename
from typing import Final, cast

import moptipy.utils.nputils as npu
import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import CSV_SEPARATOR, KeyValueLogSection
from moptipy.utils.nputils import int_range_to_dtype
from moptipy.utils.path import Path
from moptipy.utils.strings import sanitize_name
from moptipy.utils.types import check_int_range, check_to_int_range, type_error

#: the instances resource name
INSTANCES_RESOURCE: Final[str] = "2dpacklib.txt"

#: the total number of items
N_ITEMS: Final[str] = "nItems"
#: the number of different items.
N_DIFFERENT_ITEMS: Final[str] = "nDifferentItems"
#: the bin width
BIN_WIDTH: Final[str] = "binWidth"
#: the bin height
BIN_HEIGHT: Final[str] = "binHeight"
#: The internal item separator
INTERNAL_SEP: Final[str] = "," if CSV_SEPARATOR == ";" else ";"

#: the index of the width element in an item of an instance
IDX_WIDTH: Final[int] = 0
#: the index of the height element in an item of an instance
IDX_HEIGHT: Final[int] = 1
#: the index of the repetitions element in an item of an instance
IDX_REPETITION: Final[int] = 2

#: the list of instance names of the 2DPackLib bin packing set
_INSTANCES: Final[tuple[str, ...]] = (
    "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
    "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
    "a21", "a22", "a23", "a24", "a25", "a26", "a27", "a28", "a29", "a30",
    "a31", "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39", "a40",
    "a41", "a42", "a43", "beng01", "beng02", "beng03", "beng04", "beng05",
    "beng06", "beng07", "beng08", "beng09", "beng10", "cl01_020_01",
    "cl01_020_02", "cl01_020_03", "cl01_020_04", "cl01_020_05", "cl01_020_06",
    "cl01_020_07", "cl01_020_08", "cl01_020_09", "cl01_020_10", "cl01_040_01",
    "cl01_040_02", "cl01_040_03", "cl01_040_04", "cl01_040_05", "cl01_040_06",
    "cl01_040_07", "cl01_040_08", "cl01_040_09", "cl01_040_10", "cl01_060_01",
    "cl01_060_02", "cl01_060_03", "cl01_060_04", "cl01_060_05", "cl01_060_06",
    "cl01_060_07", "cl01_060_08", "cl01_060_09", "cl01_060_10", "cl01_080_01",
    "cl01_080_02", "cl01_080_03", "cl01_080_04", "cl01_080_05", "cl01_080_06",
    "cl01_080_07", "cl01_080_08", "cl01_080_09", "cl01_080_10", "cl01_100_01",
    "cl01_100_02", "cl01_100_03", "cl01_100_04", "cl01_100_05", "cl01_100_06",
    "cl01_100_07", "cl01_100_08", "cl01_100_09", "cl01_100_10", "cl02_020_01",
    "cl02_020_02", "cl02_020_03", "cl02_020_04", "cl02_020_05", "cl02_020_06",
    "cl02_020_07", "cl02_020_08", "cl02_020_09", "cl02_020_10", "cl02_040_01",
    "cl02_040_02", "cl02_040_03", "cl02_040_04", "cl02_040_05", "cl02_040_06",
    "cl02_040_07", "cl02_040_08", "cl02_040_09", "cl02_040_10", "cl02_060_01",
    "cl02_060_02", "cl02_060_03", "cl02_060_04", "cl02_060_05", "cl02_060_06",
    "cl02_060_07", "cl02_060_08", "cl02_060_09", "cl02_060_10", "cl02_080_01",
    "cl02_080_02", "cl02_080_03", "cl02_080_04", "cl02_080_05", "cl02_080_06",
    "cl02_080_07", "cl02_080_08", "cl02_080_09", "cl02_080_10", "cl02_100_01",
    "cl02_100_02", "cl02_100_03", "cl02_100_04", "cl02_100_05", "cl02_100_06",
    "cl02_100_07", "cl02_100_08", "cl02_100_09", "cl02_100_10", "cl03_020_01",
    "cl03_020_02", "cl03_020_03", "cl03_020_04", "cl03_020_05", "cl03_020_06",
    "cl03_020_07", "cl03_020_08", "cl03_020_09", "cl03_020_10", "cl03_040_01",
    "cl03_040_02", "cl03_040_03", "cl03_040_04", "cl03_040_05", "cl03_040_06",
    "cl03_040_07", "cl03_040_08", "cl03_040_09", "cl03_040_10", "cl03_060_01",
    "cl03_060_02", "cl03_060_03", "cl03_060_04", "cl03_060_05", "cl03_060_06",
    "cl03_060_07", "cl03_060_08", "cl03_060_09", "cl03_060_10", "cl03_080_01",
    "cl03_080_02", "cl03_080_03", "cl03_080_04", "cl03_080_05", "cl03_080_06",
    "cl03_080_07", "cl03_080_08", "cl03_080_09", "cl03_080_10", "cl03_100_01",
    "cl03_100_02", "cl03_100_03", "cl03_100_04", "cl03_100_05", "cl03_100_06",
    "cl03_100_07", "cl03_100_08", "cl03_100_09", "cl03_100_10", "cl04_020_01",
    "cl04_020_02", "cl04_020_03", "cl04_020_04", "cl04_020_05", "cl04_020_06",
    "cl04_020_07", "cl04_020_08", "cl04_020_09", "cl04_020_10", "cl04_040_01",
    "cl04_040_02", "cl04_040_03", "cl04_040_04", "cl04_040_05", "cl04_040_06",
    "cl04_040_07", "cl04_040_08", "cl04_040_09", "cl04_040_10", "cl04_060_01",
    "cl04_060_02", "cl04_060_03", "cl04_060_04", "cl04_060_05", "cl04_060_06",
    "cl04_060_07", "cl04_060_08", "cl04_060_09", "cl04_060_10", "cl04_080_01",
    "cl04_080_02", "cl04_080_03", "cl04_080_04", "cl04_080_05", "cl04_080_06",
    "cl04_080_07", "cl04_080_08", "cl04_080_09", "cl04_080_10", "cl04_100_01",
    "cl04_100_02", "cl04_100_03", "cl04_100_04", "cl04_100_05", "cl04_100_06",
    "cl04_100_07", "cl04_100_08", "cl04_100_09", "cl04_100_10", "cl05_020_01",
    "cl05_020_02", "cl05_020_03", "cl05_020_04", "cl05_020_05", "cl05_020_06",
    "cl05_020_07", "cl05_020_08", "cl05_020_09", "cl05_020_10", "cl05_040_01",
    "cl05_040_02", "cl05_040_03", "cl05_040_04", "cl05_040_05", "cl05_040_06",
    "cl05_040_07", "cl05_040_08", "cl05_040_09", "cl05_040_10", "cl05_060_01",
    "cl05_060_02", "cl05_060_03", "cl05_060_04", "cl05_060_05", "cl05_060_06",
    "cl05_060_07", "cl05_060_08", "cl05_060_09", "cl05_060_10", "cl05_080_01",
    "cl05_080_02", "cl05_080_03", "cl05_080_04", "cl05_080_05", "cl05_080_06",
    "cl05_080_07", "cl05_080_08", "cl05_080_09", "cl05_080_10", "cl05_100_01",
    "cl05_100_02", "cl05_100_03", "cl05_100_04", "cl05_100_05", "cl05_100_06",
    "cl05_100_07", "cl05_100_08", "cl05_100_09", "cl05_100_10", "cl06_020_01",
    "cl06_020_02", "cl06_020_03", "cl06_020_04", "cl06_020_05", "cl06_020_06",
    "cl06_020_07", "cl06_020_08", "cl06_020_09", "cl06_020_10", "cl06_040_01",
    "cl06_040_02", "cl06_040_03", "cl06_040_04", "cl06_040_05", "cl06_040_06",
    "cl06_040_07", "cl06_040_08", "cl06_040_09", "cl06_040_10", "cl06_060_01",
    "cl06_060_02", "cl06_060_03", "cl06_060_04", "cl06_060_05", "cl06_060_06",
    "cl06_060_07", "cl06_060_08", "cl06_060_09", "cl06_060_10", "cl06_080_01",
    "cl06_080_02", "cl06_080_03", "cl06_080_04", "cl06_080_05", "cl06_080_06",
    "cl06_080_07", "cl06_080_08", "cl06_080_09", "cl06_080_10", "cl06_100_01",
    "cl06_100_02", "cl06_100_03", "cl06_100_04", "cl06_100_05", "cl06_100_06",
    "cl06_100_07", "cl06_100_08", "cl06_100_09", "cl06_100_10", "cl07_020_01",
    "cl07_020_02", "cl07_020_03", "cl07_020_04", "cl07_020_05", "cl07_020_06",
    "cl07_020_07", "cl07_020_08", "cl07_020_09", "cl07_020_10", "cl07_040_01",
    "cl07_040_02", "cl07_040_03", "cl07_040_04", "cl07_040_05", "cl07_040_06",
    "cl07_040_07", "cl07_040_08", "cl07_040_09", "cl07_040_10", "cl07_060_01",
    "cl07_060_02", "cl07_060_03", "cl07_060_04", "cl07_060_05", "cl07_060_06",
    "cl07_060_07", "cl07_060_08", "cl07_060_09", "cl07_060_10", "cl07_080_01",
    "cl07_080_02", "cl07_080_03", "cl07_080_04", "cl07_080_05", "cl07_080_06",
    "cl07_080_07", "cl07_080_08", "cl07_080_09", "cl07_080_10", "cl07_100_01",
    "cl07_100_02", "cl07_100_03", "cl07_100_04", "cl07_100_05", "cl07_100_06",
    "cl07_100_07", "cl07_100_08", "cl07_100_09", "cl07_100_10", "cl08_020_01",
    "cl08_020_02", "cl08_020_03", "cl08_020_04", "cl08_020_05", "cl08_020_06",
    "cl08_020_07", "cl08_020_08", "cl08_020_09", "cl08_020_10", "cl08_040_01",
    "cl08_040_02", "cl08_040_03", "cl08_040_04", "cl08_040_05", "cl08_040_06",
    "cl08_040_07", "cl08_040_08", "cl08_040_09", "cl08_040_10", "cl08_060_01",
    "cl08_060_02", "cl08_060_03", "cl08_060_04", "cl08_060_05", "cl08_060_06",
    "cl08_060_07", "cl08_060_08", "cl08_060_09", "cl08_060_10", "cl08_080_01",
    "cl08_080_02", "cl08_080_03", "cl08_080_04", "cl08_080_05", "cl08_080_06",
    "cl08_080_07", "cl08_080_08", "cl08_080_09", "cl08_080_10", "cl08_100_01",
    "cl08_100_02", "cl08_100_03", "cl08_100_04", "cl08_100_05", "cl08_100_06",
    "cl08_100_07", "cl08_100_08", "cl08_100_09", "cl08_100_10", "cl09_020_01",
    "cl09_020_02", "cl09_020_03", "cl09_020_04", "cl09_020_05", "cl09_020_06",
    "cl09_020_07", "cl09_020_08", "cl09_020_09", "cl09_020_10", "cl09_040_01",
    "cl09_040_02", "cl09_040_03", "cl09_040_04", "cl09_040_05", "cl09_040_06",
    "cl09_040_07", "cl09_040_08", "cl09_040_09", "cl09_040_10", "cl09_060_01",
    "cl09_060_02", "cl09_060_03", "cl09_060_04", "cl09_060_05", "cl09_060_06",
    "cl09_060_07", "cl09_060_08", "cl09_060_09", "cl09_060_10", "cl09_080_01",
    "cl09_080_02", "cl09_080_03", "cl09_080_04", "cl09_080_05", "cl09_080_06",
    "cl09_080_07", "cl09_080_08", "cl09_080_09", "cl09_080_10", "cl09_100_01",
    "cl09_100_02", "cl09_100_03", "cl09_100_04", "cl09_100_05", "cl09_100_06",
    "cl09_100_07", "cl09_100_08", "cl09_100_09", "cl09_100_10", "cl10_020_01",
    "cl10_020_02", "cl10_020_03", "cl10_020_04", "cl10_020_05", "cl10_020_06",
    "cl10_020_07", "cl10_020_08", "cl10_020_09", "cl10_020_10", "cl10_040_01",
    "cl10_040_02", "cl10_040_03", "cl10_040_04", "cl10_040_05", "cl10_040_06",
    "cl10_040_07", "cl10_040_08", "cl10_040_09", "cl10_040_10", "cl10_060_01",
    "cl10_060_02", "cl10_060_03", "cl10_060_04", "cl10_060_05", "cl10_060_06",
    "cl10_060_07", "cl10_060_08", "cl10_060_09", "cl10_060_10", "cl10_080_01",
    "cl10_080_02", "cl10_080_03", "cl10_080_04", "cl10_080_05", "cl10_080_06",
    "cl10_080_07", "cl10_080_08", "cl10_080_09", "cl10_080_10", "cl10_100_01",
    "cl10_100_02", "cl10_100_03", "cl10_100_04", "cl10_100_05", "cl10_100_06",
    "cl10_100_07", "cl10_100_08", "cl10_100_09", "cl10_100_10")


def __cutsq(matrix: np.ndarray) -> list[int]:
    """
    Cut all items into squares via the CUTSQ procedure.

    :param matrix: the item matrix
    :return: the list of squares

    >>> __cutsq(np.array([[14, 12, 1]], int))
    [12, 2, 2, 2, 2, 2, 2]

    >>> __cutsq(np.array([[14, 12, 2]], int))
    [12, 12, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    """
    # create list of items in horizontal orientation
    j_sq: Final[list[int]] = []  # the list of squares (width = height)
    s: Final[list[int]] = []  # a temporary list
    for row in matrix:
        w: int = int(row[0])
        h: int = int(row[1])
        if h > w:
            w, h = h, w
        while h > 1:
            k: int = w // h
            for _ in range(k):
                s.append(h)
            w, h = h, w - (k * h)
        times: int = int(row[2])
        j_sq.extend(s * times if times > 1 else s)
        s.clear()

    j_sq.sort(reverse=True)  # sort the squares in decreasing size
    return j_sq


def __lb_q(bin_width: int, bin_height: int, q: int, j_js: list[int]) -> int:
    """
    Compute the lower bound for a given q.

    :param bin_width: the bin width
    :param bin_height: the bin height
    :param q: the parameter q
    :param j_js: the sorted square list
    :return: the lower bound

    >>> jj = [18, 18, 12, 12, 11, 11, 11, 11, 11, 7, 7, 7, 7, 7, 7]
    >>> len(jj)
    15
    >>> __lb_q(23, 20, 6, jj)
    6
    """
    m: Final[int] = len(j_js)
    half_width: Final[float] = bin_width / 2
    half_height: Final[float] = bin_height / 2
    width_m_q: Final[int] = bin_width - q

    # First we compute sets S1 to S4.
    s1: list[int] = []  # S1 from Equation 2
    s2: list[int] = []  # S2 from Equation 3
    s3: list[int] = []  # S2 from Equation 4
    s4: list[int] = []  # S2 from Equation 5

    for i in range(m):
        l_i: int = j_js[i]
        if l_i > width_m_q:
            s1.append(i)  # Equation 2
        elif l_i > half_width:
            s2.append(i)  # Equation 3
        elif l_i > half_height:
            s3.append(i)  # Equation 4
        elif l_i >= q:
            s4.append(i)  # Equation 5
        else:
            break

    # compute set S23 as in Theorem 3 under Equation 7
    height_m_q: Final[int] = bin_height - q
    s23: Final[list[int]] = [j for j in (s2 + s3) if j_js[j] > height_m_q]

    # Now we sort S2 by non-increasing value of residual space.
    s2.reverse()  # = .sort(key=lambda i: bin_width - j_js[i], reverse=True)

    # Now we compute S3 - ^S3^
    s3_minus_s3d: list[int] = s3.copy()
    for i in s2:
        residual: int = bin_width - j_js[i]
        not_found: bool = True
        for j, idx in enumerate(s3_minus_s3d):
            needs: int = j_js[idx]
            if needs <= residual:
                del s3_minus_s3d[j]
                not_found = False
                break
        if not_found:
            break

    sum_s3_l: int = sum(j_js[i] for i in s3_minus_s3d)
    b1 = sum_s3_l // bin_width
    if (b1 * bin_width) < sum_s3_l:
        b1 = b1 + 1

    len_s3: int = len(s3_minus_s3d)
    div: int = bin_width // ((bin_height // 2) + 1)
    b2 = len_s3 // div
    if (b2 * div) < len_s3:
        b2 = b2 + 1

    l_tilde: Final[int] = len(s2) + (b1 if b1 >= b2 else b2)  # Equation 6.
    bound: int = len(s1) + l_tilde

    # Now compute the final bound based on Theorem 3 / Equation 7.
    bin_size: Final[int] = bin_width * bin_height
    denom: int = sum(j_js[i] ** 2 for i in (s2 + s3 + s4)) \
        - ((bin_size * l_tilde) - sum(j_js[i] * (
            bin_height - j_js[i]) for i in s23))
    if denom > 0:
        b = denom // bin_size
        if (b * bin_size) < denom:
            b = b + 1
        bound = bound + b

    return bound


def _lower_bound_damv(bin_width: int, bin_height: int,
                      matrix: np.ndarray) -> int:
    """
    Compute the lower bound as defined by Dell'Amico et al.

    :param bin_width: the bin width
    :param bin_height: the bin height
    :param matrix: the item matrix
    :return: the lower bound

    >>> mat = np.array([[10, 5, 1], [3, 3, 1], [3, 3, 1]])
    >>> _lower_bound_damv(23, 20, mat)
    1

    >>> mat = np.array([[20, 5, 3], [13, 23, 1], [13, 9, 3]])
    >>> _lower_bound_damv(23, 20, mat)
    3
    """
    # ensure horizontal orientation (width >= height)
    if bin_height > bin_width:
        bin_width, bin_height = bin_height, bin_width
    j_sq: Final[list[int]] = __cutsq(matrix)
    return max(__lb_q(bin_width, bin_height, q, j_sq)
               for q in range(0, (bin_height // 2) + 1))


class Instance(Component, np.ndarray):
    """
    An instance of the 2D Bin Packing Problem.

    Each row of the matrix contains three values: 1. the item's width, 2. the
    item's height, 3. how often the item occurs.
    """

    #: the name of the instance
    name: str
    #: the total number of items (including repetitions)
    n_items: int
    #: the number of different items
    n_different_items: int
    #: the total area occupied by all items
    total_item_area: int
    #: the bin width
    bin_width: int
    #: the bin height
    bin_height: int
    #: the minimum number of bins that this instance requires
    lower_bound_bins: int

    def __new__(cls, name: str,
                bin_width: int, bin_height: int,
                matrix: np.ndarray | list[list[int]]) -> "Instance":
        """
        Create an instance of the 2D bin packing problem.

        :param cls: the class
        :param name: the name of the instance
        :param bin_width: the bin width
        :param bin_height: the bin height
        :param matrix: the matrix with the data (will be copied)
        """
        use_name: Final[str] = sanitize_name(name)
        if name != use_name:
            raise ValueError(f"Name {name!r} is not a valid name.")

        check_int_range(bin_width, "bin_width", 1, 1_000_000_000_000)
        check_int_range(bin_height, "bin_height", 1, 1_000_000_000_000)
        max_dim: Final[int] = max(bin_width, bin_height)
        min_dim: Final[int] = min(bin_width, bin_height)

        n_different_items: Final[int] = check_int_range(
            len(matrix), "n_different_items", 1, 100_000_000)
        use_shape: Final[tuple[int, int]] = (n_different_items, 3)

        if isinstance(matrix, np.ndarray):
            if not npu.is_np_int(matrix.dtype):
                raise ValueError(
                    "Matrix must have an integer type, but is of type "
                    f"{str(matrix.dtype)!r} in instance {name!r}.")
            if matrix.shape != use_shape:
                raise ValueError(
                    f"Invalid shape {str(matrix.shape)!r} of matrix: "
                    "must have three columns and two dimensions, must be "
                    f"equal to {use_shape} in instance {name!r}.")
        elif not isinstance(matrix, list):
            raise type_error(matrix, "matrix", np.ndarray)

        n_items: int = 0
        max_size: int = -1
        item_area: int = 0
        for i in range(n_different_items):
            row = matrix[i]
            if not isinstance(row, list | np.ndarray):
                raise type_error(
                    row, f"{row} at index {i} in {use_name!r}",
                    (list, np.ndarray))
            if len(row) != 3:
                raise ValueError(
                    f"invalid row {row} at index {i} in {use_name!r}.")
            width, height, repetitions = row
            width = check_int_range(int(width), "width", 1, max_dim)
            height = check_int_range(int(height), "height", 1, max_dim)
            repetitions = check_int_range(int(repetitions), "repetitions",
                                          1, 100_000_000)
            item_area += (width * height * repetitions)
            max_size = max(width, height)
            if (width > min_dim) and (height > min_dim):
                raise ValueError(
                    f"object with width={width} and height={height} does "
                    f"not fit into bin with width={width} and "
                    f"height={height}.")
            n_items += repetitions

        obj: Final[Instance] = super().__new__(
            cls, use_shape, int_range_to_dtype(
                min_value=0, max_value=max(max_dim + max_size, n_items),
                force_signed=True))
        for i in range(n_different_items):
            obj[i, :] = matrix[i]

        #: the name of the instance
        obj.name = use_name
        #: the number of different items
        obj.n_different_items = n_different_items
        #: the total number of items, i.e., the number of different items
        #: multiplied with their repetition counts
        obj.n_items = check_int_range(
            n_items, "n_items", n_different_items, 1_000_000_000_000)
        #: the height of the bins
        obj.bin_height = bin_height
        #: the width of the bins
        obj.bin_width = bin_width
        #: the total area occupied by all items
        obj.total_item_area = item_area

# We need at least as many bins such that their area is big enough
# for the total area of the items.
        bin_area: int = bin_height * bin_width
        lower_bound_geo: int = item_area // bin_area
        if (lower_bound_geo * bin_area) < item_area:
            lower_bound_geo += 1
        lower_bound_geo = check_int_range(
            lower_bound_geo, "lower_bound_bins_geometric",
            1, 1_000_000_000_000)

# We now compute the lower bound by Dell'Amico et al.
        lower_bound_damv = check_int_range(_lower_bound_damv(
            bin_width, bin_height, obj), "lower_bound_bins_damv",
            1, 1_000_000_000_000)

# The overall computed lower bound is the maximum of the geometric and the
# Dell'Amico lower bound.
        obj.lower_bound_bins = max(lower_bound_damv, lower_bound_geo)
        return obj

    def __str__(self):
        """
        Get the name of this instance.

        :return: the name of this instance
        """
        return self.name

    def get_standard_item_sequence(self) -> list[int]:
        """
        Get the standardized item sequence.

        :return: the standardized item sequence

        >>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 1]])
        >>> ins.get_standard_item_sequence()
        [1, 2, 3]
        >>> ins = Instance("a", 100, 50, [[10, 5, 1], [3, 3, 1], [5, 5, 2]])
        >>> ins.get_standard_item_sequence()
        [1, 2, 3, 3]
        >>> ins = Instance("a", 100, 50, [[10, 5, 2], [3, 3, 3], [5, 5, 4]])
        >>> ins.get_standard_item_sequence()
        [1, 1, 2, 2, 2, 3, 3, 3, 3]
        """
        base_string: Final[list[int]] = []
        for i in range(1, self.n_different_items + 1):
            for _ in range(self[i - 1, IDX_REPETITION]):
                base_string.append(int(i))
        return base_string

    @staticmethod
    def from_2dpacklib(file: str) -> "Instance":
        """
        Load a problem instance from the 2dpacklib format.

        :param file: the file path
        :return: the instance
        """
        path: Final[Path] = Path.file(file)
        name: str = basename(path).lower()
        if name.endswith(".ins2d"):
            name = sanitize_name(name[:-6])

        lines: Final[list[str]] = path.read_all_list()
        n_different_items: Final[int] = check_to_int_range(
            lines[0], "n_different_items", 1, 100_000_000)

        wh: Final[str] = lines[1]
        spaceidx: Final[int] = wh.index(" ")
        if spaceidx <= 0:
            raise ValueError("Did not find space in second line.")
        bin_width: Final[int] = check_to_int_range(
            wh[:spaceidx], "bin_width", 1, 1_000_000_000_000)
        bin_height: Final[int] = check_to_int_range(
            wh[spaceidx + 1:], "bin_height", 1, 1_000_000_000_000)
        del lines[0:2]

        max_dim: Final[int] = max(bin_width, bin_height)
        data: Final[list[list[int]]] = []
        old_item_id: int = 0
        for line in lines:
            text: list[str] = line.split(" ")
            itemid: int = check_to_int_range(
                text[0], "item-id", 1, n_different_items)
            if itemid != (old_item_id + 1):
                raise ValueError(
                    f"non-sequential item id {itemid} after {old_item_id}!")
            old_item_id = itemid
            width: int = check_to_int_range(text[1], "width", 1, max_dim)
            height: int = check_to_int_range(text[2], "height", 1, max_dim)
            count: int = 1 if len(text) <= 3 else \
                check_to_int_range(text[3], "count", 1, 1_000_000)
            data.append([width, height, count])
        data.sort()
        return Instance(name, bin_width, bin_height, data)

    def to_compact_str(self) -> str:
        """
        Convert the instance to a compact string.

        The format of the string is a single line of semi-colon separated
        values. The values are: `name;n_items;bin_width;bin_height`,
        followed by the sequence of items, each in the form of
        `;width,heigh[,times]`, where `times` is optional and only added
        if the item occurs more than once.

        :return: a single line string with all the instance data

        >>> ins = Instance("x", 500, 50, [[3, 5, 1], [2, 5, 2]])
        >>> ins.to_compact_str()
        'x;2;500;50;3,5;2,5,2'
        >>> ins.n_different_items
        2
        """
        lst: Final[list[str]] = [self.name, str(self.n_different_items),
                                 str(self.bin_width), str(self.bin_height)]
        for i in range(self.n_different_items):
            width: int = self[i, IDX_WIDTH]
            height: int = self[i, IDX_HEIGHT]
            repetitions: int = self[i, IDX_REPETITION]
            lst.append(
                f"{width}{INTERNAL_SEP}{height}" if repetitions == 1 else
                f"{width}{INTERNAL_SEP}{height}{INTERNAL_SEP}{repetitions}")
        return CSV_SEPARATOR.join(lst)

    @staticmethod
    def from_compact_str(data: str) -> "Instance":
        """
        Transform a compact string back to an instance.

        :param data: the string data
        :return: the instance

        >>> ins = Instance("x", 500, 50, [[3, 5, 1], [2, 5, 2]])
        >>> Instance.from_compact_str(ins.to_compact_str()).to_compact_str()
        'x;2;500;50;3,5;2,5,2'
        """
        if not isinstance(data, str):
            raise type_error(data, "data", str)
        text: Final[list[str]] = data.split(CSV_SEPARATOR)
        name: Final[str] = text[0]
        n_different_items: Final[int] = check_to_int_range(
            text[1], "n_different_items", 1, 100_000_000)
        bin_width: Final[int] = check_to_int_range(
            text[2], "bin_width", 1, 1_000_000_000_000)
        bin_height: Final[int] = check_to_int_range(
            text[3], "bin_height", 1, 1_000_000_000_000)
        max_dim: Final[int] = max(bin_width, bin_height)
        items: list[list[int]] = []
        for i in range(4, n_different_items + 4):
            s: list[str] = text[i].split(INTERNAL_SEP)
            row: list[int] = [
                check_to_int_range(s[IDX_WIDTH], "width", 1, max_dim),
                check_to_int_range(s[IDX_HEIGHT], "height", 1, max_dim),
                1 if len(s) <= IDX_REPETITION else
                check_to_int_range(
                    s[IDX_REPETITION], "times", 1, 100_000_000)]
            items.append(row)
        return Instance(name, bin_width, bin_height, items)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log the parameters describing this bin packing instance to the logger.

        :param logger: the logger for the parameters

        >>> from moptipy.utils.logger import InMemoryLogger
        >>> with InMemoryLogger() as l:
        ...     with l.key_values("I") as kv:
        ...         Instance.from_resource("beng05").log_parameters_to(kv)
        ...     print(repr('@'.join(l.get_log())))
        'BEGIN_I@name: beng05@class: moptipyapps.binpacking2d\
.instance.Instance@nItems: 100@nDifferentItems: 100@binWidth: 25\
@binHeight: 10@dtype: b@END_I'
        """
        super().log_parameters_to(logger)
        logger.key_value(N_ITEMS, self.n_items)
        logger.key_value(N_DIFFERENT_ITEMS, self.n_different_items)
        logger.key_value(BIN_WIDTH, self.bin_width)
        logger.key_value(BIN_HEIGHT, self.bin_height)
        logger.key_value(npu.KEY_NUMPY_TYPE, self.dtype.char)

    @staticmethod
    def list_resources() -> tuple[str, ...]:
        """
        Get a tuple of all the instances available as resource.

        :return: the tuple with the instance names

        >>> len(list(Instance.list_resources()))
        553
        """
        return _INSTANCES

    @staticmethod
    def from_resource(name: str) -> "Instance":
        """
        Load an instance from a resource.

        :param name: the name string
        :return: the instance

        >>> Instance.from_resource("a01").to_compact_str()
        'a01;2;2750;1220;463,386,18;1680,420,6'
        >>> Instance.from_resource("a07").to_compact_str()
        'a07;5;2750;1220;706,286,8;706,516,16;986,433,10;1120,486,\
12;1120,986,12'
        >>> Instance.from_resource("a08") is Instance.from_resource("a08")
        True
        >>> Instance.from_resource("a08") is Instance.from_resource("a09")
        False
        """
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        container: Final = Instance.from_resource
        inst_attr: Final[str] = f"__inst_{name}"
        if hasattr(container, inst_attr):  # instance loaded?
            return cast(Instance, getattr(container, inst_attr))
        text_attr: Final[str] = f"__text_{INSTANCES_RESOURCE}"
        text: list[str]
        total_attr: Final[str] = "__total_insts"
        if hasattr(container, text_attr):  # ok, we got the text already
            text = cast(list[str], getattr(container, text_attr))
        else:  # the first time we load the text
            with resources.open_text(package=str(__package__),
                                     resource=INSTANCES_RESOURCE) as stream:
                text = [line for line in (line_raw.strip() for line_raw
                                          in stream.readlines())
                        if len(line) > 0]
            setattr(container, text_attr, text)
            setattr(container, total_attr, 0)  # so far, no instance

        imax: int = len(text)
        imin: int = 0
        while imin <= imax:  # binary search for instance
            imid: int = (imax + imin) // 2
            line: str = text[imid]
            idx: int = line.index(CSV_SEPARATOR)
            prefix: str = line[0:idx]
            if prefix == name:
                instance = Instance.from_compact_str(line)
                setattr(container, inst_attr, instance)
                got: int = getattr(container, total_attr)
                got = got + 1
                if got >= len(_INSTANCES):  # got all instances, can del text
                    delattr(container, total_attr)
                    delattr(container, text_attr)
                return instance
            if prefix < name:
                imin = imid + 1
            else:
                imax = imid - 1
        raise ValueError(f"instance {name!r} not found.")
