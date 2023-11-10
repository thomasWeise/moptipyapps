"""
The Quadratic Assignment Problem (QAP).

The quadratic assignment problem represents assignments of facilities to
locations. Between each pair of facilities, there is a flow of goods. Between
each two locations, there is a distance. The goal is to assign facilities to
locations such that the overall sum of the products of distance and flow gets
minimized. Each instance therefore presents a matrix with
:attr:`~moptipyapps.qap.instance.Instance.distances` and a matrix with flows
:attr:`~moptipyapps.qap.instance.Instance.flows`. The
:mod:`~moptipyapps.qap.objective` is then to minimize said product sum.

1. Eliane Maria Loiola, Nair Maria Maia de Abreu, Paulo Oswaldo
   Boaventura-Netto, Peter Hahn, and Tania Querido. A survey for the
   Quadratic Assignment Problem. European Journal of Operational Research.
   176(2):657-690. January 2007. https://doi.org/10.1016/j.ejor.2005.09.032.
2. Rainer E. Burkard, Eranda Çela, Panos M. Pardalos, and
   Leonidas S. Pitsoulis. The Quadratic Assignment Problem. In Ding-Zhu Du,
   Panos M. Pardalos, eds., Handbook of Combinatorial Optimization,
   pages 1713-1809, 1998, Springer New York, NY, USA.
   https://doi.org/10.1007/978-1-4613-0303-9_27.
"""
