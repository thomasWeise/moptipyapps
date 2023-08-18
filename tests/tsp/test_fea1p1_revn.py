"""Test the (1+1) FEA with sub-tour reversal on the TSP."""

from moptipyapps.tests.on_tsp import validate_algorithm_on_tsp
from moptipyapps.tsp.fea1p1_revn import TSPFEA1p1revn


def test_ea1p1_revn_on_tsp() -> None:
    """Test the (1+1) FEA with sub-tour reversal on the TSP."""
    validate_algorithm_on_tsp(lambda i, _: TSPFEA1p1revn(i),
                              asymmetric=False)
